#pragma once

#include <vector>
#include <iostream>

#include "config.h"
#include "common.cuh"
#include "error_checking.hpp"

namespace gossip {

    /** Each transfer corresponds to one copy operation between two buffers. A transfer may contain one or multiple chunks
        * which must lie on consecutive blocks of memory in the source and target buffers. The transfer_template can be used
        * to create a transfer given the chunk sizes and precalculated buffer offsets.
        */ 
    struct transfer_template {            
        public:
            const std::vector<chunk_id_t> chunks; // chunk_ids start at 0 and go to num_chunks - 1; 
            const chunk_id_t src_chunk_pos;
            const chunk_id_t trg_chunk_pos;
            const gpu_id_t src_gpu;
            const gpu_id_t trg_gpu;
            const std::vector<event_id_t> evt_ids_before; // event_ids start at 1; 0 stands for no event
            const event_id_t evt_id_after;
        
            transfer_template(
                const std::vector<chunk_id_t> chunks, // chunk_ids start at 0 and go to num_chunks - 1; 
                const chunk_id_t src_chunk_pos,
                const chunk_id_t trg_chunk_pos,
                const gpu_id_t src_gpu,
                const gpu_id_t trg_gpu,
                const std::vector<event_id_t> evt_ids_before,
                const event_id_t evt_id_after
            ) : 
                chunks(chunks),
                src_chunk_pos(src_chunk_pos),
                trg_chunk_pos(trg_chunk_pos),
                src_gpu(src_gpu),
                trg_gpu(trg_gpu),
                evt_ids_before(evt_ids_before),
                evt_id_after(evt_id_after)
            {}

            /** Given the lengths of the chunks and precalculated buffer offsets returns the transfer corresponding to the packet.
            * @param chunk_lens vector of the lengths (in number of contained elements of the underlying datatype) of each chunk
            * @param src_buf_offsets buf_offsets[i][j] Gives the offset of the j-th chunk in the buffer of gpu i
            * in the phase before the transfer occurs
            * @param trg_buf_offsets buf_offsets[i][j] Gives the offset of the j-th chunk in the buffer of gpu i
            * in the phase after the transfer occurs
            * @param events vector of events, where events[i] corresponds to the event with id i+1 
            * @return transfer corresponding to the packet
            */ 
            transfer instantiate_transfer(
                const std::vector<size_t>& chunk_lens,
                const size_t* src_buf_offsets,
                const size_t* trg_buf_offsets,
                std::vector<cudaEvent_t*>& events
            ) {
                // Calculate number of elements to be transferred 
                size_t len = 0;
                for (chunk_id_t i = 0; i < chunks.size(); i++)
                    len += chunk_lens[chunks[i]];
                // Calculate pointers to the referenced events
                cudaEvent_t* evts_before[evt_ids_before.size()];
                for (event_id_t i : evt_ids_before)
                    evts_before[i] = events[i-1];
                cudaEvent_t* evt_after;
                if (evt_id_after == 0)
                    evt_after = nullptr;
                else
                    evt_after = events[evt_id_after - 1];
                // Only the event after the transfer is created by this transfer's instantiation. The events before are created
                // by the previous transfer's instantiations
                cudaSetDevice(src_gpu);
                cudaEventCreate(evt_after);
                return transfer(
                    src_gpu, src_buf_offsets[src_chunk_pos],
                    trg_gpu, trg_buf_offsets[src_chunk_pos],
                    len, evts_before, evt_ids_before.size(), evt_after
                );
            }
    };


    class transfer_plan_t {
    private:
        std::string type_;
        gpu_id_t num_gpus_;
        const size_t num_chunks_;
        const size_t num_phases_;
        size_t num_events_;
        std::vector<std::vector<transfer_template>> templates_;
        std::vector<std::vector<std::vector<chunk_id_t>>> all_buffers_;
        std::vector<gpu_id_t> src_gpus_;
        std::vector<gpu_id_t> dst_gpus_;

    public:
        transfer_plan_t(
            const std::string type,
            const gpu_id_t num_gpus,
            const size_t num_chunks,
            const std::vector<std::vector<gpu_id_t>>& sequences,
            const std::vector<std::vector<chunk_id_t>>& positions,
            const std::vector<size_t>& transfer_sizes
        ) :
            type_(type),
            num_gpus_(num_gpus),
            num_phases_(sequences[0].size()),
            num_chunks_(num_chunks)
        {
            check(sequences.size() == num_chunks_, "transfer_plan: there must be exactly one sequence for each chunk");
            check(positions.size() == num_chunks_, "transfer_plan: there must be exactly one position for each chunk for each phase");
            check(transfer_sizes.size() != num_chunks_, "transfer_plan: there must be exactly one transfer size for each chunk");
            check(num_phases_ > 1, "transfer_plan: a transfer plan needs at least one phase");
            for (chunk_id_t c = 0; c < num_chunks_; c++) {
                check(sequences[c].size() == num_phases_ + 1, "transfer_plan: sequences must have dimension num_chunks * (num_phases + 1)");
                check(positions[c].size() == num_phases_ + 1, "transfer_plan: positions must have dimension num_chunks * (num_phases + 1)");
            }

            // Find out which chunks reside in the source and destination arrays at which time
            // in_src[c][p] (in_dst) returns whether the chunk c is in the source (destination) array after phase p 
            std::vector<std::vector<bool>> in_src(num_chunks_);
            for (chunk_id_t c = 0; c < num_chunks_; c++) {
                bool src = true;
                in_src[c].push_back(true);
                for (size_t p = 1; p <= num_phases_; p++) {
                    if (src && (sequences[c][p - 1] == sequences[c][p])) {
                        in_src[c].push_back(true);
                    } else {
                        in_src[c].push_back(false);
                        src = false;
                    }
                }
            }

            std::vector<std::vector<bool>> in_dst(0);
            for (chunk_id_t c = 0; c < num_chunks_; c++) {
                in_dst.emplace_back(num_phases_, false);
                in_dst[c][num_phases_] = true;
                bool dst = true;
                for (size_t p = num_phases_ - 1; p >= 0; p--) {
                    if (dst && (sequences[c][p + 1] == sequences[c][p])) {
                        in_dst[c][p] = true;
                    } else {
                        dst = false;
                    }
                }
            }

            // all_buffers_ will hold all contents of all buffers, source and destination arrays for every phase
            // all_buffers_[p][g][i] is the id of the chunk at position i in the source array of gpu g before phase p
            // all_buffers_[p][g + num_gpus_][i] is the id of the chunk at position i in the buffer of gpu g before phase p 
            // all_buffers_[p][g + 2 * num_gpus_][i] is the id of the chunk at position i in the destination array of gpu g before phase p
            // (last p is state after the last phase)
            for (size_t p = 0; p < num_phases_ + 1; p++) {
                all_buffers_.emplace_back(num_gpus);
                for (gpu_id_t g = 0; g < num_gpus_; g++) {
                    all_buffers_[p].emplace_back(num_chunks, -1); // Chunk indexing starts at 0; -1 is no chunk
                }
                // Construct buffer state for each phase
                for (chunk_id_t c = 0; c < num_chunks_; c++) {
                    if (in_src[c][p]) {
                        all_buffers_[p][sequences[c][p]][positions[c][p]] = c;
                    } else if (in_dst[c][p]) {
                        all_buffers_[p][sequences[c][p] + 2 * num_gpus_][positions[c][p]] = c;
                    } else {
                        all_buffers_[p][sequences[c][p] + num_gpus_][positions[c][p]] = c;
                    }
                }
            }
            
            // Inititalize event array (temporary for each phase), Event indexing starts at 1; 0 means no event
            event_id_t next_evt_id = 1;
            std::vector<event_id_t> events_after(num_chunks, 0); // events_after[i] is the event that is recorded after chunk i has been transferred

            for (size_t p = 0; p < num_phases_; p++) {
                std::vector<transfer_template> tmp = {};
                templates_.push_back(tmp);
                // Iterate through buffers before the phase to construct all transfers
                chunk_id_t s_idx, t_idx, num_added_chunks;
                gpu_id_t sgpu, tgpu;
                for (sgpu = 0; sgpu < num_gpus_ * 3; sgpu++) {
                    if (!all_buffers_[p][sgpu].empty()) {
                        s_idx = 0;
                        // Each iteration creates on transfer, loops until all chunks in the buffer have a transfer (0 as chunk_id means no chunk)
                        while (s_idx < all_buffers_[p][sgpu].size() && 0 <= all_buffers_[p][sgpu][s_idx]) {
                            chunk_id_t c_id = all_buffers_[p][sgpu][s_idx];
                            // Get target GPU and target position of transfer
                            tgpu = sequences[c_id][p];
                            t_idx = positions[c_id][p];
                            // Find out how many of the chunks that lie directly after the current chunk have the same target and adjacent position
                            num_added_chunks = 1;
                            while (
                                s_idx + num_added_chunks < all_buffers_[p][sgpu].size() && 
                                0 <= all_buffers_[p][sgpu][s_idx + num_added_chunks] &&
                                all_buffers_[p][sgpu][s_idx + num_added_chunks] == all_buffers_[p+1][tgpu][t_idx + num_added_chunks]
                            ) {
                                num_added_chunks++;
                            }
                            // Now create transfer template (possibly for multiple chunks)
                            // First determine if there is an event after and if there is an event before
                            event_id_t evt_id_after;
                            std::vector<event_id_t> evt_ids_before(0);
                            if (!in_src[c_id][p])
                                evt_ids_before.emplace_back(events_after[c_id]);
                            if (!in_dst[c_id][p])
                                evt_id_after = next_evt_id++;
                            else 
                                evt_id_after = 0;
                            // Now collect chunks, collect the events before the transfer and update events_after 
                            std::vector<chunk_id_t> chunks_together(0);
                            chunks_together.emplace_back(c_id);
                            for (chunk_id_t i = 0; i < num_added_chunks; i++) {
                                chunk_id_t nc_id = all_buffers_[p][sgpu][c_id + i];
                                chunks_together.emplace_back(nc_id);
                                if (!in_src[nc_id][p]) {
                                    // If the previous chunk came with the same transfer as the current chunk (in the previous phase)
                                    // it has the same event_before and we do not need to wait for the same event twice.
                                    if (evt_ids_before.back() != events_after[nc_id])
                                        evt_ids_before.emplace_back(events_after[nc_id]);
                                }
                                if (!in_dst[nc_id][p])
                                    events_after[nc_id] = evt_id_after;
                            }
                            templates_[p].emplace_back(
                                chunks_together,
                                s_idx,
                                t_idx,
                                sgpu,
                                tgpu,
                                evt_ids_before,
                                evt_id_after
                            );
                        }
                    }
                }
            }
            num_events_ = next_evt_id - 1;
            // Determine source and target GPUs
            for (gpu_id_t g = 0; g < num_gpus_; g++) {
                // Nonempty before first phase => source 
                if (all_buffers_[0][g].size() > 0) {
                    src_gpus_.emplace_back(g);
                }
                // Nonempty after last phase => destination 
                if (all_buffers_[num_phases_][g].size() > 0) {
                    dst_gpus_.emplace_back(g);
                }
            }
        }

        std::string type() const noexcept {
            return type_;
        }

        gpu_id_t num_gpus() const noexcept {
            return num_gpus_;
        }

        gpu_id_t num_src_gpus() const noexcept {
            return src_gpus_.size();
        }

        const std::vector<gpu_id_t> src_gpus() const noexcept {
            return src_gpus_;
        }

        gpu_id_t num_dst_gpus() const noexcept {
            return dst_gpus_.size();
        }

        const std::vector<gpu_id_t> dst_gpus() const noexcept {
            return dst_gpus_;
        }

        size_t num_phases() const noexcept {
            return num_phases_;
        }

        size_t num_chunks() const noexcept {
            return num_chunks_;
        }

        bool valid_mask(const gpu_id_t* mask) {
            gpu_id_t* tmp;
            tmp = (gpu_id_t*) malloc(num_gpus_);
            for (gpu_id_t id = 0; id < num_gpus_; id++)
                tmp[id] = 0;
            for (gpu_id_t id = 0; id < num_gpus_; id++)
                tmp[mask[id]]++;
            bool all_one = true;
            for (gpu_id_t tid = 0; tid < num_gpus_; tid++)
                all_one = all_one && (tmp[tid] == 1);
            return all_one;
        }
        
        transfer_handler instantiate_handler(
            const context_t* context,
            const std::vector<size_t>& chunk_sizes, 
            const gpu_id_t* mask = nullptr // TODO: Unused
        ) {
            /*
            // If no mask is given: assume that the gpu ids in the plan are the actual gpu ids
            if (mask == nullptr) {
                mask = (gpu_id_t*) malloc(sizeof(gpu_id_t) * num_gpus_);
                for (gpu_id_t i = 0; i < num_gpus_; i++)
                    mask[i] = i;
            } else {
                check(valid_mask(mask), "instantiate_handler: mask must be a one-to-one mapping from the GPU IDs in the plan to the real GPU IDs");
            }
            */
            // Create the event vector, actual events are created during the instantiate_transfer calls 
            std::vector<cudaEvent_t*> events(num_events_);
            // Calculate prefix sums over the buffers and at the same time determine the required buffer sizes
            std::vector<size_t> req_buf_sizes(num_gpus_);
            std::vector<std::vector<std::vector<size_t>>> all_buf_offsets(num_phases_);
            for (size_t p = 0; p < num_phases_; p++) {
                all_buf_offsets[p].emplace_back(num_gpus_);
                for (size_t g = 0; g < num_gpus_; g++) {
                    all_buf_offsets[g].emplace_back(num_chunks_);
                    size_t pref_sum = 0;
                    for (chunk_id_t c = 0; c < num_chunks_; c++) {
                        all_buf_offsets[p][g][c] = pref_sum;
                        if (all_buffers_[p][g][c] > -1)
                            pref_sum += chunk_sizes[all_buffers_[p][g][c]];
                    }
                    req_buf_sizes[g] = std::max(req_buf_sizes[g], pref_sum);
                }
            }
            // Now create the actual transfers from the templates
            std::vector<std::vector<transfer>> transfers(num_phases_ + 1);
            for (size_t p = 0; p < num_phases_; p++) {
                for (auto t : templates_[p]) {
                    t.instantiate_transfer(
                        chunk_sizes,
                        all_buf_offsets[p][t.src_gpu].data(),
                        all_buf_offsets[p][t.trg_gpu].data(),
                        events
                    );
                }
            }
            // Now create transfer handler
            return transfer_handler(context, num_phases_, num_chunks_, transfers, events, req_buf_sizes);
        }


        /*
        void show_plan() const {
            if(!valid_)
                std::cout << "ERROR: invalid plan\n";

            std::cout << "INFO: Transfer plan for " << uint32_t(num_gpus_) << " gpus\n";
            std::cout << "INFO: Transfer " << uint32_t(num_chunks_) << " chunks in " << num_steps_ << " steps\n";

            if(synchronized()) {
                std::cout << "INFO: Plan synchronizes after steps ";
                for(const auto& s : sync_steps_)
                    std::cout << s << ' ';
                std::cout << '\n';
            }
            else {
                std::cout << "INFO: Plan is without synchronization\n";
            }

            for (const auto& sequence : transfer_sequences_) {
                std::cout << "\tTransfer "
                        << sequence.size
                        << " chunks via [";
                for(const auto& item : sequence.seq)
                    std::cout << uint32_t(item) << ' ';
                std::cout << "]\n";
            }
            std::cout << std::endl;
        }
        */

    };

} // namespace

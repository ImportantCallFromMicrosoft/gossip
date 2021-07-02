#pragma once

#include <vector>
#include <iostream>
#include <common.cuh>

#include "config.h"

namespace gossip {

using chunk_id_t = unsigned short int;
using event_id_t = unsigned short int;

class transfer_plan_t {
    
    /** Each transfer corresponds to one copy operation between two buffers. A transfer may contain one or multiple chunks
    * which must lie on consecutive blocks of memory in the source and target buffers. The transfer_template can be used
    * to create a transfer given the chunk sizes and precalculated buffer offsets.
    */ 
    template<int NUM_EVTS_BEFORE>
    struct transfer_template_t {
        private: 
            const std::vector<chunk_id_t> chunks; // chunk_ids start at 0 and go to num_chunks - 1; 
            const unsigned short src_chunk_pos;
            const unsigned short trg_chunk_pos;
            const gpu_id_t src_gpu;
            const gpu_id_t trg_gpu;
            const size_t len;
            event_id_t evt_ids_before[NUM_EVTS_BEFORE]; // event_ids start at 1; 0 stands for no event
            event_id_t evt_id_after;
            
        public:
            /** Given the lengths of the chunks and precalculated buffer offsets returns the transfer corresponding to the packet.
            * @param chunk_lens vector of the lengths (in bytes??) of each chunk
            * @param src_buf_offsets buf_offsets[i][j] Gives the offset (in bytes??) of the j-th chunk in the buffer of gpu i
            * in the phase before the transfer occurs
            * @param trg_buf_offsets buf_offsets[i][j] Gives the offset (in bytes??) of the j-th chunk in the buffer of gpu i
            * in the phase after the transfer occurs
            * @param events vector of events, where events[i] corresponds to the event with id i+1 
            * @return transfer corresponding to the packet
             */ 
            transfer instantiate_transfer(
                const std::vector<size_t>& chunk_lens,
                const std::vector<std::vector<size_t>> src_buf_offsets,
                const std::vector<std::vector<size_t>> trg_buf_offsets,
                const cudaEvent_t* events
                ) {
                if (evts_before.size() != NUM_EVTS_BEFORE)
                    throw std::invalid_argument("instantiate_transfer: evts_before didn't contain the expected number of events")
                // Calculate number of bytes to be transferred 
                size_t len = 0;
                for (chunk_pos_t i = 0; i < chunks.size(); i++)
                    len += chunk_lens[chunks[i]];
                // Calculate pointers to the referenced events
                std::vector<cudaEvent_t*> evts_before(NUM_EVTS_BEFORE);
                for (event_id_t i = 0; i < NUM_EVTS_BEFORE)
                    evts_before[i] = events + (evt_ids_before[i] - 1);
                if (evt_id_after)
                    evt_after = events + (evt_id_after - 1)
                // Only the event after the transfer is created by this instantiation. The events before are created by the previous 
                // transfer's instantiations
                cudaEventCreate(evt_after);
                return transfer(
                    src_gpu, src_buf_offsets[src_chunk_pos],
                    trg_gpu, trg_buf_offsets[src_chunk_pos],
                    len, evts_before, evt_after
                );
            }
    };

    std::string type_;
    gpu_id_t num_gpus_;
    gpu_id_t main_gpu_;
    size_t num_steps_;
    size_t num_chunks_;
    size_t num_events_;
    std::vector<std::vector<transfer_template_t>> templates_;
    std::vector<size_t> sync_steps_;
    bool valid_;

public:
    transfer_plan_t(
        const std::string type,
        const gpu_id_t num_gpus,
        const std::vector<std::vector<gpu_id_t>>& sequences
    ) :
        type_(type),
        num_gpus_(num_gpus),
        main_gpu_(gpu_id_t(-1)),
        num_steps_(0),
        num_chunks_(1),
        valid_(false)
    {
        if(sequences.size())
            num_steps_ = sequences[0].size()-1;
        transfer_sequences_.reserve(sequences.size());
        for(const auto& sequence : sequences)
            transfer_sequences_.push_back({sequence, 1});
    }

    transfer_plan_t(
        const std::string type,
        const gpu_id_t num_gpus,
        const std::vector<std::vector<gpu_id_t>>& sequences,
        const std::vector<std::vector<gpu_id_t>>& positions,
        const size_t num_chunks,
        const std::vector<size_t>& transfer_sizes
    ) :
        type_(type),
        num_gpus_(num_gpus),
        main_gpu_(gpu_id_t(-1)),
        num_steps_(0),
        num_chunks_(num_chunks),
        valid_(false)
    {

        if(sequences.size() != num_chunks)
            throw std::invalid_argument("transfer_plan: there must be exactly one sequence for each chunk")
        if(positions.size() != num_chunks)
            throw std::invalid_argument("transfer_plan: there must be exactly one position for each chunk for each phase")
        if(transfer_sizes.size() != num_chunks)
            throw std::invalid_argument("transfer_plan: there must be exactly one transfer size for each chunk")
        // We assume that sequences and positions have dimension num_chunks * (num_phases - 1)
        if (sequences[0].size() < 2)
            throw std::invalid_argument("transfer_plan: every transfer plan needs at least one phase")
        
        size_t num_phases = sequences[0].size() - 1;

        // These vectors are used to represent the states of all buffers before and after a phase
        std::vector<std::vector<int>> buffers_before(num_gpus); // buffers_before[i][j] is the number of the chunk at position j on gpu i
        std::vector<std::vector<int>> buffers_after(num_gpus);
        for (gpu_id_t i = 0; i < num_gpus; i++) {
            buffers_before[i] = std::vector<chunk_id_t>(num_chunks);
            buffers_after[i] = std::vector<chunk_id_t>(num_chunks)
        }
        std::fill(buffers_before.begin(), buffers_before.end(), -1); // Chunk indexing starts at 0; -1 is no chunk
        std::fill(buffers_after.begin(), buffers_after.end(), -1);
        // Initialize buffers_before with initial positions of the chunks
        for (chunk_id_t c = 0; c < num_chunks_; c++)
            buffers_before[sequences[c][p]][positions[c][p]] = c;
        
        event_id_t next_evt_id = 1;
        std::vector<event_id_t> events_before(num_chunks); // events_before[i] is the event that chunk i needs to wait for before transfer
        std::vector<event_id_t> events_after(num_chunks); // events_after[i] is the event that is recorded after chunk i has been transferred
        std::fill(events_before.begin(), buffers_before.end(), 0); // indexing starts at 1; 0 means no event
        std::fill(events_after.begin(), buffers_after.end(), 0);

        for (size_t p = 0; p < num_phases; p++) {
            // Construct buffer state after the phase
            for (chunk_id_t c = 0; c < num_chunks_; c++)
                buffers_after[sequences[c][p]][positions[c][p]] = c;
            // Now iterate through buffers_before to construct all transfers
            chunk_id_t s_idx, t_idx, num_chunks_together;
            gpu_id_t sgpu, tgpu;
            for (sgpu = 0; sgpu < num_gpus; sgpu++) {
                if (!buffers_before[sgpu].empty()) {
                    s_idx = 0;
                    while (buffers_before[sgpu][c_idx] > -1) {
                        // Get target GPU and target position of transfer
                        tgpu = sequences[buffers_before[sgpu][s_idx]][p];
                        t_idx = positions[buffers_before[sgpu][s_idx]][p];
                        // Find out how many of the chunks that lie directly after the current chunk has the same target and adjacent position
                        num_chunks_together = 1;
                        while (buffers_before[sgpu][s_idx + num_chunks_together] == buffers_after[tgpu][t_idx + num_chunks_together]) {
                            num_chunks_together++;
                        }
                        // Now create transfer template (possibly for multiple chunks)
                        // First determine if there is an event after
                        event_id_t event_id_after = 0;
                        if (p < num_phases - 1)
                            event_id_after = next_evt_id++;
                        // Now collect chunks, collect the events before the transfer and update events_after 
                        std::vector<chunk_id_t> chunks_together(num_chunks_together);
                        std::vector<event_id_t> evt_ids_before();
                        for (chunk_id_t i = 0; i < num_chunks_together; i++) {
                            chunk_id_t c = buffers_before[sgpu][c_idx + i];
                            chunks_together[i] = c;
                            events_after[c] = evt_id_after;
                            // If the previous chunk came with the same transfer as the current chunk (in the previous phase)
                            // it has the same event_before and we do not need to wait for the same event twice. We only add
                            // event_before if this is not the case.
                            if (i == 0 || evt_ids_before.back() != events_before[c])
                                evt_ids_before.emplace_back(events_before[c]);
                        }
                        templates_.emplace_back(
                            chunks_together,
                            s_idx,
                            t_idx,
                            sgpu,
                            tgpu,
                            evt_ids_before,
                            evt_id_after
                        )
                    }
                }
            }
            // after -> before; 0 -> after
            std::swap(buffers_after, buffers_before);
            std::fill(buffers_after.begin(), buffers_after.end(), -1);
            std::swap(events_after, events_before);
            std::fill(events_after.begin(), events_after.end(), 0);
        }
        num_events_ = next_evt_id - 1;
        // TODO: Source GPUs and target GPUs
    }

    transfer_plan_t(
        const std::string type,
        const gpu_id_t num_gpus,
        const std::vector<std::vector<gpu_id_t>>& sequences,
        const std::vector<std::vector<gpu_id_t>>& positions,
        const size_t num_chunks,
        const std::vector<size_t>& transfer_sizes
    ) :
        type_(type),
        num_gpus_(num_gpus),
        main_gpu_(gpu_id_t(-1)),
        num_steps_(0),
        num_chunks_(num_chunks),
        valid_(false)
    {
        if(sequences.size() == transfer_sizes.size()) {
            if(sequences.size())
                num_steps_ = sequences[0].size()-1;

            transfer_sequences_.reserve(sequences.size());

            for(size_t i = 0; i < sequences.size(); ++i)
                transfer_sequences_.push_back({sequences[i], transfer_sizes[i]});
        }
    }

public:
    std::string type() const noexcept {
        return type_;
    }

    gpu_id_t num_gpus() const noexcept {
        return num_gpus_;
    }

    gpu_id_t main_gpu() const noexcept {
        return main_gpu_;
    }

    void main_gpu(const gpu_id_t gpu) {
        main_gpu_ = gpu;
    }

    size_t num_steps() const noexcept {
        return num_steps_;
    }

    size_t num_chunks() const noexcept {
        return num_chunks_;
    }

    const std::vector<transfer_sequence>& transfer_sequences() const {
        return transfer_sequences_;
    }

    const std::vector<size_t>& sync_steps() {
        return sync_steps_;
    }

    void sync_steps(const std::vector<size_t>& steps) {
        sync_steps_ = steps;
    }

    bool synchronized() const noexcept {
        return sync_steps_.size() > 0;
    }

    bool valid() const noexcept {
        return valid_;
    }

    void validate() {
        valid_ = true;
    }

    void invalidate() {
        valid_ = false;
    }

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

};

} // namespace

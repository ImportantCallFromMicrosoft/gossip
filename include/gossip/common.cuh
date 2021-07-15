#pragma once

#include <iostream>

#include "config.h"
#include "context.cuh"

namespace gossip {
    // shared between scatter, gather, all_to_all_async
>
    struct transfer {
        const gpu_id_t src_gpu;
        const size_t src_pos;
        const gpu_id_t trg_gpu;
        const size_t trg_pos;
        const size_t len;
        const std::vector<cudaEvent_t*> events_before;
        const cudaEvent_t* event_after;

        transfer(const gpu_id_t src_gpu,
                 const size_t src_pos,
                 const gpu_id_t trg_gpu,
                 const size_t trg_pos,
                 const size_t len,
                 const std::vector<cudaEvent_t*>& events_before,
                 const cudaEvent_t* event_after = nullptr) :
            src_gpu(src_gpu),
            src_pos(src_pos),
            trg_gpu(trg_gpu),
            trg_pos(trg_pos),
            len(len),
            event_before(event_before),
            event_after(event_after)
        {}

        void show() const {
            std::string evts_before_str = "{";
            for (ptr : events_before)
                evts_before_str += to_string(e) + ",";
            evts_before_str += "}"
            std::cout <<   "src:" << int(src_gpu)
                      << ", pos:" << src_pos
                      << ", trg:" << int(trg_gpu)
                      << ", pos:" << trg_pos
                      << ", len:" << len
                      << ", events before:" << evts_before_str
                      << ", event after:" << (event_after ? event_after : 0)
                      << std::endl;
        }
    };

    struct transfer_handler {
        
        const context_t * context_;
        const size_t num_phases_;
        const size_t num_chunks_;
        std::vector<std::vector<transfer>> phases_;
        const std::vector<cudaEvent_t*> events_;
        const std::vector<size_t> buffer_sizes_;

        transfer_handler<size_t>(
            const context_t * context,
            const size_t num_phases,
            const size_t num_chunks,
            const std::vector<std::vector<transfer>>& phases,
            const std::vector<cudaEvent_t*> events; 
            const std::vector<size_t>& buffer_sizes
        ) :
            context_(context),
            num_phases_(num_phases),
            num_chunks_(num_chunks),
            phases_(phases),
            events_(events),
            buffer_sizes_(buffer_sizes_)

        {}

        transfer_handler(const transfer_handler&) = delete;
        transfer_handler(transfer_handler&&) = default;

        ~transfer_handler() {
            for(auto& e : events_)
                cudaEventDestroy(*e);
        }

        const std::vector<size_t> buffer_size() {
            return buffer_sizes_;
        }

        void show_phase(const size_t phase) const {
            for(const transfer& t : phases[phase]) {
                t.show();
            }
        }

        template<typename value_t>
        bool execute_phase(
            const size_t phase,
            const std::vector<value_t *>& srcs,
            const std::vector<value_t *>& dsts,
            const std::vector<value_t *>& bufs
        ) const {
            for (const transfer& t : phases[phase]) {
                const gpu_id_t src = context->get_device_id(t.src_gpu);
                const gpu_id_t trg = context->get_device_id(t.trg_gpu);
                const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
                cudaSetDevice(src);
                const size_t size = t.len * sizeof(value_t);
                value_t * from = (t.event_before == nullptr) ?
                                srcs[t.src_gpu] + t.src_pos :
                                bufs[t.src_gpu] + t.src_pos;
                value_t * to   = (t.event_after == nullptr) ?
                                dsts[t.trg_gpu] + t.trg_pos :
                                bufs[t.trg_gpu] + t.trg_pos;

                for (eb : t.events_before)
                    cudaStreamWaitEvent(stream, *(eb), 0);
                cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
                if(t.event_after != nullptr) 
                    cudaEventRecord(*(t.event_after), stream);
            } CUERR

            return true;
        }
    };

} // namespace

#pragma once

#include "config.h"
#include "error_checking.hpp"
#include "transfer_plan.hpp"
#include "context.cuh"

namespace gossip {

class scatter_t {

private:
    const context_t* context_;
    const transfer_plan_t* transfer_plan_;
    transfer_handler* curr_handler_;
    bool handler_valid_;

public:
    scatter_t (
        const context_t& context,
        const transfer_plan_t& transfer_plan
    ) : 
        context_(&context),
        transfer_plan_(&transfer_plan),
        handler_valid_(false)
    {
        std::string error_msg;
        // Minimalistic checks of the transfer plan
        check(context_->is_valid(), "You have to pass a valid context!");
        error_msg = std::string("scatter: transfer_plan must have exactly one source GPU not ");
        error_msg += std::to_string(transfer_plan_->num_src_gpus());

        check(transfer_plan_->num_src_gpus() == 1, error_msg.data());
        error_msg = std::string("scatter: transfer_plan must have as many destinations as GPUs not ");
        error_msg += std::to_string(transfer_plan_->num_dst_gpus());
        
        check(transfer_plan_->num_dst_gpus() == transfer_plan_->num_gpus(), error_msg.data());
        check(transfer_plan_->num_gpus() == context_->get_num_devices(), "scatter: transfer_plan must fit number of gpus of context");
    }

public:
    const std::vector<size_t> calcBufferLengths(const std::vector<size_t>& chunk_sizes) {
        if (handler_valid_) {
            return curr_handler_->buffer_sizes();
        } else {
            curr_handler_ = transfer_plan_->instantiate_handler(context_, chunk_sizes);
            handler_valid_ = true;
            return curr_handler_->buffer_sizes();
        }

    };

    /**
     * Execute scatter asynchronously using the given context.
     * The lenghts of the parameters have to match the context.
     * @param src pointer to source array. should reside on device_ids[main_gpu].
     * @param src_len src_len is length of src array.
     * @param dsts pointers to destination arrays. dsts[k] array should reside on device_ids[k].
     * @param dsts_len dsts_len[k] is length of dsts[k] array.
     * @param bufs pointers to buffer arrays. bufs[k] array should reside on device_ids[k].
     * @param bufs_len bufs_len[k] is length of bufs[k] array.
     * @param send_counts send_counts[k] elements are sent to device_ids[k]
     * @param verbose if true, show details for each transfer.
     * @return true if executed successfully.
     */
    template <
        typename value_t,
        typename index_t>
    bool execAsync (
        value_t * src,
        const index_t src_len,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& dsts_lens,
        const std::vector<value_t *>& bufs,
        const std::vector<index_t  >& bufs_lens,
        const std::vector<size_t  >& chunk_sizes,
        bool verbose = false
    ) {
        std::cout << "scatter.cuh: execAsync" << std::endl; 
        if (!check(dsts.size() == get_num_devices(),
        "dsts size does not match number of gpus."))
        return false;
        if (!check(dsts_lens.size() == get_num_devices(),
        "dsts_lens size does not match number of gpus."))
        return false;
        if (!check(bufs.size() == get_num_devices(),
        "bufs size does not match number of gpus."))
        return false;
        if (!check(bufs_lens.size() == get_num_devices(),
        "bufs_lens size does not match number of gpus."))
        return false;
        if (!check(chunk_sizes.size() == transfer_plan_->num_chunks(),
        "number of chunk sizes does not match number of gpus."))
        return false;
        
        std::cout << "scatter.cuh: Instantiate Handler" << std::endl; 
        if (!handler_valid_) {
            curr_handler_ = transfer_plan_->instantiate_handler(context_, chunk_sizes);
            handler_valid_ = true;
        } // TODO: Else check that the buffer sizes are correct for the given handler
        
        std::vector<value_t *> srcs(get_num_devices(), nullptr);
        srcs[transfer_plan_->src_gpus()[0]] = src;
        
        std::cout << "scatter.cuh: Executing phases" << std::endl; 
        for (size_t p = 0; p < curr_handler_->num_phases(); ++p)
            curr_handler_->execute_phase(p, srcs, dsts, bufs);
        
        return true;
    }

    gpu_id_t get_num_devices () const noexcept {
        return context_->get_num_devices();
    }

    void sync () const noexcept {
        context_->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context_->sync_hard();
    }

    const context_t& get_context() const noexcept {
        return *context_;
    }
};

} // namespace

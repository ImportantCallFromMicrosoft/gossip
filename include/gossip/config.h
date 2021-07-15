# pragma once


#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda.h>
// TODO: CUERR wont work
// #include "../hpc_helpers/include/cuda_helpers.cuh"

#define THROW_EXCEPTIONS 1

namespace gossip {

    using gpu_id_t = uint16_t;
    using chunk_id_t = int;
    using event_id_t = uint16_t;
    // type for offsets in bytes or numbers of elements
    using index_t = uint64_t;
    // type of multisplit counters
    using cnter_t = uint64_t;

    enum class PEER_STATUS : uint8_t {
        SLOW = 0,
        DIAG = 1,
        FAST = 2
    };

} // namespace

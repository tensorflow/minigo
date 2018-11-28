#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include <memory>

#include "cc/dual_net/dual_net.h"

namespace minigo {

// Creates a factory for DualNets which batch inference requests and forwards
// them to DualNet instances created by the impl factory.
//
// Inference requests sent to DualNet instances created from the returned
// factory may block until *all* instances have received an inference request.
std::unique_ptr<DualNetFactory> NewBatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl, size_t batch_size);

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_

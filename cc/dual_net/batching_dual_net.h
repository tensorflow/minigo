#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include <memory>

#include "cc/dual_net/dual_net.h"

namespace minigo {

std::unique_ptr<DualNetFactory> NewBatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl);

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_

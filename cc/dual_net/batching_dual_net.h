#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include <memory>

#include "cc/dual_net/factory.h"

namespace minigo {

class DualNetFactory {
 public:
  virtual ~DualNetFactory();

  virtual std::unique_ptr<DualNet> New() = 0;
};

// Creates a factory for DualNets which batch inference requests and forwards
// them to the dual_net.
//
// Inference requests sent to DualNet instances created from the returned
// factory may block until *all* instances have received an inference request.
std::unique_ptr<DualNetFactory> NewBatchingFactory(
    std::unique_ptr<DualNet> dual_net, size_t batch_size);

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_

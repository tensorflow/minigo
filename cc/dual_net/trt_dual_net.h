#ifndef CC_DUAL_NET_TRT_DUAL_NET_H_
#define CC_DUAL_NET_TRT_DUAL_NET_H_

#include "cc/dual_net/dual_net.h"

namespace minigo {

class TrtDualNetFactory : public DualNetFactory {
 public:
  TrtDualNetFactory();

  int GetBufferCount() const override;

  std::unique_ptr<DualNet> NewDualNet(const std::string& model) override;

 private:
  int device_count_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_TRT_DUAL_NET_H_

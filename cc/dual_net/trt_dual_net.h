#ifndef CC_DUAL_NET_TRT_DUAL_NET_H_
#define CC_DUAL_NET_TRT_DUAL_NET_H_

#include "cc/dual_net/dual_net.h"

namespace minigo {

std::unique_ptr<DualNet> NewTrtDualNet(const std::string& model_path);

}  // namespace minigo

#endif  // CC_DUAL_NET_TRT_DUAL_NET_H_

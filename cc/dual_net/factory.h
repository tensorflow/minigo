#ifndef MINIGO_CC_DUAL_NET_FACTORY_H_
#define MINIGO_CC_DUAL_NET_FACTORY_H_

#include <memory>
#include <string>
#include <utility>

#include "cc/dual_net/dual_net.h"

namespace minigo {

class DualNetFactory {
 public:
  explicit DualNetFactory(std::string model_path)
      : model_path_(std::move(model_path)) {}
  virtual ~DualNetFactory();
  virtual std::unique_ptr<DualNet> New() = 0;

  const std::string& model() const { return model_path_; }

 private:
  const std::string model_path_;
};

std::unique_ptr<DualNetFactory> NewDualNetFactory(std::string model_path);

}  // namespace minigo

#endif  // MINIGO_CC_DUAL_NET_FACTORY_H_

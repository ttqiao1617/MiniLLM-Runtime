#pragma once

#include "execution_backend.h"

#include <memory>

namespace Ort {
class Env;
class Session;
} // namespace Ort

namespace minillm {

class OrtBackend : public ExecutionBackend {
public:
    OrtBackend();
    ~OrtBackend() override;

    void SetModelSpec(const ModelSpec& spec) override;
    void Load(const std::string& model_path) override;
    ForwardOutputs Forward(const ForwardInputs& inputs) override;
    std::string Name() const override;

private:
    std::string model_path_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

} // namespace minillm

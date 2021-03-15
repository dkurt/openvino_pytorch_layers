// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ngraph/ngraph.hpp>

namespace TemplateExtension {

//! [cpu_implementation:header]
class UnpoolImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit UnpoolImpl(const std::shared_ptr<ngraph::Node>& node);
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                           InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig &config,
                                     InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                        std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override;
private:
    std::vector<ngraph::Shape> inShapes;
    ngraph::Shape outShape;
    std::string error;
};

class FFTImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit FFTImpl(const std::shared_ptr<ngraph::Node>& node);
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                           InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig &config,
                                     InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                        std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override;
private:
    ngraph::Shape inpShape;
    ngraph::Shape outShape;
    bool inverse;
    std::string error;
};

class GridSampleImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit GridSampleImpl(const std::shared_ptr<ngraph::Node>& node);
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                           InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig &config,
                                     InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                        std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override;
private:
    std::vector<ngraph::Shape> inShapes;
    ngraph::Shape outShape;
    std::string error;
};
//! [cpu_implementation:header]

}  // namespace TemplateExtension

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "extension.hpp"
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <ngraph/factory.hpp>
#include <ngraph/opsets/opset.hpp>
#include <onnx_import/onnx_utils.hpp>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace TemplateExtension;

Extension::Extension() {
    ngraph::onnx_import::register_operator(UnpoolOp::type_info.name, 1, "extension", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
        ngraph::OutputVector ng_inputs {node.get_ng_inputs()};
        return {std::make_shared<UnpoolOp>(ng_inputs.at(0), ng_inputs.at(1),
                                           ng_inputs.at(2), ng_inputs.at(3))};
    });
    ngraph::onnx_import::register_operator(FFTOp::type_info.name, 1, "extension", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
        ngraph::OutputVector ng_inputs {node.get_ng_inputs()};
        bool inverse = node.get_attribute_value<int64_t>("inverse");
        bool centered = node.get_attribute_value<int64_t>("centered");
        return {std::make_shared<FFTOp>(ng_inputs.at(0), inverse, centered)};
    });
    ngraph::onnx_import::register_operator(IFFTOp::type_info.name, 1, "extension", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
        ngraph::OutputVector ng_inputs {node.get_ng_inputs()};
        bool inverse = node.get_attribute_value<int64_t>("inverse");
        bool centered = node.get_attribute_value<int64_t>("centered");
        return {std::make_shared<IFFTOp>(ng_inputs.at(0), inverse, centered)};
    });
    ngraph::onnx_import::register_operator(ComplexMulOp::type_info.name, 1, "extension", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
        ngraph::OutputVector ng_inputs {node.get_ng_inputs()};
        return {std::make_shared<ComplexMulOp>(ng_inputs.at(0), ng_inputs.at(1))};
    });
    ngraph::onnx_import::register_operator(GridSampleOp::type_info.name, 1, "extension", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
        ngraph::OutputVector ng_inputs {node.get_ng_inputs()};
        return {std::make_shared<GridSampleOp>(ng_inputs.at(0), ng_inputs.at(1))};
    });
}

Extension::~Extension() {
    ngraph::onnx_import::unregister_operator(UnpoolOp::type_info.name, 1, "extension");
    ngraph::onnx_import::unregister_operator(FFTOp::type_info.name, 1, "extension");
    ngraph::onnx_import::unregister_operator(IFFTOp::type_info.name, 1, "extension");
    ngraph::onnx_import::unregister_operator(ComplexMulOp::type_info.name, 1, "extension");
    ngraph::onnx_import::unregister_operator(GridSampleOp::type_info.name, 1, "extension");
}

//! [extension:GetVersion]
void Extension::GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
        {1, 0},           // extension API version
        "1.0",
        "template_ext"    // extension description message
    };

    versionInfo = &ExtensionDescription;
}
//! [extension:GetVersion]

//! [extension:getOpSets]
std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    opset.insert<UnpoolOp>();
    opset.insert<FFTOp>();
    opset.insert<IFFTOp>();
    opset.insert<ComplexMulOp>();
    opset.insert<GridSampleOp>();
    opsets["extension"] = opset;
    return opsets;
}
//! [extension:getOpSets]

//! [extension:getImplTypes]
std::vector<std::string> Extension::getImplTypes(const std::shared_ptr<ngraph::Node> &node) {
    if (std::dynamic_pointer_cast<UnpoolOp>(node) ||
        std::dynamic_pointer_cast<ComplexMulOp>(node) ||
        std::dynamic_pointer_cast<GridSampleOp>(node) ||
        std::dynamic_pointer_cast<IFFTOp>(node) ||
        std::dynamic_pointer_cast<FFTOp>(node)) {
        return {"CPU"};
    }
    return {};
}
//! [extension:getImplTypes]

//! [extension:getImplementation]
InferenceEngine::ILayerImpl::Ptr Extension::getImplementation(const std::shared_ptr<ngraph::Node> &node, const std::string &implType) {
    if (std::dynamic_pointer_cast<UnpoolOp>(node) && implType == "CPU") {
        return std::make_shared<UnpoolImpl>(node);
    }
    if ((std::dynamic_pointer_cast<FFTOp>(node) || std::dynamic_pointer_cast<IFFTOp>(node)) && implType == "CPU") {
        return std::make_shared<FFTImpl>(node);
    }
    if (std::dynamic_pointer_cast<GridSampleOp>(node) && implType == "CPU") {
        return std::make_shared<GridSampleImpl>(node);
    }
    if (std::dynamic_pointer_cast<ComplexMulOp>(node) && implType == "CPU") {
        return std::make_shared<ComplexMulImpl>(node);
    }
    return nullptr;
}
//! [extension:getImplementation]

//! [extension:CreateExtension]
// Exported function
INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateExtension(InferenceEngine::IExtension *&ext,
                                                                                      InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        ext = new Extension();
        return OK;
    } catch (std::exception &ex) {
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}
//! [extension:CreateExtension]

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#include "ie_parallel.hpp"

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
CalculateGridImpl::CalculateGridImpl(const std::shared_ptr<ngraph::Node> &node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<CalculateGridOp>(node);
        if (!castedNode)
            THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
        if (castedNode->inputs().size() != 1 || castedNode->outputs().size() != 1)
            THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
        if (castedNode->get_input_shape(0).size() != 2 || castedNode->get_output_shape(0).size() != 2)
            THROW_IE_EXCEPTION << "Operation supports only 4d tensors for input and output.";
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
            THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
        inShapes.push_back(castedNode->get_input_shape(0));
        outShape = castedNode->get_output_shape(0);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        error = ex.what();
    }

}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode CalculateGridImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                       InferenceEngine::ResponseDesc *resp) noexcept {
    std::vector<InferenceEngine::DataConfig> inDataConfig;
    std::vector<InferenceEngine::DataConfig> outDataConfig;
    // Allow any offset before data
    size_t offset((std::numeric_limits<size_t>::max)());

    // Input shape
    for (const auto& shape : inShapes)
    {
        InferenceEngine::SizeVector order(shape.size());
        std::iota(order.begin(), order.end(), 0);

        InferenceEngine::DataConfig inpConf;
        inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});
        inDataConfig.push_back(inpConf);
    }

    // Output shape
    InferenceEngine::SizeVector order(outShape.size());
    std::iota(order.begin(), order.end(), 0);

    InferenceEngine::DataConfig outConf;
    outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
    outDataConfig.push_back(outConf);

    InferenceEngine::LayerConfig layerConfig;
    layerConfig.inConfs = inDataConfig;
    layerConfig.outConfs = outDataConfig;

    conf.push_back(layerConfig);
    return InferenceEngine::StatusCode::OK;
}
//! [cpu_implementation:getSupportedConfigurations]

//! [cpu_implementation:init]
InferenceEngine::StatusCode CalculateGridImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1) {
            THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.inConfs[0].desc.getDims().size() != 2 || config.outConfs[0].desc.getDims().size() != 2) {
            THROW_IE_EXCEPTION << "Operation can be initialized only with 2d input/output tensors!";
        }

        if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
            config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32)  {
            THROW_IE_EXCEPTION << "Operation supports only FP32 precisions!";
        }
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg)-1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }

    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

//! [cpu_implementation:execute]
InferenceEngine::StatusCode CalculateGridImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                       std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                       InferenceEngine::ResponseDesc *resp) noexcept {
    const float* inpPos = inputs[0]->cbuffer().as<float*>();
    float* out = outputs[0]->buffer().as<float*>();

    std::set<std::tuple<int, int, int> > outPos;

    const size_t numPoints = inputs[0]->getTensorDesc().getDims()[0];
    static const std::vector<std::vector<int> > filters {{-1, -1, -1}, {-1, -1, 0}, {-1, 0, -1},
                                                         {-1, 0, 0}, {0, -1, -1}, {0, -1, 0},
                                                         {0, 0, -1}, {0, 0, 0}};

    std::vector<int> pos(3);
    for (size_t i = 0; i < numPoints; ++i) {
        for (size_t j = 0; j < filters.size(); ++j) {
            bool isValid = true;
            for (size_t k = 0; k < 3; ++k) {
                int val = static_cast<int>(inpPos[i * 3 + k]) + filters[j][k];
                if (val < 0 || val % 2) {
                    isValid = false;
                    break;
                }
                pos[k] = val;
            }
            if (isValid)
                outPos.insert(std::make_tuple(pos[0], pos[1], pos[2]));
        }
    }

    int i = 0;
    for (const auto it : outPos) {
        out[i * 3] = 0.5f + std::get<0>(it);
        out[i * 3 + 1] = 0.5f + std::get<1>(it);
        out[i * 3 + 2] = 0.5f + std::get<2>(it);
        i += 1;
    }
    memset(out + i * 3, 0, sizeof(float) * 3 * (numPoints - i));
    out[i * 3] = -1.0f;
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]

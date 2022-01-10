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
SparseConvImpl::SparseConvImpl(const std::shared_ptr<ngraph::Node> &node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<SparseConvOp>(node);
        if (!castedNode)
            THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
        if (castedNode->inputs().size() != 4 || castedNode->outputs().size() != 1)
            THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
        if (castedNode->get_input_shape(0).size() != 2 || castedNode->get_output_shape(0).size() != 2)
            THROW_IE_EXCEPTION << "Operation supports only 4d tensors for input and output.";
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
            THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
        inShapes.resize(4);
        for (int i = 0; i < 4; ++i)
            inShapes[i] = castedNode->get_input_shape(i);
        outShape = castedNode->get_output_shape(0);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        error = ex.what();
    }

}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode SparseConvImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
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
InferenceEngine::StatusCode SparseConvImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        if (config.inConfs.size() != 4 || config.outConfs.size() != 1) {
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
InferenceEngine::StatusCode SparseConvImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                    std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                    InferenceEngine::ResponseDesc *resp) noexcept {
    const float* features = inputs[0]->cbuffer().as<float*>();
    const float* inpPos = inputs[1]->cbuffer().as<float*>();
    const float* kernel = inputs[2]->cbuffer().as<float*>();
    const float* offset = inputs[3]->cbuffer().as<float*>();
    float* out = outputs[0]->buffer().as<float*>();
    memset(out, 0, outputs[0]->byteSize());

    size_t numPoints = inputs[1]->getTensorDesc().getDims()[0];
    std::vector<size_t> kernelDims = inputs[2]->getTensorDesc().getDims();

    // Kernel layout is DxHxWxICxOH
    const int kd = kernelDims[0];
    const int kh = kernelDims[1];
    const int kw = kernelDims[2];
    const int IC = kernelDims[3];
    const int OC = kernelDims[4];

    // See https://github.com/isl-org/Open3D/blob/master/python/open3d/ml/torch/python/layers/convolutions.py
    float rw = kw * 0.51f;
    float rh = kh * 0.51f;
    float rd = kd * 0.51f;

    for (size_t i = 0; i < numPoints; ++i) {
        if (inpPos[i * 3] < 0) {
            numPoints = i;
            break;
        }
    }

    for (size_t i = 0; i < numPoints; ++i) {
        const float xi = inpPos[i * 3] - offset[0];
        const float yi = inpPos[i * 3 + 1] - offset[1];
        const float zi = inpPos[i * 3 + 2] - offset[2];

        // Accumulate features which inside the kernel
        for (size_t j = 0; j < numPoints; ++j) {
            const float xj = inpPos[j * 3];
            const float yj = inpPos[j * 3 + 1];
            const float zj = inpPos[j * 3 + 2];

            if (xi - rw <= xj && xj <= xi + rw &&
                yi - rh <= yj && yj <= yi + rh &&
                zi - rd <= zj && zj <= zi + rd) {

                const int w = std::min(static_cast<int>(xj - xi + kw * 0.5f), kw - 1);
                const int h = std::min(static_cast<int>(yj - yi + kh * 0.5f), kh - 1);
                const int d = std::min(static_cast<int>(zj - zi + kd * 0.5f), kd - 1);

                const float* featuresOffset = features + j * IC;
                for (size_t ic = 0; ic < IC; ++ic) {
                    const float* kernelOffset = kernel + OC * (ic + IC * (w + kw * (h + kh * d)));
                    for (size_t oc = 0; oc < OC; ++oc) {
                        out[i * OC + oc] += kernelOffset[oc] * featuresOffset[ic];
                    }
                }
            }
        }
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]

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
LSTSQImpl::LSTSQImpl(const std::shared_ptr<ngraph::Node> &node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<LSTSQOp>(node);
        if (!castedNode)
            THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
        if (castedNode->inputs().size() != 2 || castedNode->outputs().size() != 1)
            THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
        if (castedNode->get_input_shape(0).size() != 2 || castedNode->get_output_shape(0).size() != 2)
            THROW_IE_EXCEPTION << "Operation supports only 4d tensors for input and output.";
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
            THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
        inShapes.resize(2);
        for (int i = 0; i < inShapes.size(); ++i)
            inShapes[i] = castedNode->get_input_shape(i);
        outShape = castedNode->get_output_shape(0);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        error = ex.what();
    }

}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode LSTSQImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
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
InferenceEngine::StatusCode LSTSQImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        if (config.inConfs.size() != 2 || config.outConfs.size() != 1) {
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
InferenceEngine::StatusCode LSTSQImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                               std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                               InferenceEngine::ResponseDesc *resp) noexcept {
    const float* B = inputs[0]->cbuffer().as<float*>();
    const float* A = inputs[1]->cbuffer().as<float*>();
    float* out = outputs[0]->buffer().as<float*>();

    // Perform A = QR factorization. This implementation works on A with 2 columns.
    const size_t M = inputs[0]->getTensorDesc().getDims()[0];
    const size_t N = inputs[0]->getTensorDesc().getDims()[1];

    std::vector<float> Q(M * 2);
    std::vector<float> R(4, 0.0f);
    float norm0 = 0.0f;
    float product = 0.0f;  // cross-product between second column of A with first column of Q
    for (int i = 0; i < M; ++i) {
        float val = A[i * 2];
        product += A[i * 2 + 1] * val;
        norm0 += val * val;
    }
    norm0 = sqrtf(norm0);
    product /= norm0;
    R[1] = product;

    float norm1 = 0.0f;
    for (int i = 0; i < M; ++i) {
        float val = A[i * 2] / norm0;
        Q[i * 2] = val;
        R[0] += A[i * 2] * val;

        val = A[i * 2 + 1] - product * val;
        Q[i * 2 + 1] = val;
        norm1 += val * val;
        R[3] += A[i * 2 + 1] * val;
    }
    norm1 = sqrtf(norm1);
    for (int i = 0; i < M; ++i) {
        Q[i * 2 + 1] /= norm1;
    }
    R[3] /= norm1;

    // Inverse R matrix
    float scale = 1.0f / (R[0] * R[3]);
    std::vector<float> R_inv{R[3] * scale, -R[1] * scale, 0.0f, R[0] * scale};

    // Output is inverse(R) * transpose(Q) * B
    for (int i = 0; i < M; ++i) {
        Q[i * 2] = R_inv[0] * Q[i * 2] + R_inv[1] * Q[i * 2 + 1];
        Q[i * 2 + 1] *= R_inv[3];
    }

    for (int i = 0; i < N; ++i) {
        out[i] = 0.0f;
        out[N + i] = 0.0f;
        for (int j = 0; j < M; ++j) {
            out[i] += Q[j * 2] * B[j * N + i];
            out[N + i] += Q[j * 2 + 1] * B[j * N + i];
        }
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]

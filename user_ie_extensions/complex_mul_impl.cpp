// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>

// #include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>
// #include "opencv_hal_intrin.hpp"
#include "ie_parallel.hpp"

// #include <opencv2/core/core_c.h>

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
ComplexMulImpl::ComplexMulImpl(const std::shared_ptr<ngraph::Node> &node) {
    auto castedNode = std::dynamic_pointer_cast<ComplexMulOp>(node);
    if (!castedNode)
        THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
    if (castedNode->inputs().size() != 2 || castedNode->outputs().size() != 1)
        THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
    if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
        THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
    if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
        THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
    inpShapes.push_back(castedNode->get_input_shape(0));
    inpShapes.push_back(castedNode->get_input_shape(1));
    outShape = castedNode->get_output_shape(0);
    is_conj = castedNode->is_conj;
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode ComplexMulImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                         InferenceEngine::ResponseDesc *resp) noexcept {
    std::vector<InferenceEngine::DataConfig> inDataConfig;
    std::vector<InferenceEngine::DataConfig> outDataConfig;

    // Allow any offset before data
    size_t offset((std::numeric_limits<size_t>::max)());

    // Input shape
    for (const auto inpShape : inpShapes) {
        InferenceEngine::SizeVector order(inpShape.size());
        std::iota(order.begin(), order.end(), 0);

        InferenceEngine::DataConfig inpConf;
        inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, {inpShape, order, offset});
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
InferenceEngine::StatusCode ComplexMulImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    if (config.inConfs.size() != 2 || config.outConfs.size() != 1) {
        THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
    }

    if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
        config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32)  {
        THROW_IE_EXCEPTION << "Operation supports only FP32 precisions!";
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

//! [cpu_implementation:execute]
InferenceEngine::StatusCode ComplexMulImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                   std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                   InferenceEngine::ResponseDesc *resp) noexcept {
    const float* inp0 = inputs[0]->cbuffer().as<const float*>();
    const float* inp1 = inputs[1]->cbuffer().as<const float*>();
    float* out = outputs[0]->buffer().as<float*>();

    size_t channels0 = inputs[0]->getTensorDesc().getDims()[1];
    size_t channels1 = inputs[1]->getTensorDesc().getDims()[1];
    size_t spatialSize = inputs[1]->getTensorDesc().getDims()[2] * inputs[1]->getTensorDesc().getDims()[3];

    // // std::cout << "Input0" << std::endl;
    // // for (int i = 0; i < 16; ++i)
    // //     std::cout << inp0[i] << " ";
    // std::cout << "Input1" << std::endl;
    // for (int i = 0; i < 8 * channels1; ++i)
    //     std::cout << inp1[i] << " ";
    // // std::cout << " " << std::endl;
    // std::cout << " " << std::endl;
    // std::cout << spatialSize << std::endl;
    // std::cout << channels0 << " " <<channels1 << std::endl;

   
    // # x1 = x_r * y_r - x_i * y_i
    // # x2 = x_r * y_i + x_i * y_r
    if (channels0 == channels1)
        InferenceEngine::parallel_for(channels0, [&](size_t ch) {
            for (int i = 0; i < spatialSize; ++i) {
                int outIdx = (ch * spatialSize + i) * 2;
                float real0 = inp0[outIdx];
                float imag0 = inp0[outIdx + 1];
                float real1 = inp1[outIdx];
                float imag1 = inp1[outIdx + 1];
                out[outIdx] = real0 * real1 - imag0 * imag1;
                out[outIdx + 1] = real0 * imag1 + imag0 * real1;
            }
        });
    else if (channels1 < channels0)
        InferenceEngine::parallel_for(channels0, [&](size_t ch) {
                for (int i = 0; i < spatialSize; ++i) {
                    int outIdx = (ch * spatialSize + i) * 2;
                    float real0 = inp0[outIdx];
                    float imag0 = inp0[outIdx + 1];
                    float real1 = inp1[i * 2];
                    float imag1 = inp1[i * 2 + 1];
                    out[outIdx] = real0 * real1 - imag0 * imag1;
                    out[outIdx + 1] = real0 * imag1 + imag0 * real1;
                }
            });

        // std::cout << " " << std::endl;
        // std::cout << "RES" << std::endl;
        // for (int i = 0; i < 16; ++i)
        //     std::cout << out[i] << " ";
        // std::cout << " " << std::endl;
        // std::cout << " " << std::endl;


    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]


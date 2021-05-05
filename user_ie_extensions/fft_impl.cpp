// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#include "ie_parallel.hpp"

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
FFTImpl::FFTImpl(const std::shared_ptr<ngraph::Node> &node) {
    auto castedNode = std::dynamic_pointer_cast<FFTOp>(node);
    if (!castedNode)
        THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
    if (castedNode->inputs().size() != 1 || castedNode->outputs().size() != 1)
        THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
    if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
        THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
    if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
        THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
    inpShape = castedNode->get_input_shape(0);
    outShape = castedNode->get_output_shape(0);
    inverse = castedNode->inverse;
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode FFTImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                         InferenceEngine::ResponseDesc *resp) noexcept {
    std::vector<InferenceEngine::DataConfig> inDataConfig;
    std::vector<InferenceEngine::DataConfig> outDataConfig;
    InferenceEngine::SizeVector order(inpShape.size());
    std::iota(order.begin(), order.end(), 0);

    // Allow any offset before data
    size_t offset((std::numeric_limits<size_t>::max)());

    // Input shape
    InferenceEngine::DataConfig inpConf;
    inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, {inpShape, order, offset});
    inDataConfig.push_back(inpConf);

    // Output shape
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
InferenceEngine::StatusCode FFTImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    if (config.inConfs.size() != 1 || config.outConfs.size() != 1) {
        THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
    }

    if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
        config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32)  {
        THROW_IE_EXCEPTION << "Operation supports only FP32 precisions!";
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

#ifdef HAVE_OPENCV
static cv::Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();
    CV_Assert(precision == InferenceEngine::Precision::FP32);
    return cv::Mat(size, CV_32F, (void*)blob->buffer());
}

//! [cpu_implementation:execute]
InferenceEngine::StatusCode FFTImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                      std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                      InferenceEngine::ResponseDesc *resp) noexcept {
    cv::Mat inp = infEngineBlobToMat(inputs[0]);
    cv::Mat out = infEngineBlobToMat(outputs[0]);

    CV_CheckEQ(inp.size[inp.dims - 1], 2, "");

    if (inp.dims == 5) {
        const int batch = inp.size[0];
        const int channels = inp.size[1];
        int rows = inp.size[2];
        int cols = inp.size[3];
        inp = inp.reshape(1, batch * channels);
        out = out.reshape(1, batch * channels);
        InferenceEngine::parallel_for(batch * channels, [&](size_t d) {
            cv::Mat inpSlice(rows, cols, CV_32FC2, inp.ptr<float>(d));
            cv::Mat outSlice(rows, cols, CV_32FC2, out.ptr<float>(d));
            if (inverse)
                cv::idft(inpSlice, outSlice);
            else
                cv::dft(inpSlice, outSlice);
        });
        out /= sqrtf(cols * rows);
    } else {
        int rows, cols;
        if (inp.dims == 4) {
            rows = inp.size[0] * inp.size[1];
            cols = inp.size[2];
        } else if (inp.dims == 3) {
            rows = inp.size[0];
            cols = inp.size[1];
        } else {
            CV_Assert(inp.dims == 3 || inp.dims == 4);
        }
        inp = cv::Mat(rows, cols, CV_32FC2, inp.ptr<float>());
        out = cv::Mat(rows, cols, CV_32FC2, out.ptr<float>());

        if (inverse)
            cv::idft(inp, out, cv::DFT_ROWS);
        else
            cv::dft(inp, out, cv::DFT_ROWS);
        out /= sqrtf(cols);
    }

    return InferenceEngine::OK;
}

#else

InferenceEngine::StatusCode FFTImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                             std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                             InferenceEngine::ResponseDesc *resp) noexcept {
    THROW_IE_EXCEPTION << "OpenCV is required for FFT implementation!";
}

#endif  // HAVE_OPENCV
//! [cpu_implementation:execute]

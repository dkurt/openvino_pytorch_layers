// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <details/ie_so_loader.h>
#include <ie_layouts.h>

#include "ie_parallel.hpp"
#include <opencv2/core/core_c.h>

std::unique_ptr<InferenceEngine::details::SharedObjectLoader> so;
using cvCreateMatHeaderF = CvMat*(int, int, int);
using cvSetDataF = void(CvArr*, void*, int);
using cvReleaseMatF = void(CvMat**);
using cvDftF = void(const CvArr*, CvArr*, int, int);
using cvScaleF = void(const CvArr*, CvArr*, double, double);
using cvCloneMatF = CvMat*(const CvMat*);
using cvCopyF = void(const CvArr*, const CvArr*, const CvArr*);
using cvInitMatHeaderF = CvMat*(CvMat*, int, int, int, void*, int);
using cvGetRawDataF = void(const CvArr*, uchar**, int* step, CvSize* roi_size);
using cvReshapeF = CvMat*(const CvArr*, CvMat*, int, int);
using cvCreateDataF = void(CvArr*);

bool loadOpenCV() {
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        try {
#ifdef _WIN32
            so.reset(new InferenceEngine::details::SharedObjectLoader("opencv_core.dll"));
#elif defined(__APPLE__)
            so.reset(new InferenceEngine::details::SharedObjectLoader("libopencv_core.dylib"));
#else
            so.reset(new InferenceEngine::details::SharedObjectLoader("libopencv_core.so"));
#endif
        } catch (InferenceEngine::details::InferenceEngineException& ex) {
            return false;
        }
    }
    return loaded;
}

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
FFTImpl::FFTImpl(const std::shared_ptr<ngraph::Node> &node) {
    auto castedNode = std::dynamic_pointer_cast<FFTOp>(node);
    if (!castedNode)
        THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
    if (castedNode->inputs().size() != 2 || castedNode->outputs().size() != 1)
        THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
    if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
        THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
    if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
        THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
    inShapes.resize(2);
    for (int i = 0; i < 2; ++i)
        inShapes[i] = castedNode->get_input_shape(i);
    outShape = castedNode->get_output_shape(0);
    inverse = castedNode->inverse;
    centered = castedNode->centered;
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode FFTImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
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
InferenceEngine::StatusCode FFTImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    if (!loadOpenCV()) {
        THROW_IE_EXCEPTION << "Failed to load OpenCV!";
    }

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

static void fftshift(CvMat* src) {
    static auto cvCloneMat = reinterpret_cast<cvCloneMatF*>(so->get_symbol("cvCloneMat"));
    static auto cvCopy = reinterpret_cast<cvCopyF*>(so->get_symbol("cvCopy"));
    static auto cvInitMatHeader = reinterpret_cast<cvInitMatHeaderF*>(so->get_symbol("cvInitMatHeader"));
    static auto cvGetRawData = reinterpret_cast<cvGetRawDataF*>(so->get_symbol("cvGetRawData"));
    static auto cvReleaseMat = reinterpret_cast<cvReleaseMatF*>(so->get_symbol("cvReleaseMat"));


    // for dim = (2, 3):
    // tl | tr        br | bl
    // ---+---   ->   ---+---
    // bl | br        tr | tl

    float* data;
    int step;
    CvSize size;
    cvGetRawData(src, (uchar**)&data, &step, &size);

    int height = size.height;
    int width = size.width;
    int h2 = height / 2;
    int w2 = width / 2;

    CvMat* tl = new CvMat();
    CvMat* tr = new CvMat();
    CvMat* bl = new CvMat();
    CvMat* br = new CvMat();

    cvInitMatHeader(tl, h2, w2, CV_32FC2, data, step);
    cvInitMatHeader(tr, h2, w2, CV_32FC2, data + width, step);
    cvInitMatHeader(bl, h2, w2, CV_32FC2, data + height * width, step);
    cvInitMatHeader(br, h2, w2, CV_32FC2, data + height * width + width, step);

    CvArr* mask = 0;
    CvMat* tmp = cvCloneMat(tl);
    cvCopy(br, tl, mask);
    cvCopy(tmp, br, mask);

    cvCopy(tr, tmp, mask);
    cvCopy(bl, tr, mask);
    cvCopy(tmp, bl, mask);

    cvReleaseMat(&tmp);

    delete tl;
    delete tr;
    delete bl;
    delete br;
}

//! [cpu_implementation:execute]
InferenceEngine::StatusCode FFTImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                             std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                             InferenceEngine::ResponseDesc *resp) noexcept {
    static auto cvSetData = reinterpret_cast<cvSetDataF*>(so->get_symbol("cvSetData"));
    static auto cvCreateMatHeader = reinterpret_cast<cvCreateMatHeaderF*>(so->get_symbol("cvCreateMatHeader"));
    static auto cvDFT = reinterpret_cast<cvDftF*>(so->get_symbol("cvDFT"));
    static auto cvScale = reinterpret_cast<cvScaleF*>(so->get_symbol("cvConvertScale"));
    static auto cvReleaseMat = reinterpret_cast<cvReleaseMatF*>(so->get_symbol("cvReleaseMat"));
    static auto cvReshape = reinterpret_cast<cvReshapeF*>(so->get_symbol("cvReshape"));
    static auto cvCloneMat = reinterpret_cast<cvCloneMatF*>(so->get_symbol("cvCloneMat"));
    static auto cvCreateData = reinterpret_cast<cvCreateDataF*>(so->get_symbol("cvCreateData"));
    static auto cvCopy = reinterpret_cast<cvCopyF*>(so->get_symbol("cvCopy"));

    float* inpData = inputs[0]->buffer();
    float* signalDimsData = inputs[1]->buffer();
    float* outData = outputs[0]->buffer();
    std::vector<size_t> dims = inputs[0]->getTensorDesc().getDims();
    const size_t numSignalDims = inputs[1]->getTensorDesc().getDims()[0];

    if (!(dims.size() == 3 && (numSignalDims == 1 && signalDimsData[0] == 1) ||
          dims.size() == 4 && ((numSignalDims == 1 && signalDimsData[0] == 1) ||
                               (numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2)) ||
          dims.size() == 5 && ((numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) ||
                               (numSignalDims == 2 && signalDimsData[0] == 2 && signalDimsData[1] == 3)))) {
        THROW_IE_EXCEPTION << "Unsupported configuration!";
    }

    const int batch = dims[0];

    if (dims.size() == 5 && numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) {
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        const int planeSize = channels * rows * cols;
        InferenceEngine::parallel_for(batch * cols, [&](size_t d) {
            int b = d / cols;
            int col = d % cols;
            // Copy a slice from input
            CvMat* inpSlice = cvCreateMatHeader(channels * rows, 1, CV_32FC2);
            CvMat* outSlice = cvCreateMatHeader(channels * rows, 1, CV_32FC2);
            cvSetData(inpSlice, reinterpret_cast<void*>(inpData + (b * planeSize + col) * 2), cols * 2 * sizeof(float));
            cvSetData(outSlice, reinterpret_cast<void*>(outData + (b * planeSize + col) * 2), cols * 2 * sizeof(float));

            CvMat* inp_col = cvCloneMat(inpSlice);

            CvMat inp_header, *inp;
            inp = cvReshape(inp_col, &inp_header, 2, channels);

            CvMat* out = cvCreateMatHeader(channels, rows, CV_32FC2);
            cvCreateData(out);

            if (centered)
                fftshift(inp);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(channels * rows), 0);

            if (centered)
                fftshift(out);

            CvMat out_col_header, *out_col;
            out_col = cvReshape(out, &out_col_header, 2, channels * rows);

            CvArr* mask = 0;
            cvCopy(out_col, outSlice, mask);

            // cvReleaseMat(&inp);
            // cvReleaseMat(&out);
            // cvReleaseMat(&inpSlice);
            // cvReleaseMat(&outSlice);
        });
    } else if (dims.size() == 5 && numSignalDims == 2 && signalDimsData[0] == 2 && signalDimsData[1] == 3) {
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        InferenceEngine::parallel_for(batch * channels, [&](size_t d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            if (centered)
                fftshift(out);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else if (dims.size() == 4 && inputs[1]->getTensorDesc().getDims()[0] == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) {
        int rows = dims[1];
        int cols = dims[2];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        InferenceEngine::parallel_for(batch, [&](size_t d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            if (centered)
                fftshift(out);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else if (dims.size() == 4 && numSignalDims == 1 && signalDimsData[0] == 1) {
        int rows = dims[1];
        int cols = dims[2];

        const int planeSize = rows;
        InferenceEngine::parallel_for(batch * cols, [&](size_t d) {
            int b = d / cols;
            int col = d % cols;
            CvMat* inp = cvCreateMatHeader(rows, 1, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, 1, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + (b * planeSize * cols + col) * 2), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + (b * planeSize * cols + col) * 2), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(rows), 0);

            if (centered)
                fftshift(out);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else if (dims.size() == 3) {
        int rows = dims[0];
        int cols = dims[1];
        CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
        CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
        cvSetData(inp, reinterpret_cast<void*>(inpData), cols * 2 * sizeof(float));
        cvSetData(out, reinterpret_cast<void*>(outData), cols * 2 * sizeof(float));

        if (centered)
            fftshift(inp);

        if (inverse)
            cvDFT(inp, out, CV_DXT_INVERSE | CV_DXT_ROWS, 0);
        else
            cvDFT(inp, out, CV_DXT_FORWARD | CV_DXT_ROWS, 0);
        cvScale(out, out, 1.0 / sqrtf(cols), 0);

        if (centered)
            fftshift(out);

        cvReleaseMat(&inp);
        cvReleaseMat(&out);
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]

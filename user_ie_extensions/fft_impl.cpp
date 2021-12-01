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
    if (!loadOpenCV()) {
        THROW_IE_EXCEPTION << "Failed to load OpenCV!";
    }

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

static cv::Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();
    CV_Assert(precision == InferenceEngine::Precision::FP32);
    return cv::Mat(size, CV_32F, (void*)blob->buffer());
}

static void fftshift(cv::Mat& src)
{
    const int h_2 = src.rows / 2;
    const int w_2 = src.cols / 2;

    // tl | tr        br | bl
    // ---+---   ->   ---+---
    // bl | br        tr | tl
    cv::Mat tl = src({0, 0, w_2, h_2});
    cv::Mat tr = src({w_2, 0, w_2, h_2});
    cv::Mat bl = src({0, h_2, w_2, h_2});
    cv::Mat br = src({w_2, h_2, w_2, h_2});

    cv::Mat tmp = tl.clone();
    br.copyTo(tl);
    tmp.copyTo(br);

    tr.copyTo(tmp);
    bl.copyTo(tr);
    tmp.copyTo(bl);
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

    float* inpData = inputs[0]->buffer();
    float* outData = outputs[0]->buffer();
    std::vector<size_t> dims = inputs[0]->getTensorDesc().getDims();

    if (dims.size() == 5) {
        const int batch = dims[0];
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        InferenceEngine::parallel_for(batch * channels, [&](size_t d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));
            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else {
        int rows, cols;
        if (dims.size() == 4) {
            rows = dims[0] * dims[1];
            cols = dims[2];
        } else if (dims.size() == 3) {
            rows = dims[0];
            cols = dims[1];
        }
        CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
        CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
        cvSetData(inp, reinterpret_cast<void*>(inpData), cols * 2 * sizeof(float));
        cvSetData(out, reinterpret_cast<void*>(outData), cols * 2 * sizeof(float));

        if (inverse)
            cvDFT(inp, out, CV_DXT_INVERSE | CV_DXT_ROWS, 0);
        else
            cvDFT(inp, out, CV_DXT_FORWARD | CV_DXT_ROWS, 0);
        cvScale(out, out, 1.0 / sqrtf(cols), 0);

        cvReleaseMat(&inp);
        cvReleaseMat(&out);
    }
    return InferenceEngine::OK;
}

// InferenceEngine::StatusCode FFTImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
//                                              std::vector<InferenceEngine::Blob::Ptr> &outputs,
//                                              InferenceEngine::ResponseDesc *resp) noexcept {
//     static auto cvSetData = reinterpret_cast<cvSetDataF*>(so->get_symbol("cvSetData"));
//     static auto cvCreateMatHeader = reinterpret_cast<cvCreateMatHeaderF*>(so->get_symbol("cvCreateMatHeader"));
//     static auto cvDFT = reinterpret_cast<cvDftF*>(so->get_symbol("cvDFT"));
//     static auto cvScale = reinterpret_cast<cvScaleF*>(so->get_symbol("cvConvertScale"));
//     static auto cvReleaseMat = reinterpret_cast<cvReleaseMatF*>(so->get_symbol("cvReleaseMat"));

//     float* inpData = inputs[0]->buffer();
//     float* outData = outputs[0]->buffer();
//     std::vector<size_t> dims = inputs[0]->getTensorDesc().getDims();

//     if (dims.size() == 5) {
//         const int batch = dims[0];
//         const int channels = dims[1];
//         int rows = dims[2];
//         int cols = dims[3];
//         int planeSize = rows * cols * 2;  // 2 is last dimension size
//         InferenceEngine::parallel_for(batch * channels, [&](size_t d) {
//             CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
//             CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
//             cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
//             cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));
//             if (inverse)
//                 cvDFT(inp, out, CV_DXT_INVERSE, 0);
//             else
//                 cvDFT(inp, out, CV_DXT_FORWARD, 0);
//             cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

//             cvReleaseMat(&inp);
//             cvReleaseMat(&out);
//         });
//     } else {
//         int rows, cols;
//         if (dims.size() == 4) {
//             rows = dims[0] * dims[1];
//             cols = dims[2];
//         } else if (dims.size() == 3) {
//             rows = dims[0];
//             cols = dims[1];
//         }
//         CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
//         CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
//         cvSetData(inp, reinterpret_cast<void*>(inpData), cols * 2 * sizeof(float));
//         cvSetData(out, reinterpret_cast<void*>(outData), cols * 2 * sizeof(float));

//         if (inverse)
//             cvDFT(inp, out, CV_DXT_INVERSE | CV_DXT_ROWS, 0);
//         else
//             cvDFT(inp, out, CV_DXT_FORWARD | CV_DXT_ROWS, 0);
//         cvScale(out, out, 1.0 / sqrtf(cols), 0);

//         cvReleaseMat(&inp);
//         cvReleaseMat(&out);
//     }
//     return InferenceEngine::OK;
// }
//! [cpu_implementation:execute]


/*
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// ===============================================================================
// Generated file for Inference Engine extension for CPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// You need to edit this file in order to:
//  1. initialize parameters (in constructor)
//  2. implement inference logic (in execute() method)
//
// Refer to the section "Adding Your Own Kernels to the Inference Engine" in
// OpenVINO* documentation (either online or offline in
// <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
// to the corresponding section).
// ===============================================================================

// #include "ext_list.hpp"
// #include "ext_base.hpp"
// #include <cmath>
// #include <vector>
// #include <string>
// #include <algorithm>

// #include <opencv2/opencv.hpp>
// #include <inference_engine.hpp>

// namespace InferenceEngine {
// namespace Extensions {
// namespace Cpu {





// class FFT2DImpl: public ExtLayerBase {
// public:
//     explicit FFT2DImpl(const CNNLayer* layer) {
//         inverse = layer->type == "IFFT2D";
//         addConfig(layer, { { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } });
//     }

//     StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
//                        ResponseDesc *resp) noexcept override {

//         cv::Mat inp = infEngineBlobToMat(inputs[0]);
//         cv::Mat out = infEngineBlobToMat(outputs[0]);

//         const int batchSize = inp.size[0];
//         for (int i = 0; i < batchSize; ++i)
//         {
//             cv::Mat shifted_inp;
//             std::vector<cv::Mat> complex = {
//                 cv::Mat(inp.size[2], inp.size[3], CV_32F, inp.ptr<float>(i, 0)),
//                 cv::Mat(inp.size[2], inp.size[3], CV_32F, inp.ptr<float>(i, 1))
//             };
//             cv::merge(complex, shifted_inp);

//             fftshift(shifted_inp);

//             cv::Mat interleaved_out;
//             if (!inverse)
//                 cv::dft(shifted_inp, interleaved_out);
//             else
//                 cv::idft(shifted_inp, interleaved_out, cv::DFT_SCALE);

//             fftshift(interleaved_out);

//             complex = {
//                 cv::Mat(out.size[2], out.size[3], CV_32F, out.ptr<float>(i, 0)),
//                 cv::Mat(out.size[2], out.size[3], CV_32F, out.ptr<float>(i, 1))
//             };
//             cv::split(interleaved_out, complex);
//         }
//         return OK;
//     }
// private:
//     bool inverse;
// };

// REG_FACTORY_FOR(ImplFactory<FFT2DImpl>, FFT2D);
// REG_FACTORY_FOR(ImplFactory<FFT2DImpl>, IFFT2D);

// // TODO: The following code is a workaround for a bug with networks reshaping from Python.
// // Replece it to nGraph nodes management with the release after 2020.1

// class FFTShapeInferImpl : public IShapeInferImpl {
//     StatusCode inferShapes(const std::vector<Blob::CPtr>& inBlobs,
//                            const std::map<std::string, std::string>& params,
//                            const std::map<std::string, Blob::Ptr>& blobs,
//                            std::vector<SizeVector>& outShapes,
//                            ResponseDesc* desc) noexcept override
//     {
//         outShapes.resize(inBlobs.size());
//         for (size_t i = 0; i < inBlobs.size(); ++i)
//             outShapes[i] = inBlobs[i]->getTensorDesc().getDims();
//         return StatusCode::OK;
//     }
// };

// class FFTShapeInferFactory : public IShapeInferExtension {
//     virtual void SetLogCallback(IErrorListener& listener) noexcept {}
//     virtual void GetVersion(const Version*& versionInfo) const noexcept {}
//     virtual void Unload() noexcept {}
//     virtual void Release() noexcept {}

//     virtual StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
//         size = 2;
//         types = new char*[size];

//         const char* fftType = "FFT2D";
//         types[0] = new char[sizeof(fftType)];
//         memcpy(types[0], fftType, sizeof(fftType));

//         const char* ifftType = "IFFT2D";
//         types[1] = new char[sizeof(ifftType)];
//         memcpy(types[1], ifftType, sizeof(ifftType));

//         return StatusCode::OK;
//     }

//     virtual StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept {
//         impl = std::make_shared<FFTShapeInferImpl>();
//         return StatusCode::OK;
//     }
// };

// class FFT2DShapeInferFactory {
// public:
//     FFT2DShapeInferFactory() {

//       // We need to do it to trigger custom shape inference registration.
//       const std::string ir = R"V0G0N(
//       <?xml version="1.0" ?>
//       <net name="" version="10">
//         <layers>
//           <layer id="0" name="Placeholder_2" type="Parameter" version="opset1">
//             <data element_type="f32" shape="1"/>
//             <output>
//               <port id="0" precision="FP32">
//                 <dim>1</dim>
//               </port>
//             </output>
//           </layer>
//         </layers>
//       </net>
//       )V0G0N";

//       InferenceEngine::Core ie;
//       InferenceEngine::CNNNetwork net = ie.ReadNetwork(ir, Blob::CPtr());
//       net.AddExtension(std::make_shared<FFTShapeInferFactory>());
//     }
// };

// static FFT2DShapeInferFactory factory;

// }  // namespace Cpu
// }  // namespace Extensions
// }  // namespace InferenceEngine

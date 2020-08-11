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

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class MaxUnpoolImpl: public ExtLayerBase {
public:
    explicit MaxUnpoolImpl(const CNNLayer* layer) {
        try {
            
            const auto dims = layer->insData[1].lock()->getTensorDesc().getDims();
            mask.resize(dims[0]*dims[1]*dims[2]*dims[3]);

            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                             { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, 
                       std::vector<Blob::Ptr>& outputs, 
                       ResponseDesc *resp) noexcept override {
        const float* poolInp = inputs[0]->cbuffer().as<float*>();
        const float* poolOut = inputs[1]->cbuffer().as<float*>();
        const float* inp     = inputs[2]->cbuffer().as<float*>();
        float* out = outputs[0]->buffer().as<float*>();

        std::vector<size_t> poolInpDims = inputs[0]->getTensorDesc().getDims();
        std::vector<size_t> poolOutDims = inputs[1]->getTensorDesc().getDims();
        std::vector<size_t> inpDims = inputs[2]->getTensorDesc().getDims();
        std::vector<size_t> outDims = outputs[0]->getTensorDesc().getDims();

        const size_t batch    = poolInpDims[0];
        const size_t channels = poolInpDims[1];
        const size_t height   = poolInpDims[2];
        const size_t width    = poolInpDims[3];
        const size_t outHeight = outDims[2];
        const size_t outWidth  = outDims[3];
        const size_t poolOutHeight = poolOutDims[2];
        const size_t poolOutWidth  = poolOutDims[3];
        std::fill(mask.begin(), mask.end(), false);
        memset(out, 0, outputs[0]->byteSize());
        parallel_for(batch*channels, [&](size_t d) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int poolOutIdx = (d * poolOutHeight + y / 2) * poolOutWidth + x / 2;
                    int poolInpIdx = (d * height + y) * width + x;
                    int dstIdx = d * outHeight * outWidth + (y * width + x);
                    if (fabs(poolInp[poolInpIdx] - poolOut[poolOutIdx]) < 1e-5f && !mask[poolOutIdx]) {
                        out[dstIdx] = inp[poolOutIdx];
                        mask[poolOutIdx] = true;
                    }
                }
            }
        });
        return OK; 
    }
private:
    std::vector<bool> mask;
};

REG_FACTORY_FOR(ImplFactory<MaxUnpoolImpl>, MaxPoolGrad);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

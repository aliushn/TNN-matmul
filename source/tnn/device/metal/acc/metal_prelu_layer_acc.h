// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_METAL_PRELU_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_METAL_PRELU_LAYER_ACC_H_

#include "tnn/device/metal/acc/metal_layer_acc.h"

namespace TNN_NS {

// @brief prelu layer metal acc
class MetalPReluLayerAcc : public MetalLayerAcc {
public:
    // @brief virtual destrcutor
    virtual ~MetalPReluLayerAcc();

    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    std::shared_ptr<MetalLayerAcc> activate_impl_ = nullptr;
    id<MTLBuffer> buffer_slope_                   = nil;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_METAL_METAL_PRELU_LAYER_ACC_H_

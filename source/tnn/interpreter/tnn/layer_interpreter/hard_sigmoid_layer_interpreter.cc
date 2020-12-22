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

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(HardSigmoid, LAYER_HARDSIGMOID);

Status HardSigmoidLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    HardSigmoidLayerParam* layer_param = new HardSigmoidLayerParam();
    *param                             = layer_param;
    int index                          = start_index;

    if (index < layer_cfg_arr.size()) {
        layer_param->alpha = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));
    }

    if (index < layer_cfg_arr.size()) {
        layer_param->beta = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));
    }
    return TNN_OK;
}

Status HardSigmoidLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status HardSigmoidLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<HardSigmoidLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->alpha << " " << layer_param->beta << " ";
    return TNN_OK;
}

Status HardSigmoidLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(HardSigmoid, LAYER_HARDSIGMOID);

}  // namespace TNN_NS

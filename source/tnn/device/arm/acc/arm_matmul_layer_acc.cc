// Tencent is pleased to support the open source community by making TNN
// available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

// leiheng
#include "tnn/device/arm/acc/arm_matmul_layer_acc.h"

#include "tnn/core/blob_int8.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

ArmMatMulLayerAcc::~ArmMatMulLayerAcc() {}

// pack int8 kernel: round up c8, round up oc4
// static void packweight_i8(const int8_t *src, int8_t *dst, const int oc,
//                           const int ic) {
//   auto dst_step = ROUND_UP(ic, 8);

//   for (int o = 0; o < oc; o++) {
//     memcpy(dst + o * dst_step, src + o * ic, ic);
//   }
// }

// pack float and bfp16 weights
template <typename T>
static void PackWeightO4(const T *src, T *dst, const int oc, const int ic) {
  const int oc_r4 = ROUND_UP(oc, 4);
  const int ic_r4 = ROUND_UP(ic, 4);
  for (int o = 0; o < oc_r4; o++) {
    int o_inner = o % 4;
    int o_outer = o / 4 * 4;
    for (int i = 0; i < ic_r4; i++) {
      if (i >= ic || o >= oc) {
        dst[i * 4 + o_outer * ic_r4 + o_inner] = 0;
      } else {
        dst[i * 4 + o_outer * ic_r4 + o_inner] = src[o * ic + i];
      }
    }
  }
}

// get 4 result at a time
template <typename T>
static void SGEMV(T *dst, const T *src, T *weight, const int oc_r4,
                  const int ic_r4) {
  OMP_PARALLEL_FOR_
  for (int o = 0; o < oc_r4; o += 4) {
    auto weight_z = weight + o * ic_r4;
    Float4 acc(0.f);
    for (int i = 0; i < ic_r4; i += 4) {
      Float4 w0 = Float4::load(weight_z + i * 4 + 0);
      Float4 w1 = Float4::load(weight_z + i * 4 + 4);
      Float4 w2 = Float4::load(weight_z + i * 4 + 8);
      Float4 w3 = Float4::load(weight_z + i * 4 + 12);
      Float4 v0 = Float4::load(src + i);
      Float2 v0_0, v0_1;
      Float4::get_low(v0, v0_0);
      Float4::get_high(v0, v0_1);
      Float4::mla_lane0(acc, w0, v0_0);
      Float4::mla_lane1(acc, w1, v0_0);
      Float4::mla_lane0(acc, w2, v0_1);
      Float4::mla_lane1(acc, w3, v0_1);
    }
    Float4::save(dst + o, acc);
  }
}

Status ArmMatMulLayerAcc::allocateBufferWeight(
    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
  MatMulLayerParam *fc_param = dynamic_cast<MatMulLayerParam *>(param_);
  CHECK_PARAM_NULL(fc_param);
  MatMulLayerResource *fc_res = dynamic_cast<MatMulLayerResource *>(resource_);
  CHECK_PARAM_NULL(fc_res);

  if (!buffer_weight_.GetBytesSize()) {
    DimsVector dims_input = inputs[0]->GetBlobDesc().dims;
    DimsVector dims_output = outputs[0]->GetBlobDesc().dims;

    RawBuffer w_handle = fc_res->weight_handle;
    CHECK_PARAM_NULL(w_handle.force_to<void *>());

    if (w_handle.GetDataType() == DATA_TYPE_HALF)
      w_handle = ConvertHalfHandle(w_handle);

    auto weight_data_type = w_handle.GetDataType();
    int ic = dims_input[1] * dims_input[2] * dims_input[3];
    const int oc = fc_param->num_output;
    auto data_byte_size = DataTypeUtils::GetBytesSize(weight_data_type);
    if (weight_data_type == DATA_TYPE_FLOAT) {
      // transform weight dims from 4 to 2
      if (dims_input[2] != 1 || dims_input[3] != 1) {
        RawBuffer reorder_buffer =
            RawBuffer(dims_input[3] * dims_input[2] *
                      ROUND_UP(dims_input[1], 4) * oc * data_byte_size);
        for (int i = 0; i < oc; i++) {
          auto dst_ptr =
              reorder_buffer.force_to<float *>() +
              i * dims_input[3] * dims_input[2] * ROUND_UP(dims_input[1], 4);
          auto src_ptr = w_handle.force_to<float *>() + i * ic;
          PackC4(dst_ptr, src_ptr, dims_input[2] * dims_input[3],
                 dims_input[1]);
        }

        ic = dims_input[3] * dims_input[2] * ROUND_UP(dims_input[1], 4);
        w_handle = reorder_buffer;
      }

      auto weight_count = ROUND_UP(oc, 4) * ROUND_UP(ic, 4);
      buffer_weight_ = RawBuffer(weight_count * data_byte_size);
      PackWeightO4(w_handle.force_to<float *>(),
                   buffer_weight_.force_to<float *>(), oc, ic);

      // both data and weight will use type bfp16
      if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        RawBuffer bfp16_buffer(weight_count * sizeof(bfp16_t));
        ConvertFromFloatToBFP16(buffer_weight_.force_to<float *>(),
                                bfp16_buffer.force_to<void *>(), weight_count);
        buffer_weight_ = bfp16_buffer;
      }
      // } else {
      //   auto weight_count = ROUND_UP(oc, 4) * ROUND_UP(ic, 8);
      //   buffer_weight_ =
      //       RawBuffer(weight_count * data_byte_size +
      //       NEON_KERNEL_EXTRA_LOAD);
      //   packweight_i8(w_handle.force_to<int8_t *>(),
      //                 buffer_weight_.force_to<int8_t *>(), oc, ic);
    }
  }

  return TNN_OK;
}

Status ArmMatMulLayerAcc::Init(Context *context, LayerParam *param,
                               LayerResource *resource,
                               const std::vector<Blob *> &inputs,
                               const std::vector<Blob *> &outputs) {
  RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs),
                TNN_OK);
  RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
  // RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

  return TNN_OK;
}

/*
general template function for float and bfp16
in bfp16 mode, both data and weight data type are bfp16, is there any precision
problem
*/
template <typename T>
Status ArmMatMulLayerAcc::Exec(const std::vector<Blob *> &inputs,
                               const std::vector<Blob *> &outputs) {
  MatMulLayerParam *fc_param = dynamic_cast<MatMulLayerParam *>(param_);
  CHECK_PARAM_NULL(fc_param);

  auto input = inputs[0];
  auto output = outputs[0];

  auto dims_input = input->GetBlobDesc().dims;
  auto dims_output = output->GetBlobDesc().dims;
  auto ic = dims_input[3] * dims_input[2] * ROUND_UP(dims_input[1], 4);
  auto oc_r4 = ROUND_UP(dims_output[1], 4);

  auto input_origin =
      reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
  auto output_origin =
      reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));
  for (int n = 0; n < dims_output[0]; n++) {
    auto input_ptr = input_origin + n * ic;
    auto output_ptr = output_origin + n * oc_r4;

    SGEMV(output_ptr, input_ptr, buffer_weight_.force_to<T *>(), oc_r4, ic);

    // if (fc_param->has_bias) {
    //   PostAddBias<T>(output_ptr, buffer_bias_.force_to<float *>(), 1,
    //                  oc_r4 / 4);
    // }
  }

  return TNN_OK;
}

Status ArmMatMulLayerAcc::DoForward(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
  if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
    return Exec<float>(inputs, outputs);
  } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
    return Exec<bfp16_t>(inputs, outputs);
    // } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
    //   return Exec<int8_t>(inputs, outputs);
  }
  return TNNERR_LAYER_ERR;
}

REGISTER_ARM_ACC(MatMul, LAYER_MATMUL)

}  // namespace TNN_NS

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
#include <fstream>
#include <iostream>
#include <sstream>

#include "half_utils.h"
#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER(MatMul);

string OnnxOpConverterMatMul::TNNOpType(NodeProto& node,
                                        OnnxNetInfo& net_info) {
  return "MatMul";
}

string OnnxOpConverterMatMul::TNNLayerParam(NodeProto& node,
                                            OnnxNetInfo& net_info) {
  ostringstream layer_param;

  const std::string& onnx_op = node.op_type();

  const onnx::TensorProto& weights = net_info.weights_map[node.input(1)];
  int channel_output = (int)weights.dims(1);
  layer_param << channel_output << " ";

  return layer_param.str();
}

int OnnxOpConverterMatMul::WriteTNNModel(serializer* net_writer,
                                         NodeProto& node,
                                         OnnxNetInfo& net_info) {
  const std::string& onnx_op = node.op_type();
  std::string name = !node.name().empty() ? node.name() : node.output(0);
  const std::string& tnn_layer_type = TNNOpType(node, net_info);

  //写头信息
  net_writer->put_int(0);  //触发type from string
  net_writer->put_string(tnn_layer_type);
  net_writer->put_string(name);

  //写数据
  net_writer->put_string(name);
  auto B = get_node_attr_tensor(node, "B", net_info, 1);

  auto const h = B.dims(0);
  auto const w = B.dims(1);

  float* permuted_data = new float[h * w];
  auto bptr = get_tensor_proto_data(B);

  float* permuted_data_ptr = permuted_data;
  for (int j = 0; j < w; j++) {
    for (int k = 0; k < h; k++) {
      float vb = bptr[k * w + j];
      *permuted_data_ptr = vb;
      permuted_data_ptr++;
    }
  }

  WriteRawData(permuted_data, (int)(h * w), net_writer, net_info.data_type);
  delete[] permuted_data;

  //有权值写入的返回1， 没有的返回0
  return 1;
}

REGISTER_OP_CONVERTER(MatMul, MatMul);

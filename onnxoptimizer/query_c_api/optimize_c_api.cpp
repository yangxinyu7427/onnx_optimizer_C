//
// Created by xyyang on 23-11-14.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>
#include <onnxoptimizer/optimize.h>
#include <onnxoptimizer/query_c_api/model_merge.cpp>


#include <filesystem>
#include <fstream>

namespace ONNX_NAMESPACE {
namespace optimization {

void OptimizeWithModels(
    const ModelProto& mp_in1,
    const ModelProto& mp_in2,
    const std::string& mp_name1,
    const std::string& mp_name2,
    const ModelProto& mp_out) {
  model_merge()
}


}
}

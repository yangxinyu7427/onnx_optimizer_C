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

ModelProto OptimizeWithModels(
    ModelProto& mp_in1,
    ModelProto& mp_in2,
    std::string& mp_name1,
    std::string& mp_name2) {

  ModelProto model=model_merge(&mp_in1,&mp_in2,mp_name1,mp_name2);
  auto new_model = onnx::optimization::OptimizeFixed(
      model, onnx::optimization::GetFuseAndEliminationPass());
  onnx::checker::check_model(new_model);
  return new_model;
}


}
}

//
// Created by xyyang on 23-11-14.
//
#include <onnxoptimizer/query_c_api/optimize_c_api.cpp>
#include <onnxoptimizer/model_util.h>
#include <onnxoptimizer/optimize.h>
int main(int argc, char* argv[]) {
  ONNX_NAMESPACE::ModelProto model1,model2;
  onnx::optimization::loadModel(&model1, "/home/xyyang/PycharmProjects/pythonProject/model_lr.onnx", true);
  onnx::checker::check_model(model1);
  onnx::optimization::loadModel(&model2, "/home/xyyang/PycharmProjects/pythonProject/model_linear.onnx", true);
  onnx::checker::check_model(model2);
  std::string prefix1="model_lr";
  std::string prefix2="model_linear";
  auto model_merged=onnx::optimization::OptimizeWithModels(model1,model2, prefix1,prefix2);
  auto model_opted = onnx::optimization::OptimizeFixed(
      model_merged, onnx::optimization::GetFuseAndEliminationPass());
  onnx::checker::check_model(model_opted);
  onnx::optimization::saveModel(&model_opted,"/home/xyyang/PycharmProjects/pythonProject/model_opted.onnx");
}

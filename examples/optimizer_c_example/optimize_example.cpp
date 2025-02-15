//
// Created by xyyang on 23-11-14.
//
//#include <onnxoptimizer/model_util.h>

#include <string>

#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

int main(int argc, char* argv[]) {
  //  ONNX_NAMESPACE::ModelProto model1,model2;
  //  onnx::optimization::loadModel(&model1, "../examples/onnx_input_model/model_lr.onnx", true);
  //  onnx::checker::check_model(model1);
  //  onnx::optimization::loadModel(&model2, "../examples/onnx_input_model/model_linear.onnx", true);
  //  onnx::checker::check_model(model2);
  //  std::string prefix1="model_lr";
  //  std::string prefix2="model_linear";
  //  auto model_opted=onnx::optimization::OptimizeWithModels(model1,model2, prefix1,prefix2);
  //  onnx::optimization::saveModel(&model_opted,"../examples/onnx_output_model/model_opted.onnx");
  //  std::string path1="../examples/onnx_input_model/model_lr.onnx";
  //  std::string path2="../examples/onnx_input_model/model_linear.onnx";

  std::string path1="../examples/onnx_input_model/neo_dt.onnx";
  std::string path2="../examples/onnx_input_model/neo_lr2.onnx";
  std::string path3="../examples/onnx_input_model/neo_sgd2.onnx";
  //  std::string pre1="model_lr_1_";
  //  std::string pre2="model_linear_1_";
  std::string pre1="dt_1_";
  std::string pre2="lr_1_";
  std::string pre3="sgd_1_";
  std::string pre4="";
  //std::string out_path="../examples/onnx_output_model/model_opted.onnx";
  std::string out_path="../examples/onnx_output_model/model_opted.onnx";
  optimize_with_model_path(path1,path2,pre1,pre2,out_path);
  optimize_with_model_path(path3,out_path,pre3,pre4,out_path);
  //optimize_with_model_path(out_path,path1,pre4,pre3,out_path);
}

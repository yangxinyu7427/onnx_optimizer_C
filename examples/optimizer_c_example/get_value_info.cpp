//
// Created by xyyang on 23-12-15.
//
//
// Created by xyyang on 23-12-15.
//
#include <string>

#include "onnxoptimizer/query_c_api/predicate_push_down.cpp"

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
  std::string path1="../examples/onnx_input_model/model_lr.onnx";
  std::string name="count_clicks";
  onnx::optimization::get_value_info_from_onnx_model(path1,name);
}
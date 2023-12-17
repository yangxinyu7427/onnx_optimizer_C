//
// Created by xyyang on 23-12-15.
//
#include <string>
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

int main(int argc, char* argv[]) {

  std::string path="../examples/onnx_output_model/model_opted.onnx";
  std::string predicate1="Equal";
  std::string type="int";
  std::string name1="model_lr";
  std::string predicate2="Equal";
  std::string name2="model_linear";
  std::string output_model_path="../examples/onnx_output_model/model_pushed.onnx";
  std::string predicate3="And";

  merge_single_model_with_predicate(path,predicate2,type,name2);
  merge_single_model_with_predicate(output_model_path,predicate1,type,name1);
  merge_double_models_with_predicate(output_model_path,predicate3,name1,name2);
}
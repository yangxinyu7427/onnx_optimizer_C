//
// Created by xyyang on 23-11-14.
//
//#include <onnxoptimizer/model_util.h>

#include <string>
#include <vector>
#include <iostream>
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

// 1: ==, 2: <, 3: <=, 4: >, 5: >= 
int main(int argc, char* argv[]) {
  std::string path1 = "../examples/onnx_input_model/titanic_pipeline.onnx";
  std::string path2 = "../examples/onnx_input_model/iris_dataset_pipeline.onnx";
  std::string path3 = "/home/ding/duckdb_project/onnx_optimizer_C/examples/onnx_input_model/house_16H_d10_l281_n561_20240922063836.onnx";
  std::string testmodelpath = "/home/ding/duckdb_project/onnx_optimizer_C/examples/model4test/convert_test.onnx";
  std::string testmodelpath1 = "/home/ding/duckdb_project/onnx_optimizer_C/examples/model4test/clf2regtest.onnx";
  std::string testmodelpath2 = "/home/ding/duckdb_project/onnx_optimizer_C/examples/model4test/wine_quality_d11_l280_n559_20241209164224_with_zipmap.onnx";
  std::string testmodelpath3 = "/home/ding/duckdb_project/onnx_optimizer_C/examples/model4test/wine_quality_d11_l280_n559_20241209164224.onnx";
  // ?
  // std::vector<std::string> features = {};
  // std::string path3 = "../examples/onnx_input_model/house_16H_d10_l281_n561_20240922063836.onnx";
  // optimize_on_decision_tree_predicate(path1, 2, 10);
  // optimize_on_decision_tree_predicate(path2, 2, 10);
  // std::cout << optimize_on_decision_tree_predicate(path3, 4, 10, &features) << std::endl;
  std::cout << optimize_on_decision_tree_predicate(testmodelpath3, 1, 1) << std::endl;
  // std::cout << optimize_on_decision_tree_predicate(testmodelpath, 1, 1) << std::endl;
  return 0;
}

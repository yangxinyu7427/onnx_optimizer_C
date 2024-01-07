//
// Created by xyyang on 23-12-15.
//
#include <string>
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

int main(int argc, char* argv[]) {

  std::string path="../examples/onnx_output_model/model_opted.onnx";
  std::string predicate1="Greater";
  std::string predicate2="Or";
  std::string predicate3="And";
  std::string type="int";
  std::string type2="bool";
  std::string name1="model_lr_1";
  std::string name2="model_linear_1";
  std::string name3="model_lr_1_model_linear_1";
  std::string name4="model_lr_2";
  std::string name5="model_lr_1_model_linear_1_model_lr_2";
//  merge_single_model_with_predicate(path,predicate2,type,name2);
//  merge_single_model_with_predicate(path,predicate1,type,name1);
//  merge_double_models_with_predicate(path,predicate3,name1,name2);
//  merge_single_model_with_predicate(path,predicate4,type,name3);
//  merge_double_models_with_predicate(path,predicate2,name1,name2);
//  merge_single_model_with_predicate(path,predicate4,type,name3);

  merge_single_model_with_predicate(path,predicate1,type,name1,1);
  merge_single_model_with_predicate(path,predicate1,type,name1,2);
  merge_single_model_with_predicate(path,predicate1,type,name2,1);
  merge_double_models_with_predicate(path,predicate3,name1,name2);
//  merge_single_model_with_predicate(path,predicate1,type,name4,1);
//  merge_double_models_with_predicate(path,predicate2,name3,name4);

//  merge_double_models_with_predicate(path,predicate3,name1,name2);
//  merge_single_model_with_predicate(path,predicate1,type,name4);
//  merge_double_models_with_predicate(path,predicate3,name3,name4);
//  merge_single_model_with_predicate(path,predicate1,type,name5);

}
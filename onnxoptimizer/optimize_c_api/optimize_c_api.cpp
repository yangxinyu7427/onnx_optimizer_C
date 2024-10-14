//
// Created by xyyang on 23-11-14.
//
#include <filesystem>
#include <fstream>

#include "optimize_c_api.h"
//#include "onnxoptimizer/query_c_api/model_merge.cpp"
#include "onnxoptimizer/query_c_api/predicate_push_down.cpp"
#include "onnxoptimizer/query_c_api/redundant_calculation_detection.cpp"

void optimize_with_model_path(
    std::string& mp_in_path1,
    std::string& mp_in_path2,
    std::string& mp_name1,
    std::string& mp_name2,
    std::string& mp_out_path){
    onnx::optimization::OptimizeWithModels(mp_in_path1,mp_in_path2,mp_name1,mp_name2,mp_out_path);
}

void merge_single_model_with_predicate(
    std::string& onnx_model_path,
    std::string& predicate,
    std::string& value_type,
    std::string prefix,
    int count){
    onnx::optimization::merge_single_model_with_predicate(onnx_model_path,predicate,value_type,prefix,count);
}

void merge_double_models_with_predicate(std::string& onnx_model_path,std::string& predicate,
                                        std::string prefix_l,std::string prefix_r){
    onnx::optimization::merge_double_models_with_predicate(onnx_model_path,predicate,prefix_l,prefix_r);
}

void merge_with_model_path(std::string& mp_in_path1,std::string& mp_in_path2,std::string& mp_name1,std::string& mp_name2,
    std::string& mp_out_path){
    onnx::optimization::MergeWithModels(mp_in_path1,mp_in_path2,mp_name1,mp_name2,mp_out_path);
}

void optimize_on_merged_model(std::string& mp_in_path,std::string& mp_out_path){
    onnx::optimization::OptimizeOnMergedModel(mp_in_path, mp_out_path);
}

std::vector<std::string> check_redundant(std::string& changed_model_path, std::string& compared_model_path){
    return onnx::optimization::check_redundant(changed_model_path, compared_model_path);
}
void change_models(std::string& changed_model_path,std::string& output_model_path,
                   std::string& changed_input_model_path,
                   const std::vector<std::string>& output_name){
    onnx::optimization::change_models(changed_model_path, output_model_path, changed_input_model_path, output_name);
}
void add_prefix_on_model(std::string& changed_model_path, std::string& output_model_path, std::string& prefix){
    onnx::optimization::add_prefix_on_model(changed_model_path, output_model_path, prefix);
}

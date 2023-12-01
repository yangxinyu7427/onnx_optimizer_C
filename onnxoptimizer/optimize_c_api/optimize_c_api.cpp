//
// Created by xyyang on 23-11-14.
//
#include <filesystem>
#include <fstream>

#include "optimize_c_api.h"
#include "onnxoptimizer/query_c_api/model_merge.cpp"

void optimize_with_model_path(
    std::string& mp_in_path1,
    std::string& mp_in_path2,
    std::string& mp_name1,
    std::string& mp_name2,
    std::string& mp_out_path){
    onnx::optimization::OptimizeWithModels(mp_in_path1,mp_in_path2,mp_name1,mp_name2,mp_out_path);
}

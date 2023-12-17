//
// Created by xyyang on 23-12-1.
//

#ifndef ONNX_OPTIMIZER_OPTIMIZE_C_API_H
#define ONNX_OPTIMIZER_OPTIMIZE_C_API_H

void optimize_with_model_path(std::string& mp_in_path1,
                              std::string& mp_in_path2,
                              std::string& mp_name1,
                              std::string& mp_name2,
                              std::string& mp_out_path);

void merge_single_model_with_predicate(std::string& onnx_model_path, std::string& predicate,
                         std::string& value_type, std::string prefix);

void merge_double_models_with_predicate(std::string& onnx_model_path,std::string& predicate,
                                        std::string prefix_l,std::string prefix_r);
#endif  // ONNX_OPTIMIZER_OPTIMIZE_C_API_H


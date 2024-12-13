//
// Created by xyyang on 23-12-15.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/optimize.h>
#include <onnxoptimizer/model_util.h>
#include <regex>
#include <vector>

namespace onnx::optimization {


std::map<std::string, int> value_type_map = {
    {"int", 7},
    {"string", 8},
    {"bool", 9},
    {"double", 11}
};

std::map<std::string, int> predicate_result_type_map={
    {"Greater",9},
    {"Less",9},
    {"GreaterOrEqual",9},
    {"LessOrEqual",9},
    {"Equal",9},
    {"And",9},
    {"Or",9},
    {"Add",7},
    {"Mul",7},
};

void merge_single_model_with_predicate(std::string& onnx_model_path, std::string& predicate,
                                       std::string& value_type, std::string prefix, int count){
  // load model
  onnx::ModelProto onnx_model;
  onnx::optimization::loadModel(&onnx_model, onnx_model_path, true);
  onnx::checker::check_model(onnx_model);

  onnx::TensorShapeProto_Dimension input_dim1,input_dim2,output_dim;
  onnx::ValueInfoProto input,output,reshape_input;
  onnx::NodeProto node, reshape_node;
  std::string input_name=prefix+"_"+to_string(count)+"_input";
  std::string output_name=prefix+"_"+to_string(count)+"_output";
  std::string node_name=output_name+"_node";
  std::string match_str="^"+prefix;
  std::regex pattern(match_str);
  std::string match_str_end="probability$";
  std::regex pattern_end(match_str_end);
  std::regex pattern_ends("probabilities$");

  // create dim
  input_dim2.set_dim_value(1);

  // add int input, just like [0,1]
  *input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=input_dim1;
  *input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=input_dim2;
  input.mutable_type()->mutable_tensor_type()->set_elem_type(value_type_map[value_type]);
  input.set_name(input_name);

  // create output ValueInfoProto
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=output_dim;
  output.mutable_type()->mutable_tensor_type()->set_elem_type(predicate_result_type_map[predicate]);
  output.set_name(output_name);

  // create reshape node
  onnx::TensorProto tensor;
  std::string tensor_name=input_name+"_reshape_tensor";
  tensor.set_name(tensor_name);
  tensor.set_data_type(value_type_map[value_type]);
  tensor.add_dims(1);
  tensor.add_int64_data(-1);

  std::string reshape_name=input_name+"_reshape";
  std::string reshape_output_name=input_name+"_reshape_output";
  reshape_node.set_op_type("Reshape");
  reshape_node.set_name(reshape_name);
  reshape_node.add_input(input_name);
  reshape_node.add_input(tensor_name);
  reshape_node.add_output(reshape_output_name);


  // create predicate node
  node.set_op_type(predicate);
  std::string label;
  for(int i=0;i<onnx_model.graph().output_size();i++){
    if(std::regex_search(onnx_model.graph().output(i).name(), pattern)&&
        !std::regex_search(onnx_model.graph().output(i).name(), pattern_end)
        &&!std::regex_search(onnx_model.graph().output(i).name(), pattern_ends)){
      label=onnx_model.graph().output(i).name();;
    }
  }
  node.add_input(label);
  node.add_input(reshape_output_name);
  node.add_output(output_name);
  node.set_name(node_name);

  // change onnx_model
  *onnx_model.mutable_graph()->add_initializer()=tensor;
  *onnx_model.mutable_graph()->add_node()=reshape_node;
  *onnx_model.mutable_graph()->add_input()=input;
  *onnx_model.mutable_graph()->add_node()=node;
  std::regex pattern_output_label("_label$");
  std::vector<onnx::ValueInfoProto> output_list;
  std::vector<onnx::ValueInfoProto> save_output_list;
  for(int i=0;i<onnx_model.graph().output_size();i++){
    if(!std::regex_search(onnx_model.graph().output(i).name(), pattern))
      output_list.push_back(onnx_model.graph().output(i));
    else if(std::regex_search(onnx_model.graph().output(i).name(), pattern_output_label))
      save_output_list.push_back(onnx_model.graph().output(i));
  }
  onnx_model.mutable_graph()->clear_output();
  *onnx_model.mutable_graph()->add_output()=output;
  for(int i=0;i<output_list.size();i++){
    *onnx_model.mutable_graph()->add_output()=output_list.at(i);
  }
  for(int i=0;i<save_output_list.size();i++){
    *onnx_model.mutable_graph()->add_output()=save_output_list.at(i);
  }


  onnx::checker::check_model(onnx_model);
  saveModel(&onnx_model,onnx_model_path);

}

void merge_double_models_with_predicate(std::string& onnx_model_path,std::string& predicate,
                                        std::string prefix_l,std::string prefix_r){
  // load model
  onnx::ModelProto onnx_model;
  onnx::optimization::loadModel(&onnx_model, onnx_model_path, true);
  onnx::checker::check_model(onnx_model);

  onnx::TensorShapeProto_Dimension input_dim_l,input_dim_r,output_dim;
  onnx::ValueInfoProto input,output;
  onnx::NodeProto node;
  std::string match_str_l="^"+prefix_l;
  std::string match_str_r="^"+prefix_r;
  std::string match_str_end="probability$";
  std::string output_name=prefix_l+"_"+prefix_r+"_"+predicate;
  std::string node_name=output_name+"_node";
  std::regex pattern_l(match_str_l);
  std::regex pattern_r(match_str_r);
  std::regex pattern_end(match_str_end);
  std::regex pattern_ends("probabilities$");

  // search output ValueInfoProto
  onnx::ValueInfoProto output_l;
  onnx::ValueInfoProto output_r;

  for(int i=onnx_model.graph().output_size()-1;i>=0;i--){
      if(std::regex_search(onnx_model.graph().output(i).name(), pattern_l)&&
        !std::regex_search(onnx_model.graph().output(i).name(), pattern_end)&&
        !std::regex_search(onnx_model.graph().output(i).name(), pattern_ends)){
        output_l=onnx_model.graph().output(i);
      }
      else if(std::regex_search(onnx_model.graph().output(i).name(), pattern_r)&&
               !std::regex_search(onnx_model.graph().output(i).name(), pattern_end)&&
        !std::regex_search(onnx_model.graph().output(i).name(), pattern_ends)){
        output_r=onnx_model.graph().output(i);
      }
  }

  // create output ValueInfoProto
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=output_dim;
  output.mutable_type()->mutable_tensor_type()->set_elem_type(predicate_result_type_map[predicate]);
  output.set_name(output_name);

  // create predicate node
  node.set_op_type(predicate);
  node.add_input(output_l.name());
  node.add_input(output_r.name());
  node.add_output(output_name);
  node.set_name(node_name);

  // change onnx_model
  *onnx_model.mutable_graph()->add_node()=node;
  std::regex pattern_output_label("_label$");
  std::vector<onnx::ValueInfoProto> output_list;
  std::vector<onnx::ValueInfoProto> save_output_list;
  for(int i=0;i<onnx_model.graph().output_size();i++){
      std::string tmp=onnx_model.graph().output(i).name();
      if(!std::regex_search(onnx_model.graph().output(i).name(), pattern_l) &&
          !std::regex_search(onnx_model.graph().output(i).name(), pattern_r)){
        output_list.push_back(onnx_model.graph().output(i));
      } else if(std::regex_search(onnx_model.graph().output(i).name(), pattern_output_label))
        save_output_list.push_back(onnx_model.graph().output(i));
  }
  onnx_model.mutable_graph()->clear_output();
  *onnx_model.mutable_graph()->add_output()=output;
  for(int i=0;i<output_list.size();i++){
      *onnx_model.mutable_graph()->add_output()=output_list.at(i);
  }
  for(int i=0;i<save_output_list.size();i++){
      *onnx_model.mutable_graph()->add_output()=save_output_list.at(i);
  }


  onnx::checker::check_model(onnx_model);
  saveModel(&onnx_model,onnx_model_path);
}

}
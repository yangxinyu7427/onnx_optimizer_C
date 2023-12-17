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

//void get_value_info_from_onnx_model(std::string& onnx_model_path, std::string& name){
//  onnx::ModelProto onnx_model;
//  onnx::optimization::loadModel(&onnx_model, onnx_model_path, true);
//  onnx::checker::check_model(onnx_model);
//  onnx::ValueInfoProto input;
//
//  for(int i=0;i<onnx_model.graph().input_size();i++){
//    if(onnx_model.mutable_graph()->mutable_input(i)->name()==name){
//      input.CopyFrom(onnx_model.mutable_graph()->input(i));
//      saveValueInfo(&input,"new_value_info.onnx");
//      onnx::ValueInfoProto input_from_pre;
//      loadValueInfo(&input_from_pre,"new_value_info.onnx");
//      std::cout<<input_from_pre.name();
//    }
//  }
//
//}

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
};

void merge_single_model_with_predicate(std::string& onnx_model_path, std::string& predicate,
                                       std::string& value_type, std::string prefix){
  // load model
  onnx::ModelProto onnx_model;
  onnx::optimization::loadModel(&onnx_model, onnx_model_path, true);
  onnx::checker::check_model(onnx_model);

  onnx::TensorShapeProto_Dimension input_dim1,input_dim2,output_dim;
  onnx::ValueInfoProto input,output;
  onnx::NodeProto node;
  std::string input_name=prefix+"_"+predicate+"_"+value_type;
  std::string output_name=prefix+"_"+predicate;
  std::string node_name=output_name+"_node";
  std::string match_str="^"+prefix;
  std::regex pattern(match_str);

  // create dim
  // input_dim1.set_dim_value(-1);
  input_dim2.set_dim_value(1);

  // add int input, just like [0,1]
  *input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=input_dim1;
  *input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=input_dim2;
  input.mutable_type()->mutable_tensor_type()->set_elem_type(value_type_map[value_type]);
  input.set_name(input_name);

  // create output ValueInfoProto
  // output_dim.set_dim_value(-1);
  *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=output_dim;
  output.mutable_type()->mutable_tensor_type()->set_elem_type(predicate_result_type_map[predicate]);
  output.set_name(output_name);

  // create predicate node
  node.set_op_type(predicate);
  // todo how to get output label
  std::string label=prefix+"_output_label";
  node.add_input(label);
  node.add_input(input_name);
  node.add_output(output_name);
  node.set_name(node_name);

  // change onnx_model
  *onnx_model.mutable_graph()->add_input()=input;
  *onnx_model.mutable_graph()->add_node()=node;
  std::vector<onnx::ValueInfoProto> output_list;
  for(int i=0;i<onnx_model.graph().output_size();i++){
    if(!std::regex_search(onnx_model.graph().output(i).name(), pattern)){
      output_list.push_back(onnx_model.graph().output(i));
    }
  }
  onnx_model.mutable_graph()->clear_output();
  for(int i=0;i<output_list.size();i++){
    *onnx_model.mutable_graph()->add_output()=output_list.at(i);
  }
  *onnx_model.mutable_graph()->add_output()=output;

  // print
  for(int i=0;i<onnx_model.graph().output_size();i++){
      std::cout<<"this is int output "<< onnx_model.graph().output(i).name()<<std::endl;
      std::cout<<onnx_model.graph().output(i).type().tensor_type().elem_type()<<std::endl;
  }

  onnx::checker::check_model(onnx_model);
  std::string output_model_path="../examples/onnx_output_model/model_pushed.onnx";
  saveModel(&onnx_model,output_model_path);

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
  std::string output_name=prefix_l+"_"+prefix_r+"_"+predicate;
  std::string node_name=output_name+"_node";
  std::regex pattern_l(match_str_l);
  std::regex pattern_r(match_str_r);

  // search output ValueInfoProto
  onnx::ValueInfoProto output_l;
  onnx::ValueInfoProto output_r;

  for(int i=0;i<onnx_model.graph().output_size();i++){
      if(std::regex_search(onnx_model.graph().output(i).name(), pattern_l)){
        output_l=onnx_model.graph().output(i);
      } else if(std::regex_search(onnx_model.graph().output(i).name(), pattern_r)){
        output_r=onnx_model.graph().output(i);
      }
  }

  // create output ValueInfoProto
  // output_dim.set_dim_value(-1);
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
  std::vector<onnx::ValueInfoProto> output_list;
  for(int i=0;i<onnx_model.graph().output_size();i++){
      if(!std::regex_search(onnx_model.graph().output(i).name(), pattern_l) &&
          !std::regex_search(onnx_model.graph().output(i).name(), pattern_r)){
        output_list.push_back(onnx_model.graph().output(i));
      }
  }
  onnx_model.mutable_graph()->clear_output();
  for(int i=0;i<output_list.size();i++){
      *onnx_model.mutable_graph()->add_output()=output_list.at(i);
  }
  *onnx_model.mutable_graph()->add_output()=output;
  // print
  for(int i=0;i<onnx_model.graph().output_size();i++){
      std::cout<<"this is int output "<< onnx_model.graph().output(i).name()<<std::endl;
      std::cout<<onnx_model.graph().output(i).type().tensor_type().elem_type()<<std::endl;
  }

  onnx::checker::check_model(onnx_model);
  std::string output_model_path="../examples/onnx_output_model/model_pushed.onnx";
  saveModel(&onnx_model,output_model_path);
}

}
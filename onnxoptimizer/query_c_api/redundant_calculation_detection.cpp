//
// Created by xyyang's mac on 2024/7/1.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/optimize.h>
#include <onnxoptimizer/query_c_api/model_merge.cpp>
#include <onnxoptimizer/model_util.h>
#include <vector>
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/proto_utils.h"

#include "onnxoptimizer/pass_manager.h"
#include "onnxoptimizer/pass_registry.h"
namespace onnx::optimization {
  void add_prefix_on_model(std::string& changed_model_path, std::string& output_model_path, std::string& prefix){
    ModelProto model_changed, model_output;
    onnx::optimization::loadModel(&model_changed, changed_model_path, true);
    add_model_prefix(&model_changed,prefix,&model_output);
    saveModel(&model_output,output_model_path);
  }

  void mark_reachable_nodes(std::shared_ptr<Graph> graph, const std::string& input_name, std::unordered_set<std::string>& reachable_nodes) {
    for(auto node:graph->nodes()){
      if(reachable_nodes.find(node->name())!=reachable_nodes.end())
        continue;
      for(auto tmp_input:node->inputs()){
        if(tmp_input->uniqueName()==input_name){
          reachable_nodes.insert(node->name());
          for(auto output:node->outputs()){
            mark_reachable_nodes(graph,output->uniqueName(),reachable_nodes);
          }
        }
      }
    }
  }

  std::vector<std::string> check_redundant(std::string& changed_model_path, std::string& compared_model_path){
    ModelProto model_changed, model_compared;
    onnx::optimization::loadModel(&model_changed, changed_model_path, true);
    onnx::optimization::loadModel(&model_compared, compared_model_path, true);
    std::shared_ptr<Graph> g_changed(ImportModelProto(model_changed));
    std::shared_ptr<Graph> g_compared(ImportModelProto(model_compared));
    auto node_list_changed = g_changed->nodes();
    auto node_list_compared = g_compared->nodes();
    // 使用hashmap记录被比较的模型的结点
    std::unordered_map<Node *, Node *, CSENodeHash, CSEEqual> hash_map;
    for (auto it = node_list_compared.begin(); it != node_list_compared.end(); ++it) {
      auto node = *it;
      if (!node->hasUses() || !IsSupportedByCSE(node))
        continue;

      if (hash_map.find(node) == hash_map.end())
        hash_map[node] = node;
      else
        //同一个模型内应该不会有重复的算子？
        continue;
    }
    std::vector<Node *> same_node_list;
    for(auto it = node_list_changed.begin(); it != node_list_changed.end(); ++it){
      auto node = *it;
      if (!node->hasUses() || !IsSupportedByCSE(node))
        continue;
      if (hash_map.find(node) != hash_map.end()) {
        same_node_list.push_back(node);
      }
    }
    std::vector<Node *> same_node_input_list(same_node_list);
    //将那些输出均有下一个相同结点将其作为输入的结点删掉
    for(int i = 0;i<same_node_list.size(); i++){
      auto node=same_node_list[i];
      auto outputs = node->outputs();
      int tar = outputs.size();
      int count=0;
      for(int i=0;i<outputs.size();i++){
        bool found= false;
        auto output=outputs[i];
        for(Node* node1: same_node_input_list){
          if(found) break;
          auto inputs = node1->inputs();
          for(int j=0;j<inputs.size();j++){
            if(output->uniqueName()==inputs[j]->uniqueName()){
              count++;
              found= true;
              break;
            }
          }
        }
      }

      if(count==tar) {
        auto it = std::remove(same_node_list.begin(), same_node_list.end(), node);
        same_node_list.erase(it, same_node_list.end());
        i--;
      }
    }

    // 检测没有依赖的node
    std::vector<Node *> final_node_list;
    for(int i=0; i<same_node_list.size();i++){
      auto it = std::remove(same_node_input_list.begin(), same_node_input_list.end(), same_node_list[i]);
      same_node_input_list.erase(it, same_node_input_list.end());
    }

    for(int i=0; i<same_node_list.size();i++){
      auto inputs=same_node_list[i]->inputs();
      bool sign=false;
      for(int j=0;j<inputs.size();j++){
        if(sign) break;
        auto input=inputs[j];
        for(int m=0; m<same_node_input_list.size(); m++){
          if(sign) break;
          auto outputs=same_node_input_list[m]->outputs();
          for(int n=0; n<outputs.size(); n++){
            if(outputs[n]->uniqueName()==input->uniqueName()){
              sign=true;
              final_node_list.push_back(same_node_list[i]);
              break;
            }
          }
        }
      }
    }

    std::vector<std::string> value_name_list;
    //for(int i=0;i<final_node_list.size();i++){
    for(auto node: final_node_list){
      for(auto value:node->outputs()){
        value_name_list.push_back(value->uniqueName());
      }
    }
    //}
    return value_name_list;
  }

  void change_models(std::string& changed_model_path, std::string& output_model_path, std::string& changed_input_model_path,const std::vector<std::string>& output_name){
    ModelProto model_changed;
    onnx::optimization::loadModel(&model_changed, changed_model_path, true);
    std::shared_ptr<Graph> g_changed(ImportModelProto(model_changed));
    onnx::ValueInfoProto output;
    onnx::TypeProto input_value_type;
    for(const std::string& name:output_name){
      bool found=false;
      for(auto node:g_changed->nodes()){
        if(found) break;
        for(auto tmp_output:node->outputs()){
          if(tmp_output->uniqueName()==name){
            output.set_allocated_type(tmp_output->valueType());
            output.set_name(name);
            *model_changed.mutable_graph()->add_output() = output;
            input_value_type.CopyFrom(output.type());
            found= true;
            break;
          }
        }
      }
    }
    //onnx::checker::check_model(model_changed);
    saveModel(&model_changed,output_model_path);

    // below del input
    ModelProto model_input_changed;
    onnx::optimization::loadModel(&model_input_changed, changed_model_path, true);
    std::shared_ptr<Graph> g_input_changed(ImportModelProto(model_changed));
    model_input_changed.mutable_graph()->clear_input();
    onnx::ValueInfoProto input;
    input.mutable_type()->CopyFrom(input_value_type);
    input.set_name(output_name[0]);
    // 暂时固定输入形状和类型
    onnx::TensorShapeProto_Dimension input_dim1,input_dim2;
    //input_dim2.set_dim_value(1);
    *input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=input_dim1;
    *input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()=input_dim2;
    input.mutable_type()->mutable_tensor_type()->set_elem_type(1);
    *model_input_changed.mutable_graph()->add_input()=input;
    // del nodes without input
    std::unordered_set<std::string> reachable_nodes;
    mark_reachable_nodes(g_input_changed,output_name[0],reachable_nodes);
    // 支持经查询内冗余消除策略新增的input
    for(int i=0;i<model_changed.mutable_graph()->input_size();i++){
      auto input=model_changed.mutable_graph()->input(i);
      if(input.name().find("input") != std::string::npos){
        mark_reachable_nodes(g_input_changed,input.name(),reachable_nodes);
        *model_input_changed.mutable_graph()->add_input()=input;
      }
    }
    for(int i=0;i<model_input_changed.mutable_graph()->node_size();i++){
      if(reachable_nodes.find(model_input_changed.mutable_graph()->node(i).name())==reachable_nodes.end()){
        model_input_changed.mutable_graph()->mutable_node()->DeleteSubrange(i,1);
        i--;
      }
    }
    saveModel(&model_input_changed,changed_input_model_path);
  }



}

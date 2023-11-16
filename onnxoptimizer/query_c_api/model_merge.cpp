//
// Created by xyyang on 23-11-14.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>
#include <onnxoptimizer/optimize.h>

namespace ONNX_NAMESPACE {
namespace optimization {

std::string add_prefix(std::string prefix,std::string name){
  if(!name.empty())
    return prefix.append(name);
  else return name;
}

void add_graph_prefix(
    GraphProto* graph,
    std::string& prefix,
    std::set<std::string> input_names,
    bool sub_graph,
    bool inplace,
    std::map<std::string,std::string> name_map
    ){

  if(sub_graph){
    for(auto & it_node:graph->node()){

      for(auto & it_input:it_node.input())
        if(input_names.find(it_input)==input_names.end())
          name_map[it_input]= add_prefix(prefix,it_input);

      for(auto & it_output:it_node.output())
        name_map[it_output]= add_prefix(prefix,it_output);
    }//end node

    for(auto & it_input:graph->input()){
      std::string name=it_input.name();
      if(input_names.find(name)==input_names.end())
        name_map[name]= add_prefix(prefix,name);
    }

  }else{
    for(auto & it_input:graph->input())
      input_names.insert(it_input.name());

    for(auto & it_node:graph->node()){
      for(auto & it_input:it_node.input())
        if(input_names.find(it_input)==input_names.end())
          name_map[it_input]= add_prefix(prefix,it_input);

      for(auto & it_output:it_node.output())
        name_map[it_output]= add_prefix(prefix,it_output);
    }// end node
  }//end sub_graph

  for(auto & it_output:graph->output())
    name_map[it_output.name()]= add_prefix(prefix,it_output.name());

  for(auto & it_node:*graph->mutable_node()){
    it_node.set_name(add_prefix(prefix,it_node.name()));
    for(auto & it_attr:*it_node.mutable_attribute()){
      if(it_attr.has_g()){
        add_graph_prefix(it_attr.mutable_g(),prefix,input_names, true, true,name_map);
      }
    }
  }// end node

  for(auto & it_init:graph->initializer())
    name_map[it_init.name()]= add_prefix(prefix,it_init.name());

  for(auto & it_sparse_init:graph->sparse_initializer()){
    name_map[it_sparse_init.values().name()]= add_prefix(prefix,it_sparse_init.values().name());
    name_map[it_sparse_init.indices().name()]= add_prefix(prefix,it_sparse_init.indices().name());
  }

  for(auto & it_value_info:graph->value_info()){
    name_map[it_value_info.name()]= add_prefix(prefix,it_value_info.name());
  }

  for(auto it_node:*graph->mutable_node()){
    for(int i=0;i<it_node.input_size();i++)
      if(name_map.find(it_node.input(i))!=name_map.end())
        it_node.set_input(i,name_map[it_node.input(i)]);

    for(int i=0;i<it_node.output_size();i++)
      if(name_map.find(it_node.output(i))!=name_map.end())
        it_node.set_output(i,name_map[it_node.output(i)]);
  }






}

void add_model_prefix(
    ModelProto* model,
    std::string& prefix,
    ModelProto* model_with_prefix){
  model_with_prefix->CopyFrom(*model);
  GraphProto graph;
  graph.CopyFrom(model_with_prefix->graph());
  std::set<std::string> input_names;
  std::map<std::string,std::string> name_map;
  add_graph_prefix(&graph,prefix,input_names,false,true,name_map);
}


void model_merge(
    ModelProto* m1,
    ModelProto* m2,
    std::string& mp_name1,
    std::string& mp_name2,
    ModelProto* mp_merged){
  if (m1->ir_version()!=m2->ir_version()){
    std::cerr << "Warning: onnx ir versions are different! " << std::endl;
  }
  std::map<std::string,::onnx::OperatorSetIdProto> map={};
  for(const auto & it : m1->opset_import()){
    if(map.find(it.domain()) == map.end()){
      map[it.domain()]=it;
      if("ai.onnx.ml"!=it.domain()){
        if(map[it.domain()].version()>18||map[it.domain()].version()<16){
          map[it.domain()].set_version(18);
        }
      }
    }else{
      auto found_entry=map[it.domain()];
      auto found_version=found_entry.version();
      if(found_version>it.version()){
        map[it.domain()]=it;
      }
    }
  }

  for(const auto & it : m2->opset_import()){
    if(map.find(it.domain()) == map.end()){
      map[it.domain()]=it;
      if("ai.onnx.ml"!=it.domain()){
        if(map[it.domain()].version()>18||map[it.domain()].version()<16){
          map[it.domain()].set_version(18);
        }
      }
    }else{
      auto found_entry=map[it.domain()];
      auto found_version=found_entry.version();
      if(found_version>it.version()){
        map[it.domain()]=it;
      }
    }
  }
  std::vector<OperatorSetIdProto> opset_import_list;
  for(const auto & it:map){
    opset_import_list.push_back(it.second);
  }

  ModelProto m1_with_prefix,m2_with_prefix;

  add_model_prefix(m1,mp_name1,&m1_with_prefix);
  add_model_prefix(m2,mp_name2,&m2_with_prefix);
}
}
}

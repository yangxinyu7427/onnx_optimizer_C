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
      if(input_names.find(it_input.name())==input_names.end())
        name_map[it_input.name()]= add_prefix(prefix,it_input.name());
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

  for(auto it_input:*graph->mutable_input())
    if(name_map.find(it_input.name())!=name_map.end())
      it_input.set_name(name_map[it_input.name()]);

  for(auto it_output:*graph->mutable_output())
    if(name_map.find(it_output.name())!=name_map.end())
      it_output.set_name(name_map[it_output.name()]);

  for(auto it_init:*graph->mutable_initializer())
    if(name_map.find(it_init.name())!=name_map.end())
      it_init.set_name(name_map[it_init.name()]);

  for(auto it_sparse:*graph->mutable_sparse_initializer()){
    if(name_map.find(it_sparse.values().name())!=name_map.end())
      it_sparse.mutable_values()->set_name(name_map[it_sparse.values().name()]);
    if(name_map.find(it_sparse.indices().name())!=name_map.end())
      it_sparse.mutable_indices()->set_name(name_map[it_sparse.indices().name()]);
  }

  for(auto it_value_info:*graph->mutable_value_info())
    if(name_map.find(it_value_info.name())!=name_map.end())
      it_value_info.set_name(name_map[it_value_info.name()]);

}

void add_model_prefix(
    ModelProto* model,
    std::string& prefix,
    ModelProto* model_with_prefix){

  model_with_prefix->CopyFrom(*model);
  std::set<std::string> input_names;
  std::map<std::string,std::string> name_map;
  add_graph_prefix(model_with_prefix->mutable_graph(),prefix,input_names,false,true,name_map);

  std::map<std::string,std::string> f_name_map;
  for(auto & it_func:*model_with_prefix->mutable_functions()){
    f_name_map[it_func.name()]= add_prefix(prefix,it_func.name());
    it_func.set_name(add_prefix(prefix,it_func.name()));
  }

  for(auto & it_func:*model_with_prefix->mutable_functions()) {
    for(auto & it_node:*it_func.mutable_node()){
      if(f_name_map.find(it_node.op_type())!=f_name_map.end()){
        it_node.set_op_type(f_name_map[it_node.op_type()]);
      }
    }
  }

  for(auto & it_g_node:*model_with_prefix->mutable_graph()->mutable_node()){
    if(f_name_map.find(it_g_node.op_type())!=f_name_map.end())
      it_g_node.set_op_type(f_name_map[it_g_node.op_type()]);
  }

}

void merge_project_graphs(
    GraphProto* g1,
    GraphProto* g2,
    GraphProto* g_merged
    ){
  GraphProto g;
  // add node
  for(auto it_node:g1->node())
    *g.add_node()=it_node;
  for(auto it_node:g2->node())
    *g.add_node()=it_node;

  // add input
  std::set<std::string> input_names;
  for(auto it_input:g1->input())
    if(input_names.find(it_input.name())==input_names.end()){
      *g.add_input()=it_input;
      input_names.insert(it_input.name());
    }
  for(auto it_input:g2->input())
    if(input_names.find(it_input.name())==input_names.end()){
      *g.add_input()=it_input;
      input_names.insert(it_input.name());
    }

  // add output
  for(auto it_output:g1->output())
    *g.add_output()=it_output;
  for(auto it_output:g2->output())
    *g.add_output()=it_output;

  // add init
  for(auto it_init:g1->initializer())
    *g.add_initializer()=it_init;
  for(auto it_init:g2->initializer())
    *g.add_initializer()=it_init;

  // add sparse_initializer
  for(auto it_sparse:g1->sparse_initializer())
    *g.add_sparse_initializer()=it_sparse;
  for(auto it_sparse:g2->sparse_initializer())
    *g.add_sparse_initializer()=it_sparse;

  // add value_info
  for(auto it_value_info:g1->value_info())
    *g.add_value_info()=it_value_info;
  for(auto it_value_info:g2->value_info())
    *g.add_value_info()=it_value_info;

  // add name
  std::string name=g1->mutable_name()->append("_");
  name=name.append(g2->name());
  g.set_name(name);

  // add doc_string
  g.set_doc_string("graph merged");

  g_merged=&g;
}

ModelProto create_model(GraphProto* graph,
                        int64_t ir_version,
                        std::vector<OperatorSetIdProto> opset_import_list){
  ModelProto model;
  model.set_ir_version(ir_version);
  model.mutable_graph()->CopyFrom(*graph);
  for(auto it:opset_import_list)
    *model.add_opset_import()=it;
  model.set_model_version(1);
  model.set_producer_name("onnx.expr_compose.merge_models");
  model.set_producer_version("1.0");
  model.set_domain("");
  return model;
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
  auto ir_version=m1->ir_version();
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

  ModelProto *m1_with_prefix, *m2_with_prefix;
  // add prefix
  add_model_prefix(m1,mp_name1,m1_with_prefix);
  add_model_prefix(m2,mp_name2,m2_with_prefix);

  GraphProto* graph_merged;
  merge_project_graphs(m1_with_prefix->mutable_graph(),m2_with_prefix->mutable_graph(),graph_merged);

  ModelProto model=create_model(graph_merged,ir_version,opset_import_list);

  // merge model metadata props
  std::map<std::string,std::string> props_map;
  for(auto it:m1_with_prefix->metadata_props())
    props_map[it.key()]=it.value();
  for(auto it:m2_with_prefix->metadata_props()){
    if(props_map.find(it.key())!=props_map.end()){
      std::string value=props_map[it.key()];
      if(it.value()!=value)
        std::cerr << "Can't merge models with different values for the same model metadata property." << std::endl;
    }else
      props_map[it.key()]=it.value();
  }

  // add metadata_props
  model.clear_metadata_props();
  for(auto & it:props_map){
    auto entry=model.add_metadata_props();
    entry->set_key(it.first);
    entry->set_value(it.second);
  }

  // merge functions

  model.mutable_functions()->MergeFrom(*m1_with_prefix->mutable_functions());
  model.mutable_functions()->MergeFrom(*m2_with_prefix->mutable_functions());
  checker::check_model(model, false);
  mp_merged=&model;
}





}//end namespace
}//end namespace

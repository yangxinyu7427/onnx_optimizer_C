//
// Created by xyyang on 23-11-14.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>
#include <onnxoptimizer/optimize.h>

namespace ONNX_NAMESPACE {
namespace optimization {

void add_model_prefix(
    ModelProto* model,
    std::string& prefix,
    ModelProto* model_prefixed){

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

  ModelProto m1_prefix;
  add_model_prefix(m1,mp_name1,&m1_prefix);

}
}
}

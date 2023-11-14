//
// Created by xyyang on 23-11-14.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>
#include <onnxoptimizer/optimize.h>

namespace ONNX_NAMESPACE {
namespace optimization {
void model_merge(
    const ModelProto& m1,
    const ModelProto& m2,
    const std::string& mp_name1,
    const std::string& mp_name2,
    const ModelProto& mp_merged){
  if (m1.ir_version()!=m2.ir_version()){
    std::cerr << "Warning: onnx ir versions are different! " << std::endl;
  }
  std::map<std::string,::onnx::OperatorSetIdProto> map={};
  for(const auto & it : m1.opset_import()){
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

  for(const auto & it : m2.opset_import()){
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
}
}
}

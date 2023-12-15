//
// Created by xyyang on 23-12-15.
//
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/optimize.h>
#include <onnxoptimizer/model_util.h>

namespace onnx::optimization {

//std::map predicate_map={"equal":""};
void get_value_info_from_onnx_model(std::string& onnx_model_path, std::string& name){
  onnx::ModelProto onnx_model;
  onnx::optimization::loadModel(&onnx_model, onnx_model_path, true);
  onnx::checker::check_model(onnx_model);
  onnx::ValueInfoProto input;
  // todo add ValueInfoProto pre-create

    for(int i=0;i<onnx_model.graph().input_size();i++){
      if(onnx_model.mutable_graph()->mutable_input(i)->name()==name){
        input.CopyFrom(onnx_model.mutable_graph()->input(i));
        saveValueInfo(&input,"new_value_info.onnx");
        onnx::ValueInfoProto input_from_pre;
        loadValueInfo(&input_from_pre,"new_value_info.onnx");
        std::cout<<input_from_pre.name();
      }
    }

}

void merge_single_model_with_predicate(std::string& onnx_model_path, std::string& predicate,
                                       std::string& value_type, std::string prefix){
  onnx::ModelProto onnx_model;
  onnx::optimization::loadModel(&onnx_model, onnx_model_path, true);
  onnx::checker::check_model(onnx_model);
  onnx::ValueInfoProto input;
  // todo add ValueInfoProto pre-create
  if(value_type=="int"){
      loadValueInfo(&input,"int_value_info.onnx");
      input.set_name(prefix);
      std::cout<<input.name();
  }
}


}
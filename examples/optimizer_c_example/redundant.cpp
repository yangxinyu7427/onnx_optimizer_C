//
// Created by xyyang's mac on 2024/7/7.
//
#include <string>
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

int main(int argc, char* argv[]) {

  std::string path2="../examples/onnx_input_model/neo_nb2.onnx";
  std::string path3="../examples/onnx_input_model/neo_lr2.onnx";
  std::string path4="../examples/onnx_input_model/news_lr.onnx";
  std::string path5="../examples/onnx_input_model/news_nb.onnx";
  std::string path6="../examples/onnx_input_model/model_lr.onnx";
  std::string path7="../examples/onnx_input_model/model_nb.onnx";
  std::string path8="../examples/onnx_input_model/flights_lr.onnx";
  std::string path9="../examples/onnx_input_model/flights_nb.onnx";
  std::string path10="../examples/onnx_input_model/fakenews_lr_test.onnx";
  std::string path11="../examples/onnx_input_model/fakenews_nb";
  std::string path12="../examples/onnx_input_model/model_opted.onnx";
  std::string outpath="../examples/onnx_input_model/model_out.onnx";
  std::string inpath="../examples/onnx_input_model/model_in.onnx";
  std::string path13="../examples/onnx_input_model/model_reg.onnx";
  std::string pre="sgd_1_";
  std::string path_pre="../examples/onnx_input_model/model_reg_prefix.onnx";
  //add_prefix_on_model(path13, path_pre, pre);
  //std::vector<std::string> tmp = check_redundant(path_pre,path12);
  std::vector<std::string> tmp = check_redundant(path13,path6);
  //change_models(path_pre,outpath,inpath,tmp);
  change_models(path13,outpath,inpath,tmp);
}

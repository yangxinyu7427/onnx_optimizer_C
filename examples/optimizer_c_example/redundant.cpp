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
  std::string path10="../examples/onnx_input_model/fakenews_lr.onnx";
  std::string path11="../examples/onnx_input_model/fakenews_nb.onnx";
  std::string outpath="../examples/onnx_input_model/model_out.onnx";
  std::string inpath="../examples/onnx_input_model/model_in.onnx";

  std::vector<std::string> tmp = check_redundant(path10,path11);
  change_models(path10,outpath,inpath,tmp);

}

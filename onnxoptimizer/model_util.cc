/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include <algorithm>
#include <cstddef>  // size_t
#include <bitset>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "model_util.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/platform_helpers.h"
#include "onnxoptimizer/passes/logging.h"

namespace ONNX_NAMESPACE {
namespace optimization {

namespace {
struct ExternalDataInfo {
  std::string location;
  std::ofstream::off_type offset = {-1};
  int64_t length = {-1};

  ExternalDataInfo(const std::string& location) : location(location) {}

  ExternalDataInfo(const TensorProto* tensor) {
    const auto& external_data = tensor->external_data();
    for (const auto& entry : external_data) {
      if (entry.key() == "location") {
        location = entry.value();
      } else if (entry.key() == "offset") {
        offset = std::stoll(entry.value());
      } else if (entry.key() == "length") {
        length = std::stoll(entry.value());
      }
    }
  }

  void setExternalData(TensorProto* tensor, int32_t size_threshold = 0) {
    const auto& dims = tensor->dims();
    int64_t elem_cnt = std::accumulate(dims.cbegin(), dims.cend(), int64_t(1),
                                       std::multiplies<int64_t>{});
    VLOG(3) << Str("tensor name: ", tensor->name(), ",elem count: ", elem_cnt);
    /// ignore empty tensor
    if (elem_cnt == 0) {
      return;
    }

    if (!tensor->has_raw_data()) {
      if (!is_processor_little_endian()) {
        return;
      }
#define DO_CASE(pb_type, storage_field, cpp_type)                 \
  case TensorProto_DataType_##pb_type: {                          \
    std::vector<cpp_type> datas(tensor->storage_field().cbegin(), \
                                tensor->storage_field().cend());  \
    std::string raw_data;                                         \
    size_t raw_data_size = datas.size() * sizeof(cpp_type);       \
    if (raw_data_size < size_threshold) {                         \
      return;                                                     \
    }                                                             \
    raw_data.resize(raw_data_size);                               \
    memcpy(&raw_data[0], &datas[0], raw_data_size);               \
    tensor->mutable_raw_data()->swap(raw_data);                   \
    tensor->clear_##storage_field();                              \
    break;                                                        \
  }

      switch (tensor->data_type()) {
        DO_CASE(FLOAT, float_data, float)
        DO_CASE(DOUBLE, double_data, double)
        DO_CASE(INT32, int32_data, int32_t)
        DO_CASE(INT64, int64_data, int64_t)
        DO_CASE(UINT64, uint64_data, uint64_t)

        default:
          return;
      }
#undef DO_CASE
    }
    if (tensor->raw_data().size() < size_threshold) {
      return;
    }
    tensor->set_data_location(TensorProto_DataLocation_EXTERNAL);
    tensor->clear_external_data();
    std::unordered_map<std::string, std::string> entry_map;
    entry_map["location"] = location;
    if (offset >= 0) {
      entry_map["offset"] = std::to_string(offset);
    }
    if (length >= 0) {
      entry_map["length"] = std::to_string(length);
    }
    for (auto& entry : entry_map) {
      auto* e_d = tensor->add_external_data();
      e_d->set_key(entry.first);
      e_d->set_value(entry.second);
    }
  }
};

std::string genUUID() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);
  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  };
  return ss.str();
}

std::vector<TensorProto*> getInitializerTensors(ModelProto* m) {
  int size = m->graph().initializer_size();
  std::vector<TensorProto*> tensors;
  tensors.reserve(size);
  for (int i = 0; i < size; ++i) {
    tensors.push_back(m->mutable_graph()->mutable_initializer(i));
  }
  return tensors;
}

std::vector<TensorProto*> getAttributeTensors(ModelProto* m) {
  std::vector<TensorProto*> tensors;
  for (auto& node : *m->mutable_graph()->mutable_node()) {
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_t()) {
        tensors.push_back(attr.mutable_t());
      }
      for (auto& tensor : *attr.mutable_tensors()) {
        tensors.push_back(&tensor);
      }
    }
  }
  return tensors;
}

std::vector<TensorProto*> getAllTensors(ModelProto* m) {
  auto tensors = getInitializerTensors(m);
  auto attr_tensors = getAttributeTensors(m);
  std::copy(attr_tensors.begin(), attr_tensors.end(),
            std::back_inserter(tensors));
  return tensors;
}

bool usesExternalData(TensorProto* tensor) {
  return tensor->has_data_location() &&
         tensor->data_location() == TensorProto_DataLocation_EXTERNAL;
}

void loadExternalDataForTensor(TensorProto* tensor,
                               const std::filesystem::path& base_dir) {
  ExternalDataInfo info(tensor);
  auto external_data_file_path = base_dir;
  external_data_file_path /= (info.location);
  std::ifstream data_file(external_data_file_path.string(),
                          std::ios_base::binary | std::ios_base::in);
  if (!data_file) {
    throw std::runtime_error("open " + external_data_file_path.string() +
                             " failed!");
  }
  data_file.seekg(0, data_file.end);
  int64_t length = data_file.tellg();
  data_file.seekg(0, data_file.beg);
  if (info.offset > 0) {
    data_file.seekg(info.offset, data_file.beg);
    length -= info.offset;
  }
  std::string raw_data;
  if (info.length > 0) {
    raw_data.resize(info.length);
    data_file.read(&raw_data[0], info.length);
  } else {
    raw_data.resize(length);
    data_file.read(&raw_data[0], length);
  }
  tensor->mutable_raw_data()->swap(raw_data);
  tensor->set_data_location(TensorProto_DataLocation_DEFAULT);
  tensor->clear_external_data();
  data_file.close();
}

void loadExternalDataForModel(ModelProto* m,
                              const std::filesystem::path& base_dir) {
  auto tensors = getAllTensors(m);
  for (auto& tensor : tensors) {
    if (usesExternalData(tensor)) {
      loadExternalDataForTensor(tensor, base_dir);
    }
  }
}

void saveExternalData(TensorProto* tensor,
                      const std::filesystem::path& base_dir) {
  ExternalDataInfo info(tensor);
  auto external_data_file_path = base_dir;
  external_data_file_path /= (info.location);
  std::ofstream data_file(external_data_file_path.string(),
                          std::ios_base::binary | std::ios_base::app);
  if (!data_file) {
    throw std::runtime_error("open " + external_data_file_path.string() +
                             " failed!");
  }
  data_file.seekp(0, data_file.end);
  info.offset = data_file.tellp();
  const auto& raw_data = tensor->raw_data();
  data_file.write(raw_data.c_str(), raw_data.size());
  info.length = data_file.tellp() - info.offset;
  info.setExternalData(tensor);
  data_file.close();
}

void convertModelToExternalData(ModelProto* m, const std::string& location = {},
                                int32_t size_threshold = 1024,
                                bool convert_attribute = true) {
  std::vector<TensorProto*> tensors;
  if (convert_attribute) {
    tensors = getAllTensors(m);
  } else {
    tensors = getInitializerTensors(m);
  }

  auto file_name = genUUID();
  if (!location.empty()) {
    file_name = location;
  }
  for (auto& tensor : tensors) {
    ExternalDataInfo info(file_name);
    info.setExternalData(tensor, size_threshold);
  }
}

void writeExternalDataTensors(ModelProto* m,
                              const std::filesystem::path& base_dir) {
  auto tensors = getAllTensors(m);
  for (auto& tensor : tensors) {
    if (usesExternalData(tensor) && tensor->has_raw_data()) {
      saveExternalData(tensor, base_dir);
      tensor->clear_raw_data();
    }
  }
}

}  // namespace

void loadValueInfo(ValueInfoProto* v, const std::string& value_info_path) {
  LoadProtoFromPath<ValueInfoProto>(value_info_path, *v);
}

void saveValueInfo(ValueInfoProto* v, const std::string& value_info_path) {
  std::string serialize;
  v->SerializeToString(&serialize);
  std::ofstream data_file(value_info_path, std::ios_base::out |
                                          std::ios_base::trunc |
                                          std::ios_base::binary);
  if (!data_file) {
    throw std::runtime_error("open " + value_info_path + " failed!");
  }
  data_file.write(serialize.c_str(), serialize.size());
  data_file.close();
}

void loadModel(ModelProto* m, const std::string& model_path,
               bool load_external_data) {
  LoadProtoFromPath<ModelProto>(model_path, *m);
  if (load_external_data) {
    const auto parent_path = std::filesystem::path(model_path).parent_path();
    loadExternalDataForModel(m, parent_path);
  }
}

void saveModel(ModelProto* m, const std::string& model_path,
               const bool save_external_data,
               const std::string& data_file_name) {
  if (save_external_data) {
    convertModelToExternalData(m, data_file_name);
  }
  const auto parent_path = std::filesystem::path(model_path).parent_path();
  writeExternalDataTensors(m, parent_path);

  std::string serialize;
  m->SerializeToString(&serialize);

  std::ofstream data_file(model_path, std::ios_base::out |
                                          std::ios_base::trunc |
                                          std::ios_base::binary);
  if (!data_file) {
    throw std::runtime_error("open " + model_path + " failed!");
  }
  data_file.write(serialize.c_str(), serialize.size());
  data_file.close();
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE

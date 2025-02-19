#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>

#include <condition_variable>
#include <iomanip>
#include <mutex>
#include <queue>
#include <regex>
#include <thread>
#include <tuple>
#include <unordered_set>

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.cc"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/proto_utils.h"

//===--------------------------------------------------------------------===//
// ** Utils
//===--------------------------------------------------------------------===//
// **获取new model name
std::string getNewmodelname(std::string onnx_model_path, std::string suffix) {
  return onnx_model_path.substr(0, onnx_model_path.find(".onnx")) + "_" +
         suffix + ".onnx";
}
// ** use in pruning
struct NodeID {
  int id;
  std::string node;
};

//===--------------------------------------------------------------------===//
// ** Optimize Rules
//===--------------------------------------------------------------------===//

namespace onnx::optimization {

std::string saveModelWithNewName(ModelProto& mp_in,
                                 std::shared_ptr<Graph>& graph,
                                 std::string& model_path, std::string suffix) {
  ModelProto mp_out = PrepareOutput(mp_in);
  ExportModelProto(&mp_out, graph);
  checker::check_model(mp_out);
  auto new_model_path = getNewmodelname(model_path, suffix);
  saveModel(&mp_out, new_model_path);
  return new_model_path;
}

//** Rule1: 分类树转回归树
class DTConvertRule {
 public:
  static void processNode(Node* node, bool isForest, int mode) {
    auto classlabels_int64s = node->is(Symbol("classlabels_int64s"));
    auto class_treeids = node->is(Symbol("class_treeids"));
    auto class_ids = node->is(Symbol("class_ids"));
    auto class_nodeids = node->is(Symbol("class_nodeids"));
    auto class_weights = node->fs(Symbol("class_weights"));

    int n_trees = 1;
    if (isForest) {
      std::unordered_set<int> unique_treeids(class_treeids.begin(),
                                             class_treeids.end());
      n_trees = unique_treeids.size();
    }

    // ** convert clf attributes 2 reg attributes
    int64_t stride =
        classlabels_int64s.size() == 2 ? 1 : classlabels_int64s.size();
    int64_t nleaf = class_weights.size() / stride;
    // input_class_treeids 2 target_treeids
    std::vector<int64_t> target_treeids;
    target_treeids.reserve(nleaf);
    for (int64_t i = 0; i < nleaf; ++i) {
      target_treeids.push_back(class_treeids[i * stride]);
    }
    // input_class_ids 2 target_ids
    std::vector<int64_t> target_ids;
    target_ids.reserve(nleaf);
    for (int64_t i = 0; i < nleaf; ++i) {
      target_ids.push_back(class_ids[i * stride]);
    }
    // input_class_nodeids 2 target_nodeids
    std::vector<int64_t> target_nodeids;
    target_nodeids.reserve(nleaf);
    for (int64_t i = 0; i < nleaf; ++i) {
      target_nodeids.push_back(class_nodeids[i * stride]);
    }
    // input_class_weights 2 target_weights
    std::vector<double> target_weights;
    target_weights.reserve(nleaf);
    if (stride == 1) {
      for (auto w : class_weights) {
        w > 0.5 / n_trees ? target_weights.push_back(1.0)
                          : target_weights.push_back(0.0);
      }
    } else {
      for (int i = 0; i < nleaf; ++i) {
        auto start_it = class_weights.begin() + (i * stride);
        auto end_it = class_weights.begin() + ((i + 1) * stride);

        auto max_it = std::max_element(start_it, end_it);
        int index = std::distance(start_it, max_it);

        target_weights.push_back(static_cast<double>(index));
      }
    }

    node->removeAttribute(Symbol("classlabels_int64s"));
    node->removeAttribute(Symbol("class_treeids"));
    node->removeAttribute(Symbol("class_ids"));
    node->removeAttribute(Symbol("class_nodeids"));
    node->removeAttribute(Symbol("class_weights"));

    node->i_(Symbol("n_targets"), 1);
    node->is_(Symbol("target_treeids"), std::move(target_treeids));
    node->is_(Symbol("target_ids"), std::move(target_ids));
    node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
    node->fs_(Symbol("target_weights"), std::move(target_weights));
  }

  static std::string convertModelProto(ModelProto& mp_in, Node* node,
                                       std::string& model_path, int mode) {
    auto g_m = mp_in.mutable_graph();
    auto input = g_m->input();

    // 输出
    onnx::ValueInfoProto output;
    TensorShapeProto_Dimension output_dim1, output_dim2;
    output_dim2.set_dim_value(1);
    output.set_name("regression");
    output.mutable_type()->mutable_tensor_type()->set_elem_type(1);
    *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim() =
        output_dim1;
    *output.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim() =
        output_dim2;

    // 输入
    onnx::NodeProto* p_n_clf;
    for (int i = 0; i < g_m->node_size(); ++i) {
      auto node = g_m->mutable_node(i);
      if (node->op_type() == "TreeEnsembleClassifier") {
        p_n_clf = node;
      }
    }

    // node
    onnx::NodeProto p_n;
    p_n.set_op_type("TreeEnsembleRegressor");
    p_n.set_name(p_n_clf->name());
    p_n.set_domain(p_n_clf->domain());
    auto inputs = p_n_clf->input();
    for (auto input : inputs) {
      p_n.add_input(input);
    }
    p_n.add_output(output.name());
    for (auto attr_name : node->attributeNames()) {
      addAttribute(&p_n, node, attr_name);
    }

    for (int i = g_m->node_size() - 1; i >= 0; i--) {
      auto node = g_m->node(i);
      if (node.op_type() == "TreeEnsembleClassifier") {
        g_m->mutable_node()->DeleteSubrange(i, 1);
      }
      if (node.op_type() == "ZipMap") {
        g_m->mutable_node()->DeleteSubrange(i, 1);
      }
      // TreeEnsembleClassifier 之后的 Cast
      if (node.op_type() == "Cast") {
        if (node.input()[0] == "label") {
          g_m->mutable_node()->DeleteSubrange(i, 1);
        }
      }

      // if (node.op_type() == "TreeEnsembleClassifier" ||
      //     node.op_type() == "Cast" || node.op_type() == "ZipMap") {
      //   g_m->mutable_node()->DeleteSubrange(i, 1);
      // }
    }

    g_m->clear_output();
    *g_m->add_node() = p_n;
    *g_m->add_output() = output;

    onnx::checker::check_model(mp_in);
    auto new_model_path = getNewmodelname(model_path, "reg");
    saveModel(&mp_in, new_model_path);
    return new_model_path;
  }

  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           Node* node, std::string& model_path, bool isForest) {
    if (node->hasAttribute(Symbol("classlabels_int64s"))) {
      processNode(node, isForest, 0);
      return convertModelProto(mp_in, node, model_path, 0);
    } else if (node->hasAttribute(Symbol("classlabels_strings"))) {
      // todo: identity
      // processNode(node, 1);
      // return convertModelProto(mp_in, node, model_path, 1);
    }
    return model_path;
  }

  static std::string match(std::string& model_path) {
    std::string output_model_path = model_path;
    ModelProto mp_in;
    loadModel(&mp_in, model_path, true);
    std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

    bool found = false;
    bool isForest = true;
    Node* treeNode;
    graph->forEachNode([&found, &isForest, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("class_treeids"))) {
        auto class_treeids = node->is(Symbol("class_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE") {
          if (std::all_of(
                  class_treeids.begin(), class_treeids.end(),
                  std::bind(std::equal_to<>(), std::placeholders::_1, 0))) {
            isForest = false;
          }
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(mp_in, graph, treeNode, model_path, isForest);

    return output_model_path;
  }
};

//** Rule2.1: 决策树剪枝
class DTPruneRule {
 private:
  using ComparisonFunc = bool (*)(float, float);
  static ComparisonFunc comparison_funcs[];

 public:
  static int pruning(size_t node_id, size_t depth,
                     std::vector<std::string>& result_nodes, Node* treeNode,
                     uint8_t comparison_operator, float threshold) {
    const auto& left_nodes = treeNode->is(Symbol("nodes_truenodeids"));
    const auto& right_nodes = treeNode->is(Symbol("nodes_falsenodeids"));
    const auto& node_types = treeNode->ss(Symbol("nodes_modes"));
    const auto& target_nodeids = treeNode->is(Symbol("target_nodeids"));
    const auto& target_weights = treeNode->fs(Symbol("target_weights"));

    result_nodes[node_id] = node_types[node_id];
    auto is_leaf = node_types[node_id] == "LEAF";
    if (is_leaf) {
      auto target_id = -1;
      for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
        size_t ni = target_nodeids[ti];
        if (ni == node_id) {
          target_id = static_cast<int>(ti);
          break;
        }
      }
      int result = comparison_funcs[comparison_operator](
          target_weights[target_id], threshold);
      result == 1 ? result_nodes[node_id] = "LEAF_TRUE"
                  : result_nodes[node_id] = "LEAF_FALSE";
      return result;
    } else {
      auto left_node_id = left_nodes[node_id];
      auto left_result = pruning(left_node_id, depth + 1, result_nodes,
                                 treeNode, comparison_operator, threshold);
      auto right_node_id = right_nodes[node_id];
      auto right_result = pruning(right_node_id, depth + 1, result_nodes,
                                  treeNode, comparison_operator, threshold);

      if (left_result == 0 && right_result == 0) {
        result_nodes[node_id] = "LEAF_FALSE";
        result_nodes[left_node_id] = "REMOVED";
        result_nodes[right_node_id] = "REMOVED";
        return 0;
      }

      if (left_result == 1 && right_result == 1) {
        result_nodes[node_id] = "LEAF_TRUE";
        result_nodes[left_node_id] = "REMOVED";
        result_nodes[right_node_id] = "REMOVED";
        return 1;
      }
      return 2;
    }
  }

  static bool processNode(Node* node, std::vector<std::string>& removed_nodes) {
    int leaf_false_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE");
    int leaf_true_count =
        std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");
    // if (leaf_false_count == 0 || leaf_true_count == 0) {
    //   return false;
    // }
    int leaf_count = leaf_false_count + leaf_true_count;

    int64_t input_n_targets = node->i(Symbol("n_targets"));
    std::vector<int64_t> input_nodes_falsenodeids =
        node->is(Symbol("nodes_falsenodeids"));
    std::vector<int64_t> input_nodes_featureids =
        node->is(Symbol("nodes_featureids"));
    std::vector<double> input_nodes_hitrates =
        node->fs(Symbol("nodes_hitrates"));
    std::vector<int64_t> input_nodes_missing_value_tracks_true =
        node->is(Symbol("nodes_missing_value_tracks_true"));
    std::vector<std::string> input_nodes_modes =
        node->ss(Symbol("nodes_modes"));
    std::vector<int64_t> input_nodes_nodeids =
        node->is(Symbol("nodes_nodeids"));
    std::vector<int64_t> input_nodes_treeids =
        node->is(Symbol("nodes_treeids"));
    std::vector<int64_t> input_nodes_truenodeids =
        node->is(Symbol("nodes_truenodeids"));
    std::vector<double> input_nodes_values = node->fs(Symbol("nodes_values"));
    std::vector<int64_t> input_target_ids = node->is(Symbol("target_ids"));
    std::vector<int64_t> input_target_nodeids =
        node->is(Symbol("target_nodeids"));
    std::vector<int64_t> input_target_treeids =
        node->is(Symbol("target_treeids"));
    std::vector<double> input_target_weights =
        node->fs(Symbol("target_weights"));

    // 2. 构建 new_ids
    std::vector<NodeID> new_ids;
    int id_ = 0;
    for (const auto& node : removed_nodes) {
      if (node == "LEAF_FALSE" || node == "LEAF_TRUE" || node == "BRANCH_LEQ") {
        new_ids.push_back({id_, node});
        id_++;
      } else {
        new_ids.push_back({-1, node});
      }
    }

    // 4. 构建 nodes_falsenodeids
    std::vector<int64_t> nodes_falsenodeids;
    for (size_t i = 0; i < input_nodes_falsenodeids.size(); ++i) {
      int ii = input_nodes_falsenodeids[i];
      if (new_ids[i].node != "REMOVED") {
        int value = 0;
        if (ii >= 0 && static_cast<size_t>(ii) < new_ids.size()) {
          int new_id_value = new_ids[ii].id;
          value = (new_id_value != -1) ? new_id_value : 0;
        }
        nodes_falsenodeids.push_back(value);
      }
    }

    // 5. 构建 nodes_featureids
    std::vector<int64_t> nodes_featureids;
    for (size_t i = 0; i < input_nodes_featureids.size(); ++i) {
      int ii = input_nodes_featureids[i];
      if (new_ids[i].id != -1) {
        int value = (new_ids[i].node == "BRANCH_LEQ") ? ii : 0;
        nodes_featureids.push_back(value);
      }
    }

    // 6. 构建 nodes_hitrates
    std::vector<double> nodes_hitrates;
    for (size_t i = 0; i < input_nodes_hitrates.size(); ++i) {
      if (new_ids[i].id != -1) {
        nodes_hitrates.push_back(input_nodes_hitrates[i]);
      }
    }

    // 7. 构建 nodes_missing_value_tracks_true
    std::vector<int64_t> nodes_missing_value_tracks_true;
    for (size_t i = 0; i < input_nodes_missing_value_tracks_true.size(); ++i) {
      if (new_ids[i].id != -1) {
        nodes_missing_value_tracks_true.push_back(
            input_nodes_missing_value_tracks_true[i]);
      }
    }

    // 8. 构建 nodes_modes
    std::vector<std::string> nodes_modes;
    for (const auto& new_id : new_ids) {
      if (new_id.id != -1) {
        std::string mode =
            (new_id.node == "BRANCH_LEQ") ? "BRANCH_LEQ" : "LEAF";
        nodes_modes.push_back(mode);
      }
    }

    // 9. 构建 nodes_nodeids
    std::vector<int64_t> nodes_nodeids;
    for (size_t i = 0; i < input_nodes_nodeids.size(); ++i) {
      if (new_ids[i].id != -1) {
        nodes_nodeids.push_back(new_ids[i].id);
      }
    }

    // 10. 构建 nodes_treeids
    std::vector<int64_t> nodes_treeids;
    for (size_t i = 0; i < input_nodes_treeids.size(); ++i) {
      if (new_ids[i].id != -1) {
        nodes_treeids.push_back(input_nodes_treeids[i]);
      }
    }

    // 11. 构建 nodes_truenodeids
    std::vector<int64_t> nodes_truenodeids;
    for (size_t i = 0; i < input_nodes_truenodeids.size(); ++i) {
      int ii = input_nodes_truenodeids[i];
      if (new_ids[i].node != "REMOVED") {
        int value = 0;
        if (ii >= 0 && static_cast<size_t>(ii) < new_ids.size()) {
          int new_id_value = new_ids[ii].id;
          value = (new_id_value != -1) ? new_id_value : 0;
        }
        nodes_truenodeids.push_back(value);
      }
    }

    // 12. 构建 nodes_values
    std::vector<double> nodes_values;
    for (size_t i = 0; i < input_nodes_values.size(); ++i) {
      if (new_ids[i].id != -1) {
        double value =
            (new_ids[i].node == "BRANCH_LEQ") ? input_nodes_values[i] : 0.0f;
        nodes_values.push_back(value);
      }
    }

    // 14. 构建 target_ids
    std::vector<int64_t> target_ids(leaf_count, 0);

    // 15. 构建 target_nodeids
    std::vector<int64_t> target_nodeids;
    for (const auto& new_id : new_ids) {
      if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
        target_nodeids.push_back(new_id.id);
      }
    }

    // 16. 构建 target_treeids
    std::vector<int64_t> target_treeids(leaf_count, 0);

    // 17. 构建 target_weights
    std::vector<double> target_weights;
    for (const auto& new_id : new_ids) {
      if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
        double weight = (new_id.node == "LEAF_TRUE") ? 1.0f : 0.0f;
        target_weights.push_back(weight);
      }
    }

    node->is_(Symbol("nodes_falsenodeids"), std::move(nodes_falsenodeids));
    node->is_(Symbol("nodes_featureids"), std::move(nodes_featureids));
    node->fs_(Symbol("nodes_hitrates"), std::move(nodes_hitrates));
    node->is_(Symbol("nodes_missing_value_tracks_true"),
              std::move(nodes_missing_value_tracks_true));
    node->ss_(Symbol("nodes_modes"), std::move(nodes_modes));
    node->is_(Symbol("nodes_nodeids"), std::move(nodes_nodeids));
    node->is_(Symbol("nodes_treeids"), std::move(nodes_treeids));
    node->is_(Symbol("nodes_truenodeids"), std::move(nodes_truenodeids));
    node->fs_(Symbol("nodes_values"), std::move(nodes_values));
    node->is_(Symbol("target_ids"), std::move(target_ids));
    node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
    node->is_(Symbol("target_treeids"), std::move(target_treeids));
    node->fs_(Symbol("target_weights"), std::move(target_weights));

    return true;
  }

  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode,
                           uint8_t comparison_operator, float threshold) {
    size_t length = treeNode->ss(Symbol("nodes_modes")).size();
    std::vector<std::string> removed_nodes{length, ""};
    pruning(0, 0, removed_nodes, treeNode, comparison_operator, threshold);
    if (processNode(treeNode, removed_nodes)) {
      return saveModelWithNewName(mp_in, graph, model_path, "pruned");
    } else {
      std::string new_model_path = model_path;
      return new_model_path;
    }
  }

  static std::string match(std::string& model_path, uint8_t comparison_operator,
                           float threshold) {
    std::string output_model_path = model_path;
    ModelProto mp_in;
    loadModel(&mp_in, model_path, true);
    std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

    bool found = false;
    Node* treeNode;
    graph->forEachNode([&found, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("target_treeids"))) {
        auto target_treeids = node->is(Symbol("target_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE" &&
            std::all_of(
                target_treeids.begin(), target_treeids.end(),
                std::bind(std::equal_to<>(), std::placeholders::_1, 0))) {
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(mp_in, graph, model_path, treeNode,
                                comparison_operator, threshold);

    return output_model_path;
  }
};

DTPruneRule::ComparisonFunc DTPruneRule::comparison_funcs[] = {
    [](float x, float y) { return x == y; },
    [](float x, float y) { return x < y; },
    [](float x, float y) { return x <= y; },
    [](float x, float y) { return x > y; },
    [](float x, float y) { return x >= y; }};

//** Rule2.2: 随机森林剪枝
class RFPruneRule {
 private:
  using ComparisonFunc = bool (*)(float, float);
  static ComparisonFunc comparison_funcs[];

 public:
  static std::vector<std::tuple<int, int>> getTreeIntervals(Node* node) {
    std::vector<int> tree_roots;
    std::vector<int64_t> nodes_treeids = node->is(Symbol("nodes_treeids"));
    int next_tree_id = 0;
    for (int i = 0; i < nodes_treeids.size(); ++i) {
      if (nodes_treeids[i] == next_tree_id) {
        next_tree_id++;
        tree_roots.push_back(i);
      }
    }

    std::vector<std::tuple<int, int>> tree_intervals;
    for (int i = 0; i < tree_roots.size(); ++i) {
      int end;
      if (i == tree_roots.size() - 1) {
        end = nodes_treeids.size();
      } else {
        end = tree_roots[i + 1];
      }
      tree_intervals.push_back(std::make_tuple(tree_roots[i], end));
    }

    return tree_intervals;
  }

  static int pruning(size_t tree_no, const std::tuple<int, int>& tree_interval,
                     size_t node_id, size_t depth,
                     std::vector<std::string>& result_nodes, Node* treeNode,
                     uint8_t comparison_operator, float threshold) {
    int tree_start = std::get<0>(tree_interval);
    int tree_end = std::get<1>(tree_interval);

    const auto& left_nodess = treeNode->is(Symbol("nodes_truenodeids"));
    const auto& right_nodess = treeNode->is(Symbol("nodes_falsenodeids"));
    const auto& node_typess = treeNode->ss(Symbol("nodes_modes"));

    std::vector<int64_t> left_nodes(left_nodess.begin() + tree_start,
                                    left_nodess.begin() + tree_end);
    std::vector<int64_t> right_nodes(right_nodess.begin() + tree_start,
                                     right_nodess.begin() + tree_end);
    std::vector<std::string> node_types(node_typess.begin() + tree_start,
                                        node_typess.begin() + tree_end);

    const auto& target_treeids = treeNode->is(Symbol("target_treeids"));
    const auto& target_nodeids = treeNode->is(Symbol("target_nodeids"));
    const auto& target_weights = treeNode->fs(Symbol("target_weights"));

    result_nodes[node_id] = node_types[node_id];
    bool is_leaf = (node_types[node_id] == "LEAF");

    if (is_leaf) {
      int target_idx = -1;
      for (int ti = 0; ti < target_nodeids.size(); ++ti) {
        int ni = target_nodeids[ti];
        if (ni == node_id && target_treeids[ti] == tree_no) {
          target_idx = ti;
          break;
        }
      }

      int result = comparison_funcs[comparison_operator](
          target_weights[target_idx], threshold);
      result_nodes[node_id] = (result == 1) ? "LEAF_TRUE" : "LEAF_FALSE";

      return result;
    
    } else {
      size_t left_node_id = left_nodes[node_id];
      int left_result =
          pruning(tree_no, tree_interval, left_node_id, depth + 1, result_nodes,
                  treeNode, comparison_operator, threshold);

      size_t right_node_id = right_nodes[node_id];
      int right_result =
          pruning(tree_no, tree_interval, right_node_id, depth + 1,
                  result_nodes, treeNode, comparison_operator, threshold);

      if (left_result == 0 && right_result == 0) {
        result_nodes[node_id] = "LEAF_FALSE";
        result_nodes[left_node_id] = "REMOVED";
        result_nodes[right_node_id] = "REMOVED";
        return 0;
      }

      if (left_result == 1 && right_result == 1) {
        result_nodes[node_id] = "LEAF_TRUE";
        result_nodes[left_node_id] = "REMOVED";
        result_nodes[right_node_id] = "REMOVED";
        return 1;
      }

      return 2;
    }
  }

  static bool processNode(
      Node* node, std::vector<std::vector<std::string>>& result_nodes_list,
      std::vector<std::tuple<int, int>>& tree_intervals) {
    const int tree_count = tree_intervals.size();
    // int pruned_tree_count = tree_intervals.size();
    std::vector<int> tree_leaf_counts;
    for (const auto& removed_nodes : result_nodes_list) {
      int leaf_false_count =
          std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE");
      int leaf_true_count =
          std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");
      // if (leaf_false_count == 0 || leaf_true_count == 0) {
      //   pruned_tree_count--;
      // }
      tree_leaf_counts.push_back(leaf_false_count + leaf_true_count);
    }
    // if (pruned_tree_count == 0) {
    //   return false;
    // }

    int64_t input_n_targets = node->i(Symbol("n_targets"));
    std::vector<int64_t> input_nodes_falsenodeids =
        node->is(Symbol("nodes_falsenodeids"));
    std::vector<int64_t> input_nodes_featureids =
        node->is(Symbol("nodes_featureids"));
    std::vector<double> input_nodes_hitrates =
        node->fs(Symbol("nodes_hitrates"));
    std::vector<int64_t> input_nodes_missing_value_tracks_true =
        node->is(Symbol("nodes_missing_value_tracks_true"));
    std::vector<std::string> input_nodes_modes =
        node->ss(Symbol("nodes_modes"));
    std::vector<int64_t> input_nodes_nodeids =
        node->is(Symbol("nodes_nodeids"));
    std::vector<int64_t> input_nodes_treeids =
        node->is(Symbol("nodes_treeids"));
    std::vector<int64_t> input_nodes_truenodeids =
        node->is(Symbol("nodes_truenodeids"));
    std::vector<double> input_nodes_values = node->fs(Symbol("nodes_values"));
    std::vector<int64_t> input_target_ids = node->is(Symbol("target_ids"));
    std::vector<int64_t> input_target_nodeids =
        node->is(Symbol("target_nodeids"));
    std::vector<int64_t> input_target_treeids =
        node->is(Symbol("target_treeids"));
    std::vector<double> input_target_weights =
        node->fs(Symbol("target_weights"));

    std::vector<std::vector<NodeID>> new_ids_list;
    for (const auto& removed_nodes : result_nodes_list) {
      std::vector<NodeID> new_ids;
      int id_ = 0;
      for (const auto& node : removed_nodes) {
        if (node == "LEAF_FALSE" || node == "LEAF_TRUE" ||
            node == "BRANCH_LEQ") {
          new_ids.push_back({id_, node});
          id_++;
        } else {
          new_ids.push_back({-1, node});
        }
      }
      new_ids_list.push_back(new_ids);
    }

    // 4. 构建 nodes_falsenodeids
    std::vector<int64_t> nodes_falsenodeids;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].node != "REMOVED") {
          int64_t falsenodeid = new_ids[input_nodes_falsenodeids[ii]].id;
          nodes_falsenodeids.push_back(falsenodeid != -1 ? falsenodeid : 0);
        }
        ++i;
      }
    }

    // 5. 构建 nodes_featureids
    std::vector<int64_t> nodes_featureids;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].id != -1) {
          nodes_featureids.push_back(
              new_ids[i].node == "BRANCH_LEQ" ? input_nodes_featureids[ii] : 0);
        }
        ++i;
      }
    }

    // 6. 构建 nodes_hitrates
    std::vector<double> nodes_hitrates;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].id != -1) {
          nodes_hitrates.push_back(input_nodes_hitrates[ii]);
        }
        ++i;
      }
    }

    // 7. 构建 nodes_missing_value_tracks_true
    std::vector<int64_t> nodes_missing_value_tracks_true;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].id != -1) {
          nodes_missing_value_tracks_true.push_back(
              input_nodes_missing_value_tracks_true[ii]);
        }
        ++i;
      }
    }

    // 8. 构建 nodes_modes
    std::vector<std::string> nodes_modes;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& new_ids = new_ids_list[tree_no];

      for (const auto& new_id : new_ids) {
        if (new_id.id != -1) {
          nodes_modes.push_back(new_id.node == "BRANCH_LEQ" ? "BRANCH_LEQ"
                                                            : "LEAF");
        }
      }
    }

    // 9. 构建 nodes_nodeids
    std::vector<int64_t> nodes_nodeids;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].id != -1) {
          nodes_nodeids.push_back(new_ids[i].id);
        }
        ++i;
      }
    }

    // 10. 构建 nodes_treeids
    std::vector<int64_t> nodes_treeids;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].id != -1) {
          nodes_treeids.push_back(input_nodes_treeids[ii]);
        }
        ++i;
      }
    }
    // 11. 构建 nodes_truenodeids
    std::vector<int64_t> nodes_truenodeids;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].node != "REMOVED") {
          int truenodeid = new_ids[input_nodes_truenodeids[ii]].id;
          nodes_truenodeids.push_back(truenodeid != -1 ? truenodeid : 0);
        }
        ++i;
      }
    }

    // 12. 构建 nodes_values
    std::vector<double> nodes_values;
    for (size_t tree_no = 0; tree_no < tree_count; ++tree_no) {
      const auto& [tree_start, tree_end] = tree_intervals[tree_no];
      const auto& new_ids = new_ids_list[tree_no];
      int i = 0;
      for (size_t ii = tree_start; ii < tree_end; ++ii) {
        if (new_ids[i].id != -1) {
          nodes_values.push_back(
              new_ids[i].node == "BRANCH_LEQ" ? input_nodes_values[ii] : 0);
        }
        ++i;
      }
    }

    // 14. 构建 target_ids
    std::vector<int64_t> target_ids(
        std::accumulate(tree_leaf_counts.begin(), tree_leaf_counts.end(), 0),
        0);

    // 15. 构建 target_nodeids
    std::vector<int64_t> target_nodeids;
    for (const auto& new_ids : new_ids_list) {
      for (const auto& new_id : new_ids) {
        if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
          target_nodeids.push_back(new_id.id);
        }
      }
    }

    // 16. 构建 target_treeids
    std::vector<int64_t> target_treeids;
    for (size_t tree_no = 0; tree_no < tree_leaf_counts.size(); ++tree_no) {
      target_treeids.insert(target_treeids.end(), tree_leaf_counts[tree_no],
                            tree_no);
    }

    // 17. 构建 target_weights
    std::vector<double> target_weights;
    for (const auto& new_ids : new_ids_list) {
      for (const auto& new_id : new_ids) {
        if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
          target_weights.push_back(
              new_id.node == "LEAF_TRUE" ? 1.0f / tree_count : 0.0f);
        }
      }
    }

    node->is_(Symbol("nodes_falsenodeids"), std::move(nodes_falsenodeids));
    node->is_(Symbol("nodes_featureids"), std::move(nodes_featureids));
    node->fs_(Symbol("nodes_hitrates"), std::move(nodes_hitrates));
    node->is_(Symbol("nodes_missing_value_tracks_true"),
              std::move(nodes_missing_value_tracks_true));
    node->ss_(Symbol("nodes_modes"), std::move(nodes_modes));
    node->is_(Symbol("nodes_nodeids"), std::move(nodes_nodeids));
    node->is_(Symbol("nodes_treeids"), std::move(nodes_treeids));
    node->is_(Symbol("nodes_truenodeids"), std::move(nodes_truenodeids));
    node->fs_(Symbol("nodes_values"), std::move(nodes_values));
    node->is_(Symbol("target_ids"), std::move(target_ids));
    node->is_(Symbol("target_nodeids"), std::move(target_nodeids));
    node->is_(Symbol("target_treeids"), std::move(target_treeids));
    node->fs_(Symbol("target_weights"), std::move(target_weights));

    return true;
  }

  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode,
                           uint8_t comparison_operator, float threshold) {
    auto tree_intervals = getTreeIntervals(treeNode);
    threshold /= tree_intervals.size();

    std::vector<std::vector<std::string>> result_nodes_list;
    for (const auto& interval : tree_intervals) {
      int start = std::get<0>(interval);
      int end = std::get<1>(interval);
      result_nodes_list.push_back(std::vector<std::string>(end - start, ""));
    }
    // auto start = std::chrono::high_resolution_clock::now();
    for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
      auto& result_nodes = result_nodes_list[tree_no];
      const auto& tree_interval = tree_intervals[tree_no];
      pruning(tree_no, tree_interval, 0, 0, result_nodes, treeNode,
              comparison_operator, threshold);
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << "pruning time cost (s): " << duration.count() / 1000
    // << std::endl;
    if (processNode(treeNode, result_nodes_list, tree_intervals)) {
      return saveModelWithNewName(mp_in, graph, model_path, "pruned");
    } else {
      std::string new_model_path = model_path;
      return new_model_path;
    }
  }

  static std::string match(std::string& model_path, uint8_t comparison_operator,
                           float threshold) {
    std::string output_model_path = model_path;
    ModelProto mp_in;
    loadModel(&mp_in, model_path, true);
    std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

    bool found = false;
    Node* treeNode;
    graph->forEachNode([&found, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("target_treeids"))) {
        auto target_treeids = node->is(Symbol("target_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE" &&
            !std::all_of(
                target_treeids.begin(), target_treeids.end(),
                std::bind(std::equal_to<>(), std::placeholders::_1, 0))) {
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(mp_in, graph, model_path, treeNode,
                                comparison_operator, threshold);

    return output_model_path;
  }
};

RFPruneRule::ComparisonFunc RFPruneRule::comparison_funcs[] = {
    [](float x, float y) { return x == y; },
    [](float x, float y) { return x < y; },
    [](float x, float y) { return x <= y; },
    [](float x, float y) { return x > y; },
    [](float x, float y) { return x >= y; }};

//** 特征下推-未启用
class DTPushdownRule {
 public:
  static void pushdown(std::shared_ptr<Graph>& graph, Node* node) {
    // 1. get used featureids
    std::vector<int64_t> input_nodes_featureids =
        node->is(Symbol("nodes_featureids"));
    std::set<int64_t> unique_featureids(input_nodes_featureids.begin(),
                                        input_nodes_featureids.end());
    // // std::vector<int64_t> used_nodes_featureids(unique_featureids.begin(),
    // //                                            unique_featureids.end());
    auto features_num = unique_featureids.size();

    // reconstruct graph input
  }

  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           Node* node, std::string& model_path,
                           std::vector<std::string>* features) {
    pushdown(graph, node);

    return saveModelWithNewName(mp_in, graph, model_path, "pushdowned");
  }

  static std::string match(std::string& model_path,
                           std::vector<std::string>* features) {
    if (features->empty()) {
      return model_path;
    }

    std::string output_model_path = model_path;
    ModelProto mp_in;
    loadModel(&mp_in, model_path, true);
    std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

    bool found = false;
    Node* treeNode;
    graph->forEachNode([&found, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("target_treeids"))) {
        auto target_treeids = node->is(Symbol("target_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE" &&
            std::all_of(
                target_treeids.begin(), target_treeids.end(),
                std::bind(std::equal_to<>(), std::placeholders::_1, 0))) {
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(mp_in, graph, treeNode, model_path, features);

    return output_model_path;
  }
};

//** Rule3: 合并
// -------------------------
class TreeNode {
 public:
  int id;          // 节点ID
  int feature_id;  // 特征ID
  std::string
      mode;      // 节点类型，"LEAF" 表示叶子节点，"BRANCH_LEQ" 表示非叶子节点
  double value;  // 阈值，叶子节点的值为0
  std::optional<int> target_id;         // 叶子节点的 target ID，可能为空
  std::optional<double> target_weight;  // 叶子节点的权重，即预测值，可能为空
  int samples;                          // 节点的样本数

  TreeNode* parent;
  TreeNode* left;
  TreeNode* right;

  TreeNode(int _id, int _feature_id, const std::string& _mode, double _value,
           std::optional<int> _target_id, std::optional<double> _target_weight,
           int _samples)
      : id(_id),
        feature_id(_feature_id),
        mode(_mode),
        value(_value),
        target_id(_target_id),
        target_weight(_target_weight),
        samples(_samples),
        parent(nullptr),
        left(nullptr),
        right(nullptr) {}

  int branch_samples() {
    int samples_ = samples;
    if (left) {
      samples_ += left->branch_samples();
    }
    if (right) {
      samples_ += right->branch_samples();
    }
    return samples_;
  }
};

TreeNode* model2tree(Node* treeNode, int64_t node_id, TreeNode* parent) {
  const auto& input_n_targets = treeNode->i(Symbol("n_targets"));
  const auto& input_nodes_falsenodeids =
      treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& input_nodes_featureids = treeNode->is(Symbol("nodes_featureids"));
  const auto& input_nodes_hitrates = treeNode->fs(Symbol("nodes_hitrates"));
  const auto& input_nodes_missing_value_tracks_true =
      treeNode->is(Symbol("nodes_missing_value_tracks_true"));
  const auto& input_nodes_modes = treeNode->ss(Symbol("nodes_modes"));
  const auto& input_nodes_nodeids = treeNode->is(Symbol("nodes_nodeids"));
  const auto& input_nodes_treeids = treeNode->is(Symbol("nodes_treeids"));
  const auto& input_nodes_truenodeids =
      treeNode->is(Symbol("nodes_truenodeids"));
  const auto& input_nodes_values = treeNode->fs(Symbol("nodes_values"));
  const auto& input_target_ids = treeNode->is(Symbol("target_ids"));
  const auto& input_target_nodeids = treeNode->is(Symbol("target_nodeids"));
  const auto& input_target_treeids = treeNode->is(Symbol("target_treeids"));
  const auto& input_target_weights = treeNode->fs(Symbol("target_weights"));

  // node_id -> target_id
  std::unordered_map<int, int> input_target_nodeid_map;
  for (size_t i = 0; i < input_target_nodeids.size(); ++i) {
    input_target_nodeid_map[input_target_nodeids[i]] = static_cast<int>(i);
  }

  size_t id = node_id;
  int64_t feature_id = input_nodes_featureids[id];
  std::string mode = input_nodes_modes[id];
  double value = input_nodes_values[id];
  int samples = static_cast<int>(input_nodes_hitrates[id]);

  std::optional<int64_t> target_id;
  auto it = input_target_nodeid_map.find(id);
  if (it != input_target_nodeid_map.end()) {
    target_id = it->second;
  } else {
    target_id = std::nullopt;
  }
  std::optional<double> target_weight;
  if (target_id.has_value()) {
    target_weight = input_target_weights[target_id.value()];
  } else {
    target_weight = std::nullopt;
  }

  TreeNode* node = new TreeNode(node_id, feature_id, mode, value, target_id,
                                target_weight, samples);
  node->parent = parent;

  if (mode != "LEAF") {
    int64_t left_node_id = input_nodes_truenodeids[id];
    TreeNode* left_node = model2tree(treeNode, left_node_id, node);
    node->left = left_node;

    int64_t right_node_id = input_nodes_falsenodeids[id];
    TreeNode* right_node = model2tree(treeNode, right_node_id, node);
    node->right = right_node;
  }

  return node;
}

void delete_tree(TreeNode* node) {
  if (node == nullptr) {
    return;
  }
  delete_tree(node->left);
  delete_tree(node->right);
  delete node;
}

class TreeEnsembleRegressor {
 public:
  int n_targets;
  std::vector<int64_t> nodes_falsenodeids;
  std::vector<int64_t> nodes_featureids;
  std::vector<double> nodes_hitrates;
  std::vector<int64_t> nodes_missing_value_tracks_true;
  std::vector<std::string> nodes_modes;
  std::vector<int64_t> nodes_nodeids;
  std::vector<int64_t> nodes_treeids;
  std::vector<int64_t> nodes_truenodeids;
  std::vector<double> nodes_values;
  std::string post_transform;
  std::vector<int64_t> target_ids;
  std::vector<int64_t> target_nodeids;
  std::vector<int64_t> target_treeids;
  std::vector<double> target_weights;

  TreeEnsembleRegressor() : n_targets(1), post_transform("NONE") {};
  //   std::string to_model(const onnx::ModelProto& input_model);
  static TreeEnsembleRegressor from_tree(TreeNode* root) {
    TreeEnsembleRegressor regressor;
    from_tree_internal(regressor, root);

    std::unordered_map<int, int> id_map;
    for (size_t i = 0; i < regressor.nodes_nodeids.size(); ++i) {
      int old_id = regressor.nodes_nodeids[i];
      id_map[old_id] = static_cast<int>(i);
    }

    std::vector<bool> is_leaf;
    for (const auto& mode : regressor.nodes_modes) {
      is_leaf.push_back(mode == "LEAF");
    }

    for (size_t i = 0; i < regressor.nodes_falsenodeids.size(); ++i) {
      if (is_leaf[i]) {
        regressor.nodes_falsenodeids[i] = 0;
      } else {
        regressor.nodes_falsenodeids[i] =
            id_map[regressor.nodes_falsenodeids[i]];
      }
    }
    for (size_t i = 0; i < regressor.nodes_truenodeids.size(); ++i) {
      if (is_leaf[i]) {
        regressor.nodes_truenodeids[i] = 0;
      } else {
        regressor.nodes_truenodeids[i] = id_map[regressor.nodes_truenodeids[i]];
      }
    }
    for (size_t i = 0; i < regressor.nodes_nodeids.size(); ++i) {
      regressor.nodes_nodeids[i] = id_map[regressor.nodes_nodeids[i]];
    }
    for (size_t i = 0; i < regressor.target_nodeids.size(); ++i) {
      regressor.target_nodeids[i] = id_map[regressor.target_nodeids[i]];
    }

    return regressor;
  }

 private:
  static void from_tree_internal(TreeEnsembleRegressor& regressor,
                                 TreeNode* node) {
    bool is_leaf = node->mode == "LEAF";

    int falsenodeid = (!is_leaf && node->right) ? node->right->id : 0;
    int truenodeid = (!is_leaf && node->left) ? node->left->id : 0;

    regressor.nodes_falsenodeids.push_back(falsenodeid);
    regressor.nodes_featureids.push_back(node->feature_id);
    regressor.nodes_hitrates.push_back(static_cast<double>(node->samples));
    regressor.nodes_missing_value_tracks_true.push_back(0);
    regressor.nodes_modes.push_back(node->mode);
    regressor.nodes_nodeids.push_back(node->id);
    regressor.nodes_treeids.push_back(0);
    regressor.nodes_truenodeids.push_back(truenodeid);
    regressor.nodes_values.push_back(node->value);

    if (is_leaf) {
      regressor.target_ids.push_back(0);
      regressor.target_nodeids.push_back(node->id);
      regressor.target_treeids.push_back(0);
      regressor.target_weights.push_back(node->target_weight.value_or(0.0));
    }

    if (!is_leaf) {
      from_tree_internal(regressor, node->left);
      from_tree_internal(regressor, node->right);
    }
  }
};

void toTree(Node* treeNode, TreeEnsembleRegressor& regressor) {
  treeNode->is_(Symbol("nodes_falsenodeids"),
                std::move(regressor.nodes_falsenodeids));
  treeNode->is_(Symbol("nodes_featureids"),
                std::move(regressor.nodes_featureids));
  treeNode->fs_(Symbol("nodes_hitrates"), std::move(regressor.nodes_hitrates));
  treeNode->is_(Symbol("nodes_missing_value_tracks_true"),
                std::move(regressor.nodes_missing_value_tracks_true));
  treeNode->ss_(Symbol("nodes_modes"), std::move(regressor.nodes_modes));
  treeNode->is_(Symbol("nodes_nodeids"), std::move(regressor.nodes_nodeids));
  treeNode->is_(Symbol("nodes_treeids"), std::move(regressor.nodes_treeids));
  treeNode->is_(Symbol("nodes_truenodeids"),
                std::move(regressor.nodes_truenodeids));
  treeNode->fs_(Symbol("nodes_values"), std::move(regressor.nodes_values));
  treeNode->is_(Symbol("target_ids"), std::move(regressor.target_ids));
  treeNode->is_(Symbol("target_nodeids"), std::move(regressor.target_nodeids));
  treeNode->is_(Symbol("target_treeids"), std::move(regressor.target_treeids));
  treeNode->fs_(Symbol("target_weights"), std::move(regressor.target_weights));
}

struct pair_hash {
  std::size_t operator()(const std::pair<int, bool>& p) const {
    return std::hash<int>{}(p.first) ^ (std::hash<bool>{}(p.second) << 1);
  }
};

class MergeChain {
 public:
  TreeNode* start_node;
  TreeNode* end_node;
  std::optional<int> value;

  MergeChain(TreeNode* _start_node, TreeNode* _end_node,
             std::optional<int> _value)
      : start_node(_start_node), end_node(_end_node), value(_value) {}

  bool left_leaf_value(TreeNode* node) {
    if (!value.has_value()) {
      return false;
    }
    if (!node->left || node->left->mode != "LEAF" ||
        !node->left->target_weight.has_value()) {
      return false;
    }
    return static_cast<int>(node->left->target_weight.value()) == value.value();
  }

  bool has_same_feature() {
    if (!value.has_value() || (value.value() != 0 && value.value() != 1)) {
      return false;
    }
    std::unordered_set<int> features;
    TreeNode* node = end_node;
    while (node != start_node->parent) {
      if (features.find(node->feature_id) != features.end()) {
        return true;
      }
      features.insert(node->feature_id);
      node = node->parent;
    }
    return false;
  }
  // intra-chain merge
  void merge() {
    if (!value.has_value() || (value.value() != 0 && value.value() != 1)) {
      return;
    }

    std::unordered_map<std::pair<int, bool>, TreeNode*, pair_hash> node_map;
    TreeNode* node = start_node;

    while (true) {
      int feature_id = node->feature_id;
      bool left_leaf = left_leaf_value(node);
      auto key = std::make_pair(feature_id, left_leaf);

      auto it = node_map.find(key);
      if (it == node_map.end()) {
        node_map[key] = node;
      } else {
        TreeNode* ancestor_node = it->second;
        ancestor_node->value = node->value;

        // std::cout << "intra-chain merging..." << std::endl;

        TreeNode* parent = node->parent;
        if (left_leaf) {
          if (ancestor_node->left && node->left) {
            ancestor_node->left->samples += node->left->samples;
          }

          if (parent->left == node) {
            parent->left = node->right;
          } else {
            parent->right = node->right;
          }

          if (node->right) {
            node->right->parent = parent;
          }
        } else {
          if (ancestor_node->right && node->right) {
            ancestor_node->right->samples += node->right->samples;
          }

          if (parent->left == node) {
            parent->left = node->left;
          } else {
            parent->right = node->left;
          }

          if (node->left) {
            node->left->parent = parent;
          }
        }

        if (node == end_node) {
          end_node = parent;
          break;
        }
      }

      if (node == end_node) {
        break;
      }

      node = left_leaf ? node->right : node->left;
    }

    update_samples();
    check_and_update_value();
  }

  void update_samples() {
    if (!value.has_value()) {
      return;
    }

    TreeNode* node = end_node;
    while (true) {
      int left_samples = node->left ? node->left->samples : 0;
      int right_samples = node->right ? node->right->samples : 0;
      node->samples = left_samples + right_samples;

      if (node == start_node) {
        break;
      }
      node = node->parent;
    }
  }

  void print() {
    if (!value.has_value()) {
      std::cout << std::endl;
      return;
    }

    std::string ret;
    TreeNode* node = end_node;
    while (node != start_node->parent) {
      std::string s = node_str(node->left) + ", " + node_str(node->right);
      ret = s + "\n" + ret;
      node = node->parent;
    }
    ret = node_str(start_node) + "\n" + ret;

    std::cout << ret << std::endl;
  }

  static std::string node_str(TreeNode* node) {
    if (node->mode == "LEAF") {
      int target_weight = node->target_weight.has_value()
                              ? static_cast<int>(node->target_weight.value())
                              : 0;
      return "[LEAF: " + std::to_string(target_weight) + ", (" +
             std::to_string(node->samples) + ")]";
    }
    std::ostringstream oss;
    oss << "[x" << node->feature_id << " <= " << std::fixed
        << std::setprecision(6) << node->value << ", (" << node->samples
        << ")]";
    return oss.str();
  }

  void check_and_update_value() {
    if (start_node == end_node && start_node->left &&
        start_node->left->mode == "LEAF" && start_node->right &&
        start_node->right->mode == "LEAF") {
      value = 2;
    }
  }
};

using MergeChainPtr = std::unique_ptr<MergeChain>;

std::pair<TreeNode*, int> find_merge_chains_(
    TreeNode* node, std::vector<MergeChainPtr>& merge_chains) {
  if (node == nullptr) {
    return {nullptr, 3};
  }

  bool left_is_leaf = node->left && node->left->mode == "LEAF";
  bool right_is_leaf = node->right && node->right->mode == "LEAF";

  if (left_is_leaf && right_is_leaf) {
    return {node, 2};
  }

  if (left_is_leaf) {
    if (!node->left->target_weight.has_value()) {
      return {node, 3};
    }

    int chain_value = static_cast<int>(node->left->target_weight.value());
    auto right_result = find_merge_chains_(node->right, merge_chains);
    TreeNode* end_node = right_result.first;
    int right_chain_value = right_result.second;

    if (right_chain_value == 2 || right_chain_value == chain_value) {
      return {end_node, chain_value};
    }

    return {node, chain_value};
  }

  if (right_is_leaf) {
    if (!node->right->target_weight.has_value()) {
      return {node, 3};
    }

    int chain_value = static_cast<int>(node->right->target_weight.value());
    auto left_result = find_merge_chains_(node->left, merge_chains);
    TreeNode* end_node = left_result.first;
    int left_chain_value = left_result.second;

    if (left_chain_value == 2 || left_chain_value == chain_value) {
      return {end_node, chain_value};
    }

    return {node, chain_value};
  }

  auto left_result = find_merge_chains_(node->left, merge_chains);
  TreeNode* left_end_node = left_result.first;
  int left_chain_value = left_result.second;

  if ((left_chain_value >= 0 && left_chain_value <= 2) &&
      left_end_node != node->left) {
    merge_chains.emplace_back(std::make_unique<MergeChain>(
        node->left, left_end_node, left_chain_value));
  }

  auto right_result = find_merge_chains_(node->right, merge_chains);
  TreeNode* right_end_node = right_result.first;
  int right_chain_value = right_result.second;

  if ((right_chain_value >= 0 && right_chain_value <= 2) &&
      right_end_node != node->right) {
    merge_chains.emplace_back(std::make_unique<MergeChain>(
        node->right, right_end_node, right_chain_value));
  }

  return {nullptr, 3};
}

void find_merge_chains(TreeNode* node,
                       std::vector<MergeChainPtr>& merge_chains) {
  std::pair<TreeNode*, int> result = find_merge_chains_(node, merge_chains);
  TreeNode* end_node = result.first;
  int chain_value = result.second;
  if (chain_value >= 0 && chain_value <= 2) {
    merge_chains.emplace_back(
        std::make_unique<MergeChain>(node, end_node, chain_value));
  }
}

struct TreeNodePtrHash {
  std::size_t operator()(const TreeNode* node) const {
    return std::hash<int>{}(node->id);
  }
};

struct TreeNodePtrEqual {
  bool operator()(const TreeNode* lhs, const TreeNode* rhs) const {
    return lhs == rhs;
  }
};

std::unordered_map<TreeNode*, std::vector<MergeChain*>, TreeNodePtrHash,
                   TreeNodePtrEqual>
list_to_parent_map(const std::vector<MergeChainPtr>& merge_chains) {
  std::unordered_map<TreeNode*, std::vector<MergeChain*>, TreeNodePtrHash,
                     TreeNodePtrEqual>
      parent_map;

  for (auto& chain_ptr : merge_chains) {
    MergeChain* chain = chain_ptr.get();
    TreeNode* parent = chain->start_node->parent;
    if (parent != nullptr) {
      parent_map[parent].push_back(chain);
    }
  }

  for (auto& [parent, chains] : parent_map) {
    if (chains.size() > 1) {
      if (chains[0]->start_node != parent->left) {
        std::swap(chains[0], chains[1]);
      }
    }
  }

  return parent_map;
}

void inter_chain_merge(TreeNode* parent, MergeChain* left_chain,
                       MergeChain* right_chain) {
  if (parent->left != left_chain->start_node ||
      parent->right != right_chain->start_node) {
    throw std::invalid_argument("Parent not match");
  }

  int feature_id = parent->feature_id;

  TreeNode* left_node = left_chain->end_node;
  while (left_node != parent) {
    if (left_node->feature_id == feature_id &&
        left_node->right->mode == "LEAF") {
      int target_weight =
          left_node->right->target_weight.has_value()
              ? static_cast<int>(left_node->right->target_weight.value())
              : -1;
      if (left_chain->value == 2 ||
          left_chain->value.value() == target_weight) {
        break;
      }
      return;
    }
    left_node = left_node->parent;
  }
  if (left_node == parent) {
    return;
  }

  TreeNode* right_node = right_chain->end_node;
  while (right_node != parent) {
    if (right_node->feature_id == feature_id &&
        right_node->left->mode == "LEAF") {
      int target_weight =
          right_node->left->target_weight.has_value()
              ? static_cast<int>(right_node->left->target_weight.value())
              : -1;
      if (right_chain->value == 2 ||
          right_chain->value.value() == target_weight) {
        break;
      }
      return;
    }
    right_node = right_node->parent;
  }
  if (right_node == parent) {
    return;
  }

  // std::cout << "inter-chain merging..." << std::endl;

  int left_length = 0;
  TreeNode* node = left_node;
  while (node != parent) {
    left_length++;
    node = node->parent;
  }

  int right_length = 0;
  node = right_node;
  while (node != parent) {
    right_length++;
    node = node->parent;
  }

  if (left_length <= right_length) {
    parent->value = right_node->value;

    if (right_node == right_node->parent->left) {
      right_node->parent->left = right_node->right;
    } else {
      right_node->parent->right = right_node->right;
    }
    if (right_node->right) {
      right_node->right->parent = right_node->parent;
    }

    if (left_node->right && right_node->left) {
      left_node->right->samples += right_node->left->samples;
    }

    if (right_chain->start_node == right_chain->end_node) {
      right_chain->start_node = nullptr;
      right_chain->end_node = nullptr;
      right_chain->value.reset();
    } else {
      if (right_node == right_chain->start_node) {
        right_chain->start_node = right_node->right;
      }
      if (right_node == right_chain->end_node) {
        right_chain->end_node = right_node->parent;
      }
      right_chain->check_and_update_value();
    }
  } else {
    parent->value = left_node->value;

    if (left_node == left_node->parent->left) {
      left_node->parent->left = left_node->left;
    } else {
      left_node->parent->right = left_node->left;
    }
    if (left_node->left) {
      left_node->left->parent = left_node->parent;
    }

    if (right_node->left && left_node->right) {
      right_node->left->samples += left_node->right->samples;
    }

    if (left_chain->start_node == left_chain->end_node) {
      left_chain->start_node = nullptr;
      left_chain->end_node = nullptr;
      left_chain->value.reset();
    } else {
      if (left_node == left_chain->start_node) {
        left_chain->start_node = left_node->left;
      }
      if (left_node == left_chain->end_node) {
        left_chain->end_node = left_node->parent;
      }
      left_chain->check_and_update_value();
    }
  }

  left_chain->update_samples();
  right_chain->update_samples();
}

// --------------------------
class DTMergeRule {
 public:
  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode) {
    auto root = model2tree(treeNode, 0, nullptr);
    // // std::vector<MergeChain> merge_chains;

    // double reduced_cost = 0.0;

    std::vector<MergeChainPtr> merge_chains;
    find_merge_chains(root, merge_chains);

    // for (size_t i = 0; i < merge_chains.size(); ++i) {
    //   std::cout << i << std::endl;
    //   merge_chains[i]->print();
    //   reduced_cost += merge_chains[i]->start_node->branch_samples();
    //   merge_chains[i]->merge();
    //   reduced_cost -= merge_chains[i]->start_node->branch_samples();
    //   merge_chains[i]->print();
    // }

    auto chain_parent_map = list_to_parent_map(merge_chains);

    int i = 0;
    for (auto& [parent, chains] : chain_parent_map) {
      if (chains.size() > 1) {
        // std::cout << i << std::endl;
        // ++i;
        // std::cout << "parent: " << MergeChain::node_str(parent) << std::endl;
        // std::cout << "left:" << std::endl;
        // chains[0]->print();
        // std::cout << "right:" << std::endl;
        // chains[1]->print();
        // reduced_cost +=
        //     parent->left->branch_samples() + parent->right->branch_samples();
        inter_chain_merge(parent, chains[0], chains[1]);
        // reduced_cost -=
        //     parent->left->branch_samples() + parent->right->branch_samples();
        // std::cout << "parent: " << MergeChain::node_str(parent) << std::endl;
        // std::cout << "left:" << std::endl;
        // chains[0]->print();
        // std::cout << "right:" << std::endl;
        // chains[1]->print();
      }
    }
    auto regressor = TreeEnsembleRegressor::from_tree(root);

    // input: Node* treeNode, TreeEnsembleRegressor regressor
    // copy regressor attribute from Node* treeNode, graph and model structure
    // remain unchanged
    toTree(treeNode, regressor);
    delete_tree(root);

    return saveModelWithNewName(mp_in, graph, model_path, "merged");
  }

  static std::string match(std::string& model_path) {
    std::string output_model_path = model_path;
    ModelProto mp_in;
    loadModel(&mp_in, model_path, true);
    std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

    bool found = false;
    Node* treeNode;
    graph->forEachNode([&found, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("target_treeids"))) {
        auto target_treeids = node->is(Symbol("target_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE" &&
            std::all_of(
                target_treeids.begin(), target_treeids.end(),
                std::bind(std::equal_to<>(), std::placeholders::_1, 0))) {
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(mp_in, graph, model_path, treeNode);

    return output_model_path;
  }
};

}  // namespace onnx::optimization
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>

#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
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

#define M_FALSE 0
#define M_TRUE 1
#define M_NO 2

#define CONSERVATIVE 0

namespace onnx::optimization {

float round(float x) {
  return std::round(x * 1e6) / 1e6;
}

struct NodeID {
  int id;
  std::string node;
};

std::string getNewmodelname(std::string onnx_model_path, std::string suffix) {
  return onnx_model_path.substr(0, onnx_model_path.find(".onnx")) + "_" +
         suffix + ".onnx";
}

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

class ThreadPool {
 public:
  explicit ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
              return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
      worker.join();
  }

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;

  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

/**
 * @brief 分类树转回归树(单棵决策树、随机森林)
 */
class DTConvertRule {
 public:
  static void processNode(Node* node, int mode) {
    auto classlabels_int64s = node->is(Symbol("classlabels_int64s"));
    auto class_treeids = node->is(Symbol("class_treeids"));
    auto class_ids = node->is(Symbol("class_ids"));
    auto class_nodeids = node->is(Symbol("class_nodeids"));
    auto class_weights = node->fs(Symbol("class_weights"));

    std::unordered_set<int> unique_treeids(class_treeids.begin(),
                                             class_treeids.end());
    int n_trees = unique_treeids.size();
    
    // ** convert clf attributes 2 reg attributes
    int64_t stride =
        classlabels_int64s.size() == 2 ? 1 : classlabels_int64s.size();
    int nleaf = static_cast<int>(class_weights.size() / stride);

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
        target_weights.push_back((w > 0.5 / n_trees) ? 1.0 : 0.0);
        // target_weights.push_back(static_cast<double>(std::round(w * n_trees)));
      }
    } else {
      for (int i = 0; i < nleaf; ++i) {
        auto start_it = class_weights.begin() + (i * stride);
        auto end_it = class_weights.begin() + ((i + 1) * stride);

        auto max_it = std::max_element(start_it, end_it);
        int index = std::distance(start_it, max_it);
        // wine_quality classlabels_int64s [3,4,5,6,7,8]
        target_weights.push_back(static_cast<double>(classlabels_int64s[index]));
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
                           Node* node, std::string& model_path) {
    if (node->hasAttribute(Symbol("classlabels_int64s"))) {
      processNode(node, 0);
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
    Node* treeNode;
    graph->forEachNode([&found, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("class_treeids"))) {
        auto class_treeids = node->is(Symbol("class_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE") {
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(mp_in, graph, treeNode, model_path);

    return output_model_path;
  }
};

/**
 * @brief 决策树剪枝(单棵决策树、随机森林)
 * @deprecated DTPruneRule
 */
//** Rule2.1: 单棵树剪枝
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

    // int64_t input_n_targets = node->i(Symbol("n_targets"));
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
      return saveModelWithNewName(mp_in, graph, model_path, "pruned" + std::to_string(threshold));
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
    for (size_t i = 0; i < nodes_treeids.size(); ++i) {
      if (nodes_treeids[i] == next_tree_id) {
        next_tree_id++;
        tree_roots.push_back(i);
      }
    }

    std::vector<std::tuple<int, int>> tree_intervals;
    for (size_t i = 0; i < tree_roots.size(); ++i) {
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
                     size_t node_id, std::vector<std::string>& result_nodes,
                     Node* treeNode, uint8_t comparison_operator,
                     float threshold) {
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
      for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
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
          pruning(tree_no, tree_interval, left_node_id, result_nodes, treeNode,
                  comparison_operator, threshold);

      size_t right_node_id = right_nodes[node_id];
      int right_result =
          pruning(tree_no, tree_interval, right_node_id, result_nodes, treeNode,
                  comparison_operator, threshold);

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

  static void pruning_loop(size_t tree_no,
                           const std::tuple<int, int>& tree_interval,
                           std::vector<std::string>& result_nodes,
                           Node* treeNode, uint8_t comparison_operator,
                           float threshold) {
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

    std::vector<int> computed_result(node_types.size(), -1);

    struct StackFrame {
      int64_t node_id;
      bool visited;
    };
    std::stack<StackFrame> s;
    s.push(StackFrame{0, false});

    while (!s.empty()) {
      auto& frame = s.top();
      size_t curr_id = frame.node_id;
      if (!frame.visited) {
        result_nodes[curr_id] = node_types[curr_id];
        frame.visited = true;

        if (node_types[curr_id] == "LEAF") {
          int target_idx = -1;
          for (size_t ti = 0; ti < target_nodeids.size(); ++ti) {
            if (target_nodeids[ti] == curr_id &&
                target_treeids[ti] == tree_no) {
              target_idx = ti;
              break;
            }
          }
          int result = comparison_funcs[comparison_operator](target_weights[target_idx], threshold);
          result_nodes[curr_id] = (result == 1) ? "LEAF_TRUE" : "LEAF_FALSE";
          computed_result[curr_id] = result;
          s.pop();
        } else {
          s.push(StackFrame{right_nodes[curr_id], false});
          s.push(StackFrame{left_nodes[curr_id], false});
        }
      } else {
        size_t left_id = left_nodes[curr_id];
        size_t right_id = right_nodes[curr_id];
        int left_res = computed_result[left_id];
        int right_res = computed_result[right_id];

        if (left_res == 0 && right_res == 0) {
          result_nodes[curr_id] = "LEAF_FALSE";
          result_nodes[left_id] = "REMOVED";
          result_nodes[right_id] = "REMOVED";
          computed_result[curr_id] = 0;
        } else if (left_res == 1 && right_res == 1) {
          result_nodes[curr_id] = "LEAF_TRUE";
          result_nodes[left_id] = "REMOVED";
          result_nodes[right_id] = "REMOVED";
          computed_result[curr_id] = 1;
        } else {
          computed_result[curr_id] = 2;
        }
        s.pop();
      }
    }
  }

  static bool processNode(
      Node* node, std::vector<std::vector<std::string>>& result_nodes_list,
      std::vector<std::tuple<int, int>>& tree_intervals) {
    const int tree_count = tree_intervals.size();
    std::vector<int> tree_leaf_counts;
    for (const auto& removed_nodes : result_nodes_list) {
      int leaf_false_count =
          std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_FALSE");
      int leaf_true_count =
          std::count(removed_nodes.begin(), removed_nodes.end(), "LEAF_TRUE");
      tree_leaf_counts.push_back(leaf_false_count + leaf_true_count);
    }

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

    // 构建 nodes_falsenodeids
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

    // 构建 nodes_featureids
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

    // 构建 nodes_hitrates
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

    // 构建 nodes_missing_value_tracks_true
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

    // 构建 nodes_modes
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

    // 构建 nodes_nodeids
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

    // 构建 nodes_treeids
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
    // 构建 nodes_truenodeids
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

    // 构建 nodes_values
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

    // 构建 target_ids
    std::vector<int64_t> target_ids(
        std::accumulate(tree_leaf_counts.begin(), tree_leaf_counts.end(), 0),
        0);

    // 构建 target_nodeids
    std::vector<int64_t> target_nodeids;
    for (const auto& new_ids : new_ids_list) {
      for (const auto& new_id : new_ids) {
        if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
          target_nodeids.push_back(new_id.id);
        }
      }
    }

    // 构建 target_treeids
    std::vector<int64_t> target_treeids;
    for (size_t tree_no = 0; tree_no < tree_leaf_counts.size(); ++tree_no) {
      target_treeids.insert(target_treeids.end(), tree_leaf_counts[tree_no],
                            tree_no);
    }

    // 构建 target_weights
    std::vector<double> target_weights;
    double tw = 1.0 / tree_count;
    for (const auto& new_ids : new_ids_list) {
      for (const auto& new_id : new_ids) {
        if (new_id.node == "LEAF_FALSE" || new_id.node == "LEAF_TRUE") {
          target_weights.push_back(
              new_id.node == "LEAF_TRUE" ? tw : 0.0f);
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

  static std::string apply(int threads_count, ModelProto& mp_in,
                           std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode,
                           uint8_t comparison_operator, float threshold) {
    auto tree_intervals = getTreeIntervals(treeNode);
    // 由分类树转成的回归树无需处理threshold
    if (model_path.find("reg") == std::string::npos){
      threshold /= tree_intervals.size();
    }
    std::vector<std::vector<std::string>> result_nodes_list;
    for (const auto& interval : tree_intervals) {
      int start = std::get<0>(interval);
      int end = std::get<1>(interval);
      result_nodes_list.push_back(std::vector<std::string>(end - start, ""));
    }
    if (threads_count == 1) {
      for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
        auto& result_nodes = result_nodes_list[tree_no];
        const auto& tree_interval = tree_intervals[tree_no];
        pruning_loop(tree_no, tree_interval, result_nodes, treeNode,
                     comparison_operator, threshold);
      }
    } else {
      ThreadPool pool(threads_count);
      std::vector<std::future<void>> futures;
      futures.reserve(result_nodes_list.size());
      for (size_t tree_no = 0; tree_no < result_nodes_list.size(); ++tree_no) {
        futures.push_back(pool.enqueue([&, tree_no]() {
          auto& result_nodes = result_nodes_list[tree_no];
          const auto& tree_interval = tree_intervals[tree_no];
          pruning_loop(tree_no, tree_interval, result_nodes, treeNode,
                       comparison_operator, threshold);
        }));
      }
      for (auto& fut : futures) {
        fut.get();
      }
    }

    if (processNode(treeNode, result_nodes_list, tree_intervals)) {
      return saveModelWithNewName(mp_in, graph, model_path, "pruned" + std::to_string(threshold));
    } else {
      std::string new_model_path = model_path;
      return new_model_path;
    }
  }

  static std::string match(std::string& model_path, uint8_t comparison_operator,
                           float threshold, int threads_count) {
    std::string output_model_path = model_path;
    ModelProto mp_in;
    loadModel(&mp_in, model_path, true);
    std::shared_ptr<Graph> graph = std::move(ImportModelProto(mp_in));

    bool found = false;
    Node* treeNode;
    graph->forEachNode([&found, &treeNode](Node* node) {
      if (node->hasAttribute(Symbol("target_treeids"))) {
        auto target_treeids = node->is(Symbol("target_treeids"));
        if (node->s(Symbol("post_transform")) == "NONE") {
          found = true;
          treeNode = node;
        }
      }
    });

    if (found)
      output_model_path = apply(threads_count, mp_in, graph, model_path,
                                treeNode, comparison_operator, threshold);

    return output_model_path;
  }
};

RFPruneRule::ComparisonFunc RFPruneRule::comparison_funcs[] = {
    [](float x, float y) { return x == y; },
    [](float x, float y) { return x < y; },
    [](float x, float y) { return x <= y; },
    [](float x, float y) { return x > y; },
    [](float x, float y) { return x >= y; }};

/**
 * @brief 决策树合并（随机森林）
 */
class TreeNode {
 public:
  int id;
  int feature_id;
  std::string mode;
  double value;
  std::optional<int> target_id;
  std::optional<double> target_weight;
  int samples;

  std::weak_ptr<TreeNode> parent;
  std::shared_ptr<TreeNode> left;
  std::shared_ptr<TreeNode> right;

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
        // parent(nullptr),
        left(nullptr),
        right(nullptr) {}
};

static std::vector<std::tuple<int, int>> get_target_tree_intervals(Node* node) {
  std::vector<int> target_tree_roots;
  std::vector<int64_t> target_treeids = node->is(Symbol("target_treeids"));
  int next_tree_id = 0;
  for (int i = 0; i < target_treeids.size(); ++i) {
    if (target_treeids[i] == next_tree_id) {
      next_tree_id++;
      target_tree_roots.push_back(i);
    }
  }

  std::vector<std::tuple<int, int>> target_tree_intervals;
  for (int i = 0; i < target_tree_roots.size(); ++i) {
    int end;
    if (i == target_tree_roots.size() - 1) {
      end = target_treeids.size();
    } else {
      end = target_tree_roots[i + 1];
    }
    target_tree_intervals.push_back(std::make_tuple(target_tree_roots[i], end));
  }
  return target_tree_intervals;
}

std::shared_ptr<TreeNode> model2tree(Node* treeNode, int64_t node_id, std::shared_ptr<TreeNode> parent,
                     std::tuple<int, int>& tree_interval,
                     std::tuple<int, int>& target_tree_interval) {
  int tree_start = std::get<0>(tree_interval);
  int tree_end = std::get<1>(tree_interval);
  int target_tree_start = std::get<0>(target_tree_interval);
  int target_tree_end = std::get<1>(target_tree_interval);

  const auto& input_nodes_falsenodeids_ =
      treeNode->is(Symbol("nodes_falsenodeids"));
  const auto& input_nodes_featureids_ =
      treeNode->is(Symbol("nodes_featureids"));
  const auto& input_nodes_hitrates_ = treeNode->fs(Symbol("nodes_hitrates"));
  const auto& input_nodes_modes_ = treeNode->ss(Symbol("nodes_modes"));
  const auto& input_nodes_truenodeids_ =
      treeNode->is(Symbol("nodes_truenodeids"));
  const auto& input_nodes_values_ = treeNode->fs(Symbol("nodes_values"));
  const auto& input_target_nodeids_ = treeNode->is(Symbol("target_nodeids"));
  const auto& input_target_weights_ = treeNode->fs(Symbol("target_weights"));

  std::vector<int64_t> input_nodes_falsenodeids(
      input_nodes_falsenodeids_.begin() + tree_start,
      input_nodes_falsenodeids_.begin() + tree_end);
  std::vector<int64_t> input_nodes_featureids(
      input_nodes_featureids_.begin() + tree_start,
      input_nodes_featureids_.begin() + tree_end);
  std::vector<double> input_nodes_hitrates(
      input_nodes_hitrates_.begin() + tree_start,
      input_nodes_hitrates_.begin() + tree_end);
  std::vector<std::string> input_nodes_modes(
      input_nodes_modes_.begin() + tree_start,
      input_nodes_modes_.begin() + tree_end);
  std::vector<int64_t> input_nodes_truenodeids(
      input_nodes_truenodeids_.begin() + tree_start,
      input_nodes_truenodeids_.begin() + tree_end);
  std::vector<double> input_nodes_values(
      input_nodes_values_.begin() + tree_start,
      input_nodes_values_.begin() + tree_end);
  std::vector<int64_t> input_target_nodeids(
      input_target_nodeids_.begin() + target_tree_start,
      input_target_nodeids_.begin() + target_tree_end);
  std::vector<double> input_target_weights(
      input_target_weights_.begin() + target_tree_start,
      input_target_weights_.begin() + target_tree_end);

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

  std::shared_ptr<TreeNode> node = std::make_shared<TreeNode>(
    id, feature_id, mode, value, target_id, target_weight, samples);
  node->parent = parent;

  if (mode != "LEAF") {
    int64_t left_node_id = input_nodes_truenodeids[id];
    std::shared_ptr<TreeNode> left_node = model2tree(treeNode, left_node_id, node,
      tree_interval, target_tree_interval);
    node->left = left_node;

    int64_t right_node_id = input_nodes_falsenodeids[id];
    std::shared_ptr<TreeNode> right_node = model2tree(treeNode, right_node_id, node,
      tree_interval, target_tree_interval);
    node->right = right_node;
  }

  return node;
}

std::vector<std::shared_ptr<TreeNode>> model2trees(Node* treeNode) {
  auto tree_intervals = RFPruneRule::getTreeIntervals(treeNode);
  auto target_tree_intervals = get_target_tree_intervals(treeNode);
  std::vector<std::shared_ptr<TreeNode>> trees;
  for (size_t i = 0; i < tree_intervals.size(); i++) {
    std::shared_ptr<TreeNode> root = model2tree(treeNode, 0, nullptr, tree_intervals[i],
                                target_tree_intervals[i]);
    trees.push_back(root);
  }
  return trees;
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

  static TreeEnsembleRegressor from_trees(std::vector<std::shared_ptr<TreeNode>> roots) {
    TreeEnsembleRegressor regressor;
    std::vector<TreeEnsembleRegressor> regressors;
    for (size_t i = 0; i < roots.size(); i++) {
      regressors.push_back(from_tree(roots[i], i));
    }
    for (auto& r : regressors) {
      regressor.nodes_falsenodeids.insert(regressor.nodes_falsenodeids.end(),
                                          r.nodes_falsenodeids.begin(),
                                          r.nodes_falsenodeids.end());
      regressor.nodes_featureids.insert(regressor.nodes_featureids.end(),
                                        r.nodes_featureids.begin(),
                                        r.nodes_featureids.end());
      regressor.nodes_hitrates.insert(regressor.nodes_hitrates.end(),
                                      r.nodes_hitrates.begin(),
                                      r.nodes_hitrates.end());
      regressor.nodes_missing_value_tracks_true.insert(
          regressor.nodes_missing_value_tracks_true.end(),
          r.nodes_missing_value_tracks_true.begin(),
          r.nodes_missing_value_tracks_true.end());
      regressor.nodes_modes.insert(regressor.nodes_modes.end(),
                                   r.nodes_modes.begin(), r.nodes_modes.end());
      regressor.nodes_nodeids.insert(regressor.nodes_nodeids.end(),
                                     r.nodes_nodeids.begin(),
                                     r.nodes_nodeids.end());
      regressor.nodes_treeids.insert(regressor.nodes_treeids.end(),
                                     r.nodes_treeids.begin(),
                                     r.nodes_treeids.end());
      regressor.nodes_truenodeids.insert(regressor.nodes_truenodeids.end(),
                                         r.nodes_truenodeids.begin(),
                                         r.nodes_truenodeids.end());
      regressor.nodes_values.insert(regressor.nodes_values.end(),
                                    r.nodes_values.begin(),
                                    r.nodes_values.end());
      regressor.target_ids.insert(regressor.target_ids.end(),
                                  r.target_ids.begin(), r.target_ids.end());
      regressor.target_nodeids.insert(regressor.target_nodeids.end(),
                                      r.target_nodeids.begin(),
                                      r.target_nodeids.end());
      regressor.target_treeids.insert(regressor.target_treeids.end(),
                                      r.target_treeids.begin(),
                                      r.target_treeids.end());
      regressor.target_weights.insert(regressor.target_weights.end(),
                                      r.target_weights.begin(),
                                      r.target_weights.end());
    }
    return regressor;
  }

  static TreeEnsembleRegressor from_tree(std::shared_ptr<TreeNode> root, int tree_no = 0) {
    TreeEnsembleRegressor regressor;
    from_tree_internal(regressor, root, tree_no);

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
    std::shared_ptr<TreeNode> node, int tree_no = 0) {
    bool is_leaf = node->mode == "LEAF";

    int falsenodeid = (!is_leaf && node->right) ? node->right->id : 0;
    int truenodeid = (!is_leaf && node->left) ? node->left->id : 0;

    regressor.nodes_falsenodeids.push_back(falsenodeid);
    regressor.nodes_featureids.push_back(node->feature_id);
    regressor.nodes_hitrates.push_back(static_cast<double>(node->samples));
    regressor.nodes_missing_value_tracks_true.push_back(0);
    regressor.nodes_modes.push_back(node->mode);
    regressor.nodes_nodeids.push_back(node->id);
    regressor.nodes_treeids.push_back(tree_no);
    regressor.nodes_truenodeids.push_back(truenodeid);
    regressor.nodes_values.push_back(node->value);

    if (is_leaf) {
      regressor.target_ids.push_back(0);
      regressor.target_nodeids.push_back(node->id);
      regressor.target_treeids.push_back(tree_no);
      regressor.target_weights.push_back(node->target_weight.value_or(0.0));
    }

    if (!is_leaf) {
      from_tree_internal(regressor, node->left, tree_no);
      from_tree_internal(regressor, node->right, tree_no);
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

void update_result(std::shared_ptr<TreeNode> node, int path_length, std::shared_ptr<TreeNode> root,
  std::vector<std::tuple<std::shared_ptr<TreeNode>, int>>& result) {
  if (node->mode == "LEAF")
    return;
  assert(node->feature_id == root->feature_id);

  if (result.empty()) {
    result.push_back(std::make_tuple(node, path_length));
    return;
  }

  auto last = std::get<0>(result.back());
  assert(node->feature_id == last->feature_id);
  float new_delta = std::abs(round(round(node->value) - round(root->value)));
  float old_delta = std::abs(round(round(last->value) - round(root->value)));

  if (new_delta == old_delta) {
    result.push_back(std::make_tuple(node, path_length));
    return;
  }
  if (new_delta < old_delta) {
    result.clear();
    result.push_back(std::make_tuple(node, path_length));
    return;
  }
  return;
}

int find_merge_nodes(std::shared_ptr<TreeNode> node, int path_length, std::shared_ptr<TreeNode> root,
                     bool left_branch,
                     std::vector<std::tuple<std::shared_ptr<TreeNode>, int>>& result) {
  if (node->mode == "LEAF") {
    return (node->target_weight.value() == 0) ? M_FALSE : M_TRUE;
  }

  bool same_feature = (node->feature_id == root->feature_id);
  int left_merge_stats = M_NO, right_merge_stats = M_NO;

  if (!same_feature || (same_feature && !left_branch)) {
    left_merge_stats = find_merge_nodes(node->left, path_length + 1, root,
                                        left_branch, result);
    if (left_merge_stats == M_NO)
      return M_NO;
  }

  if (!same_feature || (same_feature && left_branch)) {
    right_merge_stats = find_merge_nodes(node->right, path_length + 1, root,
                                         left_branch, result);
    if (right_merge_stats == M_NO)
      return M_NO;
  }

  if (!same_feature) {
    if (left_merge_stats != right_merge_stats)
      return M_NO;
    else {
      // update_result(node, path_length, root, result);
      return left_merge_stats;
    }
  }

  if (!left_branch) {
    if (left_merge_stats != M_NO)
      update_result(node, path_length, root, result);
    return left_merge_stats;
  }

  if (left_branch) {
    if (right_merge_stats != M_NO)
      update_result(node, path_length, root, result);
    return right_merge_stats;
  }
  return M_NO;
}

void merge(std::shared_ptr<TreeNode> root, std::vector<std::shared_ptr<TreeNode>> nodes, bool left_branch) {
  assert(root->mode == "BRANCH_LEQ");
  assert(!nodes.empty());
  for (auto node : nodes) {
    assert(node->mode == "BRANCH_LEQ");
    assert(node->feature_id == root->feature_id);
  }

  root->value = nodes[0]->value;

  for (auto node : nodes) {
    if (left_branch) {
      auto parent = node->parent.lock();
      if (!parent) {
        throw std::runtime_error("Parent node is no longer available.");
      }

      std::shared_ptr<TreeNode> left = node->left;

      if (node == parent->left)
        parent->left = left;
      else
        parent->right = left;
      left->parent = parent;

    } else {
      auto parent = node->parent.lock();
      if (!parent) {
        throw std::runtime_error("Parent node is no longer available.");
      }
      std::shared_ptr<TreeNode> right = node->right;

      if (node == parent->left)
        parent->left = right;
      else
        parent->right = right;
      right->parent = parent;
    }

    node->parent.reset();
    node->left.reset();
    node->right.reset();
  }
}

void dfs(std::shared_ptr<TreeNode> node) {
  if (node->mode == "LEAF") {
    return;
  }
  dfs(node->left);
  dfs(node->right);
  std::vector<std::tuple<std::shared_ptr<TreeNode>, int>> left_merge_nodes, right_merge_nodes;
  int left_merge_stats =
      find_merge_nodes(node->left, 1, node, true, left_merge_nodes);
  if (left_merge_stats == M_NO) {
    return;
  }

  int right_merge_stats =
      find_merge_nodes(node->right, 1, node, false, right_merge_nodes);
  if (right_merge_stats == M_NO) {
    return;
  }

  if (left_merge_stats != right_merge_stats) {
    return;
  }

  int max_left_path_length = 0, max_right_path_length = 0;
  for (auto& entry : left_merge_nodes) {
    max_left_path_length = std::max(max_left_path_length, std::get<1>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    max_right_path_length = std::max(max_right_path_length, std::get<1>(entry));
  }

  std::vector<std::shared_ptr<TreeNode>> left_merge_nodes_only, right_merge_nodes_only;
  for (auto& entry : left_merge_nodes) {
    left_merge_nodes_only.push_back(std::get<0>(entry));
  }
  for (auto& entry : right_merge_nodes) {
    right_merge_nodes_only.push_back(std::get<0>(entry));
  }

  if (!left_merge_nodes_only.empty() && !right_merge_nodes_only.empty()) {
    if (CONSERVATIVE) {
      return;
    }
    if (max_left_path_length > max_right_path_length) {
      merge(node, left_merge_nodes_only, true);
    } else {
      merge(node, right_merge_nodes_only, false);
    }
    return;
  }

  if (!left_merge_nodes_only.empty()) {
    merge(node, left_merge_nodes_only, true);
    return;
  }

  merge(node, right_merge_nodes_only, false);
}

class DTMergeRule {
 public:
  static std::string apply(ModelProto& mp_in, std::shared_ptr<Graph>& graph,
                           std::string& model_path, Node* treeNode) {
    auto roots = model2trees(treeNode);
    for (size_t i = 0; i < roots.size(); i++) {
      dfs(roots[i]);
    }
    auto regressor = TreeEnsembleRegressor::from_trees(roots);
    toTree(treeNode, regressor);
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
        if (node->s(Symbol("post_transform")) == "NONE") {
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
#!/bin/bash

# # 删除旧的 build 目录
# rm -rf build

# # 创建新的 build 目录并配置 Debug 构建
# cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 构建项目，Debug 模式，先清理再并行编译
cmake --build build  --parallel 50

# ./build/optimize_dt_example
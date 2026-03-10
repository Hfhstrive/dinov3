import onnx
from onnx import helper, checker
import numpy as np
from collections import defaultdict


def fix_dropout_and_rebuild_graph(input_path, output_path):
    # 加载模型
    model = onnx.load(input_path)
    graph = model.graph

    # 第一步：创建节点之间的依赖关系图
    node_inputs = {}  # 每个节点使用的输入
    node_outputs = {}  # 每个节点产生的输出
    output_to_node = {}  # 输出名称 -> 产生它的节点

    # 记录每个输出是由哪个节点产生的
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node
        node_inputs[node.name] = set(node.input)
        node_outputs[node.name] = set(node.output)

    # 第二步：处理Dropout节点，创建映射关系
    dropout_mapping = {}  # Dropout的输出 -> Dropout的输入
    dropout_nodes_to_remove = []

    for node in graph.node:
        if node.op_type == "Dropout":
            if len(node.input) > 0 and len(node.output) > 0:
                dropout_mapping[node.output[0]] = node.input[0]
                dropout_nodes_to_remove.append(node)
                print(f"Found Dropout node: {node.name}, mapping {node.output[0]} -> {node.input[0]}")

    # 如果没有Dropout节点，直接返回
    if not dropout_nodes_to_remove:
        print("No Dropout nodes found")
        onnx.save(model, output_path)
        return

    # 第三步：重建所有节点的连接
    new_nodes = []
    processed_outputs = set()

    # 首先添加所有非Dropout节点
    for node in graph.node:
        if node.op_type != "Dropout":
            new_nodes.append(node)
            for output in node.output:
                processed_outputs.add(output)

    # 第四步：修复所有使用Dropout输出的节点的输入
    for node in new_nodes:
        new_inputs = []
        for input_name in node.input:
            # 如果输入是Dropout的输出，替换为对应的输入
            if input_name in dropout_mapping:
                new_input = dropout_mapping[input_name]
                print(f"Fixing input for node {node.name}: {input_name} -> {new_input}")
                new_inputs.append(new_input)
            else:
                new_inputs.append(input_name)

        # 更新节点的输入
        node.ClearField('input')
        node.input.extend(new_inputs)

    # 第五步：删除所有initializer中可能引用的无效输出
    # 收集所有有效的输入和输出
    valid_names = set()
    for node in new_nodes:
        for input_name in node.input:
            valid_names.add(input_name)
        for output_name in node.output:
            valid_names.add(output_name)

    # 保留initializer中在valid_names中的tensor
    new_initializer = []
    for init in graph.initializer:
        if init.name in valid_names:
            new_initializer.append(init)

    graph.ClearField('initializer')
    graph.initializer.extend(new_initializer)

    # 第六步：处理value_info，移除可能无效的信息
    graph.ClearField('value_info')

    # 第七步：验证并修复拓扑顺序
    # 创建节点依赖图
    node_deps = {}
    node_by_output = {}

    for node in new_nodes:
        for output in node.output:
            node_by_output[output] = node

    # 建立依赖关系
    for node in new_nodes:
        deps = set()
        for input_name in node.input:
            if input_name in node_by_output:
                deps.add(node_by_output[input_name].name)
        node_deps[node.name] = deps

    # 简单的拓扑排序
    sorted_nodes = []
    visited = set()

    def dfs(node_name):
        if node_name in visited:
            return
        visited.add(node_name)

        # 先处理依赖
        for dep_name in node_deps[node_name]:
            # 找到依赖的节点
            for node in new_nodes:
                if node.name == dep_name:
                    dfs(dep_name)
                    break

        # 添加当前节点
        for node in new_nodes:
            if node.name == node_name and node not in sorted_nodes:
                sorted_nodes.append(node)

    # 执行拓扑排序
    node_names = [node.name for node in new_nodes]
    for node_name in node_names:
        dfs(node_name)

    # 更新图的节点
    graph.ClearField('node')
    graph.node.extend(sorted_nodes)

    # 第八步：验证模型
    try:
        checker.check_model(model)
        print("Model validation passed!")
    except Exception as e:
        print(f"Model validation failed: {e}")
        # 打印更详细的调试信息
        print("\nDebug information:")
        for node in graph.node:
            print(f"Node: {node.name} ({node.op_type})")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")

    # 保存模型
    onnx.save(model, output_path)
    print(f"Fixed model saved to {output_path}")
    print(f"Removed {len(dropout_nodes_to_remove)} Dropout nodes")


# 使用
input_model = "/media/inno/work_dirs/mmseg-inno/mmseg1.x/dinov3-vit-small_e16-80.01/end2end_sim.onnx"
output_model = "/media/inno/work_dirs/mmseg-inno/mmseg1.x/dinov3-vit-small_e16-80.01/end2end_fixed.onnx"

fix_dropout_and_rebuild_graph(input_model, output_model)

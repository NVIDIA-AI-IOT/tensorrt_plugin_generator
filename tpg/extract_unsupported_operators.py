# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from tpg.log import *

import argparse
import os
import sys
import onnx
import numpy as np
import yaml

try:
    import onnx_graphsurgeon as gs
except ModuleNotFoundError as e:
    print("Automatic installing onnx_graphsurgeon.")
    import pip
    pip.main("install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com".split(" "))
    import onnx_graphsurgeon as gs

template_dir = os.path.dirname(os.path.realpath(__file__)) + "/plugin_templates/"
def get_supported_op_list(version):
    op_lists = []
    if version == '8.4':
        with open(os.path.join(template_dir, "supported_operator_trt84.txt")) as f:
            lines = f.readlines()
            for line in lines:
                op_lists.append(line.rstrip('\n'))
    else:
        assert False, 'unsupported TRT version, available: 8.4'
    return op_lists

def get_shape_str(shape):
    ret_str = ""
    if shape == None:
        return ret_str
    # WAR for 1D shape dim become 'dim'
    if len(shape) == 1:
        dim = shape[0]
        if type(dim) is not int:
            dim = -1
        return dim
        
    for dim in shape:
        if type(dim) is not int:
            dim = -1
        ret_str += str(dim)
        ret_str += "x"
    # remove last "x"
    ret_str = ret_str[:-1]
    return ret_str

def get_value_datetype(v):
    if type(v) is int:
        return "int32"
    elif type(v) is float:
        return "float32"
    elif type(v) is bool:
        return "bool"
    # always treat string as char array
    elif type(v) is str:
        return "char[]"
    else:
        print(type(v))
        assert False, "unsupported datatype"

def get_attr_datatype(object):
    ret_str = ''
    if type(object) is list:
        ret_str += get_value_datetype(object[0])
        ret_str += '[]'
    else:
        ret_str += get_value_datetype(object)
    return ret_str

def get_input_key_name(i):
    return 'tpg_input_' + str(i)

def get_output_key_name(i):
    return 'tpg_output_' + str(i)

def convert_node_info_to_dict(d, node):
    logging.info("generate yaml for node: "+node.op)
    d[node.op] = {}
    d[node.op]["plugin_type"] = "IPluginV2DynamicExt"
    logging.warning("please specify support_format_combination for node: "+node.op)
    d[node.op]["support_format_combination"] = ["need_user_to_specify"]
    d[node.op]["inputs"] = {}
    for i in range(len(node.inputs)):
        tensor = node.inputs[i]
        key_name = get_input_key_name(i)
        d[node.op]["inputs"][key_name] = {}
        if tensor.shape != None:
            d[node.op]["inputs"][key_name]['shape'] = get_shape_str(tensor.shape)
        else:
            d[node.op]["inputs"][key_name]['shape'] = "need_user_to_specify"

    d[node.op]["outputs"] = {}
    for i in range(len(node.outputs)):
        tensor = node.outputs[i]
        key_name = get_output_key_name(i)
        d[node.op]["outputs"][key_name] = {}
        if tensor.shape != None:
            d[node.op]["outputs"][key_name]['shape'] = get_shape_str(tensor.shape)
        else:
            d[node.op]["outputs"][key_name]['shape'] = "need_user_to_specify"

    if len(node.attrs) > 0:
        d[node.op]["attributes"] = {}
        for attr in node.attrs:
            d[node.op]["attributes"][attr] = {}
            d[node.op]["attributes"][attr]["datatype"] = get_attr_datatype(node.attrs[attr])

def update_node_info(d, node):
    for i in range(len(node.inputs)):
        tensor = node.inputs[i]
        key_name = get_input_key_name(i)
        if (d[node.op]["inputs"][key_name]['shape'] != "need_user_to_specify"
                and d[node.op]["inputs"][key_name]['shape'] != get_shape_str(tensor.shape)):
            d[node.op]["inputs"][key_name]['shape'] = "need_user_to_specify"
    for i in range(len(node.outputs)):
        tensor = node.outputs[i]
        key_name = get_output_key_name(i)
        if (d[node.op]["outputs"][key_name]['shape'] != "need_user_to_specify"
                and d[node.op]["outputs"][key_name]['shape'] != get_shape_str(tensor.shape)):
            d[node.op]["outputs"][key_name]['shape'] = "need_user_to_specify"

def parse_io_tensors(io_tensors_list):
    io_tensors = {}
    for io_tensors_str in io_tensors_list:
        op_name = io_tensors_str.split(":")[0]
        inputs = io_tensors_str.split(":")[1].split(",")
        outputs = io_tensors_str.split(":")[2].split(",")
        io_tensors[op_name] = {}
        io_tensors[op_name]["inputs"] = inputs
        io_tensors[op_name]["outputs"] = outputs
    return io_tensors

def write_custom_shape(d, io_tensors):
    for op_name in io_tensors:
        for i in range(len(io_tensors[op_name]["inputs"])):
            key_name = get_input_key_name(i)
            d[op_name]["inputs"][key_name]['shape'] = io_tensors[op_name]["inputs"][i]
        for i in range(len(io_tensors[op_name]["outputs"])):
            key_name = get_output_key_name(i)
            d[op_name]["outputs"][key_name]['shape'] = io_tensors[op_name]["outputs"][i]

def parse_format_combinations(format_combinations_list):
    format_combinations = {}
    for format_combinations_str in format_combinations_list:
        op_name = format_combinations_str.split(":")[0]
        format_combination = format_combinations_str.split(":")[1].split(",")
        format_combinations[op_name] = {}
        format_combinations[op_name]["support_format_combination"] = format_combination
    return format_combinations

def write_custom_format_combinations(d, format_combinations):
    for op_name in format_combinations:
        d[op_name]["support_format_combination"] = format_combinations[op_name]["support_format_combination"]

def generate_yaml(args):
    graph = gs.import_onnx(onnx.load(args.onnx))
    d = {}
    op_parsed = []
    if args.custom_operators != "":
        target_nodes = args.custom_operators.split(",")
        for node in graph.nodes:
            if node.op in op_parsed:
                update_node_info(d, node)
            if node.op in target_nodes and node.op not in op_parsed:
                convert_node_info_to_dict(d, node)
                op_parsed.append(node.op)
    else:
        supported_ops = get_supported_op_list(args.trt_version)
        for node in graph.nodes:
            if node.op in op_parsed:
                update_node_info(d, node)
            if node.op not in supported_ops and node.op not in op_parsed:
                convert_node_info_to_dict(d, node)
                op_parsed.append(node.op)
    if args.io_tensors is not None and len(args.io_tensors) > 0:
        io_tensors = parse_io_tensors(args.io_tensors)
        write_custom_shape(d, io_tensors)
    if args.format_combinations is not None and  len(args.format_combinations) > 0:
        format_combinations = parse_format_combinations(args.format_combinations)
        write_custom_format_combinations(d, format_combinations)
    print(d)
    with open(args.yaml, 'w') as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False)

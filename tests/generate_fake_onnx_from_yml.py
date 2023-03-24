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

import os
import glob
import sys
sys.path.append("../src/")
split_lv1 = '------------------------------> '
split_lv2 = '---------------> '
from log import *
# Env setup
logging.info(split_lv1 + "Install dependencies...")
os.system("pip3 install pyyaml onnx numpy")
os.system("pip3 install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com")

from yaml_parser import read_yaml

import argparse
import shutil
import subprocess
import onnx_graphsurgeon as gs
import numpy as np
import onnx

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate fake onnx from TPG config yaml',
        usage='%(prog)s <yaml_file> <onnx_save_dir>')

    parser.add_argument('yaml_file', type=str,
                        help='Path to the YAML file')
    parser.add_argument('onnx_save_dir', type=str,
                        help='Directory to the save ONNX files')

    args = parser.parse_args()
    return args

def get_tensor_dtype(format):
    if format == "float32" or format == "float16" or format == "int8":
        return np.float32
    elif format == "int32":
        return np.int32
    else:
        raise ValueError(f"Unsupported format {format}")

def get_np_shape(dims):
    new_dims = []
    for dim in dims:
        if dim == -1:
            dim = 1
        new_dims.append(dim)
    return tuple(new_dims)

def create_onnx(save_dir, plugin_name, inputs, outputs, attrs, support_format_combination):
    # Inputs
    node_inputs = []
    for i in range(len(inputs)):
        t = inputs[i]
        dtype = get_tensor_dtype(support_format_combination[0][i])
        shape = get_np_shape(t.dims)
        v = gs.Variable(name=t.name, dtype=dtype, shape=shape)
        node_inputs.append(v)

    # Outputs
    node_outputs = []
    for i in range(len(outputs)):
        t = outputs[i]
        dtype = get_tensor_dtype(support_format_combination[0][i+len(inputs)])
        shape = get_np_shape(t.dims)
        v = gs.Variable(name=t.name, dtype=dtype, shape=shape)
        node_outputs.append(v)

    # Attributes
    node_attrs = {}
    for attr in attrs:
        assert attr.default_value is not None, f"Can not find default value for attr {attr.name}"
        node_attrs[attr.name] = attr.default_value

    nodes = [
        gs.Node(op=plugin_name, inputs=node_inputs, outputs=node_outputs, attrs=node_attrs),
    ]

    graph = gs.Graph(nodes=nodes, inputs=node_inputs, outputs=node_outputs)

    save_path = os.path.join(save_dir, plugin_name+".onnx")
    logging.info(split_lv2 + "Create ONNX " + save_path)
    onnx.save(gs.export_onnx(graph), save_path)

def generate_onnx(configs, save_dir):
    for config in configs:
        create_onnx(save_dir, config.plugin_name, config.inputs, config.outputs, config.attrs,
                    config.support_format_combination)

def mkdir_empty(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

def rmdir(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

def compile_and_test(test_dir, configs):
    for config in configs:
        onnx_name = config.plugin_name
        plugin_type = config.plugin_type
        onnx_path = os.path.join(test_dir, onnx_name+".onnx")
        engine_path = onnx_path+".plan"
        plugin_dir = os.path.join(test_dir, onnx_name+plugin_type)
        lib_path = plugin_dir+"/lib"+onnx_name+plugin_type+".so"
        logging.info(split_lv2 + onnx_path)
        os.system(f"cd {plugin_dir} && make")
        os.system(f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --fp16 --int8 --warmUp=0 --duration=0 --iterations=2 --saveEngine={engine_path} --plugins={lib_path} 2>&1 | grep TensorRT.trtexec")
        os.system(f"/usr/src/tensorrt/bin/trtexec --warmUp=0 --duration=0 --iterations=2 --loadEngine={engine_path} --plugins={lib_path} 2>&1 | grep TensorRT.trtexec")

def main():
    # Read config
    yaml_file = "./test.yml"
    configs = read_yaml(yaml_file)

    # Create dir for testing
    test_dir = "/tmp/tpg_test_tmp"
    onnx_save_dir = test_dir
    mkdir_empty(onnx_save_dir)

    # Generate ONNX model
    logging.info(split_lv1 + "Generate ONNX model...")
    generate_onnx(configs, onnx_save_dir)

    # Generate Plugin Code
    logging.info(split_lv1 + "Generate plugin...")
    os.system(f'python3 ../tpg.py generate --yaml {yaml_file} --output {test_dir} > /dev/null 2>&1')

    # Compile and test with trtexec
    logging.info(split_lv1 + "Compile and test...")
    compile_and_test(test_dir, configs)

    # Finish
    logging.info(split_lv1 + "Test finished")

if __name__ == '__main__':
    main()

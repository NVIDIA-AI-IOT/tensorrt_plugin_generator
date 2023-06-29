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


import argparse

import sys
import os
from tpg.extract_unsupported_operators import generate_yaml
from tpg.generate import *

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def main():
    parser = MyParser()
    subparsers = parser.add_subparsers(dest='func')
    parser_extract = subparsers.add_parser("extract", help="generate yml from onnx")
    parser_extract.add_argument("--onnx", type=str, default='',
                        help="onnx for extraction")
    parser_extract.add_argument("--custom_operators", type=str, default='',
                        help="user-specify custom operator, split by \",\"")
    parser_extract.add_argument("--io_tensors", nargs='+',
                        help="specify io tensor explicitly so that you don't edit the yaml file.\
                        usage: --io_tensors op_name_1:1x1x1x1,2x2x2,-1x-1x-1:7x7x7,-1x-1 op_name_2:inputs_shapes:output_shapes")
    parser_extract.add_argument("--format_combinations", nargs='+',
                        help="specify support format combination explicitly so that you don't edit the yaml file.\
                        usage: --format_combinations op_name_1:float32+int8+int8,float16+int8+int8 op_name_2:support_format_combination_str")
    parser_extract.add_argument("--yaml", type=str, default='',
                        help="output yaml")
    parser_extract.add_argument("--trt_version", type=str, default='8.4',
                        help="filter unsupported op based on the version")
    parser_extract.set_defaults(func=generate_yaml)

    parser_generate = subparsers.add_parser("generate", help="generate plugin codes from yml")
    parser_generate.add_argument("--yaml", type=str, default='',
                        help="output yaml config")
    parser_generate.add_argument("--no_makefile", action='store_false',
                        help="don't generate make file")
    parser_generate.add_argument("--trt_lib_dir", type=str, default=DEFAULT_TRT_LIB_DIR,
                        help="directory contains the libnvinfer_plugin.so")
    parser_generate.add_argument("--trt_include_dir", type=str, default=DEFAULT_TRT_INCLUDE_DIR,
                        help="directory contains the NvInferPlugin.h")
    parser_generate.add_argument("--cuda_include_dir", type=str, default=DEFAULT_CUDA_INCLUDE_DIR,
                        help="CUDA include path")
    parser_generate.add_argument("-o", "--output", type=str, default='./',
                        help="output directory which the generated plugin directory located")
    parser_generate.add_argument("--print", action='store_true',
                        help="to print the generate h, cpp and makefile to console")
    parser_generate.set_defaults(func=generate)
    args = parser.parse_args()

    if args.func is None:
        parser.print_help()
        exit()

    args.func(args)

if __name__ == "__main__":
    main()

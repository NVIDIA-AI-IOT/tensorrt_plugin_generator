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

import yaml
from tpg.config import *
from tpg.log import *

def read_yaml(path):
    configs = []
    with open(path, 'r') as stream:
        logging.info('load yaml: ' + path)
        data = yaml.safe_load(stream)
        for node_name in data:
            logging.info('parse operator: ' + node_name)
            node = data[node_name]
            config = Config()
            config.SetPluginName(node_name)
            config.SetPluginType(node['plugin_type'])
            inputs = node['inputs']
            for input in inputs:
                config.AddInput(input, inputs[input]['shape'])
            outputs = node['outputs']
            for output in outputs:
                config.AddOutput(output, outputs[output]['shape'])
            support_format_combinations = node['support_format_combination']
            for combination in support_format_combinations:
                config.AddSupportFormatCombination(combination)
            if 'attributes' in node:
                attrs = node['attributes']
                for attr in attrs:
                    default_value = None
                    if 'default_value' in attrs[attr]:
                        default_value = attrs[attr]['default_value']
                    config.AddAttr(attr, attrs[attr]['datatype'], default_value)
            configs.append(config)
    return configs

if __name__ == "__main__":
    config = read_yaml("../sample.yml")
    print(config)
    print('passed')



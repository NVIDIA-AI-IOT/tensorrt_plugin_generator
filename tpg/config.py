
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

class Tensor:
    def __init__(self):
        self.name = ""
        self.dims = []

class Attribute:
    def __init__(self):
        self.name = ""
        self.type = ""
        self.default_value = None

class Format:
    def __init__(self, datatype, tensor_format):
        self.datatype = datatype
        self.tensor_format = tensor_format.upper()
    def validate(self):
        assert self.datatype in ['float32', 'float16', 'int8', 'int32'], 'Unsupported datatype'
        assert self.tensor_format in ['LINEAR', 'CHW32', 'CHW2', 'HWC8', 'HWC16', 'DHWC8', 'CHW4'], 'Unsupported data format'
        # refer to https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#ipluginv2
        if self.datatype == 'float32':
            assert self.tensor_format in ['LINEAR', 'CHW32'], 'Unsupported data format for float32'
        if self.datatype == 'float16':
            assert self.tensor_format in ['LINEAR', 'CHW2', 'HWC8', 'HWC16', 'DHWC8', 'CHW4'], 'Unsupported data format for float16'
        if self.datatype == 'int8':
            assert self.tensor_format in ['LINEAR', 'CHW32', 'CHW4'], 'Unsupported data format for int8'
        if self.datatype == 'int32':
            assert self.tensor_format in ['LINEAR'], 'Unsupported data format for int32'

class Config:
    '''Config for a custom plugin
    Attributes
        plugin_name: identifier of the operator, as known as the type of an onnx operator.
        plugin_type: TensorRT's plugin type, support type:
            IPluginV2DynamicExt, IPluginV2IOExt
        inputs: inputs information, including name and dimension
        outputs: outputs information, including name and dimension
        attrs: can be empty if the operator doesn’t have any attributes, for nonempty attributes, 
            we need to know their name, datatype the attribute datatype can be 
            scalar(int, float, or string) or vector(int[], float[])

    '''
    def __init__(self):
        self.plugin_name = ''
        self.plugin_type = ''
        self.support_format_combination = []
        self.inputs = []
        self.outputs = []
        self.attrs = []

    def SetPluginName(self, plugin_name):
        self.plugin_name = plugin_name

    def SetPluginType(self, plugin_type):
        self.plugin_type = plugin_type

    def AddInput(self, input_name, input_dims):
        t = Tensor()
        t.name = input_name
        assert input_dims != "need_user_to_specify", "failed to parse "+t.name+", please specify the input dims"
        dims = str(input_dims).split('x')
        for dim in dims:
            dim = int(float(dim))
            t.dims.append(dim)
        self.inputs.append(t)

    def AddOutput(self, output_name, output_dims):
        t = Tensor()
        t.name = output_name
        assert output_dims != "need_user_to_specify", "failed to parse "+t.name+", please specify the output dims"
        dims = str(output_dims).split('x')
        for dim in dims:
            dim = int(dim)
            t.dims.append(dim)
        self.outputs.append(t)

    def AddSupportFormatCombination(self, format_combination):
        l = []
        assert format_combination != "need_user_to_specify", "please specify the format combination"
        items = format_combination.split('+')
        formats = []
        for item in items:
            datatype = item.split(':')[0]
            tensor_format = item.split(':')[1] if len(item.split(':')) > 1 else "LINEAR"
            format = Format(datatype, tensor_format)
            format.validate()
            formats.append(format)
        assert len(formats) == (len(self.inputs) + len(self.outputs)), 'Error: formats mismatch'
        self.support_format_combination.append(formats)

    def AddAttr(self, attr_name, attr_type, default_value = None):
        a = Attribute()
        a.name = attr_name
        a.type = attr_type
        a.default_value = default_value
        datatype = a.type.split('[')[0]
        assert datatype in ['float64', 'float32', 'float16', 'int32', 'int16', 'int8', 'char']
        self.attrs.append(a)


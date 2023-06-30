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

from tpg.config import *
from tpg.log import *
from tpg.yaml_parser import read_yaml

import os
import sys
import shutil
from string import Template
import argparse
import re

DEFAULT_TRT_LIB_DIR = '/usr/lib/x86_64-linux-gnu/'
DEFAULT_TRT_INCLUDE_DIR = '/usr/include/x86_64-linux-gnu/'
DEFAULT_CUDA_INCLUDE_DIR = '/usr/local/cuda/include/'

PARAMS_NAME = "params"
PRIVATE_PARAMS_NAME = "mParams"

template_dir = os.path.dirname(os.path.realpath(__file__)) + "/plugin_templates/"
def load_template(path):
    with open(path, 'r') as f:
        template = f.read()
        return template

# replace move all special character, capitalize each word and then remove space to make a new name
# e.g. "my_name&is_bob" -> "MyNameIsBob"
def get_capital_name(name):
    return re.sub('[^a-zA-Z0-9 \_]', ' ', name).title().replace(" ", "")

def get_valid_cxx_variable_name(name):
    return re.sub('[^a-zA-Z0-9]', ' ', name).replace(" ", "_")

# make a string all uppercase
def make_uppercase(str):
    return str.upper()

def get_size_prefix():
    return 'num_'

def is_std_vector(variable):
    return ('std::vector' in variable)

def get_cxx_type(t):
    if t.startswith("int32"):
        return 'int32_t'
    elif t.startswith('int16'):
        return 'int16_t'
    elif t.startswith('int8'):
        return 'int8_t'
    elif t.startswith('float32'):
        return 'float'
    elif t.startswith('float16'):
        return 'half'
    elif t.startswith('char'):
        return 'char'
    else:
        assert False, 'unsupported datatype: ' + t
        return ''

def parse_attr_to_cxx_variable(n, t, with_counts=True):
    n = get_valid_cxx_variable_name(n)
    variables = []
    # add counts for multiple value
    if with_counts:
        if '[' in t:
            variables.append('size_t ' + get_size_prefix() + n)
    param = ''

    param += get_cxx_type(t)
    if '[' in t:
        param = 'std::vector<' + param + '>'
    param = param + ' ' + n
    variables.append(param)
    return variables

def concat_variables(cxx_variables, delimiter, truncate_end=False):
    str_params = ''
    if len(cxx_variables) == 0:
        return str_params
    for param in cxx_variables:
        str_params += str(param)
        str_params += delimiter
    if truncate_end:
        str_params = str_params[:len(str_params) - len(delimiter)]
    return str_params

def generate_attribute_struct(cxx_variables, plugin_name):
    ret_str = ''
    if len(cxx_variables) == 0:
        return ret_str
    ret_str += ("struct {}Attrs\n".format(plugin_name))
    ret_str += ("{\n")
    ret_str += ("    ")
    ret_str += concat_variables(cxx_variables, ";\n    ", truncate_end=True)
    ret_str += (";\n};")
    return ret_str

def attrs_to_cxx_variables(attrs, with_counts=True):
    cxx_variables = []
    for attr in attrs:
        n = attr.name
        t = attr.type
        for param in  parse_attr_to_cxx_variable(n, t, with_counts):
            cxx_variables.append(param)
    return cxx_variables

def convert_python_value_to_cxx_value(v):
    if isinstance(v, str):
        # a -> 'a'
        return ('\'' + str(v) + '\'')
    else:
        return str(v)

def attrs_to_cxx_variables_with_initialization(attrs, with_counts=True):
    cxx_variables = []
    for attr in attrs:
        n = attr.name
        t = attr.type
        v = attr.default_value
        v_str = ''
        num = 0
        if v is not None:
            v_str += ("{")
            if isinstance(v, list):
                num = len(v)
                for element in v:
                    v_str += convert_python_value_to_cxx_value(element)
                    v_str += ", "
                # remove last ", "
                v_str = v_str[:-2]
            else:
                v_str += convert_python_value_to_cxx_value(v)
            v_str += ("}")
        else:
            v_str += ("{}")
        params = parse_attr_to_cxx_variable(n, t, with_counts)
        if len(params) == 2:
            params[0] += ('{'+str(num)+'}')
            cxx_variables.append(params[0])
            params[1] += v_str
            cxx_variables.append(params[1])
        elif len(params) == 1:
            params[0] += v_str
            cxx_variables.append(params[0])
        else:
            assert False, 'parse_attr_to_cxx_variable failed'
    return cxx_variables

def get_constructor_params(plugin_attributes_struct, name):
    if plugin_attributes_struct == '':
        return ''
    else:
        return plugin_attributes_struct.split()[1] + " " + name

def get_plugin_field_type_str(str):
    if str.startswith("float16"):
        return 'kFLOAT16'
    elif str.startswith("float32"):
        return 'kFLOAT32'
    elif str.startswith("float64"):
        return 'kFLOAT64'
    elif str.startswith("int8"):
        return 'kINT8'
    elif str.startswith("int16"):
        return 'kINT16'
    elif str.startswith("int32"):
        return 'kINT32'
    elif str.startswith("char"):
        return 'kCHAR'
    elif str.startswith("dims"):
        return 'kDIMS'
    else:
        return 'kUNKNOWN'

def get_trt_datatype(str):
    if str.startswith("float16"):
        return 'kHALF'
    elif str.startswith("float32"):
        return 'kFLOAT'
    elif str.startswith("int8"):
        return 'kINT8'
    elif str.startswith("int32"):
        return 'kINT32'
    elif str.startswith("bool"):
        return 'kBOOL'
    else:
        assert False, 'error: wrong datatype: '+str

def codegen_plugin_attributes_emplace_back(config):
    template = "mPluginAttributes.emplace_back(PluginField(\"$attr_name\", nullptr, PluginFieldType::$field_Type, 1));"
    ret_str = ''
    for attr in config.attrs:
        line = template.replace("$attr_name", attr.name)
        line = line.replace("$field_Type", get_plugin_field_type_str(attr.type))
        ret_str += (line + '\n    ')
    return ret_str

def codegen_assign_params(plugin_attributes_struct):
    if plugin_attributes_struct == '':
        return ''
    else:
        return '    {} = {};'.format(PRIVATE_PARAMS_NAME, PARAMS_NAME)

def codegen_read_deserialized_buffer(cxx_variables):
    ret_str = ''
    for i in range(len(cxx_variables)):
        variable = cxx_variables[i]
        name = variable.split()[1]
        if is_std_vector(variable):
            assert cxx_variables[i-1].split()[0] == 'size_t', 'could not find the count for ' + variable
            size = cxx_variables[i-1].split()[1]
            ret_str += ('    {}.{}.resize({}.{});\n'.format(PRIVATE_PARAMS_NAME, name, PRIVATE_PARAMS_NAME, size))
            ret_str += ('    for(size_t i=0; i<{}.{};i++)\n'.format(PRIVATE_PARAMS_NAME, size))
            ret_str += ('    {\n')
            ret_str += ('        readFromBuffer(d, {}.{}[i]);\n'.format(PRIVATE_PARAMS_NAME, name))
            ret_str += ('    }\n')
        else:
            ret_str += ('    readFromBuffer(d, {}.{});\n'.format(PRIVATE_PARAMS_NAME, name))
    return ret_str

def get_raw_type(t):
    return t.split('<')[1].split('>')[0]

def codegen_get_serialization_size(cxx_variables):
    ret_str = ''
    for i in range(len(cxx_variables)):
        variable = cxx_variables[i]
        t = variable.split()[0]
        ret_str += ('    // {}\n'.format(variable))
        if is_std_vector(variable):
            size = cxx_variables[i-1].split()[1]
            # std::vector<T> -> T
            t = get_raw_type(t)
            ret_str += ('    size += (sizeof({}) * {}.{});\n'.format(t, PRIVATE_PARAMS_NAME,size))
        else:
            ret_str += ('    size += sizeof(' + t + ');\n' )
    return ret_str

def codegen_serialize_to_buffer(cxx_variables):
    ret_str = ''
    for i in range(len(cxx_variables)):
        variable = cxx_variables[i]
        name = variable.split()[1]
        if is_std_vector(variable):
            assert cxx_variables[i-1].split()[0] == 'size_t', 'could not find the count for ' + variable
            size = cxx_variables[i-1].split()[1]
            ret_str += ('    assert({}.{}.size() == {}.{});\n'.format(PRIVATE_PARAMS_NAME, name, PRIVATE_PARAMS_NAME, size))
            ret_str += ('    for(size_t i=0; i<{}.{}; i++)\n'.format(PRIVATE_PARAMS_NAME, size))
            ret_str += ('    {\n')
            ret_str += ('        writeToBuffer(d, {}.{}[i]);\n'.format(PRIVATE_PARAMS_NAME, name))
            ret_str += ('    }\n')
        else:
            ret_str += ('    writeToBuffer(d, {}.{});\n'.format(PRIVATE_PARAMS_NAME, name))
    return ret_str

def get_variable_name(cxx_variables):
    ret_list = []
    for variable in cxx_variables:
        ret_list.append(variable.split()[1])
    return ret_list

# 'int a, float b' -> 'a, b'
def codegen_copy_params(cxx_variables):
    variable_name = get_variable_name(cxx_variables)
    names = concat_variables(variable_name, ", ", True)
    return names

def codegen_create_plugin(cxx_variables, onnx_variables, constructor_params):
    ret_str = ''
    if len(onnx_variables) == 0:
        return ret_str
    ret_str += ('    const PluginField* fields = fc->fields;\n')
    ret_str += ('    int nbFields = fc->nbFields;\n')
    # add default initialization here to avoid compile warning
    ret_str += ('    {};\n'.format(constructor_params))
    # we already have 4 spaces after final variable declaration
    ret_str += ('    for (int i = 0; i < nbFields; ++i)\n')
    ret_str += ('    {\n')
    for i in range(len(onnx_variables)):
        variable = onnx_variables[i]
        cxx_type = variable.split()[0]
        name = variable.split()[1]

        if is_std_vector(variable):
            ret_str += ('        if (!strcmp(fields[i].name, \"' + name + '\"))\n')
            ret_str += ('        {\n')
            ret_str += ('            {}.{} = fields[i].length;\n'.format(PARAMS_NAME, get_size_prefix()+name))
            ret_str += ('            {}.{}.resize({}.{});\n'.format(PARAMS_NAME, name, PARAMS_NAME, get_size_prefix()+name))
            ret_str += ('            for(int j = 0; j < fields[i].length; j++)\n')
            ret_str += ('            {\n')
            ret_str += ('                {}.{}[j] = (reinterpret_cast<const {}*>(fields[i].data))[j];\n'.format(PARAMS_NAME, name, get_raw_type(cxx_type)))
            ret_str += ('            }\n')
            ret_str += ('        }\n')
        else:
            ret_str += ('        if (!strcmp(fields[i].name, \"' + name + '\"))\n')
            ret_str += ('        {\n')
            ret_str += ('            {}.{} = *(reinterpret_cast<const {}*>(fields[i].data));\n'.format(PARAMS_NAME, name, cxx_type))
            ret_str += ('        }\n')
    ret_str += ('    }\n')

    return ret_str

def get_input_combination_str(param_name, format_combination, num_input, tensor_format):
    ret_str = ''
    for i in range(num_input):
        datatype = get_trt_datatype(format_combination[i])
        ret_str += f'{param_name}[{i}].type == DataType::{datatype} && {param_name}[{i}].format == TensorFormat::{tensor_format} &&\n            '
    return ret_str

def get_input_combination_str_v2(param_name, format_combination, num_input):
    ret_str = ''
    for i in range(num_input):
        ret_str += '{}[{}] == DataType::{} &&\n           '.format(param_name, i, get_trt_datatype(format_combination[i]))
    return ret_str

def get_dims_str(dims):
    ret_str = '{'
    ret_str += concat_variables(dims, ", ", True)
    ret_str += '}'
    return ret_str


MAX_DIMS = 8
def codegen_get_output_dimensions(config):
    ret_str = ''
    outputs = config.outputs
    inputs = config.inputs
    for t in inputs:
        assert t.dims[0] == -1, 'first dimension of {} must be specify as -1 if use IPluginIOExt'.format(t.name)
    for t in outputs:
        assert t.dims[0] == -1, 'first dimension of {} must be specify as -1 if use IPluginIOExt'.format(t.name)
    ret_str += ('    assert(index < {});\n'.format(len(outputs)))
    ret_str += ('    Dims output;\n')
    for i in range(len(outputs)):
        assert len(outputs[i].dims) <= MAX_DIMS, 'output {}: only support dimension less than {}'.format(outputs[i].name, MAX_DIMS)
        ret_str += ('    if(index == {})\n'.format(i))
        ret_str += ('    {\n')
        ret_str += ('        output.nbDims = {};\n'.format(len(outputs[i].dims)-1))
        for j in range(len(outputs[i].dims)-1):
            ret_str += ('        output.d[{}] = {};\n'.format(j, outputs[i].dims[j+1]))
        ret_str += ('    }\n')
    ret_str += ('    return output;')
    return ret_str

def is_dynamic_tensor(tensor):
    for dim in tensor.dims:
        if dim < 0:
            return True
    return False

def codegen_get_output_dimensions_dynamic(config):
    ret_str = ''
    outputs = config.outputs
    ret_str += ('    assert(outputIndex < {});\n'.format(len(outputs)))
    ret_str += ('    assert(inputs != nullptr);\n')
    ret_str += ('    assert(nbInputs == {});\n'.format(len(config.inputs)))
    ret_str += ('    DimsExprs output;\n')
    for i in range(len(outputs)):
        assert len(outputs[i].dims) <= MAX_DIMS, 'output {}: only support dimension less than {}'.format(outputs[i].name, MAX_DIMS)
        ret_str += ('    if(outputIndex == {})\n'.format(i))
        ret_str += ('    {\n')
        if is_dynamic_tensor(outputs[i]):
            ret_str += ('        // Need implement by user\n')
            ret_str += ('        assert(false && \"Please implement this function first\");\n')
        else:
            ret_str += ('        output.nbDims = {};\n'.format(len(outputs[i].dims)))
            for j in range(len(outputs[i].dims)):
                ret_str += ('        output.d[{}] = exprBuilder.constant({});\n'.format(j, outputs[i].dims[j]))
        ret_str += ('    }\n')
    ret_str += ('    return output;\n')
    return ret_str


def codegen_deduce_output_datatype(config):
    ret_str = ''
    num_input = len(config.inputs)
    num_output = len(config.outputs)
    support_format_combination = config.support_format_combination
    ret_str += ('    assert(index < {});\n'.format(num_output))
    ret_str += ('    assert(inputTypes != nullptr);\n')
    ret_str += ('    assert(nbInputs == {});\n'.format(len(config.inputs)))
    for j in range(num_output):
        ret_str += ('    if(index == {})\n'.format(j))
        ret_str += ('    {\n')
        for i in range(len(support_format_combination)):
            format_combination = support_format_combination[i]
            # remove last 11 ' &&\n           '
            str = get_input_combination_str_v2('inputTypes', format_combination, num_input)[:-15]
            ret_str += ('        if({})\n'.format(str))
            ret_str += ('        {\n')
            t = get_trt_datatype(support_format_combination[i][j+num_input])
            ret_str += ('            return DataType::{};\n'.format(t))
            ret_str += ('        }\n')
        ret_str += ('    }\n')
    ret_str += ('    // Default: return the datatype of the first input or datatype kFLOAT\n')
    ret_str += ('    return nbInputs==0 ? DataType::kFLOAT : inputTypes[0];\n')
    return ret_str

def codegen_get_support_format_combination(config):
    ret_str = ''
    num_input = len(config.inputs)
    num_output = len(config.outputs)
    num_io = num_input + num_output
    support_format_combination = config.support_format_combination
    for i in range(num_io):
        tensor_format = 'kLINEAR'
        if i < num_input:
            ret_str += ('    if(pos == {})\n'.format(i))
            ret_str += ('    {\n')
            ret_str += ('        is_supported =\n')
            for j in range(len(support_format_combination) - 1):
                option = get_trt_datatype(support_format_combination[j][i])
                ret_str += ('            (inOut[pos].type == DataType::{} && inOut[pos].format == TensorFormat::{}) ||\n'.format(option, tensor_format))
            option = get_trt_datatype(support_format_combination[len(support_format_combination) - 1][i])
            ret_str += ('            (inOut[pos].type == DataType::{} && inOut[pos].format == TensorFormat::{});\n'.format(option, tensor_format))
            ret_str += ('    }\n')
        else:
            ret_str += ('    if(pos == {})\n'.format(i))
            ret_str += ('    {\n')
            ret_str += ('        is_supported =\n')
            for j in range(len(support_format_combination) - 1):
                inputs_option = get_input_combination_str('inOut', support_format_combination[j], num_input, tensor_format)
                option = get_trt_datatype(support_format_combination[j][i])
                ret_str += ('            ({}inOut[pos].type == DataType::{} && inOut[pos].format == TensorFormat::{}) ||\n'.format(inputs_option, option, tensor_format))
            inputs_option = get_input_combination_str('inOut', support_format_combination[len(support_format_combination) - 1], num_input, tensor_format)
            option = get_trt_datatype(support_format_combination[len(support_format_combination) - 1][i])
            ret_str += ('            ({}inOut[pos].type == DataType::{} && inOut[pos].format == TensorFormat::{});\n'.format(inputs_option, option, tensor_format))
            ret_str += ('    }\n')
    return ret_str

def generate_plugin_common_api(config, h, cpp):
    capitalized_name = get_capital_name(config.plugin_name)
    # parse attribute
    cxx_variables = attrs_to_cxx_variables(config.attrs)
    onnx_variables = attrs_to_cxx_variables(config.attrs, with_counts=False)

    # replace plugin name
    h = h.replace("$plugin_name", capitalized_name)
    cpp = cpp.replace("$plugin_name", capitalized_name)
    cpp = cpp.replace("$parse_name", config.plugin_name)

    # replace define header
    h = h.replace("${plugin_name_uppercase}", make_uppercase(capitalized_name))

    # pack plugin attributes to a struct
    initialized_cxx_variables = attrs_to_cxx_variables_with_initialization(config.attrs)
    plugin_attributes_struct = generate_attribute_struct(initialized_cxx_variables, capitalized_name)
    h = h.replace("$plugin_attributes_struct", plugin_attributes_struct)

    # replace attribute in constructor
    constructor_params = get_constructor_params(plugin_attributes_struct, PARAMS_NAME)
    h = h.replace("$constructor_params", constructor_params)
    cpp = cpp.replace("$constructor_params", constructor_params)
    # add cxx_variables to IPluginV2 private attributes
    plugin_private_attributes = get_constructor_params(plugin_attributes_struct, PRIVATE_PARAMS_NAME) + ';'
    h = h.replace("$plugin_private_attributes", plugin_private_attributes)
    # codegen Constructor($constructor_params)
    assign_params = codegen_assign_params(plugin_attributes_struct)
    cpp = cpp.replace("$assign_params", assign_params)
    # codegen Constructor(const void* data, size_t length)
    read_deserialized_buffer = codegen_read_deserialized_buffer(cxx_variables)
    cpp = cpp.replace("$read_deserialized_buffer", read_deserialized_buffer)
    # codegen getSerializationSize()
    get_serialization_size = codegen_get_serialization_size(cxx_variables)
    cpp = cpp.replace("$get_serialization_size", get_serialization_size)
    # codegen serialize(void* buffer)
    serialize_to_buffer = codegen_serialize_to_buffer(cxx_variables)
    cpp = cpp.replace("$serialize_to_buffer", serialize_to_buffer)
    # codegen clone() and createPlugin(const char* name, const PluginFieldCollection* fc) return
    if(len(config.attrs) > 0):
        cpp = cpp.replace("$copy_params", PRIVATE_PARAMS_NAME)
    else:
        cpp = cpp.replace("$copy_params", "")
    # codegen getNbOutputs()
    number_of_outputs = str(len(config.outputs))
    cpp = cpp.replace("$number_of_outputs", number_of_outputs)
    # codegen getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs)
    deduce_output_datatype = codegen_deduce_output_datatype(config)
    cpp = cpp.replace("$deduce_output_datatype", deduce_output_datatype)

    # codegen IPluginCreator()
    plugin_attributes_emplace_back = codegen_plugin_attributes_emplace_back(config)
    cpp = cpp.replace("$plugin_attributes_emplace_back", plugin_attributes_emplace_back)
    # codegen IPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    create_plugin = codegen_create_plugin(cxx_variables, onnx_variables, constructor_params)
    cpp = cpp.replace("$create_plugin", create_plugin)
    if(len(config.attrs) > 0):
        cpp = cpp.replace("$new_plugin_params", PARAMS_NAME)
    else:
        cpp = cpp.replace("$new_plugin_params", "")

    return h,cpp

def generate_ioext(config):
    logging.info("generating plugin for node: " + config.plugin_name)
    h = load_template(os.path.join(template_dir, "IPluginV2IOExt.h"))
    cpp = load_template(os.path.join(template_dir, "IPluginV2IOExt.cpp"))

    h, cpp = generate_plugin_common_api(config, h, cpp)
    # codegen getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    get_output_dimensions = codegen_get_output_dimensions(config)
    cpp = cpp.replace("$get_output_dimensions", get_output_dimensions)
    # codegen supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs)
    get_support_format_combination = codegen_get_support_format_combination(config)
    cpp = cpp.replace("$get_support_format_combination", get_support_format_combination)

    return h, cpp

def generate_dynamic_ext(config):
    logging.info("generating plugin for node: " + config.plugin_name)
    h = load_template(os.path.join(template_dir, "IPluginV2DynamicExt.h"))
    cpp = load_template(os.path.join(template_dir, "IPluginV2DynamicExt.cpp"))
    # generate common api
    h, cpp = generate_plugin_common_api(config, h, cpp)
    #### TODO OPTIMIZE HERE
    # codegen getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder)
    get_output_dimensions = codegen_get_output_dimensions_dynamic(config)
    cpp = cpp.replace("$get_output_dimensions", get_output_dimensions)
    # codegen supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs)
    get_support_format_combination = codegen_get_support_format_combination(config)
    cpp = cpp.replace("$get_support_format_combination", get_support_format_combination)

    return h, cpp

def generate_makefile(output_file_name, args):
    makefile = load_template(os.path.join(template_dir, "makefile"))

    if args.trt_lib_dir == DEFAULT_TRT_LIB_DIR:
        logging.warning("Didn't specify --trt_lib_dir for makefile. so using default value "+DEFAULT_TRT_LIB_DIR)
    makefile = makefile.replace('$trt_lib_dir', args.trt_lib_dir)
    if args.trt_include_dir == DEFAULT_TRT_INCLUDE_DIR:
        logging.warning("Didn't specify --trt_include_dir for makefile. so using default value "+DEFAULT_TRT_INCLUDE_DIR)
    makefile = makefile.replace('$trt_include_dir', args.trt_include_dir)
    if args.cuda_include_dir == DEFAULT_CUDA_INCLUDE_DIR:
        logging.warning("Didn't specify --cuda_include_dir for makefile. so using default value "+DEFAULT_CUDA_INCLUDE_DIR)
    makefile = makefile.replace('$cuda_include_dir', args.cuda_include_dir)
    makefile = makefile.replace('$target_name', 'lib{}.so'.format(output_file_name))

    return makefile

def codegen(config, args):
    output_dir = args.output
    with_makefile = args.no_makefile
    # load template
    h = None
    cpp = None

    if config.plugin_type == "IPluginV2IOExt":
        h, cpp = generate_ioext(config)
    elif config.plugin_type == "IPluginV2DynamicExt":
        h, cpp = generate_dynamic_ext(config)
    else:
        assert False, 'unsupported plugin type'

    # write back to file
    output_file_name = get_capital_name(config.plugin_name) + config.plugin_type
    plugin_dir = os.path.join(output_dir, output_file_name)
    if os.path.exists(plugin_dir):
        logging.warning("detect plugin directory already exist, remove it and re-generate")
        shutil.rmtree(plugin_dir)
    os.makedirs(plugin_dir)
    h_path = os.path.join(plugin_dir, output_file_name + '.h')
    cpp_path = os.path.join(plugin_dir, output_file_name + '.cpp')
    with open(h_path, 'w') as f_h, open(cpp_path, 'w') as f_cpp:
        f_h.write(h)
        f_cpp.write(cpp)

    # codegen makefile
    if with_makefile:
        makefile = generate_makefile(output_file_name, args)
        # save makefile
        makefile_path = os.path.join(plugin_dir, 'Makefile')
        with open(makefile_path, 'w') as f_makefile:
            f_makefile.write(makefile)

    if args.print:
        print(h)
        print(cpp)
        print(makefile)

def generate(args):
    configs = None
    if args.yaml != '':
        configs = read_yaml(args.yaml)
    else:
        assert False, 'must provide the yaml config'
    for config in configs:
        codegen(config, args)



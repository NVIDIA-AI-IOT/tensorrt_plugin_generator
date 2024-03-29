- [YAML Config](#yaml-config)
  - [Plugin Name](#plugin-name)
  - [Plugin Type](#plugin-type)
  - [Inputs and Outputs](#inputs-and-outputs)
  - [Support Format Combination](#support-format-combination)
  - [Attributes](#attributes)

You can refer to sample.yml if you don't know where to starts.

## YAML Config

A compete config yaml contains following items:

```yaml
{plugin_name}:
    plugin_type: {plugin_type}
    inputs:
        {input_name_a}:
            shape: {shape}
        {...other inputs}
    outputs:
        {output_name_a}:
            shape: {shape}
        {...other outputs}
    support_format_combination: {support_format_combination}
    attributes:
        {attribute_a}:
            datatype: {attribute_datatype}
            default_value: {default_value}
        {...other attributes}

{... more plugins}
```

### Plugin Name

`{plugin_name}` is the operator type for a onnx operator, e.g. Conv, Add.

### Plugin Type

`{plugin_type}` can be `IPluginV2DynamicExt` or `IPluginV2IOExt`. please always use `IPluginV2DynamicExt` if you are not write a plugin for safety, because `IPluginV2DynamicExt` use explicit dims and so it's friendly to onnx while `IPluginV2IOExt` use implicit dimension and it has more restriction when using.

### Inputs and Outputs

`inputs` is the inputs of the plugin, it looks like:
```
    inputs:
        inputs_0:
            shape: 1x2x3
        inputs_0:
            shape: -1x-1
```
the `{input_name_a}` here doesn't matter, you can replace it to any name you think it's more meaningful, the shape here must specify like axb(split by character **'x'**) while a and b must be integer between [-1, INT_MAX]. please specify dimension as **-1** if they are dynamic, and the best practice is specify the dimension as much clear as possible, for example `-1x3x224x224` is better than `-1x-1x-1x-1` if you know the h and w if always fixed, because TPG can generate more codes for you in the first case and help you do more shape checks.

`outputs` is similar to `inputs`, the name doesn't matter and please specify the dimension as much clear as possible.

### Support Format Combination

`{support_format_combination}` is very important and **must be specified** by the user. to specify it, first ask yourself what the inputs of the your custom plugin like and what precision you want to implement? for example we have a custom operator that has 2 inputs and 2 outputs, the first input is FP32 and the second input is INT32 by default, and all the outputs are FP32. Suppose I want to implement the plugin that can also support both of the FP16 and INT8 precision provided by TensorRT. then I can specify the `support_format_combination` as below:
```yaml
support_format_combination: ['float32+int32+float32+float32', 'float16+int32+float16+float16', 'int8+int32+int8+int8']
```
or equivalent in yaml's syntax(this is what yaml config generated by `python tpg.py extract` looks like):
```yaml
    support_format_combination:
        - "float32+float32+float32+float32"
        - "float16+float32+float32+float16"
        - "int8+float32+float32+int8"
```
both of the config are valid [yaml syntax for a list](https://yaml.org/). you can use any of the syntax you like. Because we have 2 inputs and 2 outputs, so each entry string has 4 precision specified and split by a character **'+'**. please note that the inputs and outputs specify **in order**, like `the_fist_input+the_second_input+the_fist_output+the_second_output`. And because the FP16 and INT8 only works for FP32, so the precision of the second input is always `int32`. we strongly recommend to always implement the `float32+int32+float32+float32`, aka the FP32 implementation so that you have a baseline to check the accuracy of the plugin. Currently all the format specify here will be treated as **linear format**

We also support specify dat format for each inputs/outputs, it looks like:
```
    support_format_combination:
        - "float32:LINEAR+float32:LINEAR+float32:LINEAR+float32:LINEAR"
        - "float16:HWC8+float32+float32+float16:CHW16"
        - "int8:CHW4+float32+float32+int8:CHW32"
```

The available data format for TRT can be found on [api doc](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html), search `TensorFormat`

TRT plugin has some limitations on the IO format, please refer to [IPluginV2 API Description](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#ipluginv2). If the data format is not support by TRT, tpg will throw an error.

```
Plugin layers can support the following data formats:
LINEAR single-precision (FP32), half-precision (FP16), integer (INT8), and integer (INT32) tensors
CHW32 single-precision (FP32) and integer (INT8) tensors
CHW2, HWC8,HWC16, and DHWC8 half-precision (FP16) tensors
CHW4 half-precision (FP16), and integer (INT8) tensors
```

### Attributes

`attributes` will be auto-generated by `python tpg.py extract` and usually doesn't need to specify by the user manually. If it's create from scratch then you need to specify the attribute name and attribute datatype. unlike `inputs` and `outputs`, attributes name must be specify as you seen in the onnx attribute, and so as the datatype. optionally, you can specify a default value for an attribute, it's useful when your custom operator has some default attributes.

please refer to sample.yml for as an example. it provide many example for different use case
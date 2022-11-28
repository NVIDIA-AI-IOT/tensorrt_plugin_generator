# Demonstrate trt-plugin-generator using TRT's sampleOnnxMnistCoordConvAC sample

- [Demonstrate trt-plugin-generator using TRT's sampleOnnxMnistCoordConvAC sample](#demonstrate-trt-plugin-generator-using-trts-sampleonnxmnistcoordconvac-sample)
  - [How does this sample work?](#how-does-this-sample-work)
  - [Prerequisites](#prerequisites)
  - [Running the sample](#running-the-sample)
    - [Generate the config yaml form modified.onnx](#generate-the-config-yaml-form-modifiedonnx)
    - [modify the generated yaml](#modify-the-generated-yaml)
    - [Generate the plugin code](#generate-the-plugin-code)
    - [compile && running with trtexec](#compile--running-with-trtexec)
    - [Add necessary function implementation to the generated code](#add-necessary-function-implementation-to-the-generated-code)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## How does this sample work?

This sample convert the original mnist_with_coordconv.onnx models to a new one(simply change the plugin name in the onnx so it will not recognized by TRT's official sample). and use trt-plugin to generate plugin config yaml. after do some minor modification, we can automatically generate the plugin code. then with little modification on the getOutputDimension(). We can compile the generated plugin code and load it using trtexec to demonstrate the usage of trt-plugin-generator.

## Prerequisites

The first steps is prepare the onnx model, please generate the model by following https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMnistCoordConvAC#running-the-sample. after the first step, you should get the mnist_with_coordconv.onnx. move it to this directory.

The next step is modify the original operator name from "CoordConvAC" to "MyCoordConvAC" so that it won't recognize by TRT's OSS plugin registry.

```
python modify_onnx.py
```

Then you should see modified.onnx under the same directory.

## Running the sample

### Generate the config yaml form modified.onnx

```
python3 ../../tpg.py extract --onnx modified.onnx --custom_operators MyCoordConvAC --yaml sampleOnnxMnistCoordConvAC.yml
```

### modify the generated yaml

Now we can see the generated sampleOnnxMnistCoordConvAC.yaml as follow:

```yaml
MyCoordConvAC:
  inputs:
    tpg_input_0:
      shape: need_user_to_specify
  outputs:
    tpg_output_0:
      shape: need_user_to_specify
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - need_user_to_specify
```

We need to fill the item that the value is "need_user_to_specify", because we know how our operator works, how's its inputs/outputs like, what precision we want to implement it, we are doing more things here: let the plugin also support FP16 and INT8

```yaml
MyCoordConvAC:
  inputs:
    tpg_input_0:
      shape: -1x-1x-1x-1
  outputs:
    tpg_output_0:
      shape: -1x-1x-1x-1
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - "float32+float32"
  - "float16+float16"
  - "int8+int8"
```

### Generate the plugin code

Now let's generate the plugin code.

```
python ../../tpg.py generate --yaml sampleOnnxMnistCoordConvAC.yml --output ./
```

You can see the generated plugin code is under `./MycoordconvacIPluginV2DynamicExt`

### compile && running with trtexec

The generated code can be compile now, e.g. I can compile it using TRT's official docker image `nvcr.io/nvidia/tensorrt:22.07-py3`

```
docker run -it --gpus all --rm -v /path/to/this/sample:/onnx_packnet nvcr.io/nvidia/tensorrt:22.07-py3 /bin/bash
# in container
cd /sampleOnnxMnistCoordConvAC/MycoordconvacIPluginV2DynamicExt
make
cd ..
```

Let's see if it works!
```
trtexec --onnx=modified.onnx --plugins=MycoordconvacIPluginV2DynamicExt/libMycoordconvacIPluginV2DynamicExt.so
```

Oops, it throw an error

```
trtexec: MycoordconvacIPluginV2DynamicExt.cpp:194: virtual nvinfer1::DimsExprs MycoordconvacIPluginV2DynamicExt::getOutputDimensions(int32_t, const nvinfer1::DimsExprs*, int32_t, nvinfer1::IExprBuilder&): Assertion `false && "Please implement this part first"' failed.
Aborted (core dumped)
```

### Add necessary function implementation to the generated code

To solve the above error, we need to implement a getOutputDimensions() in MycoordconvacIPluginV2DynamicExt.cpp. Because TRT don't know how your custom plugin works, TRT don't know how to get the output dimension of it and assign memory for it(we mark the output dimension as all unknown in previous section). Now let's provide the implementation:

```
DimsExprs MycoordconvacIPluginV2DynamicExt::getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) IS_NOEXCEPT
{
    DEBUG_LOG();
    assert(outputIndex < 1);
    DimsExprs output;
    if(outputIndex == 0)
    {
        // Need implement by user
        output.nbDims = inputs[0].nbDims;
        output.d[0] = inputs[0].d[0];
        output.d[1] = exprBuilder.operation(DimensionOperation::kSUM, *(inputs[0].d[1]), *(exprBuilder.constant(2)));
        output.d[2] = inputs[0].d[2];
        output.d[3] = inputs[0].d[3];
    }
    return output;

}
```

Now let's recompile the code and try it again, also let's test all supported precision:

```
cd /onnx_packnet/MygroupnormalizationIPluginV2DynamicExt
make
cd ..
trtexec --onnx=modified.onnx --plugins=MycoordconvacIPluginV2DynamicExt/libMycoordconvacIPluginV2DynamicExt.so --verbose
trtexec --onnx=modified.onnx --plugins=MycoordconvacIPluginV2DynamicExt/libMycoordconvacIPluginV2DynamicExt.so --verbose --ifp16 --precisionConstraints=prefer --layerPrecisions=*:fp16
trtexec --onnx=modified.onnx --plugins=MycoordconvacIPluginV2DynamicExt/libMycoordconvacIPluginV2DynamicExt.so --verbose --int8 --precisionConstraints=prefer --layerPrecisions=*:int8
```

We will see the model is running successfully, although we do nothing in the enqueue function!

```
...
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=modified.onnx --plugins=MycoordconvacIPluginV2DynamicExt/libMycoordconvacIPluginV2DynamicExt.so --verbose --int8 --precisionConstraints=prefer --layerPrecisions=*:int8
 ----> debug <---- call [MycoordconvacIPluginV2DynamicExt.cpp][destroy][Line 131]
...
```

Now it's your turn to finish the enqueue and other necessary function!

# Additional resources

The following resources provide a deeper understanding about this sample and trt-plugin-generator:

**Original sample**
- [coordConvACPlugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin/coordConvACPlugin)
- [Implementing CoordConv in TensorRT with a custom plugin](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMnistCoordConvAC)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

August 2022: First Version

# Known issues

There are no known issues in this sample

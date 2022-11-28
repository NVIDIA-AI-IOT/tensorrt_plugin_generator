# Demonstrate trt-plugin-generator using TRT's onnx_packnet sample

## How does this sample work?

This sample convert the original packnet onnx models to a new one(simply change the GroupNormalizationPlugin name in the onnx so it will not recognized by TRT's official sample). and use trt-plugin to generate plugin config yaml. after do some minor modification, we can automatically generate the plugin code. then with little modification on the getOutputDimension(). We can compile the generated plugin code and load it using trtexec to demonstrate the usage of trt-plugin-generator.

## Prerequisites

The first steps is prepare the onnx model, please generate the model by following https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_packnet. after the first step, you should get the model.onnx. move it to this directory.

The next step is modify the original operator name from "GroupNormalizationPlugin" to "MyGroupNormalization" so that it won't recognize by TRT's OSS plugin registry. also I delete 2 unused attributes since they won't get parsed even in the original implementation.

```
python modify_onnx.py
```

Then you should see modified.onnx under the same directory.

## Running the sample

### Generate the config yaml form modified.onnx

```
python3 ../../tpg.py extract --onnx modified.onnx --custom_operators MyGroupNormalization --yaml onnx_packnet.yml
```

### modify the generated yaml

Now we can see the generated onnx_packnet.yaml as follow:

```yaml
MyGroupNormalization:
  attributes:
    eps:
      datatype: float32
    num_groups:
      datatype: int32
  inputs:
    tpg_input_0:
      shape: need_user_to_specify
    tpg_input_1:
      shape: need_user_to_specify
    tpg_input_2:
      shape: need_user_to_specify
  outputs:
    tpg_output_0:
      shape: need_user_to_specify
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - need_user_to_specify
```

We need to fill the item that the value is "need_user_to_specify", because we know how our operator works, how's its inputs/outputs like, what precision we want to implement it, next we will modify the yaml and save it as onnx_packnet_modified.yml:

```yaml
MyGroupNormalization:
  attributes:
    eps:
      datatype: float32
    num_groups:
      datatype: int32
  inputs:
    tpg_input_0:
      shape: -1x-1x-1x-1
    tpg_input_1:
      shape: -1x1x1
    tpg_input_2:
      shape: -1x1x1
  outputs:
    tpg_output_0:
      shape: -1x-1x-1x-1
  plugin_type: IPluginV2DynamicExt
  # suppose we will start with fp32 implementation
  support_format_combination: ["float32+float32+float32+float32"]
```

### Generate the plugin code

Now let's generate the plugin code.

```
python ../../tpg.py generate --yaml onnx_packnet_modified.yml --output ./
```

You can see the generated plugin code is under `./MygroupnormalizationIPluginV2DynamicExt`

### compile && running with trtexec

The generated code can be compile now, e.g. I can compile it using TRT's official docker image `nvcr.io/nvidia/tensorrt:22.07-py3`

```
docker run -it --gpus all --rm -v /path/to/this/sample:/onnx_packnet nvcr.io/nvidia/tensorrt:22.07-py3 /bin/bash
# in container
cd /onnx_packnet/MygroupnormalizationIPluginV2DynamicExt
make
cd ..
```

Let's see if it works!
```
trtexec --onnx=modified.onnx --plugins=MygroupnormalizationIPluginV2DynamicExt/libMygroupnormalizationIPluginV2DynamicExt.so
```

Oops, it throw an error

```
 ----> debug <---- call [MygroupnormalizationIPluginV2DynamicExt.cpp][getOutputDimensions][Line 186]
trtexec: MygroupnormalizationIPluginV2DynamicExt.cpp:192: virtual nvinfer1::DimsExprs MygroupnormalizationIPluginV2DynamicExt::getOutputDimensions(int32_t, const nvinfer1::DimsExprs*, int32_t, nvinfer1::IExprBuilder&): Assertion `false && "Please implement this part first"' failed.
Aborted (core dumped)
```

### Add necessary function implementation to the generated code

To solve the above error, we need to implement a getOutputDimensions() in MygroupnormalizationIPluginV2DynamicExt.cpp. Because TRT don't know how your custom plugin works, TRT don't know how to get the output dimension of it and assign memory for it(we mark the output dimension as all unknown in previous section). Now let's provide the implementation, we know the output dimension of this operator is always equal the the first input. so we can modify it like:

```
DimsExprs MygroupnormalizationIPluginV2DynamicExt::getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) IS_NOEXCEPT
{
    DEBUG_LOG();
    assert(outputIndex < 1);
    DimsExprs output;
    if(outputIndex == 0)
    {
        output = inputs[0];
    }
    return output;

}
```

Now let's recompile the code and try it again:

```
cd /onnx_packnet/MygroupnormalizationIPluginV2DynamicExt
make
cd ..
trtexec --onnx=modified.onnx --plugins=MygroupnormalizationIPluginV2DynamicExt/libMygroupnormalizationIPluginV2DynamicExt.so
```

We will see the model is running successfully, although we do nothing in the enqueue function!

```
[08/09/2022-06:54:41] [I] === Performance summary ===
[08/09/2022-06:54:41] [I] Throughput: 23.4642 qps
[08/09/2022-06:54:41] [I] Latency: min = 41.3601 ms, max = 43.2434 ms, mean = 42.2227 ms, median = 42.1411 ms, percentile(99%) = 43.2434 ms
[08/09/2022-06:54:41] [I] Enqueue Time: min = 2.53491 ms, max = 3.69177 ms, mean = 2.96428 ms, median = 2.9234 ms, percentile(99%) = 3.69177 ms
[08/09/2022-06:54:41] [I] H2D Latency: min = 0.129883 ms, max = 0.156128 ms, mean = 0.138414 ms, median = 0.138527 ms, percentile(99%) = 0.156128 ms
[08/09/2022-06:54:41] [I] GPU Compute Time: min = 41.1855 ms, max = 43.0625 ms, mean = 42.0431 ms, median = 41.9604 ms, percentile(99%) = 43.0625 ms
[08/09/2022-06:54:41] [I] D2H Latency: min = 0.0400391 ms, max = 0.0430908 ms, mean = 0.0412402 ms, median = 0.0410156 ms, percentile(99%) = 0.0430908 ms
[08/09/2022-06:54:41] [I] Total Host Walltime: 3.15374 s
[08/09/2022-06:54:41] [I] Total GPU Compute Time: 3.11119 s
[08/09/2022-06:54:41] [W] * GPU compute time is unstable, with coefficient of variance = 1.04731%.
[08/09/2022-06:54:41] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[08/09/2022-06:54:41] [I] Explanations of the performance metrics are printed in the verbose logs.
[08/09/2022-06:54:41] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=modified.onnx --plugins=MygroupnormalizationIPluginV2DynamicExt/libMygroupnormalizationIPluginV2DynamicExt.so
 ----> debug <---- call [MygroupnormalizationIPluginV2DynamicExt.cpp][destroy][Line 139]
 ----> debug <---- call [MygroupnormalizationIPluginV2DynamicExt.cpp][destroy][Line 139]
```

Now it's your turn to finish the enqueue and other necessary function!

# Additional resources

The following resources provide a deeper understanding about this sample and trt-plugin-generator:

**Original sample**
- [TensorRT Inference of ONNX models with custom layers.](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/onnx_packnet)

**Other**
- [Estimating Depth with ONNX Models and Custom Layers Using NVIDIA TensorRT](https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

August 2022: First Version

# Known issues

There are no known issues in this sample

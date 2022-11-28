# Demonstrate trt-plugin-generator using TRT's efficientdet sample

## How does this sample work?

This sample convert the original efficientdet onnx models to a new one(simply change the EfficientNMS_TRT name in the onnx so it will not recognized by TRT's official sample). and use trt-plugin to generate plugin config yaml. after do some minor modification, we can automatically generate the plugin code. then with little modification on the getOutputDimension(). We can compile the generated plugin code and load it using trtexec to demonstrate the usage of trt-plugin-generator.

## Prerequisites

The first steps is prepare the onnx model, please generate the model by following https://github.com/NVIDIA/TensorRT/tree/main/samples/python/efficientdet. after the first step, you should get the model.onnx. move it to this directory.

The next step is modify the original operator name from "EfficientNMS_TRT" to "MyEfficientNMS" so that it won't recognize by TRT's OSS plugin registry.

```
python modify_onnx.py
```

Then you should see modified.onnx under the same directory.

## Running the sample

### Generate the config yaml form modified.onnx

```
python3 ../../tpg.py extract --onnx modified.onnx --custom_operators MyEfficientNMS --yaml efficientdet.yml
```

### modify the generated yaml

Now we can see the generated efficientdet.yaml as follow:

```yaml
MyEfficientNMS:
  attributes:
    background_class:
      datatype: int32
    box_coding:
      datatype: int32
    iou_threshold:
      datatype: float32
    max_output_boxes:
      datatype: int32
    plugin_version:
      datatype: char[]
    score_activation:
      datatype: int32
    score_threshold:
      datatype: float32
  inputs:
    tpg_input_0:
      shape: -1x49104x4
    tpg_input_1:
      shape: -1x49104x90
    tpg_input_2:
      shape: 1x49104x4
  outputs:
    tpg_output_0:
      shape: need_user_to_specify
    tpg_output_1:
      shape: need_user_to_specify
    tpg_output_2:
      shape: need_user_to_specify
    tpg_output_3:
      shape: need_user_to_specify
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - need_user_to_specify
```

We need to fill the item that the value is "need_user_to_specify", because we know how our operator works, how's its inputs/outputs like, what precision we want to implement it, next we will modify the yaml as below:

```yaml
MyEfficientNMS:
  attributes:
    background_class:
      datatype: int32
    box_coding:
      datatype: int32
    iou_threshold:
      datatype: float32
    max_output_boxes:
      datatype: int32
    plugin_version:
      datatype: char[]
    score_activation:
      datatype: int32
    score_threshold:
      datatype: float32
  inputs:
    tpg_input_0:
      shape: -1x49104x4
    tpg_input_1:
      shape: -1x49104x90
    tpg_input_2:
      shape: 1x49104x4
  outputs:
    tpg_output_0:
      shape: -1x-1
    tpg_output_1:
      shape: -1x-1x-1
    tpg_output_2:
      shape: -1x-1
    tpg_output_3:
      shape: -1x-1
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - "float32+float32+float32+int32+float32+float32+int32"
```

### Generate the plugin code

Now let's generate the plugin code.

```
python ../../tpg.py generate --yaml efficientdet.yml --output ./
```

You can see the generated plugin code is under `./MyefficientnmsIPluginV2DynamicExt`

### compile && running with trtexec

The generated code can be compile now, e.g. I can compile it using TRT's official docker image `nvcr.io/nvidia/tensorrt:22.07-py3`

```
docker run -it --gpus all --rm -v /path/to/this/sample:/efficientdet nvcr.io/nvidia/tensorrt:22.07-py3 /bin/bash
# in container
cd /efficientdet/MyefficientnmsIPluginV2DynamicExt
make
cd ..
```

Let's see if it works!
```
trtexec --onnx=modified.onnx --plugins=MyefficientnmsIPluginV2DynamicExt/libMyefficientnmsIPluginV2DynamicExt.so
```

Oops, it throw an error

```
 ----> debug <---- call [MyefficientnmsIPluginV2DynamicExt.cpp][getOutputDimensions][Line 243]
trtexec: MyefficientnmsIPluginV2DynamicExt.cpp:249: virtual nvinfer1::DimsExprs MyefficientnmsIPluginV2DynamicExt::getOutputDimensions(int32_t, const nvinfer1::DimsExprs*, int32_t, nvinfer1::IExprBuilder&): Assertion `false && "Please implement this part first"' failed.
Aborted (core dumped)
```

### Add necessary function implementation to the generated code

To solve the above error, we need to implement a getOutputDimensions(). Because TRT don't know how your custom plugin works, TRT don't know how to get the output dimension of it and assign memory for it:

```
DimsExprs MyefficientnmsIPluginV2DynamicExt::getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) IS_NOEXCEPT
{
    DEBUG_LOG();
    assert(outputIndex < 4);
    DimsExprs output;
    // num_detections
    if (outputIndex == 0)
    {
        output.nbDims = 2;
        output.d[0] = inputs[0].d[0];
        output.d[1] = exprBuilder.constant(1);
    }
    // detection_boxes
    else if (outputIndex == 1)
    {
        output.nbDims = 3;
        output.d[0] = inputs[0].d[0];
        output.d[1] = exprBuilder.constant(this->max_output_boxes);
        output.d[2] = exprBuilder.constant(4);
    }
    // detection_scores
    else if (outputIndex == 2)
    {
        output.nbDims = 2;
        output.d[0] = inputs[0].d[0];
        output.d[1] = exprBuilder.constant(this->max_output_boxes);
    }
    // detection_classes
    else if (outputIndex == 3)
    {
        output.nbDims = 2;
        output.d[0] = inputs[0].d[0];
        output.d[1] = exprBuilder.constant(this->max_output_boxes);
    }
    return output;

}
```

Now let's recompile the code and try it again:

```
cd /efficientdet/MyefficientnmsIPluginV2DynamicExt
make
cd ..
trtexec --onnx=modified.onnx --plugins=MyefficientnmsIPluginV2DynamicExt/libMyefficientnmsIPluginV2DynamicExt.so
```

We will see the model is running successfully, although we do nothing in the enqueue function!

```
[08/11/2022-06:55:33] [I] === Performance summary ===
[08/11/2022-06:55:33] [I] Throughput: 233.689 qps
[08/11/2022-06:55:33] [I] Latency: min = 4.51123 ms, max = 5.53139 ms, mean = 4.54566 ms, median = 4.52756 ms, percentile(99%) = 5.51622 ms
[08/11/2022-06:55:33] [I] Enqueue Time: min = 3.5542 ms, max = 4.63188 ms, mean = 4.1058 ms, median = 4.16818 ms, percentile(99%) = 4.52371 ms
[08/11/2022-06:55:33] [I] H2D Latency: min = 0.260254 ms, max = 0.277344 ms, mean = 0.263636 ms, median = 0.263184 ms, percentile(99%) = 0.272461 ms
[08/11/2022-06:55:33] [I] GPU Compute Time: min = 4.23773 ms, max = 5.25439 ms, mean = 4.26986 ms, median = 4.25195 ms, percentile(99%) = 5.24138 ms
[08/11/2022-06:55:33] [I] D2H Latency: min = 0.00610352 ms, max = 0.0246582 ms, mean = 0.0121628 ms, median = 0.012207 ms, percentile(99%) = 0.0195312 ms
[08/11/2022-06:55:33] [I] Total Host Walltime: 3.01255 s
[08/11/2022-06:55:33] [I] Total GPU Compute Time: 3.00598 s
[08/11/2022-06:55:33] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[08/11/2022-06:55:33] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[08/11/2022-06:55:33] [W] * GPU compute time is unstable, with coefficient of variance = 2.9705%.
[08/11/2022-06:55:33] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[08/11/2022-06:55:33] [I] Explanations of the performance metrics are printed in the verbose logs.
[08/11/2022-06:55:33] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=modified.onnx --plugins=MyefficientnmsIPluginV2DynamicExt/libMyefficientnmsIPluginV2DynamicExt.so
 ----> debug <---- call [MyefficientnmsIPluginV2DynamicExt.cpp][destroy][Line 175]
 ----> debug <---- call [MyefficientnmsIPluginV2DynamicExt.cpp][terminate][Line 129]
 ----> debug <---- call [MyefficientnmsIPluginV2DynamicExt.cpp][destroy][Line 175]
```

Now it's your turn to finish the enqueue and other necessary function!

# Additional resources

The following resources provide a deeper understanding about this sample and trt-plugin-generator:

**Original sample**
- [EfficientDet Object Detection in TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/efficientdet)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

August 2022: First Version

# Known issues

There are no known issues in this sample

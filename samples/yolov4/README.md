# Demonstrate trt-plugin-generator using yolov4

## How does this sample work?

This sample convert the original TRT yolov4 onnx models to a new one(simply change the BatchedNMS_TRT name in the onnx so it will not recognized by TRT's official sample). and use trt-plugin to generate plugin config yaml. after do some minor modification, we can automatically generate the plugin code. then with little modification on the getOutputDimension(). We can compile the generated plugin code and load it using trtexec to demonstrate the usage of trt-plugin-generator.

## Prerequisites

The first steps is prepare the onnx model, please generate the model by following https://github.com/NVIDIA-AI-IOT/yolov4_deepstream/tree/master/tensorrt_yolov4. after the first step, you should get the yolov4_1_3_416_416_static.onnx.nms.onnx. move it to this directory.

The next step is modify the original operator name from "BatchedNMS_TRT" to "MyBatchedNMS" so that it won't recognize by TRT's OSS plugin registry.

```
python modify_onnx.py
```

Then you should see modified.onnx under the same directory.

## Running the sample

### Generate the config yaml form modified.onnx

```
python3 ../../tpg.py extract --onnx modified.onnx --custom_operators MyBatchedNMS --yaml yolov4.yml
```

### modify the generated yaml

Now we can see the generated yolov4.yaml as follow:

```yaml
MyBatchedNMS:
  attributes:
    backgroundLabelId:
      datatype: int32
    clipBoxes:
      datatype: int32
    iouThreshold:
      datatype: float32
    isNormalized:
      datatype: int32
    keepTopK:
      datatype: int32
    numClasses:
      datatype: int32
    plugin_version:
      datatype: char[]
    scoreThreshold:
      datatype: float32
    shareLocation:
      datatype: int32
    topK:
      datatype: int32
  inputs:
    tpg_input_0:
      shape: 1x10647x1x4
    tpg_input_1:
      shape: 1x10647x80
  outputs:
    tpg_output_0:
      shape: 1x1
    tpg_output_1:
      shape: 1x1000x4
    tpg_output_2:
      shape: 1x1000
    tpg_output_3:
      shape: 1x1000
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - need_user_to_specify
```

We need to fill the item that the value is "need_user_to_specify", because we know how our operator works, how's its inputs/outputs like, what precision we want to implement it, next we will modify the yaml as below:

```yaml
MyBatchedNMS:
  attributes:
    backgroundLabelId:
      datatype: int32
    clipBoxes:
      datatype: int32
    iouThreshold:
      datatype: float32
    isNormalized:
      datatype: int32
    keepTopK:
      datatype: int32
    numClasses:
      datatype: int32
    plugin_version:
      datatype: char[]
    scoreThreshold:
      datatype: float32
    shareLocation:
      datatype: int32
    topK:
      datatype: int32
  inputs:
    tpg_input_0:
      shape: 1x10647x1x4
    tpg_input_1:
      shape: 1x10647x80
  outputs:
    tpg_output_0:
      shape: 1x1
    tpg_output_1:
      shape: 1x1000x4
    tpg_output_2:
      shape: 1x1000
    tpg_output_3:
      shape: 1x1000
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - "float32+float32+int32+float32+float32+float32"
```

### Generate the plugin code

Now let's generate the plugin code.

```
python ../../tpg.py generate --yaml yolov4.yml --output ./
```

You can see the generated plugin code is under `./MyBatchedNMSIPluginV2DynamicExt`

### compile && running with trtexec

The generated code can be compile now, e.g. I can compile it using TRT's official docker image `nvcr.io/nvidia/tensorrt:22.07-py3`

```
docker run -it --gpus all --rm -v /path/to/this/sample:/yolov4 nvcr.io/nvidia/tensorrt:22.07-py3 /bin/bash
# in container
cd /yolov4/MyBatchedNMSIPluginV2DynamicExt
make
cd ..
```

Let's see if it works!
```
trtexec --onnx=modified.onnx --plugins=MyBatchedNMSIPluginV2DynamicExt/libMyBatchedNMSIPluginV2DynamicExt.so
```

We will see the model is running successfully, although we do nothing in the enqueue function!

```
[08/23/2022-06:21:06] [I] === Performance summary ===
[08/23/2022-06:21:06] [I] Throughput: 128.422 qps
[08/23/2022-06:21:06] [I] Latency: min = 7.94055 ms, max = 8.0965 ms, mean = 7.96362 ms, median = 7.95883 ms, percentile(99%) = 8.08612 ms
[08/23/2022-06:21:06] [I] Enqueue Time: min = 2.25098 ms, max = 3.06787 ms, mean = 2.68443 ms, median = 2.66418 ms, percentile(99%) = 3.04663 ms
[08/23/2022-06:21:06] [I] H2D Latency: min = 0.177185 ms, max = 0.213013 ms, mean = 0.189113 ms, median = 0.188232 ms, percentile(99%) = 0.203857 ms
[08/23/2022-06:21:06] [I] GPU Compute Time: min = 7.74718 ms, max = 7.89645 ms, mean = 7.76549 ms, median = 7.76099 ms, percentile(99%) = 7.88892 ms
[08/23/2022-06:21:06] [I] D2H Latency: min = 0.0078125 ms, max = 0.0111084 ms, mean = 0.00901728 ms, median = 0.00878906 ms, percentile(99%) = 0.0107422 ms
[08/23/2022-06:21:06] [I] Total Host Walltime: 3.02128 s
[08/23/2022-06:21:06] [I] Total GPU Compute Time: 3.01301 s
[08/23/2022-06:21:06] [I] Explanations of the performance metrics are printed in the verbose logs.
[08/23/2022-06:21:06] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=modified.onnx --plugins=MybatchednmsIPluginV2DynamicExt/libMybatchednmsIPluginV2DynamicExt.so
 ----> debug <---- call [MybatchednmsIPluginV2DynamicExt.cpp][destroy][Line 187]
 ----> debug <---- call [MybatchednmsIPluginV2DynamicExt.cpp][terminate][Line 135]
 ----> debug <---- call [MybatchednmsIPluginV2DynamicExt.cpp][destroy][Line 187]

```

Now it's your turn to finish the enqueue and other necessary function!

# Additional resources

The following resources provide a deeper understanding about this sample and trt-plugin-generator:

**Original sample**
- [yolov4 Object Detection in TensorRT](https://github.com/NVIDIA-AI-IOT/yolov4_deepstream/tree/master/tensorrt_yolov4)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

August 2022: First Version

# Known issues

There are no known issues in this sample

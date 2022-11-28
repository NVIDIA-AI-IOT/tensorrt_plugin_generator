# Demonstrate trt-plugin-generator using TRT's efficientdet sample

## How does this sample work?

This sample generate plugin code for a custom pointpillar model. and demonstrate the compile and execution.

## Prerequisites

It contains 2 custom operators, and we will convert the code for them all in one!

## Running the sample

### Generate the config yaml form modified.onnx

```
python3 ../../tpg.py extract --onnx pointpillar_mod.onnx --custom_operators PFP,ScatterBEV --yaml pointpillar.yml
```

### modify the generated yaml

Now we can see the generated pointpillar.yaml as follow:

```yaml
PFP:
  inputs:
    tpg_input_0:
      shape: 320000x4
    tpg_input_1:
      shape: 1x1
  outputs:
    tpg_output_0:
      shape: 10000x32x10
    tpg_output_1:
      shape: 1x1x10000x4
    tpg_output_2:
      shape: 1x1x1x5
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - need_user_to_specify
ScatterBEV:
  inputs:
    tpg_input_0:
      shape: need_user_to_specify
    tpg_input_1:
      shape: 1x1x10000x4
    tpg_input_2:
      shape: 1x1x1x5
  outputs:
    tpg_output_0:
      shape: 1x64x496x432
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - need_user_to_specify
```

We need to fill the item that the value is "need_user_to_specify", because we know how our operator works, how's its inputs/outputs like, what precision we want to implement it, next we will modify the yaml as below:

```yaml
PFP:
  inputs:
    tpg_input_0:
      shape: 320000x4
    tpg_input_1:
      shape: 1x1
  outputs:
    tpg_output_0:
      shape: 10000x32x10
    tpg_output_1:
      shape: 1x1x10000x4
    tpg_output_2:
      shape: 1x1x1x5
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - 'float32+int32+float32+float32+float32'
ScatterBEV:
  inputs:
    tpg_input_0:
      shape: 10000x64
    tpg_input_1:
      shape: 1x1x10000x4
    tpg_input_2:
      shape: 1x1x1x5
  outputs:
    tpg_output_0:
      shape: 1x64x496x432
  plugin_type: IPluginV2DynamicExt
  support_format_combination:
  - 'float32+float32+float32+float32'
```

### Generate the plugin code

Now let's generate the plugin code.

```
python ../../tpg.py generate --yaml pointpillar.yml --output ./
```

You can see the generated plugin code is under `./PfpIPluginV2DynamicExt` and `./ScatterbevIPluginV2DynamicExt`

### compile && running with trtexec

The generated code can be compile now, e.g. I can compile it using TRT's official docker image `nvcr.io/nvidia/tensorrt:22.07-py3`

```
docker run -it --gpus all --rm -v /path/to/this/sample:/efficientdet nvcr.io/nvidia/tensorrt:22.07-py3 /bin/bash
# in container
cd /pointpillar/ScatterbevIPluginV2DynamicExt
make
cd ../PfpIPluginV2DynamicExt
make
cd ..
```

Let's see if it works!
```
trtexec --onnx=pointpillar_mod.onnx --plugins=PfpIPluginV2DynamicExt/libPfpIPluginV2DynamicExt.so --plugins=ScatterbevIPluginV2DynamicExt/libScatterbevIPluginV2DynamicExt.so
```

Running successfully

```
[08/12/2022-07:07:02] [I] === Performance summary ===
[08/12/2022-07:07:02] [I] Throughput: 128.144 qps
[08/12/2022-07:07:02] [I] Latency: min = 9.38477 ms, max = 9.5282 ms, mean = 9.4244 ms, median = 9.42297 ms, percentile(99%) = 9.49927 ms
[08/12/2022-07:07:02] [I] Enqueue Time: min = 0.443848 ms, max = 0.810425 ms, mean = 0.702947 ms, median = 0.699951 ms, percentile(99%) = 0.795898 ms
[08/12/2022-07:07:02] [I] H2D Latency: min = 0.430054 ms, max = 0.464844 ms, mean = 0.4475 ms, median = 0.448242 ms, percentile(99%) = 0.46106 ms
[08/12/2022-07:07:02] [I] GPU Compute Time: min = 7.74902 ms, max = 7.86877 ms, mean = 7.78232 ms, median = 7.7793 ms, percentile(99%) = 7.86011 ms
[08/12/2022-07:07:02] [I] D2H Latency: min = 1.17432 ms, max = 1.21045 ms, mean = 1.19459 ms, median = 1.19342 ms, percentile(99%) = 1.20874 ms
[08/12/2022-07:07:02] [I] Total Host Walltime: 3.02003 s
[08/12/2022-07:07:02] [I] Total GPU Compute Time: 3.01176 s
[08/12/2022-07:07:02] [I] Explanations of the performance metrics are printed in the verbose logs.
[08/12/2022-07:07:02] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=pointpillar_mod.onnx --plugins=PfpIPluginV2DynamicExt/libPfpIPluginV2DynamicExt.so --plugins=ScatterbevIPluginV2DynamicExt/libScatterbevIPluginV2DynamicExt.so
 ----> debug <---- call [ScatterbevIPluginV2DynamicExt.cpp][destroy][Line 131]
 ----> debug <---- call [PfpIPluginV2DynamicExt.cpp][destroy][Line 131]
 ----> debug <---- call [PfpIPluginV2DynamicExt.cpp][terminate][Line 105]
 ----> debug <---- call [PfpIPluginV2DynamicExt.cpp][destroy][Line 131]
 ----> debug <---- call [ScatterbevIPluginV2DynamicExt.cpp][terminate][Line 105]
 ----> debug <---- call [ScatterbevIPluginV2DynamicExt.cpp][destroy][Line 131]
```

# Additional resources

The following resources provide a deeper understanding about this sample and trt-plugin-generator:

**Reference**
- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

August 2022: First Version

# Known issues

There are no known issues in this sample

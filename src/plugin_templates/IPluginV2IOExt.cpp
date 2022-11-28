/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef DEBUG_PLUGIN
#define DEBUG_PLUGIN 1 // set debug mode, if you want to see the api call, set it to 1
#endif

#include "NvInfer.h"
#include "$plugin_nameIPluginV2IOExt.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>

#if DEBUG_PLUGIN
#define DEBUG_LOG(...) {\
    std::cout << " ----> debug <---- call " << "[" << __FILE__ << ":" \
          << __LINE__ << "][" << __FUNCTION__ << "]" << std::endl;\
    }
#else
#define DEBUG_LOG(...)
#endif

using namespace nvinfer1;

namespace
{
const char* PLUGIN_VERSION{"1"};
const char* PLUGIN_NAME{"$parse_name"};
} // namespace

// Static class fields initialization
PluginFieldCollection $plugin_nameIPluginV2IOExtCreator::mFC{};
std::vector<PluginField> $plugin_nameIPluginV2IOExtCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN($plugin_nameIPluginV2IOExtCreator);

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
void readFromBuffer(const char*& buffer, T& val)
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

$plugin_nameIPluginV2IOExt::$plugin_nameIPluginV2IOExt($constructor_params)
{
    DEBUG_LOG();
$assign_params
}

$plugin_nameIPluginV2IOExt::$plugin_nameIPluginV2IOExt(const void* data, size_t length)
{
    DEBUG_LOG();
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;

$read_deserialized_buffer
    assert(d == (a + length));
}

// -------------------- IPluginV2 ----------------------

const char* $plugin_nameIPluginV2IOExt::getPluginType() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_NAME;
}

const char* $plugin_nameIPluginV2IOExt::getPluginVersion() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_VERSION;
}

int $plugin_nameIPluginV2IOExt::getNbOutputs() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return $number_of_outputs;
}

Dims $plugin_nameIPluginV2IOExt::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) IS_NOEXCEPT
{
    DEBUG_LOG();
$get_output_dimensions
}

// IMPORTANT: Memory allocated in the plug-in must be freed to ensure no memory leak. 
// If resources are acquired in the initialize() function, they must be released in the terminate() function. 
// All other memory allocations should be freed, preferably in the plug-in class destructor or in the destroy() method.

// Initialize the layer for execution.
// e.g. if the plugin require some extra device memory for execution. allocate in this function.
// for details please refer to
// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2.html
int $plugin_nameIPluginV2IOExt::initialize() IS_NOEXCEPT
{
    DEBUG_LOG();
    return 0;
}

void $plugin_nameIPluginV2IOExt::terminate() IS_NOEXCEPT
{
    DEBUG_LOG();
    // Release resources acquired during plugin layer initialization
}

// TODO: implement by user
// The actual plugin execution func.
int $plugin_nameIPluginV2IOExt::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) IS_NOEXCEPT
{
    DEBUG_LOG();
    return 0;
}

size_t $plugin_nameIPluginV2IOExt::getSerializationSize() const IS_NOEXCEPT
{
    DEBUG_LOG();
    size_t size = 0;

$get_serialization_size
    // the size should equal to the length when deserializing.
    return size;
}

void $plugin_nameIPluginV2IOExt::serialize(void* buffer) const IS_NOEXCEPT
{
    DEBUG_LOG();
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    $serialize_to_buffer
    assert(d == a + getSerializationSize());
}

size_t $plugin_nameIPluginV2IOExt::getWorkspaceSize(int ) const IS_NOEXCEPT
{
    DEBUG_LOG();
    // Find the workspace size required by the layer.
    // This function is called during engine startup, after initialize().
    // The workspace size returned should be sufficient for any batch size up to the maximum.
    return 0;
}

void $plugin_nameIPluginV2IOExt::destroy() IS_NOEXCEPT
{
    DEBUG_LOG();
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void $plugin_nameIPluginV2IOExt::setPluginNamespace(const char* pluginNamespace) IS_NOEXCEPT
{
    DEBUG_LOG();
    mNamespace = pluginNamespace;
}

const char* $plugin_nameIPluginV2IOExt::getPluginNamespace() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return mNamespace.c_str();
}

// -------------------- IPluginV2Ext --------------------

DataType $plugin_nameIPluginV2IOExt::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs)  const IS_NOEXCEPT
{
    DEBUG_LOG();
$deduce_output_datatype
}

bool $plugin_nameIPluginV2IOExt::isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs)  const IS_NOEXCEPT
{
    DEBUG_LOG();
    return false;
}

bool $plugin_nameIPluginV2IOExt::canBroadcastInputAcrossBatch(int32_t inputIndex)  const IS_NOEXCEPT
{
    DEBUG_LOG();
    return false;
}

// -------------------- IPluginV2IOExt ------------------

IPluginV2Ext* $plugin_nameIPluginV2IOExt::clone() const IS_NOEXCEPT
{
    DEBUG_LOG();
    auto plugin = new $plugin_nameIPluginV2IOExt($copy_params);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}


void $plugin_nameIPluginV2IOExt::configurePlugin(PluginTensorDesc const *in, int32_t nbInput, PluginTensorDesc const *out, int32_t nbOutput) IS_NOEXCEPT
{
    DEBUG_LOG();
    // This function is called by the builder prior to initialize().
    // It provides an opportunity for the layer to make algorithm choices on the basis of I/O PluginTensorDesc
}

bool $plugin_nameIPluginV2IOExt::supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) const IS_NOEXCEPT
{
    DEBUG_LOG();
    bool is_supported = false;
$get_support_format_combination
    return is_supported;
}

// -------------------- IPluginCreator ------------------

$plugin_nameIPluginV2IOExtCreator::$plugin_nameIPluginV2IOExtCreator()
{
    DEBUG_LOG();
    mPluginAttributes.clear();
    // Describe $plugin_nameIPluginV2IOExt's required PluginField arguments
    $plugin_attributes_emplace_back
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* $plugin_nameIPluginV2IOExtCreator::getPluginName() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_NAME;
}

const char* $plugin_nameIPluginV2IOExtCreator::getPluginVersion() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_VERSION;
}

const PluginFieldCollection* $plugin_nameIPluginV2IOExtCreator::getFieldNames() IS_NOEXCEPT
{
    DEBUG_LOG();
    return &mFC;
}

IPluginV2IOExt* $plugin_nameIPluginV2IOExtCreator::createPlugin(const char* name, const PluginFieldCollection* fc) IS_NOEXCEPT
{
    DEBUG_LOG();
$create_plugin
    return new $plugin_nameIPluginV2IOExt($new_plugin_params);
}

IPluginV2IOExt* $plugin_nameIPluginV2IOExtCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT
{
    DEBUG_LOG();
    return new $plugin_nameIPluginV2IOExt(serialData, serialLength);
}

void $plugin_nameIPluginV2IOExtCreator::setPluginNamespace(const char* libNamespace) IS_NOEXCEPT
{
    DEBUG_LOG();
    mNamespace = libNamespace;
}

const char* $plugin_nameIPluginV2IOExtCreator::getPluginNamespace() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return mNamespace.c_str();
}

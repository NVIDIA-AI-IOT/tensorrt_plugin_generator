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

#ifndef ${plugin_name_uppercase}_PLUGINV2_H
#define ${plugin_name_uppercase}_PLUGINV2_H

#if NV_TENSORRT_MAJOR >= 8
    #define IS_NOEXCEPT noexcept
#else
    #define IS_NOEXCEPT
#endif

#include "NvInferPlugin.h"
#include <string>
#include <vector>

using namespace nvinfer1;

$plugin_attributes_struct

class $plugin_nameIPluginV2IOExt : public IPluginV2IOExt
{
public:
    $plugin_nameIPluginV2IOExt($constructor_params);

    $plugin_nameIPluginV2IOExt(const void* data, size_t length);

    // -------------------- IPluginV2 --------------------

    int getNbOutputs() const IS_NOEXCEPT override;

    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs);

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) IS_NOEXCEPT override;

    int initialize() IS_NOEXCEPT override;

    void terminate() IS_NOEXCEPT override;

    size_t getWorkspaceSize(int ) const IS_NOEXCEPT override;

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) IS_NOEXCEPT override;

    size_t getSerializationSize() const IS_NOEXCEPT override;

    void serialize(void* buffer) const IS_NOEXCEPT override;

    const char* getPluginType() const IS_NOEXCEPT override;

    const char* getPluginVersion() const IS_NOEXCEPT override;

    void destroy() IS_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) IS_NOEXCEPT override;

    const char* getPluginNamespace() const IS_NOEXCEPT override;

    // -------------------- IPluginV2Ext --------------------

    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const IS_NOEXCEPT override;

    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const IS_NOEXCEPT override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const IS_NOEXCEPT override;

    IPluginV2Ext* clone() const IS_NOEXCEPT override;

    // -------------------- IPluginV2IOExt --------------------

    void configurePlugin(PluginTensorDesc const *in, int32_t nbInput, PluginTensorDesc const *out, int32_t nbOutput) IS_NOEXCEPT override;

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) const IS_NOEXCEPT override;

private:
    // plugin attributes
    std::string mNamespace;
    $plugin_private_attributes
};

class $plugin_nameIPluginV2IOExtCreator : public IPluginCreator
{
public:
    $plugin_nameIPluginV2IOExtCreator();

    const char* getPluginName() const IS_NOEXCEPT override;

    const char* getPluginVersion() const IS_NOEXCEPT override;

    const PluginFieldCollection* getFieldNames() IS_NOEXCEPT override;

    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) IS_NOEXCEPT override;

    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) IS_NOEXCEPT override;

    const char* getPluginNamespace() const IS_NOEXCEPT override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // #define ${plugin_name_uppercase}_PLUGINV2_H

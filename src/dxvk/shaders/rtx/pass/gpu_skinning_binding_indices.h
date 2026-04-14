/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include "rtx/utility/shader_types.h"

// NV-DXVK (DX11 port): shift bindings past the D3D11 slot range (0..1151) so the
// skinning compute shader does not share m_rc[] slots with D3D11 graphics PS CBs.
// Collision previously left game vertex-buffer slices in slots read as dynamic
// UBOs on subsequent graphics draws -> VUID-01971 -> GPU hang.
#define BINDING_SKINNING_CONSTANTS    1160
#define BINDING_POSITION_OUTPUT       1161
#define BINDING_POSITION_INPUT        1162
#define BINDING_BLEND_WEIGHT_INPUT    1163
#define BINDING_BLEND_INDICES_INPUT   1164
#define BINDING_NORMAL_OUTPUT         1165
#define BINDING_NORMAL_INPUT          1166

/**
* \brief Args required to perform skinning
*/
struct SkinningArgs {
  mat4 bones[256]; // 256 is the max bone count in DX (swvp)

  uint dstPositionOffset;
  uint dstPositionStride;
  uint srcPositionOffset;
  uint srcPositionStride;

  uint dstNormalOffset;
  uint dstNormalStride;
  uint srcNormalOffset;
  uint srcNormalStride;

  uint blendWeightOffset;
  uint blendWeightStride;
  uint blendIndicesOffset;
  uint blendIndicesStride;

  uint numVertices;
  uint useIndices;
  uint numBones;
  uint useOctahedralNormals;
};
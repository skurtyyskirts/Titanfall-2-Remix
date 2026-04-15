#include "d3d11_device.h"
#include "d3d11_shader.h"

#include <cstring>

namespace dxvk {
  
  D3D11CommonShader:: D3D11CommonShader() { }
  D3D11CommonShader::~D3D11CommonShader() { }
  
  
  D3D11CommonShader::D3D11CommonShader(
          D3D11Device*    pDevice,
    const DxvkShaderKey*  pShaderKey,
    const DxbcModuleInfo* pDxbcModuleInfo,
    const void*           pShaderBytecode,
          size_t          BytecodeLength) {
    const std::string name = pShaderKey->toString();
    Logger::debug(str::format("Compiling shader ", name));
    
    DxbcReader reader(
      reinterpret_cast<const char*>(pShaderBytecode),
      BytecodeLength);
    
    DxbcModule module(reader);
    
    // If requested by the user, dump both the raw DXBC
    // shader and the compiled SPIR-V module to a file.
    const std::string dumpPath = env::getEnvVar("DXVK_SHADER_DUMP_PATH");
    
    if (dumpPath.size() != 0) {
      reader.store(std::ofstream(str::tows(str::format(dumpPath, "/", name, ".dxbc").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc));
    }
    
    // Decide whether we need to create a pass-through
    // geometry shader for vertex shader stream output
    bool passthroughShader = pDxbcModuleInfo->xfb != nullptr
      && (module.programInfo().type() == DxbcProgramType::VertexShader
       || module.programInfo().type() == DxbcProgramType::DomainShader);

    if (module.programInfo().shaderStage() != pShaderKey->type() && !passthroughShader)
      throw DxvkError("Mismatching shader type.");

    m_shader = passthroughShader
      ? module.compilePassthroughShader(*pDxbcModuleInfo, name)
      : module.compile                 (*pDxbcModuleInfo, name);
    m_shader->setShaderKey(*pShaderKey);
    
    if (dumpPath.size() != 0) {
      std::ofstream dumpStream(
        str::tows(str::format(dumpPath, "/", name, ".spv").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc);
      
      m_shader->dump(dumpStream);
    }
    
    // Create shader constant buffer if necessary
    if (m_shader->shaderConstants().data() != nullptr) {
      DxvkBufferCreateInfo info;
      info.size   = m_shader->shaderConstants().sizeInBytes();
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      info.stages = util::pipelineStages(m_shader->stage());
      info.access = VK_ACCESS_UNIFORM_READ_BIT;
      
      VkMemoryPropertyFlags memFlags
        = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      
      m_buffer = pDevice->GetDXVKDevice()->createBuffer(info, memFlags, DxvkMemoryStats::Category::AppBuffer, "d3d11 shader constants");

      std::memcpy(m_buffer->mapPtr(0),
        m_shader->shaderConstants().data(),
        m_shader->shaderConstants().sizeInBytes());
    }

    pDevice->GetDXVKDevice()->registerShader(m_shader);

    // NV-DXVK: parse RDEF chunk so ExtractTransforms can look up cbuffer
    // bind slots + field offsets deterministically instead of guessing.
    parseRdef(pShaderBytecode, BytecodeLength);
  }


  // NV-DXVK: minimal DXBC RDEF parser. Extracts:
  //   - Each declared cbuffer's name, byte size, field list (name + offset)
  //   - Each cbuffer's Vulkan/D3D bind register (via the Resource Bindings table)
  // Only handles SM 4.0/4.1 variable layout (24-byte entries) — SM 5.0 RD11
  // subheader adds 16 more bytes which we skip over. Titanfall 2 ships vs_4_0.
  //
  // DXBC file format (Microsoft-documented):
  //   "DXBC" | 16B hash | u32 version | u32 totalSize | u32 chunkCount | u32 chunkOffsets[chunkCount]
  //   Each chunk: 4B tag | u32 size | data...
  // RDEF chunk body begins right after the 8B chunk header. All internal offsets
  // inside RDEF are relative to that body start (not the file start).
  void D3D11CommonShader::parseRdef(const void* pShaderBytecode, size_t BytecodeLength) {
    if (pShaderBytecode == nullptr || BytecodeLength < 32) return;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(pShaderBytecode);

    auto rdU32 = [&](size_t off) -> uint32_t {
      uint32_t v;
      std::memcpy(&v, base + off, sizeof(v));
      return v;
    };
    auto rdStr = [&](const uint8_t* chunkBody, size_t bodySize, uint32_t off) -> std::string {
      if (off >= bodySize) return {};
      const char* p = reinterpret_cast<const char*>(chunkBody + off);
      size_t maxLen = bodySize - off;
      size_t len = strnlen(p, maxLen);
      return std::string(p, len);
    };

    // DXBC header sanity
    if (std::memcmp(base, "DXBC", 4) != 0) return;
    uint32_t chunkCount = rdU32(28);
    if (chunkCount == 0 || 32 + chunkCount * 4 > BytecodeLength) return;

    // Find the RDEF chunk.
    const uint8_t* rdefBody = nullptr;
    size_t         rdefSize = 0;
    for (uint32_t i = 0; i < chunkCount; i++) {
      uint32_t chunkOff = rdU32(32 + i * 4);
      if (chunkOff + 8 > BytecodeLength) continue;
      if (std::memcmp(base + chunkOff, "RDEF", 4) == 0) {
        uint32_t csize = rdU32(chunkOff + 4);
        if (chunkOff + 8 + csize > BytecodeLength) return;
        rdefBody = base + chunkOff + 8;
        rdefSize = csize;
        break;
      }
    }
    if (!rdefBody) return;  // No RDEF (fxc stripped it, or pass-through shader).

    auto bodyU32 = [&](size_t off) -> uint32_t {
      if (off + 4 > rdefSize) return 0;
      uint32_t v;
      std::memcpy(&v, rdefBody + off, sizeof(v));
      return v;
    };

    // RDEF header (28 bytes): cbCount, cbOff, resBindCount, resBindOff, shaderVer, flags, creatorOff
    if (rdefSize < 28) return;
    uint32_t cbCount       = bodyU32(0);
    uint32_t cbTableOff    = bodyU32(4);
    uint32_t resBindCount  = bodyU32(8);
    uint32_t resBindOff    = bodyU32(12);

    // Detect SM5+ by looking for the RD11 subheader at body offset 28 — this is
    // the actual on-disk indicator of the extended layout. In SM5+, variable
    // entries are 40B instead of 24B (extra fields appended). Reading them with
    // the wrong stride scrambles every field offset including c_cameraOrigin's.
    bool isSm5Plus = false;
    if (rdefSize >= 28 + 4) {
      uint32_t maybeMagic = bodyU32(28);
      // 'RD11' = 0x31314452 little-endian
      if (maybeMagic == 0x31314452u) isSm5Plus = true;
    }

    // First pass: every resource binding -> name -> slot. This covers cbuffers,
    // SRVs (textures, structured buffers), UAVs, and samplers. The slot value
    // is the physical register index (cN/tN/uN/sN) the shader actually reads.
    std::unordered_map<std::string, uint32_t> nameToSlot;
    constexpr uint32_t D3D_SIT_CBUFFER = 0;
    const size_t resBindEntrySize = 32;  // 8 * u32
    // Resource binding entry on-disk layout (32 bytes):
    //   off  0: nameOffset
    //   off  4: type (D3D_SHADER_INPUT_TYPE)
    //   off  8: returnType
    //   off 12: dimension
    //   off 16: numSamples
    //   off 20: bindPoint   <-- physical register (cN/tN/uN/sN)
    //   off 24: bindCount   <-- consecutive registers used (>=1)
    //   off 28: flags
    for (uint32_t i = 0; i < resBindCount; i++) {
      size_t off = size_t(resBindOff) + size_t(i) * resBindEntrySize;
      if (off + resBindEntrySize > rdefSize) break;
      uint32_t nameOff = bodyU32(off + 0);
      uint32_t type    = bodyU32(off + 4);
      uint32_t bindPt  = bodyU32(off + 20);
      std::string n = rdStr(rdefBody, rdefSize, nameOff);
      if (n.empty()) continue;
      if (type == D3D_SIT_CBUFFER) {
        nameToSlot[n] = bindPt;  // for cbuffer match-up below
      }
      // Mirror every named binding into the public lookup so callers can find
      // SRV/UAV slots by HLSL name (e.g. "g_modelInst" -> 31).
      m_resourceSlots[n] = bindPt;
    }

    // Second pass: cbuffer table -> name, size, field list.
    const size_t cbEntrySize    = 24;  // 6 * u32
    const size_t varEntrySize   = isSm5Plus ? 40 : 24;
    for (uint32_t i = 0; i < cbCount; i++) {
      size_t off = size_t(cbTableOff) + size_t(i) * cbEntrySize;
      if (off + cbEntrySize > rdefSize) break;
      uint32_t nameOff   = bodyU32(off + 0);
      uint32_t varCount  = bodyU32(off + 4);
      uint32_t varOff    = bodyU32(off + 8);
      uint32_t cbSize    = bodyU32(off + 12);
      // off+16 flags, off+20 type — we only care about regular cbuffers but
      // the resource-binding type filter above already scoped us correctly.

      std::string cbName = rdStr(rdefBody, rdefSize, nameOff);
      if (cbName.empty()) continue;

      D3D11CbufferInfo info;
      info.size = cbSize;
      auto slotIt = nameToSlot.find(cbName);
      if (slotIt != nameToSlot.end()) info.bindSlot = slotIt->second;

      for (uint32_t v = 0; v < varCount; v++) {
        size_t voff = size_t(varOff) + size_t(v) * varEntrySize;
        if (voff + varEntrySize > rdefSize) break;
        uint32_t vNameOff = bodyU32(voff + 0);
        uint32_t vStart   = bodyU32(voff + 4);
        uint32_t vSize    = bodyU32(voff + 8);
        std::string vName = rdStr(rdefBody, rdefSize, vNameOff);
        if (!vName.empty())
          info.fields[vName] = { vStart, vSize };
      }

      m_cbuffers[std::move(cbName)] = std::move(info);
    }
  }


  D3D11ShaderModuleSet:: D3D11ShaderModuleSet() { }
  D3D11ShaderModuleSet::~D3D11ShaderModuleSet() { }
  
  
  HRESULT D3D11ShaderModuleSet::GetShaderModule(
          D3D11Device*        pDevice,
    const DxvkShaderKey*      pShaderKey,
    const DxbcModuleInfo*     pDxbcModuleInfo,
    const void*               pShaderBytecode,
          size_t              BytecodeLength,
          D3D11CommonShader*  pShader) {
    // Use the shader's unique key for the lookup
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto entry = m_modules.find(*pShaderKey);
      if (entry != m_modules.end()) {
        *pShader = entry->second;
        return S_OK;
      }
    }
    
    // This shader has not been compiled yet, so we have to create a
    // new module. This takes a while, so we won't lock the structure.
    D3D11CommonShader module;
    
    try {
      module = D3D11CommonShader(pDevice, pShaderKey,
        pDxbcModuleInfo, pShaderBytecode, BytecodeLength);
    } catch (const DxvkError& e) {
      Logger::err(e.message());
      return E_INVALIDARG;
    }
    
    // Insert the new module into the lookup table. If another thread
    // has compiled the same shader in the meantime, we should return
    // that object instead and discard the newly created module.
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto status = m_modules.insert({ *pShaderKey, module });
      if (!status.second) {
        *pShader = status.first->second;
        return S_OK;
      }
    }
    
    *pShader = std::move(module);
    return S_OK;
  }
  
}

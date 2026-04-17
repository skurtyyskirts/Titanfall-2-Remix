#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../dxbc/dxbc_module.h"
#include "../dxvk/dxvk_device.h"

#include "../util/sha1/sha1_util.h"

#include "../util/util_env.h"

#include "d3d11_device_child.h"
#include "d3d11_interfaces.h"

namespace dxvk {
  
  class D3D11Device;

  // NV-DXVK: RDEF-derived cbuffer metadata. Used so per-shader ExtractTransforms
  // looks up slots/offsets deterministically from the shader's own declarations,
  // instead of guessing with size/content heuristics. See parseRdef() in cpp.
  struct D3D11CbufferField {
    uint32_t offset;   // byte offset within the cbuffer
    uint32_t size;     // bytes
  };
  struct D3D11CbufferInfo {
    uint32_t bindSlot = UINT32_MAX;   // the cbN register the cbuffer is bound to
    uint32_t size     = 0;            // cbuffer size in bytes
    // fieldName -> {offset, size}
    std::unordered_map<std::string, D3D11CbufferField> fields;
  };

  /**
   * \brief Common shader object
   *
   * Stores the compiled SPIR-V shader and the SHA-1
   * hash of the original DXBC shader, which can be
   * used to identify the shader.
   */
  class D3D11CommonShader {

  public:

    D3D11CommonShader();
    D3D11CommonShader(
            D3D11Device*    pDevice,
      const DxvkShaderKey*  pShaderKey,
      const DxbcModuleInfo* pDxbcModuleInfo,
      const void*           pShaderBytecode,
            size_t          BytecodeLength);
    ~D3D11CommonShader();

    Rc<DxvkShader> GetShader() const {
      return m_shader;
    }

    Rc<DxvkBuffer> GetIcb() const {
      return m_buffer;
    }

    std::string GetName() const {
      return m_shader->debugName();
    }

    // NV-DXVK: lookup a cbuffer by name (the HLSL-declared name, e.g.
    // "CBufCommonPerCamera"). Returns nullptr if the shader doesn't bind it.
    const D3D11CbufferInfo* FindCBuffer(const std::string& name) const {
      auto it = m_cbuffers.find(name);
      return it != m_cbuffers.end() ? &it->second : nullptr;
    }
    // NV-DXVK: returns the bind slot (txx / cxx / uxx) the shader uses for a
    // resource by HLSL name, e.g. "g_modelInst" -> 31 if the VS reads t31.
    // Returns UINT32_MAX if the shader does not declare that resource.
    uint32_t FindResourceSlot(const std::string& name) const {
      auto it = m_resourceSlots.find(name);
      return it != m_resourceSlots.end() ? it->second : UINT32_MAX;
    }
    // Convenience: return {slot, offset, size} for a field, or std::nullopt.
    struct CBFieldLoc { uint32_t slot, offset, size; };
    std::optional<CBFieldLoc> FindCBField(const std::string& cbName,
                                          const std::string& fieldName) const {
      auto cb = FindCBuffer(cbName);
      if (!cb) return std::nullopt;
      auto it = cb->fields.find(fieldName);
      if (it == cb->fields.end()) return std::nullopt;
      return CBFieldLoc{ cb->bindSlot, it->second.offset, it->second.size };
    }
    // NV-DXVK: RDEF diagnostic — list all cbuffer names the shader declares,
    // plus their bind slot. Used to identify merged-bucket VS variants where
    // the objectToCameraRelative cbuffer uses a non-default HLSL name.
    std::vector<std::pair<std::string, uint32_t>> GetCBufferNamesAndSlots() const {
      std::vector<std::pair<std::string, uint32_t>> out;
      out.reserve(m_cbuffers.size());
      for (const auto& kv : m_cbuffers) out.emplace_back(kv.first, kv.second.bindSlot);
      return out;
    }

  private:

    void parseRdef(const void* pShaderBytecode, size_t BytecodeLength);

    Rc<DxvkShader> m_shader;
    Rc<DxvkBuffer> m_buffer;

    // NV-DXVK: cbName -> info. Populated from DXBC RDEF chunk at ctor time.
    std::unordered_map<std::string, D3D11CbufferInfo> m_cbuffers;
    // NV-DXVK: resourceName -> bind slot (covers SRVs, UAVs, samplers).
    // Used by Remix to identify which physical slot a named structured buffer
    // (g_modelInst at t31, g_boneMatrix at t30, etc.) is read from per shader.
    std::unordered_map<std::string, uint32_t> m_resourceSlots;
  };
  
  
  /**
   * \brief Common shader interface
   * 
   * Implements methods for all D3D11*Shader
   * interfaces and stores the actual shader
   * module object.
   */
  template<typename D3D11Interface>
  class D3D11Shader : public D3D11DeviceChild<D3D11Interface> {

  public:
    
    D3D11Shader(D3D11Device* device, const D3D11CommonShader& shader)
    : D3D11DeviceChild<D3D11Interface>(device),
      m_shader(shader) { }
    
    ~D3D11Shader() { }
    
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) final {
      *ppvObject = nullptr;
      
      if (riid == __uuidof(IUnknown)
       || riid == __uuidof(ID3D11DeviceChild)
       || riid == __uuidof(D3D11Interface)) {
        *ppvObject = ref(this);
        return S_OK;
      }
      
      Logger::warn("D3D11Shader::QueryInterface: Unknown interface query");
      return E_NOINTERFACE;
    }
    
    const D3D11CommonShader* GetCommonShader() const {
      return &m_shader;
    }

  private:
    
    D3D11CommonShader m_shader;
    
  };
  
  using D3D11VertexShader   = D3D11Shader<ID3D11VertexShader>;
  using D3D11HullShader     = D3D11Shader<ID3D11HullShader>;
  using D3D11DomainShader   = D3D11Shader<ID3D11DomainShader>;
  using D3D11GeometryShader = D3D11Shader<ID3D11GeometryShader>;
  using D3D11PixelShader    = D3D11Shader<ID3D11PixelShader>;
  using D3D11ComputeShader  = D3D11Shader<ID3D11ComputeShader>;
  
  
  /**
   * \brief Shader module set
   * 
   * Some applications may compile the same shader multiple
   * times, so we should cache the resulting shader modules
   * and reuse them rather than creating new ones. This
   * class is thread-safe.
   */
  class D3D11ShaderModuleSet {
    
  public:
    
    D3D11ShaderModuleSet();
    ~D3D11ShaderModuleSet();
    
    HRESULT GetShaderModule(
            D3D11Device*        pDevice,
      const DxvkShaderKey*      pShaderKey,
      const DxbcModuleInfo*     pDxbcModuleInfo,
      const void*               pShaderBytecode,
            size_t              BytecodeLength,
            D3D11CommonShader*  pShader);
    
  private:
    
    dxvk::mutex m_mutex;
    
    std::unordered_map<
      DxvkShaderKey,
      D3D11CommonShader,
      DxvkHash, DxvkEq> m_modules;
    
  };
  
}

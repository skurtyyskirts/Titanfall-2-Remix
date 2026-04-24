#include <cstdlib>
#include <cstring>

#include "d3d11_device.h"
#include "d3d11_initializer.h"

// HR patch: Patch 11 Edit B1 — hash CPU upload data so sampled textures get a
// deterministic content-hash for the Remix asset catalogue. Mirrors
// D3D9CommonTexture::SetupForRtxFrom. — see CHANGELOG.md 2026-04-22
#include "../util/xxHash/xxhash.h"

namespace dxvk {

  // HR patch: Patch 13 — classify sampled textures that are non-albedo PBR inputs
  // (normal maps, AO, roughness, metal masks). Heavy Rain's baked-in PBR data fights
  // Remix's path-traced shading — duplicated normal perturbation and occlusion on top
  // of ray-traced normals/AO produces the washed-out, overly-smooth look seen in
  // session 2026-04-24. Classifier runs once per device-local texture init; when it
  // returns true the Edit B1 hash assignment is skipped, Edit B2 sees hash==0 and
  // never calls ImGUI::AddTexture, and the Remix material slot for that draw falls
  // through to its albedo-only default (flat normals, no pre-baked AO).
  //
  // Controlled by env HR_SKIP_NON_ALBEDO (default "1" = enabled; set to "0" to disable
  // without rebuilding). Reason is surfaced in the [HR-TexSkip] log for post-mortem.
  // See CHANGELOG.md 2026-04-24.
  namespace {
    enum class NonAlbedoReason : uint8_t {
      None = 0,
      FormatBC5,   // 2-channel compressed: standard normal-map encoding.
      FormatBC4,   // 1-channel compressed: AO / roughness / height.
      FormatR,     // 1-channel uncompressed: mask / height / AO.
      FormatRG,    // 2-channel uncompressed: often normal XY.
      ContentNormal,    // avg ≈ (128, 128, 255) in UNORM space.
      ContentGrayscale, // R == G == B across sampled pixels.
    };

    const char* NonAlbedoReasonStr(NonAlbedoReason r) {
      switch (r) {
        case NonAlbedoReason::FormatBC5:        return "fmt_bc5";
        case NonAlbedoReason::FormatBC4:        return "fmt_bc4";
        case NonAlbedoReason::FormatR:          return "fmt_r";
        case NonAlbedoReason::FormatRG:         return "fmt_rg";
        case NonAlbedoReason::ContentNormal:    return "content_normal";
        case NonAlbedoReason::ContentGrayscale: return "content_grayscale";
        default:                                return "none";
      }
    }

    bool IsHrSkipNonAlbedoEnabled() {
      // Resolved once at first call, cached for the rest of the session.
      static const bool enabled = []() {
        const char* v = std::getenv("HR_SKIP_NON_ALBEDO");
        return v == nullptr || std::strcmp(v, "0") != 0;
      }();
      return enabled;
    }

    NonAlbedoReason ClassifyNonAlbedo(
        VkFormat                         format,
        const VkExtent3D&                extent,
        const D3D11_SUBRESOURCE_DATA&    data) {
      // Format-based shortcuts. Compressed formats are not decoded — decoding BC/ETC
      // would need a decompressor here; the format alone is high-confidence.
      switch (format) {
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
          return NonAlbedoReason::FormatBC5;
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
          return NonAlbedoReason::FormatBC4;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_SFLOAT:
          return NonAlbedoReason::FormatR;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_SFLOAT:
          return NonAlbedoReason::FormatRG;
        default:
          break;
      }

      // Content classification only for uncompressed RGBA8 layouts where we can
      // read pixels safely. BC3/BC7/other compressed 4-channel formats fall through
      // as "not non-albedo" — treat as albedo by default so we don't over-skip.
      int bpp = 0;
      int rIdx = 0, gIdx = 1, bIdx = 2;
      switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
          bpp = 4; rIdx = 0; gIdx = 1; bIdx = 2;
          break;
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
          bpp = 4; rIdx = 2; gIdx = 1; bIdx = 0;
          break;
        default:
          return NonAlbedoReason::None;
      }

      const uint32_t pitch = data.SysMemPitch;
      const auto* bytes = static_cast<const uint8_t*>(data.pSysMem);
      if (bytes == nullptr || pitch < uint32_t(bpp)) return NonAlbedoReason::None;

      const uint32_t w = std::max(1u, extent.width);
      const uint32_t h = std::max(1u, extent.height);
      const uint32_t stepX = std::max(1u, w / 8u);
      const uint32_t stepY = std::max(1u, h / 8u);

      uint64_t sumR = 0, sumG = 0, sumB = 0;
      uint32_t count = 0;
      uint32_t grayscaleCount = 0;

      for (uint32_t y = 0; y < h && count < 64; y += stepY) {
        const uint8_t* row = bytes + y * pitch;
        for (uint32_t x = 0; x < w && count < 64; x += stepX) {
          const uint8_t* pix = row + x * bpp;
          const int r = pix[rIdx], g = pix[gIdx], b = pix[bIdx];
          sumR += r; sumG += g; sumB += b;
          if (std::abs(r - g) <= 4 && std::abs(g - b) <= 4) {
            ++grayscaleCount;
          }
          ++count;
        }
      }
      if (count == 0) return NonAlbedoReason::None;

      // All-grayscale: AO / roughness / specular mask stored in RGBA.
      if (grayscaleCount == count) {
        return NonAlbedoReason::ContentGrayscale;
      }

      // Normal map in tangent space: avg very close to (128, 128, 255).
      // Tight bounds avoid false-positives on legitimate blue-dominant albedo
      // (night sky, blue fabric) — real normals cluster tightly here because
      // tangent-space Z is usually near 1.
      const uint32_t avgR = uint32_t(sumR / count);
      const uint32_t avgG = uint32_t(sumG / count);
      const uint32_t avgB = uint32_t(sumB / count);
      if (avgR >= 110 && avgR <= 150
       && avgG >= 110 && avgG <= 150
       && avgB >= 220) {
        return NonAlbedoReason::ContentNormal;
      }

      return NonAlbedoReason::None;
    }
  } // namespace

  D3D11Initializer::D3D11Initializer(
          D3D11Device*                pParent)
  : m_parent(pParent),
    m_device(pParent->GetDXVKDevice()),
    m_context(m_device->createContext()) {
    m_context->beginRecording(
      m_device->createCommandList());
  }

  
  D3D11Initializer::~D3D11Initializer() {

  }


  void D3D11Initializer::Flush() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (m_transferCommands != 0)
      FlushInternal();
  }

  void D3D11Initializer::InitBuffer(
          D3D11Buffer*                pBuffer,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    VkMemoryPropertyFlags memFlags = pBuffer->GetBuffer()->memFlags();

    (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      ? InitHostVisibleBuffer(pBuffer, pInitialData)
      : InitDeviceLocalBuffer(pBuffer, pInitialData);
  }
  

  void D3D11Initializer::InitTexture(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    (pTexture->GetMapMode() == D3D11_COMMON_TEXTURE_MAP_MODE_DIRECT)
      ? InitHostVisibleTexture(pTexture, pInitialData)
      : InitDeviceLocalTexture(pTexture, pInitialData);
  }


  void D3D11Initializer::InitUavCounter(
          D3D11UnorderedAccessView*   pUav) {
    auto counterBuffer = pUav->GetCounterSlice();

    if (!counterBuffer.defined())
      return;

    std::lock_guard<dxvk::mutex> lock(m_mutex);
    m_transferCommands += 1;

    const uint32_t zero = 0;
    m_context->updateBuffer(
      counterBuffer.buffer(),
      0, sizeof(zero), &zero);

    FlushImplicit();
  }


  void D3D11Initializer::InitDeviceLocalBuffer(
          D3D11Buffer*                pBuffer,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    DxvkBufferSlice bufferSlice = pBuffer->GetBufferSlice();

    if (pInitialData != nullptr && pInitialData->pSysMem != nullptr) {
      m_transferMemory   += bufferSlice.length();
      m_transferCommands += 1;
      
      m_context->uploadBuffer(
        bufferSlice.buffer(),
        pInitialData->pSysMem);
    } else {
      m_transferCommands += 1;

      m_context->clearBuffer(
        bufferSlice.buffer(),
        bufferSlice.offset(),
        bufferSlice.length(),
        0u);
    }

    FlushImplicit();
  }


  void D3D11Initializer::InitHostVisibleBuffer(
          D3D11Buffer*                pBuffer,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    // If the buffer is mapped, we can write data directly
    // to the mapped memory region instead of doing it on
    // the GPU. Same goes for zero-initialization.
    DxvkBufferSlice bufferSlice = pBuffer->GetBufferSlice();

    if (pInitialData != nullptr && pInitialData->pSysMem != nullptr) {
      std::memcpy(
        bufferSlice.mapPtr(0),
        pInitialData->pSysMem,
        bufferSlice.length());
    } else {
      std::memset(
        bufferSlice.mapPtr(0), 0,
        bufferSlice.length());
    }
  }


  void D3D11Initializer::InitDeviceLocalTexture(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    
    Rc<DxvkImage> image = pTexture->GetImage();

    auto mapMode = pTexture->GetMapMode();
    auto desc = pTexture->Desc();

    VkFormat packedFormat = m_parent->LookupPackedFormat(desc->Format, pTexture->GetFormatMode()).Format;
    auto formatInfo = imageFormatInfo(packedFormat);

    if (pInitialData != nullptr && pInitialData->pSysMem != nullptr) {
      // HR patch: Patch 11 Edit B1 — hash subresource 0 (top mip, layer 0) CPU data and
      // store it on the underlying DxvkImage. This is the content hash that the SRV path
      // (Edit B2) will pass to ImGUI::AddTexture, populating the Remix asset catalogue.
      // Skipped if the image already has a hash (RT/DSV/UAV are tagged elsewhere or
      // intentionally excluded — they don't represent replaceable game-asset textures).
      if (image->getHash() == 0
          && !(desc->BindFlags & (D3D11_BIND_RENDER_TARGET | D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_UNORDERED_ACCESS))) {
        const VkExtent3D ext0 = pTexture->MipLevelExtent(0);

        // HR patch: Patch 13 — classify non-albedo PBR inputs and skip hashing them.
        // Leaving hash=0 makes the SRV ctor (Edit B2) skip ImGUI::AddTexture, so these
        // textures never enter the Remix asset catalogue and the material slots fall
        // through to albedo-only path-traced shading. — see CHANGELOG.md 2026-04-24
        NonAlbedoReason skipReason = NonAlbedoReason::None;
        if (IsHrSkipNonAlbedoEnabled()) {
          skipReason = ClassifyNonAlbedo(image->info().format, ext0, pInitialData[0]);
        }

        if (skipReason != NonAlbedoReason::None) {
          static uint32_t s_skipLogCount = 0;
          if (s_skipLogCount < 50) {
            ++s_skipLogCount;
            Logger::info(str::format(
              "[HR-TexSkip] reason=", NonAlbedoReasonStr(skipReason),
              " fmt=", uint32_t(image->info().format),
              " extent=", ext0.width, "x", ext0.height, "x", ext0.depth));
          }
          // Intentionally leave hash at 0 — Edit B2 sees this and skips AddTexture.
        } else {
          // Source-data byte size: prefer the slice pitch (game-supplied 2D-slice byte count);
          // fall back to row pitch × height for legacy callers that leave slice pitch zero.
          const uint64_t srcSize = uint64_t(pInitialData[0].SysMemSlicePitch != 0
            ? pInitialData[0].SysMemSlicePitch
            : uint64_t(pInitialData[0].SysMemPitch) * std::max(1u, ext0.height));
          if (srcSize > 0) {
            XXH64_hash_t hash = XXH3_64bits(pInitialData[0].pSysMem, srcSize);
            image->setHash(hash);
            static uint32_t s_logCount = 0;
            if (s_logCount < 50) {
              ++s_logCount;
              Logger::info(str::format(
                "[HR-TexHash] kind=Sampled hash=0x", std::hex, hash, std::dec,
                " size=", srcSize,
                " extent=", ext0.width, "x", ext0.height, "x", ext0.depth,
                " format=", image->info().format));
            }
          }
        }
      }

      // pInitialData is an array that stores an entry for
      // every single subresource. Since we will define all
      // subresources, this counts as initialization.
      for (uint32_t layer = 0; layer < desc->ArraySize; layer++) {
        for (uint32_t level = 0; level < desc->MipLevels; level++) {
          const uint32_t id = D3D11CalcSubresource(
            level, layer, desc->MipLevels);

          VkOffset3D mipLevelOffset = { 0, 0, 0 };
          VkExtent3D mipLevelExtent = pTexture->MipLevelExtent(level);

          if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_STAGING) {
            m_transferCommands += 1;
            m_transferMemory   += pTexture->GetSubresourceLayout(formatInfo->aspectMask, id).Size;
            
            VkImageSubresourceLayers subresourceLayers;
            subresourceLayers.aspectMask     = formatInfo->aspectMask;
            subresourceLayers.mipLevel       = level;
            subresourceLayers.baseArrayLayer = layer;
            subresourceLayers.layerCount     = 1;
            
            if (formatInfo->aspectMask != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
              m_context->uploadImage(
                image, subresourceLayers,
                pInitialData[id].pSysMem,
                pInitialData[id].SysMemPitch,
                pInitialData[id].SysMemSlicePitch);
            } else {
              m_context->updateDepthStencilImage(
                image, subresourceLayers,
                VkOffset2D { mipLevelOffset.x,     mipLevelOffset.y      },
                VkExtent2D { mipLevelExtent.width, mipLevelExtent.height },
                pInitialData[id].pSysMem,
                pInitialData[id].SysMemPitch,
                pInitialData[id].SysMemSlicePitch,
                packedFormat);
            }
          }

          if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_NONE) {
            util::packImageData(pTexture->GetMappedBuffer(id)->mapPtr(0),
              pInitialData[id].pSysMem, pInitialData[id].SysMemPitch, pInitialData[id].SysMemSlicePitch,
              0, 0, pTexture->GetVkImageType(), mipLevelExtent, 1, formatInfo, formatInfo->aspectMask);
          }
        }
      }
    } else {
      if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_STAGING) {
        m_transferCommands += 1;
        
        // While the Microsoft docs state that resource contents are
        // undefined if no initial data is provided, some applications
        // expect a resource to be pre-cleared. We can only do that
        // for non-compressed images, but that should be fine.
        VkImageSubresourceRange subresources;
        subresources.aspectMask     = formatInfo->aspectMask;
        subresources.baseMipLevel   = 0;
        subresources.levelCount     = desc->MipLevels;
        subresources.baseArrayLayer = 0;
        subresources.layerCount     = desc->ArraySize;

        if (formatInfo->flags.any(DxvkFormatFlag::BlockCompressed, DxvkFormatFlag::MultiPlane)) {
          m_context->clearCompressedColorImage(image, subresources);
        } else {
          if (subresources.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT) {
            VkClearColorValue value = { };

            m_context->clearColorImage(
              image, value, subresources);
          } else {
            VkClearDepthStencilValue value;
            value.depth   = 0.0f;
            value.stencil = 0;
            
            m_context->clearDepthStencilImage(
              image, value, subresources);
          }
        }
      }

      if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_NONE) {
        for (uint32_t i = 0; i < pTexture->CountSubresources(); i++) {
          auto buffer = pTexture->GetMappedBuffer(i);
          std::memset(buffer->mapPtr(0), 0, buffer->info().size);
        }
      }
    }

    FlushImplicit();
  }


  void D3D11Initializer::InitHostVisibleTexture(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    Rc<DxvkImage> image = pTexture->GetImage();

    for (uint32_t layer = 0; layer < image->info().numLayers; layer++) {
      for (uint32_t level = 0; level < image->info().mipLevels; level++) {
        VkImageSubresource subresource;
        subresource.aspectMask = image->formatInfo()->aspectMask;
        subresource.mipLevel   = level;
        subresource.arrayLayer = layer;

        VkExtent3D blockCount = util::computeBlockCount(
          image->mipLevelExtent(level),
          image->formatInfo()->blockSize);

        VkSubresourceLayout layout = image->querySubresourceLayout(subresource);

        auto initialData = pInitialData
          ? &pInitialData[D3D11CalcSubresource(level, layer, image->info().mipLevels)]
          : nullptr;

        for (uint32_t z = 0; z < blockCount.depth; z++) {
          for (uint32_t y = 0; y < blockCount.height; y++) {
            auto size = blockCount.width * image->formatInfo()->elementSize;
            auto dst = image->mapPtr(layout.offset + y * layout.rowPitch + z * layout.depthPitch);

            if (initialData) {
              auto src = reinterpret_cast<const char*>(initialData->pSysMem)
                       + y * initialData->SysMemPitch
                       + z * initialData->SysMemSlicePitch;
              std::memcpy(dst, src, size);
            } else {
              std::memset(dst, 0, size);
            }
          }
        }
      }
    }

    // Initialize the image on the GPU
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    VkImageSubresourceRange subresources = image->getAvailableSubresources();
    
    m_context->initImage(image, subresources, VK_IMAGE_LAYOUT_PREINITIALIZED);

    m_transferCommands += 1;
    FlushImplicit();
  }


  void D3D11Initializer::FlushImplicit() {
    if (m_transferCommands > MaxTransferCommands
     || m_transferMemory   > MaxTransferMemory)
      FlushInternal();
  }


  void D3D11Initializer::FlushInternal() {
    m_context->flushCommandList();
    
    m_transferCommands = 0;
    m_transferMemory   = 0;
  }

}
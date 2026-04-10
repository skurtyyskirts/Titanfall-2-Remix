/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*/
#include "dxbc_spirv_cache.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "../util/log/log.h"
#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../dxvk/dxvk_limits.h"
#include "../dxvk/dxvk_pipelayout.h"
#include "../spirv/spirv_code_buffer.h"

namespace dxvk {

  // ---- On-disk format -------------------------------------------------------
  //
  // The cache is one file per shader.  The file name is the hex-encoded
  // XXH64 hash of the DXBC bytecode, with extension .dxvk-spv, living in a
  // per-exe subdirectory so two different games can't stomp each other's
  // caches.  Bump kCacheFormatVersion whenever ANY of the following changes
  // shape on disk:
  //
  //   * DxvkShader's constructor signature
  //   * DxvkResourceSlot layout
  //   * DxvkInterfaceSlots layout
  //   * DxvkShaderOptions (for the subset we store)
  //   * The DxbcCompiler output (any codegen change would invalidate all
  //     cached SPIR-V -- bump this even for minor changes)
  //
  // The cache loader aggressively rejects any file whose version/magic
  // don't match exactly.  The worst case is a full re-translation on the
  // next launch, which is exactly the status-quo cost we're trying to
  // eliminate going forward.
  //
  // File layout (all little-endian):
  //
  //   u32 magic              = 'DXSP'
  //   u32 version            = kCacheFormatVersion
  //   u64 dxbcBytecodeHash   (self-check against caller hash)
  //   u32 shaderStage        (VkShaderStageFlagBits)
  //   u32 slotCount
  //   {   u32 slot
  //       u32 type            (VkDescriptorType)
  //       u32 view            (VkImageViewType)
  //       u32 access          (VkAccessFlags)
  //       u32 count
  //       u32 flags           (VkDescriptorBindingFlags)
  //   } * slotCount
  //   u32 iface.inputSlots
  //   u32 iface.outputSlots
  //   u32 iface.pushConstOffset
  //   u32 iface.pushConstSize
  //   i32 options.rasterizedStream
  //   u32 options.xfbStrides[MaxNumXfbBuffers]   (= 4 entries)
  //   u32 constData dword count
  //   u32 constData[constData dword count]
  //   u32 spirv dword count
  //   u32 spirv[spirv dword count]
  //
  // Note: DxvkShaderOptions::extraLayouts is deliberately NOT serialized.
  // That field only contains runtime VkDescriptorSetLayout handles which
  // have no stable representation on disk anyway, and it is always empty
  // for shaders produced by the DXBC front-end translator (it's used by
  // hand-authored Remix RTX shaders, which never reach this cache path).

  namespace {
    constexpr uint32_t kCacheMagic         = 0x50534844; // 'DXSP' little-endian
    // NV-DXVK: Version history:
    //   v1 - initial; used stringstream for SPIR-V (corrupt on MSVC)
    //   v2 - switched to decompressCode() for SPIR-V serialization
    //   v3 - fixed DxvkShaderConstData(0, nullptr) producing non-null
    //        data pointer with zero size -> divide-by-zero in DxvkBuffer
    //        constructor (dxvk_buffer.cpp:52) when d3d11_shader.cpp
    //        creates a constant buffer for a shader with no const data.
    //        Fix: use default constructor when constDwords == 0.
    constexpr uint32_t kCacheFormatVersion = 3;
    constexpr char     kCacheExt[]         = ".dxvk-spv";

    // POD helpers that write/read little-endian scalars to an in-memory blob.
    // The blob is built in memory so we can write atomically via a single
    // fwrite + rename(), avoiding partial-file corruption from a crash or
    // worker thread exit mid-write.
    struct Blob {
      std::vector<uint8_t> bytes;
      void put32(uint32_t v) {
        const uint8_t b[4] = { uint8_t(v), uint8_t(v >> 8),
                               uint8_t(v >> 16), uint8_t(v >> 24) };
        bytes.insert(bytes.end(), b, b + 4);
      }
      void put64(uint64_t v) {
        put32(uint32_t(v));
        put32(uint32_t(v >> 32));
      }
      void putRaw(const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + size);
      }
    };

    struct Reader {
      const uint8_t* p;
      const uint8_t* end;
      bool ok = true;
      bool need(size_t n) {
        if (size_t(end - p) < n) { ok = false; return false; }
        return true;
      }
      uint32_t get32() {
        if (!need(4)) return 0;
        uint32_t v = uint32_t(p[0])
                   | (uint32_t(p[1]) << 8)
                   | (uint32_t(p[2]) << 16)
                   | (uint32_t(p[3]) << 24);
        p += 4;
        return v;
      }
      uint64_t get64() {
        uint64_t lo = get32();
        uint64_t hi = get32();
        return lo | (hi << 32);
      }
      void getRaw(void* dst, size_t n) {
        if (!need(n)) return;
        std::memcpy(dst, p, n);
        p += n;
      }
    };

    // Render a 64-bit hash as a zero-padded 16-character lowercase hex
    // string.  Used as both the cache filename and part of the atomic-
    // rename temp file name.
    std::string hashToHex(XXH64_hash_t hash) {
      char buf[17];
      std::snprintf(buf, sizeof(buf), "%016llx",
                    static_cast<unsigned long long>(hash));
      return std::string(buf);
    }
  }

  // ---- Singleton + directory discovery --------------------------------------

  DxbcSpirvCache& DxbcSpirvCache::get() {
    // Magic static -> C++11 thread-safe initialization.  The constructor
    // is cheap (just string assembly + one mkdir), so no danger running
    // it under the first-lookup lock.
    static DxbcSpirvCache instance;
    return instance;
  }


  DxbcSpirvCache::DxbcSpirvCache() {
    // NV-DXVK: kill-switch.  Set DXVK_DISABLE_SPIRV_CACHE=1 to bypass
    // the cache entirely without rebuilding (e.g. for debugging).
    if (env::getEnvVar("DXVK_DISABLE_SPIRV_CACHE") == "1") {
      Logger::warn("DxbcSpirvCache: Disabled via DXVK_DISABLE_SPIRV_CACHE=1");
      m_enabled = false;
      return;
    }

    // Mirror DxvkStateCache's directory discovery: honour
    // DXVK_STATE_CACHE_PATH if set, otherwise the game exe's directory.
    // This keeps the SPIR-V cache next to the existing *.dxvk-cache file
    // so the two are lifetime-matched.
    std::string baseDir = env::getEnvVar("DXVK_STATE_CACHE_PATH");
    if (baseDir.empty()) {
      // Fall back to the directory containing the currently-running exe.
      try {
        auto exePath = std::filesystem::path(env::getExePath());
        baseDir = exePath.parent_path().string();
      } catch (...) {
        baseDir.clear();
      }
    }

    if (baseDir.empty()) {
      // No valid location -> disable silently, the cache is optional.
      m_enabled = false;
      return;
    }

    const std::string exeName = env::getExeBaseName();
    // <baseDir>/<exeName>.dxvk-spirv-cache/
    m_cacheDir = baseDir;
    if (!m_cacheDir.empty() && m_cacheDir.back() != '/' && m_cacheDir.back() != '\\')
      m_cacheDir += '/';
    m_cacheDir += exeName;
    m_cacheDir += ".dxvk-spirv-cache/";

    std::error_code ec;
    std::filesystem::create_directories(m_cacheDir, ec);
    if (ec) {
      Logger::warn(str::format(
          "DxbcSpirvCache: Failed to create cache directory '",
          m_cacheDir, "': ", ec.message(), ". SPIR-V cache disabled."));
      m_enabled = false;
      return;
    }

    m_enabled = true;
    Logger::info(str::format(
        "DxbcSpirvCache: Enabled, cache directory '", m_cacheDir, "'"));

    // NV-DXVK: Sweep up any leftover .tmp.* files from previous sessions.
    // store() writes to a temp file then renames atomically; if the process
    // crashed mid-write or the rename failed (e.g. because the destination
    // existed and the libstdc++ rename wrapper didn't pass the Windows
    // MOVEFILE_REPLACE_EXISTING flag) the temp file is left behind.  These
    // never get loaded by lookup() because pathForHash() generates a
    // .dxvk-spv extension only, but they accumulate and waste disk space,
    // and they're a confusing signal when poking at the cache by hand.
    // Sweeping them away on each init keeps the directory tidy.
    try {
      uint32_t cleaned = 0;
      for (const auto& entry : std::filesystem::directory_iterator(m_cacheDir)) {
        if (!entry.is_regular_file()) continue;
        const std::string filename = entry.path().filename().string();
        // Match anything containing ".tmp." in the filename.
        if (filename.find(".tmp.") != std::string::npos) {
          std::error_code ignore;
          std::filesystem::remove(entry.path(), ignore);
          ++cleaned;
        }
      }
      if (cleaned > 0) {
        Logger::info(str::format(
            "DxbcSpirvCache: Cleaned up ", cleaned,
            " leftover temp file(s) from previous sessions."));
      }
    } catch (...) {
      // Sweeping is best-effort; silent failure is fine.
    }
  }


  std::string DxbcSpirvCache::pathForHash(XXH64_hash_t hash) const {
    return m_cacheDir + hashToHex(hash) + kCacheExt;
  }


  // ---- Lookup ---------------------------------------------------------------

  Rc<DxvkShader> DxbcSpirvCache::lookup(XXH64_hash_t hash) {
    if (!m_enabled)
      return nullptr;

    m_lookupAttempts.fetch_add(1, std::memory_order_relaxed);

    // NV-DXVK helper: log a one-shot diagnostic the first time a given
    // failure reason fires, with the offending hash printed in hex so the
    // user can grep for the matching .dxvk-spv file in the cache directory.
    // We always increment the appropriate cumulative counter regardless of
    // whether we logged this time, so the running total stays accurate.
    auto logOnceWith = [&](std::atomic<bool>& latch, const char* reason,
                           const std::string& detail) {
      if (!latch.exchange(true)) {
        Logger::warn(str::format(
            "DxbcSpirvCache: lookup miss reason '", reason,
            "' first seen for hash 0x", hashToHex(hash),
            " (path: ", pathForHash(hash), ") - ", detail,
            ". Further misses with the same reason will be silently counted ",
            "and reported in the next attempt summary."));
      }
    };

    const std::string path = pathForHash(hash);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      // Clean miss: file doesn't exist yet.  This is the EXPECTED case for
      // shaders that haven't been cached yet, so we don't log it as an
      // error.  But we still count it as a miss to track hit rate.
      m_lookupMissesNoFile.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    // Read the whole file in one gulp -- per-shader files are small
    // (typically a few KB for the SPIR-V plus a few dozen bytes of metadata)
    // and reading as one blob avoids any repeated syscall overhead.
    f.seekg(0, std::ios::end);
    const std::streampos fsize = f.tellg();
    f.seekg(0, std::ios::beg);
    if (fsize <= 0) {
      logOnceWith(m_loggedShortRead, "empty-or-unseekable",
          str::format("file size = ", static_cast<int64_t>(fsize)));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    std::vector<uint8_t> buf(static_cast<size_t>(fsize));
    f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    if (!f) {
      logOnceWith(m_loggedShortRead, "short-read",
          str::format("requested ", buf.size(), " bytes, got ", f.gcount()));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    f.close();

    Reader r{ buf.data(), buf.data() + buf.size() };

    // Header
    const uint32_t magic   = r.get32();
    const uint32_t version = r.get32();
    const uint64_t dxbcH   = r.get64();
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-header",
          "file is too small to contain magic/version/hash header");
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (magic != kCacheMagic) {
      logOnceWith(m_loggedBadMagic, "bad-magic",
          str::format("expected 0x", std::hex, kCacheMagic,
                      " got 0x", std::hex, magic));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (version != kCacheFormatVersion) {
      logOnceWith(m_loggedBadVersion, "bad-version",
          str::format("expected v", kCacheFormatVersion, " got v", version,
                      ". This is the EXPECTED behaviour after a code change "
                      "that bumped kCacheFormatVersion -- the entry will be ",
                      "regenerated."));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (dxbcH != hash) {
      logOnceWith(m_loggedHashMismatch, "hash-mismatch",
          str::format("file says it belongs to hash 0x",
                      hashToHex(static_cast<XXH64_hash_t>(dxbcH)),
                      " but we asked for 0x", hashToHex(hash),
                      ". Indicates a hash collision OR a stale file from a ",
                      "different game with the same exe-base name."));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    // Shader stage
    const uint32_t stage = r.get32();

    // Resource slots
    const uint32_t slotCount = r.get32();
    // Sanity check: reject absurd slot counts before allocating.  Vulkan
    // caps descriptor sets at a few thousand bindings total; anything with
    // more than 4096 in a single DXBC shader is definitely corruption.
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-slot-count",
          "file ended before slotCount field");
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (slotCount > 4096) {
      logOnceWith(m_loggedSlotCountSane, "slot-count-insane",
          str::format("slotCount=", slotCount,
                      " exceeds sanity cap of 4096 -> file is corrupt"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    std::vector<DxvkResourceSlot> slots(slotCount);
    for (uint32_t i = 0; i < slotCount; i++) {
      DxvkResourceSlot& s = slots[i];
      s.slot   = r.get32();
      s.type   = static_cast<VkDescriptorType>(r.get32());
      s.view   = static_cast<VkImageViewType>(r.get32());
      s.access = r.get32();
      s.count  = r.get32();
      s.flags  = r.get32();
    }
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-slots",
          str::format("file ended while reading ", slotCount, " slots"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    // Interface slots
    DxvkInterfaceSlots iface;
    iface.inputSlots      = r.get32();
    iface.outputSlots     = r.get32();
    iface.pushConstOffset = r.get32();
    iface.pushConstSize   = r.get32();

    // Shader options (everything except extraLayouts, which is reconstructed
    // as empty -- see the note in the header for why that's correct).
    DxvkShaderOptions options;
    options.rasterizedStream = static_cast<int32_t>(r.get32());
    for (uint32_t i = 0; i < MaxNumXfbBuffers; i++)
      options.xfbStrides[i] = r.get32();
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-iface-or-options",
          "file ended while reading interface/options");
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    // Const data (dword count + dwords).
    const uint32_t constDwords = r.get32();
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-const-count",
          "file ended before constDwords field");
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (constDwords > (1u << 20)) { // sanity: <= 4 MB
      logOnceWith(m_loggedConstSane, "const-dwords-insane",
          str::format("constDwords=", constDwords,
                      " exceeds sanity cap of ", (1u << 20),
                      " (4 MB worth) -> file is corrupt"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    std::vector<uint32_t> constDwordsVec(constDwords);
    for (uint32_t i = 0; i < constDwords; i++)
      constDwordsVec[i] = r.get32();
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-const-data",
          str::format("file ended while reading ", constDwords, " constData dwords"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    // NV-DXVK: MUST use the default constructor (which sets m_data=nullptr
    // and m_size=0) when there's no const data.  The sized constructor
    // DxvkShaderConstData(0, nullptr) calls `new uint32_t[0]` which
    // returns a NON-NULL pointer per the C++ spec, and downstream code in
    // d3d11_shader.cpp checks `data() != nullptr` to decide whether to
    // create a constant buffer.  A non-null pointer with sizeInBytes()==0
    // creates a zero-size DxvkBuffer, which divides by zero computing
    // m_physSliceStride at dxvk_buffer.cpp:52 → instant crash.
    DxvkShaderConstData constData = constDwordsVec.empty()
        ? DxvkShaderConstData()
        : DxvkShaderConstData(constDwordsVec.size(), constDwordsVec.data());

    // SPIR-V (dword count + dwords).
    const uint32_t spvDwords = r.get32();
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-spv-count",
          "file ended before spvDwords field");
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (spvDwords > (1u << 22)) { // sanity: <= 16 MB of SPIR-V
      logOnceWith(m_loggedSpvSane, "spv-dwords-insane",
          str::format("spvDwords=", spvDwords,
                      " exceeds sanity cap of ", (1u << 22),
                      " (16 MB worth) -> file is corrupt"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    std::vector<uint32_t> spv(spvDwords);
    for (uint32_t i = 0; i < spvDwords; i++)
      spv[i] = r.get32();
    if (!r.ok) {
      logOnceWith(m_loggedTruncated, "truncated-spv",
          str::format("file ended while reading ", spvDwords, " SPIR-V dwords"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    SpirvCodeBuffer code(spvDwords, spv.data());

    try {
      Rc<DxvkShader> shader = new DxvkShader(
          static_cast<VkShaderStageFlagBits>(stage),
          slotCount,
          slots.empty() ? nullptr : slots.data(),
          iface,
          std::move(code),
          options,
          std::move(constData));
      m_lookupHits.fetch_add(1, std::memory_order_relaxed);
      return shader;
    } catch (const std::exception& e) {
      logOnceWith(m_loggedCtorException, "ctor-exception",
          str::format("DxvkShader constructor threw '", e.what(),
                      "' on cached SPIR-V (likely malformed bytecode in the ",
                      "cached file)"));
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    } catch (...) {
      logOnceWith(m_loggedCtorException, "ctor-unknown-exception",
          "DxvkShader constructor threw an unknown exception type "
          "(likely malformed bytecode in the cached file)");
      m_lookupMissesCorrupt.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
  }


  // ---- Store ----------------------------------------------------------------

  void DxbcSpirvCache::store(XXH64_hash_t hash, const Rc<DxvkShader>& shader) {
    if (!m_enabled || shader == nullptr)
      return;

    // Build the blob in memory first so we can do a single atomic write +
    // rename at the end.  If the process crashes mid-write we want to leave
    // either the previous valid file or nothing at all -- never a half-
    // written file that another thread/launch would try to parse.
    Blob blob;
    blob.bytes.reserve(16 * 1024); // typical shader is 4-12KB

    // Header
    blob.put32(kCacheMagic);
    blob.put32(kCacheFormatVersion);
    blob.put64(static_cast<uint64_t>(hash));

    // Stage
    blob.put32(static_cast<uint32_t>(shader->stage()));

    // Resource slots.  DxvkShader::resourceSlots() (added by this change)
    // gives us the full array that the original DxbcCompiler::finalize()
    // wrote into the shader.  Without these, a cached shader reconstructed
    // on the next launch would get slotCount=0 and its resulting Vulkan
    // descriptor layout would mismatch the bindings the game expects ->
    // completely broken rendering.  Walk the array and serialize each
    // entry as six uint32s, matching the on-disk layout documented at
    // the top of this file.
    const auto& slots = shader->resourceSlots();
    const uint32_t slotCount = static_cast<uint32_t>(slots.size());
    blob.put32(slotCount);
    for (uint32_t i = 0; i < slotCount; i++) {
      const DxvkResourceSlot& s = slots[i];
      blob.put32(s.slot);
      blob.put32(static_cast<uint32_t>(s.type));
      blob.put32(static_cast<uint32_t>(s.view));
      blob.put32(static_cast<uint32_t>(s.access));
      blob.put32(s.count);
      blob.put32(static_cast<uint32_t>(s.flags));
    }

    // Interface slots
    const DxvkInterfaceSlots iface = shader->interfaceSlots();
    blob.put32(iface.inputSlots);
    blob.put32(iface.outputSlots);
    blob.put32(iface.pushConstOffset);
    blob.put32(iface.pushConstSize);

    // Shader options (minus extraLayouts)
    const DxvkShaderOptions options = shader->shaderOptions();
    blob.put32(static_cast<uint32_t>(options.rasterizedStream));
    for (uint32_t i = 0; i < MaxNumXfbBuffers; i++)
      blob.put32(options.xfbStrides[i]);

    // Const data
    const DxvkShaderConstData& constData = shader->shaderConstants();
    const uint32_t constDwords =
        static_cast<uint32_t>(constData.sizeInBytes() / sizeof(uint32_t));
    blob.put32(constDwords);
    blob.putRaw(constData.data(), constData.sizeInBytes());

    // SPIR-V bytecode -- this is the payload that actually saves us the
    // multi-minute DxbcCompiler run on subsequent launches.
    // NV-DXVK: Use decompressCode() to get the raw SPIR-V dwords directly
    // instead of going through dump(ostream) + stringstream.  The previous
    // stringstream path was the likely cause of the cache-load crash:
    // std::stringstream's str() on MSVC can silently truncate or mishandle
    // binary data containing embedded nulls or large blocks, and the
    // resulting file would pass our header checks but contain garbage
    // SPIR-V that crashes the DxvkShader constructor's instruction
    // iterator on the next launch.  The direct path is also faster (no
    // string copy, no stream overhead).
    SpirvCodeBuffer spvCode = shader->decompressCode();
    const uint32_t spvDwords = spvCode.dwords();
    blob.put32(spvDwords);
    blob.putRaw(spvCode.data(), spvDwords * sizeof(uint32_t));

    // ---- Atomic write: write to temp file, then rename into place ----
    //
    // Two different threads translating the same shader (which can happen
    // if Source's job system dispatches the same CreateVertexShader call to
    // two workers before the first has finished, or if the cache is shared
    // across two games running at once) will both win-the-rename-race with
    // byte-identical content, so the file left on disk is always valid.
    const std::string finalPath = pathForHash(hash);
    const std::string tempPath  = finalPath + ".tmp." + hashToHex(
        static_cast<XXH64_hash_t>(reinterpret_cast<uintptr_t>(&blob)));

    {
      std::ofstream out(tempPath, std::ios::binary | std::ios::trunc);
      if (!out.is_open()) {
        if (!m_ioErrorLogged.exchange(true))
          Logger::warn(str::format(
              "DxbcSpirvCache: store() failed to open temp file '", tempPath,
              "' for write (errno=", errno, "). Cache writes will be ",
              "skipped this session. Further I/O errors are suppressed."));
        return;
      }
      out.write(reinterpret_cast<const char*>(blob.bytes.data()),
                blob.bytes.size());
      if (!out) {
        if (!m_ioErrorLogged.exchange(true))
          Logger::warn(str::format(
              "DxbcSpirvCache: store() short-write to '", tempPath,
              "' (errno=", errno, "). Cache writes will be skipped this session."));
        out.close();
        std::error_code ignore;
        std::filesystem::remove(tempPath, ignore);
        return;
      }
    }

    std::error_code ec;
    std::filesystem::rename(tempPath, finalPath, ec);
    if (ec) {
      // On Windows, rename over an existing file fails.  std::filesystem's
      // rename on Windows wraps MoveFileEx which does allow replacing IF we
      // ask for it, but libstdc++'s wrapper historically didn't pass the
      // MOVEFILE_REPLACE_EXISTING flag.  Fall back to copy-then-delete
      // on the slow path; the benign race is that two threads both writing
      // the same file will briefly flicker, but content is identical.
      const std::string firstErr = ec.message();
      std::error_code ec2;
      std::filesystem::copy_file(tempPath, finalPath,
          std::filesystem::copy_options::overwrite_existing, ec2);
      std::error_code ignore;
      std::filesystem::remove(tempPath, ignore);
      if (ec2 && !m_ioErrorLogged.exchange(true)) {
        Logger::warn(str::format(
            "DxbcSpirvCache: store() rename '", tempPath, "' -> '",
            finalPath, "' failed: '", firstErr,
            "', then copy_file fallback also failed: '", ec2.message(),
            "'. Cache writes will be skipped this session."));
      }
    }
  }

}

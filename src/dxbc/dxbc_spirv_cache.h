/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*/
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "../dxvk/dxvk_shader.h"
#include "../util/xxHash/xxhash.h"

namespace dxvk {

  /**
   * \brief On-disk cache of DXBC -> DxvkShader (SPIR-V + layout) translations.
   *
   * DXVK's existing state cache (DxvkStateCache / *.dxvk-cache) stores only
   * the back-end (Vulkan pipeline state keyed on shader-hash + render state).
   * Every launch still has to re-run DxbcModule::compile() on every game
   * shader that gets used, which for Source-engine titles is the
   * dominant cost of the first several minutes of loading -- tens of
   * thousands of shader variants, each running the full DxbcCompiler to
   * parse DXBC, analyze control flow, and emit SPIR-V, on whichever thread
   * happened to call ID3D11Device::CreateVertexShader / CreatePixelShader.
   *
   * This cache makes that work idempotent across launches by writing out a
   * small binary file per shader (keyed on a hash of the raw DXBC bytecode)
   * the first time it's translated.  On subsequent launches DxbcModule::
   * compile() tries the cache first and, on hit, reconstructs a complete
   * DxvkShader with no further DXBC processing.
   *
   * Design constraints the implementation respects:
   *
   *   - **Thread safety without locks.**  DxbcModule::compile() is called
   *     from many game threads concurrently (Source dispatches shader
   *     creation to its job pool).  Using one file per shader means two
   *     threads translating different shaders never touch the same path,
   *     and two threads racing on the same shader both write correct
   *     content via an atomic rename-into-place.
   *
   *   - **Auto-invalidation on translator changes.**  The header carries
   *     a format version that this code bumps whenever the DxvkShader
   *     layout changes.  A cache written by an older build is ignored on
   *     read.  Bumping the build script "clean" path can also wipe the
   *     whole directory if desired (rm of <game>.dxvk-spirv-cache/).
   *
   *   - **No new library dependencies.**  The cache lives in libdxbc.a
   *     alongside the compiler itself and only uses headers that libdxbc
   *     already pulls in (DxvkShader is exposed via dxvk_shader.h which
   *     dxbc_module.h already includes).
   *
   *   - **Reconstruction uses only the public DxvkShader constructor.**
   *     There is no friendship / header poking -- a cached DxvkShader is
   *     produced by running the exact same `new DxvkShader(stage, slotCount,
   *     slotInfos, iface, code, options, std::move(constData))` that the
   *     translator itself would have called, just with the parameters read
   *     from disk.
   */
  class DxbcSpirvCache {
  public:
    /// Get the process-wide singleton.  Lazy-initialized; first use picks
    /// the cache directory based on the current exe's location.
    static DxbcSpirvCache& get();

    /// Compute a stable hash of a DXBC bytecode blob to use as a cache key.
    /// Uses xxhash64 to match the hashing style the rest of DXVK/Remix uses.
    static XXH64_hash_t hashBytecode(const void* data, size_t size) {
      return XXH64(data, size, 0);
    }

    /// Try to load a previously cached DxvkShader for \a hash.
    /// Returns nullptr on miss, corruption, version mismatch, or any I/O
    /// error -- the caller then runs the real DxbcModule::compile() path
    /// and feeds the result back to \ref store.
    Rc<DxvkShader> lookup(XXH64_hash_t hash);

    /// Persist \a shader to disk under \a hash.  Failures (disk full,
    /// permission denied, cache directory not writable) are logged once and
    /// then silently ignored -- the in-memory shader is already valid, the
    /// cache is purely an optimization.
    void store(XXH64_hash_t hash, const Rc<DxvkShader>& shader);

    /// Returns true if this cache session should skip disk I/O entirely
    /// (e.g. directory couldn't be created at startup).  When disabled,
    /// \ref lookup always returns nullptr and \ref store is a no-op.
    bool isEnabled() const { return m_enabled; }

  private:
    DxbcSpirvCache();

    /// Full path of the cache file for \a hash inside \ref m_cacheDir.
    std::string pathForHash(XXH64_hash_t hash) const;

    std::string m_cacheDir;            ///< Absolute directory path, trailing slash
    bool        m_enabled = false;     ///< Disk I/O gate
    std::atomic<bool> m_ioErrorLogged{ false }; ///< One-shot log-failure latch

    // NV-DXVK: Per-failure-reason one-shot log latches plus a global hit
    // counter, used by lookup() to tell us exactly WHY shader cache loads
    // are failing without flooding the log with thousands of identical
    // messages.  Each latch flips false->true on first occurrence and the
    // matching log line is emitted; subsequent failures of the same kind
    // increment a counter that's reported in the next log line.  This gives
    // us "first time" diagnostic detail and a running total without spam.
    std::atomic<bool>     m_loggedOpenFail{ false };
    std::atomic<bool>     m_loggedShortRead{ false };
    std::atomic<bool>     m_loggedBadMagic{ false };
    std::atomic<bool>     m_loggedBadVersion{ false };
    std::atomic<bool>     m_loggedHashMismatch{ false };
    std::atomic<bool>     m_loggedTruncated{ false };
    std::atomic<bool>     m_loggedSlotCountSane{ false };
    std::atomic<bool>     m_loggedConstSane{ false };
    std::atomic<bool>     m_loggedSpvSane{ false };
    std::atomic<bool>     m_loggedCtorException{ false };
    std::atomic<uint64_t> m_lookupAttempts{ 0 };
    std::atomic<uint64_t> m_lookupHits{ 0 };
    std::atomic<uint64_t> m_lookupMissesNoFile{ 0 };
    std::atomic<uint64_t> m_lookupMissesCorrupt{ 0 };

  public:
    /// Get cumulative lookup hit/miss totals for the per-frame HUD or
    /// final shutdown summary.  Cheap atomic loads, safe to call from
    /// anywhere.
    uint64_t getLookupAttempts()      const { return m_lookupAttempts.load(); }
    uint64_t getLookupHits()          const { return m_lookupHits.load(); }
    uint64_t getLookupMissesNoFile()  const { return m_lookupMissesNoFile.load(); }
    uint64_t getLookupMissesCorrupt() const { return m_lookupMissesCorrupt.load(); }
  };

}

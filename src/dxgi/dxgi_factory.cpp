#include "dxgi_factory.h"
#include "dxgi_swapchain.h"
#include "dxgi_swapchain_dispatcher.h"

namespace dxvk {

  DxgiFactory::DxgiFactory(UINT Flags)
  : m_instance    (new DxvkInstance()),
    m_monitorInfo (this),
    m_options     (m_instance->config()),
    m_flags       (Flags) {
    for (uint32_t i = 0; m_instance->enumAdapters(i) != nullptr; i++)
      m_instance->enumAdapters(i)->logAdapterInfo();
  }
  
  
  DxgiFactory::~DxgiFactory() {
    
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::QueryInterface(REFIID riid, void** ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;
    
    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(IDXGIObject)
     || riid == __uuidof(IDXGIFactory)
     || riid == __uuidof(IDXGIFactory1)
     || riid == __uuidof(IDXGIFactory2)
     || riid == __uuidof(IDXGIFactory3)
     || riid == __uuidof(IDXGIFactory4)
     || riid == __uuidof(IDXGIFactory5)
     || riid == __uuidof(IDXGIFactory6)
     || riid == __uuidof(IDXGIFactory7)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(IDXGIVkMonitorInfo)) {
      *ppvObject = ref(&m_monitorInfo);
      return S_OK;
    }
    
    Logger::warn("DxgiFactory::QueryInterface: Unknown interface query");
    Logger::warn(str::format(riid));
    return E_NOINTERFACE;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::GetParent(REFIID riid, void** ppParent) {
    InitReturnPtr(ppParent);
    
    Logger::warn("DxgiFactory::GetParent: Unknown interface query");
    return E_NOINTERFACE;
  }
  
  
  BOOL STDMETHODCALLTYPE DxgiFactory::IsWindowedStereoEnabled() {
    // We don't support Stereo 3D at the moment
    return FALSE;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::CreateSoftwareAdapter(
          HMODULE         Module,
          IDXGIAdapter**  ppAdapter) {
    InitReturnPtr(ppAdapter);
    
    if (ppAdapter == nullptr)
      return DXGI_ERROR_INVALID_CALL;
    
    Logger::err("DXGI: CreateSoftwareAdapter: Software adapters not supported");
    return DXGI_ERROR_UNSUPPORTED;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::CreateSwapChain(
          IUnknown*             pDevice,
          DXGI_SWAP_CHAIN_DESC* pDesc,
          IDXGISwapChain**      ppSwapChain) {
    // NV-DXVK: aggressive tracing for Titanfall 2 crash investigation.
    // The swap chain creation path is the last thing running before the
    // game null-derefs qword_1814EE258, so we trace every branch/argument.
    Logger::info(str::format(
      "[DXGI] DxgiFactory::CreateSwapChain entry"
      " pDevice=0x", std::hex, (uintptr_t)pDevice,
      " pDesc=0x", (uintptr_t)pDesc,
      " ppSwapChain=0x", (uintptr_t)ppSwapChain, std::dec));
    if (pDesc) {
      Logger::info(str::format(
        "[DXGI]   BufferDesc ", pDesc->BufferDesc.Width, "x", pDesc->BufferDesc.Height,
        " fmt=", pDesc->BufferDesc.Format,
        " refresh=", pDesc->BufferDesc.RefreshRate.Numerator, "/", pDesc->BufferDesc.RefreshRate.Denominator,
        " scanline=", pDesc->BufferDesc.ScanlineOrdering,
        " scaling=", pDesc->BufferDesc.Scaling));
      Logger::info(str::format(
        "[DXGI]   Sample {", pDesc->SampleDesc.Count, ",", pDesc->SampleDesc.Quality, "}",
        " usage=0x", std::hex, pDesc->BufferUsage, std::dec,
        " bufcount=", pDesc->BufferCount,
        " hwnd=0x", std::hex, (uintptr_t)pDesc->OutputWindow, std::dec,
        " windowed=", pDesc->Windowed,
        " effect=", pDesc->SwapEffect,
        " flags=0x", std::hex, pDesc->Flags, std::dec));
    }
    if (ppSwapChain == nullptr || pDesc == nullptr || pDevice == nullptr) {
      Logger::err("[DXGI] DxgiFactory::CreateSwapChain: null arg -> DXGI_ERROR_INVALID_CALL");
      return DXGI_ERROR_INVALID_CALL;
    }

    DXGI_SWAP_CHAIN_DESC1 desc;
    desc.Width              = pDesc->BufferDesc.Width;
    desc.Height             = pDesc->BufferDesc.Height;
    desc.Format             = pDesc->BufferDesc.Format;
    desc.Stereo             = FALSE;
    desc.SampleDesc         = pDesc->SampleDesc;
    desc.BufferUsage        = pDesc->BufferUsage;
    desc.BufferCount        = pDesc->BufferCount;
    desc.Scaling            = DXGI_SCALING_STRETCH;
    desc.SwapEffect         = pDesc->SwapEffect;
    desc.AlphaMode          = DXGI_ALPHA_MODE_IGNORE;
    desc.Flags              = pDesc->Flags;
    
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC descFs;
    descFs.RefreshRate      = pDesc->BufferDesc.RefreshRate;
    descFs.ScanlineOrdering = pDesc->BufferDesc.ScanlineOrdering;
    descFs.Scaling          = pDesc->BufferDesc.Scaling;
    descFs.Windowed         = pDesc->Windowed;
    
    IDXGISwapChain1* swapChain = nullptr;
    HRESULT hr = CreateSwapChainForHwnd(
      pDevice, pDesc->OutputWindow,
      &desc, &descFs, nullptr,
      &swapChain);

    // NV-DXVK: trace the outcome of the forwarded call
    Logger::info(str::format(
      "[DXGI] DxgiFactory::CreateSwapChain -> CreateSwapChainForHwnd returned hr=0x",
      std::hex, (uint32_t)hr, std::dec,
      " swapChain=0x", std::hex, (uintptr_t)swapChain, std::dec));

    *ppSwapChain = swapChain;
    return hr;
  }


  HRESULT STDMETHODCALLTYPE DxgiFactory::CreateSwapChainForHwnd(
          IUnknown*             pDevice,
          HWND                  hWnd,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
          IDXGIOutput*          pRestrictToOutput,
          IDXGISwapChain1**     ppSwapChain) {
    // NV-DXVK: entry trace
    Logger::info(str::format(
      "[DXGI] DxgiFactory::CreateSwapChainForHwnd entry"
      " pDevice=0x", std::hex, (uintptr_t)pDevice,
      " hWnd=0x", (uintptr_t)hWnd,
      " pDesc=0x", (uintptr_t)pDesc, std::dec));
    InitReturnPtr(ppSwapChain);

    if (!ppSwapChain || !pDesc || !hWnd || !pDevice) {
      Logger::err(str::format(
        "[DXGI] CreateSwapChainForHwnd: null arg -> DXGI_ERROR_INVALID_CALL",
        " ppSwapChain=", (ppSwapChain != nullptr),
        " pDesc=", (pDesc != nullptr),
        " hWnd=", (hWnd != nullptr),
        " pDevice=", (pDevice != nullptr)));
      return DXGI_ERROR_INVALID_CALL;
    }

    Com<IWineDXGISwapChainFactory> wineDevice;

    HRESULT qiHr = pDevice->QueryInterface(
      __uuidof(IWineDXGISwapChainFactory),
      reinterpret_cast<void**>(&wineDevice));
    Logger::info(str::format(
      "[DXGI] CreateSwapChainForHwnd: QI(IWineDXGISwapChainFactory) hr=0x",
      std::hex, (uint32_t)qiHr, std::dec,
      " wineDevice=0x", std::hex, (uintptr_t)wineDevice.ptr(), std::dec));

    if (SUCCEEDED(qiHr)) {
      IDXGISwapChain4* frontendSwapChain = nullptr;

      Logger::info("[DXGI] CreateSwapChainForHwnd: calling wineDevice->CreateSwapChainForHwnd");
      HRESULT hr = wineDevice->CreateSwapChainForHwnd(
        this, hWnd, pDesc, pFullscreenDesc,
        pRestrictToOutput, reinterpret_cast<IDXGISwapChain1**>(&frontendSwapChain));
      Logger::info(str::format(
        "[DXGI] CreateSwapChainForHwnd: wineDevice returned hr=0x",
        std::hex, (uint32_t)hr, std::dec,
        " frontend=0x", std::hex, (uintptr_t)frontendSwapChain, std::dec));

      // No ref as that's handled by the object we're wrapping
      // which was ref'ed on creation.
      if (SUCCEEDED(hr))
        *ppSwapChain = new DxgiSwapChainDispatcher(frontendSwapChain);

      return hr;
    }

    Logger::err("DXGI: CreateSwapChainForHwnd: Unsupported device type");
    return DXGI_ERROR_UNSUPPORTED;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::CreateSwapChainForCoreWindow(
          IUnknown*             pDevice,
          IUnknown*             pWindow,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
          IDXGIOutput*          pRestrictToOutput,
          IDXGISwapChain1**     ppSwapChain) {
    InitReturnPtr(ppSwapChain);
    
    Logger::err("DxgiFactory::CreateSwapChainForCoreWindow: Not implemented");
    return E_NOTIMPL;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::CreateSwapChainForComposition(
          IUnknown*             pDevice,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
          IDXGIOutput*          pRestrictToOutput,
          IDXGISwapChain1**     ppSwapChain) {
    InitReturnPtr(ppSwapChain);
    
    Logger::err("DxgiFactory::CreateSwapChainForComposition: Not implemented");
    return E_NOTIMPL;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::EnumAdapters(
          UINT            Adapter,
          IDXGIAdapter**  ppAdapter) {
    InitReturnPtr(ppAdapter);
    
    if (ppAdapter == nullptr)
      return DXGI_ERROR_INVALID_CALL;
    
    IDXGIAdapter1* handle = nullptr;
    HRESULT hr = this->EnumAdapters1(Adapter, &handle);
    *ppAdapter = handle;
    return hr;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::EnumAdapters1(
          UINT            Adapter,
          IDXGIAdapter1** ppAdapter) {
    InitReturnPtr(ppAdapter);
    
    if (ppAdapter == nullptr)
      return DXGI_ERROR_INVALID_CALL;
    
    Rc<DxvkAdapter> dxvkAdapter
      = m_instance->enumAdapters(Adapter);
    
    if (dxvkAdapter == nullptr)
      return DXGI_ERROR_NOT_FOUND;
    
    *ppAdapter = ref(new DxgiAdapter(this, dxvkAdapter, Adapter));
    return S_OK;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::EnumAdapterByLuid(
          LUID                  AdapterLuid,
          REFIID                riid,
          void**                ppvAdapter) {
    InitReturnPtr(ppvAdapter);
    uint32_t adapterId = 0;

    while (true) {
      Com<IDXGIAdapter> adapter;
      HRESULT hr = EnumAdapters(adapterId++, &adapter);

      if (FAILED(hr))
        return hr;
      
      DXGI_ADAPTER_DESC desc;
      adapter->GetDesc(&desc);

      if (!std::memcmp(&AdapterLuid, &desc.AdapterLuid, sizeof(LUID)))
        return adapter->QueryInterface(riid, ppvAdapter);
    }

    // This should be unreachable
    return DXGI_ERROR_NOT_FOUND;
  }

  
  HRESULT STDMETHODCALLTYPE DxgiFactory::EnumAdapterByGpuPreference(
          UINT                  Adapter,
          DXGI_GPU_PREFERENCE   GpuPreference,
          REFIID                riid,
          void**                ppvAdapter) {
    InitReturnPtr(ppvAdapter);
    uint32_t adapterCount = m_instance->adapterCount();

    if (Adapter >= adapterCount)
      return DXGI_ERROR_NOT_FOUND;

    // We know that the backend lists dedicated GPUs before
    // any integrated ones, so just list adapters in reverse
    // order. We have no other way to estimate performance.
    if (GpuPreference == DXGI_GPU_PREFERENCE_MINIMUM_POWER)
      Adapter = adapterCount - Adapter - 1;

    Com<IDXGIAdapter> adapter;
    HRESULT hr = this->EnumAdapters(Adapter, &adapter);

    if (FAILED(hr))
      return hr;

    return adapter->QueryInterface(riid, ppvAdapter);
  }


  HRESULT STDMETHODCALLTYPE DxgiFactory::EnumWarpAdapter(
          REFIID                riid,
          void**                ppvAdapter) {
    InitReturnPtr(ppvAdapter);

    static bool s_errorShown = false;

    if (!std::exchange(s_errorShown, true))
      Logger::warn("DxgiFactory::EnumWarpAdapter: WARP not supported, returning first hardware adapter");

    Com<IDXGIAdapter1> adapter;
    HRESULT hr = EnumAdapters1(0, &adapter);

    if (FAILED(hr))
      return hr;

    return adapter->QueryInterface(riid, ppvAdapter);
  }


  HRESULT STDMETHODCALLTYPE DxgiFactory::GetWindowAssociation(HWND *pWindowHandle) {
    if (pWindowHandle == nullptr)
      return DXGI_ERROR_INVALID_CALL;
    
    *pWindowHandle = m_associatedWindow;
    return S_OK;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::GetSharedResourceAdapterLuid(
          HANDLE                hResource,
          LUID*                 pLuid) {
    Logger::err("DxgiFactory::GetSharedResourceAdapterLuid: Not implemented");
    return E_NOTIMPL;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::MakeWindowAssociation(HWND WindowHandle, UINT Flags) {
    Logger::warn("DXGI: MakeWindowAssociation: Ignoring flags");
    m_associatedWindow = WindowHandle;
    return S_OK;
  }
  
  
  BOOL STDMETHODCALLTYPE DxgiFactory::IsCurrent() {
    return TRUE;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::RegisterOcclusionStatusWindow(
          HWND                  WindowHandle,
          UINT                  wMsg,
          DWORD*                pdwCookie) {
    Logger::err("DxgiFactory::RegisterOcclusionStatusWindow: Not implemented");
    return E_NOTIMPL;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::RegisterStereoStatusEvent(
          HANDLE                hEvent,
          DWORD*                pdwCookie) {
    Logger::err("DxgiFactory::RegisterStereoStatusEvent: Not implemented");
    return E_NOTIMPL;
  }
  
  
  HRESULT STDMETHODCALLTYPE DxgiFactory::RegisterStereoStatusWindow(
          HWND                  WindowHandle,
          UINT                  wMsg,
          DWORD*                pdwCookie) {
    Logger::err("DxgiFactory::RegisterStereoStatusWindow: Not implemented");
    return E_NOTIMPL;
  }
  

  HRESULT STDMETHODCALLTYPE DxgiFactory::RegisterOcclusionStatusEvent(
          HANDLE                hEvent,
          DWORD*                pdwCookie) {
    Logger::err("DxgiFactory::RegisterOcclusionStatusEvent: Not implemented");
    return E_NOTIMPL;
  }
  

  void STDMETHODCALLTYPE DxgiFactory::UnregisterStereoStatus(
          DWORD                 dwCookie) {
    Logger::err("DxgiFactory::UnregisterStereoStatus: Not implemented");
  }
  
  
  void STDMETHODCALLTYPE DxgiFactory::UnregisterOcclusionStatus(
          DWORD                 dwCookie) {
    Logger::err("DxgiFactory::UnregisterOcclusionStatus: Not implemented");
  }


  UINT STDMETHODCALLTYPE DxgiFactory::GetCreationFlags() {
    return m_flags;
  }


  HRESULT STDMETHODCALLTYPE DxgiFactory::CheckFeatureSupport(
          DXGI_FEATURE          Feature,
          void*                 pFeatureSupportData,
          UINT                  FeatureSupportDataSize) {
    switch (Feature) {
      case DXGI_FEATURE_PRESENT_ALLOW_TEARING: {
        auto info = static_cast<BOOL*>(pFeatureSupportData);

        if (FeatureSupportDataSize != sizeof(*info))
          return E_INVALIDARG;
        
        *info = TRUE;
      } return S_OK;

      default:
        Logger::err(str::format("DxgiFactory: CheckFeatureSupport: Unknown feature: ", uint32_t(Feature)));
        return E_INVALIDARG;
    }
  }


  HRESULT STDMETHODCALLTYPE DxgiFactory::RegisterAdaptersChangedEvent(
          HANDLE                hEvent,
          DWORD*                pdwCookie) {
    Logger::err("DxgiFactory: RegisterAdaptersChangedEvent: Stub");
    return E_NOTIMPL;
  }


  HRESULT STDMETHODCALLTYPE DxgiFactory::UnregisterAdaptersChangedEvent(
          DWORD                 Cookie) {
    Logger::err("DxgiFactory: UnregisterAdaptersChangedEvent: Stub");
    return E_NOTIMPL;
  }


}

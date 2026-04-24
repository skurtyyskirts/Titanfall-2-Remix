@echo off
setlocal EnableDelayedExpansion

rem Adapted from build-remixDx11 - Copy.bat for the Heavy Rain target.
rem Differences from the Titanfall 2 script:
rem   * PROJECT_DIR / GAME_DIR point into this workspace
rem   * GAME_RUNTIME_SUBDIR is empty because HeavyRain.exe sits at the root
rem     of the game folder (no bin\x64_retail\ equivalent).  Everything is
rem     deployed flat, per README.md:121-126 "Standard" deployment.
rem   * taskkill targets HeavyRain.exe instead of Titanfall2*.exe
rem Everything else (meson flags, shader cache preservation, log routing)
rem is intentionally unchanged.

echo #############################################################
echo # Setting up Visual Studio x64 Build Environment...         #
echo #############################################################
echo.

rem This box has VS 2026 Community (internal version "18") at the path below.
rem The upstream README targets VS 2019/2022; VS 2026's MSVC toolchain is
rem backward-compatible and expected to build dxvk-remix without changes.
rem If that turns out to be wrong, revert to VS 2022 Community and point
rem VS_SETUP_SCRIPT at the 2022 path.
set "VS_SETUP_SCRIPT=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VS_SETUP_SCRIPT%" (
    echo ERROR: Visual Studio setup script not found at:
    echo %VS_SETUP_SCRIPT%
    echo Please verify your Visual Studio Community installation path.
    goto :error_exit
)

call "%VS_SETUP_SCRIPT%" x64
if errorlevel 1 (
    echo ERROR: Failed to initialize the Visual Studio 2022 command prompt environment.
    goto :error_exit
)

echo.
echo #############################################################
echo # Environment configured. Navigating to project directory...#
echo #############################################################
echo.

set "PROJECT_DIR=C:\Users\skurtyy\Documents\GitHub\HeavyRainRTX\Titanfall-2-Remix"
rem Steam launches HeavyRain.exe from its registered install path (A:\), not
rem the GitHub working copy under C:\.  Even if you type HeavyRain.exe at a
rem cmd in the C:\ copy, Steam's DRM (via steam_api64.dll init) redirects the
rem live process so it ends up loading DLLs from A:\.  Deploy directly there.
set "GAME_DIR=A:\SteamLibrary\steamapps\common\HEAVY RAIN"
set "GAME_RUNTIME_SUBDIR="
set "GAME_RUNTIME_DIR=%GAME_DIR%"
set "GAME_SHADER_DIR=%GAME_RUNTIME_DIR%\rtx_shaders"
set "GAME_LOG_DIR=%GAME_DIR%\rtx-remix\logs"
if not exist "%PROJECT_DIR%" (
    echo ERROR: Project directory not found: %PROJECT_DIR%
    goto :error_exit
)
pushd "%PROJECT_DIR%"

echo.
echo #############################################################
echo # Unlocking potentially locked files...                     #
echo #############################################################
echo.

rem Kill any processes that might lock build files
echo Checking for running game processes...
taskkill /F /IM "HeavyRain.exe" >nul 2>&1

rem Clear read-only attributes on build directories
echo Clearing read-only attributes on build output...
if exist "nv-private\hdremix\bin\debug" (
    attrib -R "nv-private\hdremix\bin\debug\*.*" /S /D >nul 2>&1
)
if exist "_Comp64Debug" (
    attrib -R "_Comp64Debug\*.*" /S /D >nul 2>&1
)

rem Force unlock any file handles (best effort)
echo Attempting to unlock file handles...
rem Wait a moment for file system to settle
timeout /t 1 /nobreak >nul 2>&1

echo.
echo #############################################################
echo # Starting/Updating the Remix Runtime build...              #
echo #############################################################
echo.

rem enable_dxgi=true is REQUIRED: Heavy Rain (like most DX11 games) calls
rem IDXGIFactory::CreateSwapChain directly, which only works if Remix ships
rem its own dxgi.dll wrapper.  Without it, the game lands on Microsoft's
rem real DXGI with a Vulkan-backed device it can't handle and crashes.
call meson setup --buildtype=debugoptimized --reconfigure --backend=ninja -Denable_dxgi=true _Comp64Debug
if errorlevel 1 (
    echo ERROR: Meson setup failed.
    goto :error_build
)

ninja -j6 -C _Comp64Debug
if errorlevel 1 (
    echo ERROR: The build process failed.
    goto :error_build
)

echo.
echo #############################################################
echo # Installing build artifacts...                             #
echo #############################################################
echo.

meson install -C _Comp64Debug
if errorlevel 1 (
    echo ERROR: The Meson install process failed due to locked files.
    echo Please close any applications that may have files locked and try again.
    goto :error_build
)

echo.
echo #############################################################
echo # Copying all build artifacts to _output directory...       #
echo #############################################################
echo.

rem --- Define source and destination directories ---
set "BUILD_DIR=_Comp64Debug"
set "OUTPUT_DIR=%PROJECT_DIR%\_output"
set "SOURCE_DIR=%PROJECT_DIR%\%BUILD_DIR%\tests\rtx\unit"
set "SHADER_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\dxvk\rtx_shaders"
set "BUILD_LOG_DIR=%PROJECT_DIR%\%BUILD_DIR%\meson-logs"

echo Cleaning and creating output directory: "%OUTPUT_DIR%"
if exist "%OUTPUT_DIR%" rd /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%"
echo.

if not exist "%SOURCE_DIR%" (
    echo ERROR: Build output directory not found at "%SOURCE_DIR%"
    goto :error_copy
)

echo Copying all files and folders from "%SOURCE_DIR%" to "%OUTPUT_DIR%"...
xcopy "%SOURCE_DIR%" "%OUTPUT_DIR%" /E /I /Y /Q
echo.

rem Ensure the freshly built d3d11.dll is in _output (it lives in src\d3d11 after the build)
set "D3D11_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\d3d11"
if exist "%D3D11_BUILD_DIR%\d3d11.dll" (
    echo Copying d3d11.dll from "%D3D11_BUILD_DIR%" to "%OUTPUT_DIR%"...
    copy /Y "%D3D11_BUILD_DIR%\d3d11.dll" "%OUTPUT_DIR%\d3d11.dll" >nul
    if exist "%D3D11_BUILD_DIR%\d3d11.pdb" copy /Y "%D3D11_BUILD_DIR%\d3d11.pdb" "%OUTPUT_DIR%\d3d11.pdb" >nul
) else (
    echo WARNING: d3d11.dll not found at "%D3D11_BUILD_DIR%" - deployment may be incomplete.
)

rem Optional: copy dxgi.dll if enable_dxgi was turned on at meson setup
set "DXGI_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\dxgi"
if exist "%DXGI_BUILD_DIR%\dxgi.dll" (
    echo Copying dxgi.dll from "%DXGI_BUILD_DIR%" to "%OUTPUT_DIR%"...
    copy /Y "%DXGI_BUILD_DIR%\dxgi.dll" "%OUTPUT_DIR%\dxgi.dll" >nul
    if exist "%DXGI_BUILD_DIR%\dxgi.pdb" copy /Y "%DXGI_BUILD_DIR%\dxgi.pdb" "%OUTPUT_DIR%\dxgi.pdb" >nul
)

if not exist "%SHADER_BUILD_DIR%" goto :skip_shader_copy
echo Copying RTX shader binaries to "%OUTPUT_DIR%\rtx_shaders"...
mkdir "%OUTPUT_DIR%\rtx_shaders" >nul
robocopy "%SHADER_BUILD_DIR%" "%OUTPUT_DIR%\rtx_shaders" *.spv /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy RTX shader binaries.
    goto :error_copy
)
goto :shader_copy_done
:skip_shader_copy
echo WARNING: Compiled shader directory not found.
:shader_copy_done
echo.

echo Collecting build logs...
mkdir "%OUTPUT_DIR%\logs" >nul
if not exist "%BUILD_LOG_DIR%" goto :skip_log_copy
mkdir "%OUTPUT_DIR%\logs\build" >nul
robocopy "%BUILD_LOG_DIR%" "%OUTPUT_DIR%\logs\build" *.* /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy Meson build logs.
    goto :error_copy
)
goto :log_copy_done
:skip_log_copy
echo WARNING: Meson log directory not found.
:log_copy_done
if exist "%PROJECT_DIR%\%BUILD_DIR%\.ninja_log" copy "%PROJECT_DIR%\%BUILD_DIR%\.ninja_log" "%OUTPUT_DIR%\logs\.ninja_log" >nul
if exist "%PROJECT_DIR%\%BUILD_DIR%\.ninja_deps" copy "%PROJECT_DIR%\%BUILD_DIR%\.ninja_deps" "%OUTPUT_DIR%\logs\.ninja_deps" >nul
set "README_LINE_1=Build logs copied from !BUILD_LOG_DIR!."
echo !README_LINE_1! > "%OUTPUT_DIR%\logs\README.txt"
set "README_LINE_2=To gather runtime DXVK / Remix logs, set the environment variable DXVK_LOG_PATH to a writable folder before launching the game."
echo !README_LINE_2! >> "%OUTPUT_DIR%\logs\README.txt"


echo.
echo #############################################################
echo # Deploying artifacts to game directory...                  #
echo #############################################################
echo.

if not exist "%GAME_DIR%" (
    echo ERROR: Game directory not found at "%GAME_DIR%".
    goto :error_copy
)

>"%GAME_DIR%\__remix_write_test.tmp" echo.
if errorlevel 1 (
    echo ERROR: Unable to write to "%GAME_DIR%". Please run as Administrator.
    goto :error_copy
)
del "%GAME_DIR%\__remix_write_test.tmp" >nul

rem DXVK state caches are shader-hash keyed, so entries auto-invalidate when
rem their inputs change.  Keeping the cache across DLL rebuilds saves the
rem multi-minute first-run pipeline compile on every iteration.  Pass
rem "clean" (or "cleancache") to this script to force a full cache wipe
rem when you suspect the cache itself is corrupt.
if /i "%1"=="clean" goto :wipe_dxvk_cache
if /i "%1"=="cleancache" goto :wipe_dxvk_cache
echo Preserving DXVK shader caches (pass "clean" to wipe).
goto :skip_cache_wipe
:wipe_dxvk_cache
echo Clearing DXVK shader caches...
del "%GAME_DIR%\*.dxvk-cache" 2>nul
del "%GAME_RUNTIME_DIR%\*.dxvk-cache" 2>nul
:skip_cache_wipe

echo Copying runtime package to "%GAME_RUNTIME_DIR%"...
if not exist "%GAME_RUNTIME_DIR%" (
    mkdir "%GAME_RUNTIME_DIR%" >nul
)
rem This command copies the entire _output folder (d3d11.dll, dxgi.dll, the
rem full Remix runtime, usd\plugins\) directly next to HeavyRain.exe so the
rem Windows loader picks up the Remix DX11 bridge.
robocopy "%OUTPUT_DIR%" "%GAME_RUNTIME_DIR%" *.* /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to deploy runtime files.
    goto :error_copy
)

if exist "%OUTPUT_DIR%\rtx_shaders" (
    echo Syncing shader binaries to "%GAME_SHADER_DIR%"...
    if not exist "%GAME_SHADER_DIR%" (
        mkdir "%GAME_SHADER_DIR%" >nul
    )
    robocopy "%OUTPUT_DIR%\rtx_shaders" "%GAME_SHADER_DIR%" *.spv /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    set "ROBOCOPY_EXIT=%ERRORLEVEL%"
    if !ROBOCOPY_EXIT! GEQ 8 (
        echo ERROR: Failed to update shader binaries.
        goto :error_copy
    )
)

rem HR patch: Patch 14 — overwrite HR's bundled MSVC runtime (VCRUNTIME140 /
rem MSVCP140 at 14.16 from VS 2017 Update 9, 2018) with the modern system copy
rem (14.40+) and drop in VCRUNTIME140_1.dll (which HR does not ship at all).
rem Our d3d11.dll imports all three; when HR ships 14.16 VCRUNTIME140 next to
rem its exe, Windows resolves that via exe-dir-first search rule, but resolves
rem VCRUNTIME140_1.dll from System32 at 14.50 -- the cross-version mix is
rem ABI-incompatible and access-violates at VCRUNTIME140+0x122e during C++
rem exception unwind or dispatched helpers.  Copying all three from System32
rem forces a consistent 14.50 runtime.  Originals are preserved under
rem _crt_14_16_backup\ before the first overwrite.  Idempotent: already-14.50
rem copies are left alone.
rem See CHANGELOG.md 2026-04-24.
echo.
echo #############################################################
echo # Patch 14: sync MSVC CRT to System32 (fix VCRUNTIME mix)...#
echo #############################################################
set "CRT_BACKUP_DIR=%GAME_RUNTIME_DIR%\_crt_14_16_backup"
if not exist "%CRT_BACKUP_DIR%" mkdir "%CRT_BACKUP_DIR%" >nul 2>&1
for %%F in (VCRUNTIME140.dll MSVCP140.dll) do (
    if exist "%GAME_RUNTIME_DIR%\%%F" (
        if not exist "%CRT_BACKUP_DIR%\%%F" (
            copy /Y "%GAME_RUNTIME_DIR%\%%F" "%CRT_BACKUP_DIR%\%%F" >nul
            echo Backed up %%F ^(pre-patch original^) to _crt_14_16_backup
        )
    )
)
for %%F in (VCRUNTIME140.dll VCRUNTIME140_1.dll MSVCP140.dll) do (
    if exist "C:\Windows\System32\%%F" (
        copy /Y "C:\Windows\System32\%%F" "%GAME_RUNTIME_DIR%\%%F" >nul
        echo Synced %%F from System32
    ) else (
        echo WARNING: C:\Windows\System32\%%F not present -- CRT sync skipped for this file
    )
)
echo.

rem Always (re)point DXVK_LOG_PATH at THIS game's log directory, regardless of
rem whether it already exists.  Without this, a persistent DXVK_LOG_PATH left
rem over from a previous game (e.g. Titanfall 2) will silently redirect
rem remix-dxvk.log to the wrong folder and hide the real crash output.
if not exist "%GAME_LOG_DIR%" mkdir "%GAME_LOG_DIR%" >nul 2>&1
echo Pointing DXVK_LOG_PATH at "%GAME_LOG_DIR%"...
setx DXVK_LOG_PATH "%GAME_LOG_DIR%" >nul
if errorlevel 1 (
    echo WARNING: Failed to configure DXVK_LOG_PATH automatically.
) else (
    rem setx only affects NEW processes; update the current shell too so any
    rem follow-up commands in this session see the new value.
    set "DXVK_LOG_PATH=%GAME_LOG_DIR%"
)

echo.
echo Done copying artifacts.
goto :success


:error_build
echo.
echo AN ERROR OCCURRED during the build process.
goto :error_exit

:error_copy
echo.
echo AN ERROR OCCURRED during the copy process.
goto :error_exit

:error_exit
echo.
echo SCRIPT FAILED.
popd
pause
exit /b 1

:success
echo.
echo #############################################################
echo # Build process finished successfully.                      #
echo #############################################################
echo.
popd
pause
exit /b 0

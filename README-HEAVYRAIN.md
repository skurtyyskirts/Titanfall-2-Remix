# Heavy Rain — Remix DX11 Bring-up

Companion to [`build-heavyrain.bat`](build-heavyrain.bat) and [`gametargets.conf`](gametargets.conf). Full context lives in the approved plan at `C:\Users\skurtyy\.claude\plans\look-at-the-titanfall-2-remix-crystalline-island.md`.

## Prerequisites (install once)

1. **Visual Studio 2022 Community** at `C:\Program Files\Microsoft Visual Studio\2022\Community\` with the **Desktop development with C++** workload. The batch hard-codes this path ([build-heavyrain.bat:20](build-heavyrain.bat#L20)).
2. **Windows SDK 10.0.19041.0** (via the VS installer).
3. **Vulkan SDK 1.4.313.2+** from [LunarG](https://vulkan.lunarg.com/sdk/home#windows). Uninstall any older SDK first.
4. **Python 3.9+** from [python.org](https://www.python.org/downloads/) — not the Microsoft Store version.
5. **Meson 1.8.2** via `pip install meson==1.8.2`. Reboot after install (Meson's installer will ask).
6. **Ninja** on `PATH` (`winget install Ninja-build.Ninja` or `pip install ninja`).
7. **Git LFS** once per machine: `git lfs install`.
8. **DirectX End-User Runtime (June 2010)**.
9. PowerShell execution policy: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` in an elevated PowerShell, then close and reopen any PowerShell windows.

## One-time repo setup

From an admin `cmd` in this folder:

```
git submodule update --init --recursive
```

Submodules include `rtxdi`, `rtxcr`, `nvapi`, `nrc`, and more. Missing submodules cause cryptic meson errors.

## Build + deploy

```
build-heavyrain.bat
```

- Run from an **admin** `cmd` (the deploy step needs write access to `HEAVY RAIN\`).
- First build is 20–40 minutes cold (USD + RTXDI + shader compiles).
- Incremental rebuilds are fast — ninja picks up only changed sources, and the DXVK shader cache is preserved across runs. Pass `clean` as an argument to wipe it when you suspect cache corruption: `build-heavyrain.bat clean`.

On success the script will have:
- Produced `d3d11.dll`, `dxgi.dll`, and the full Remix runtime in `_output\`.
- Mirrored `_output\` flat into `C:\Users\skurtyy\Documents\GitHub\HeavyRainRTX\HEAVY RAIN\` next to `HeavyRain.exe`.
- Placed compiled SPIR-V at `HEAVY RAIN\rtx_shaders\`.
- `setx DXVK_LOG_PATH C:\Users\skurtyy\Documents\GitHub\HeavyRainRTX\HEAVY RAIN\rtx-remix\logs` so Remix/DXVK logs land inside this tree.

## First launch

The user is keeping the game at **native 4K fullscreen** (`Resolution=3840 x 2160`, `ScreenMode=1` in `HEAVY RAIN\user_setting.ini`). If the first boot black-screens or crashes before the Alt+X overlay appears, the recovery sequence is:

1. Open `HEAVY RAIN\dxvk.conf`, uncomment `dxgi.deferSurfaceCreation = True` (see the comment at the top of that file).
2. In cmd: `setx DXVK_LOG_LEVEL debug`, open a new cmd, then relaunch.
3. Tail `HEAVY RAIN\rtx-remix\logs\remix-dxvk.log` in another window while relaunching to see the last operation before the crash.

If the game still won't produce a picture at 4K fullscreen, switching to windowed 1080p **only for the duration of one diagnostic boot** is the fastest way to see the Remix overlay and the Process Explorer DLL list clearly. Edit `user_setting.ini` for that one boot, then restore `3840 x 2160` / `ScreenMode=1` afterwards.

## Verification checklist

1. Launch `HEAVY RAIN\HeavyRain.exe` directly (not via Steam) for the first bring-up.
2. In Process Explorer (Sysinternals) → View → Lower Pane → DLLs:
   - `d3d11.dll` must show path `...\HEAVY RAIN\d3d11.dll` (NOT `C:\Windows\System32\d3d11.dll`).
   - `dxgi.dll` must show path `...\HEAVY RAIN\dxgi.dll`.
   - `vulkan-1.dll`, `NRD.dll`, `usd.dll`, `nvngx_dlss.dll` all loaded from `...\HEAVY RAIN\`.
3. Press **Alt+X** — Remix developer menu should overlay.
4. Bink intro logos may look flat; this is expected and documented.
5. Main menu should render with path-traced lighting.
6. Start Origami Killer (Ethan on the sofa). Walk around and rotate camera. Watch for: geometry in correct positions, UI on top (not baked in), character textures present.

## Iteration ladder

Full table is in the plan file. Short form:

| Symptom | First knob to turn |
|---|---|
| Crash in `d3d11.dll` / `dxgi.dll` | `dxgi.deferSurfaceCreation = True` in `dxvk.conf`, set `DXVK_LOG_LEVEL=debug`, read `remix-dxvk.log` |
| Crash in `HeavyRain.exe` itself | Task Manager → Create Dump File. Run `python -m retools.throwmap "HEAVY RAIN\HeavyRain.exe" <dump>` from the `Vibe-Reverse-Engineering/` folder |
| Black screen, no Alt+X | Verify DLLs loaded from local dir (step 2 above). If not, create empty `HEAVY RAIN\HeavyRain.exe.local` to force local-first DLL resolution |
| UI rendered into world | Inspect `src\d3d11\d3d11_vs_classifier.cpp`, tune VS heuristic |
| World geometry floats | Alt+X → Debug → per-draw matrices. Likely `d3d11_rtx.cpp` matrix scan needs extending |
| Textures black | Check log for "SRV rejected". Inspect `src\d3d11\d3d11_view_srv.cpp` |
| Need to live-trace a D3D11 call | `python -m livetools.client --attach HeavyRain.exe` from `Vibe-Reverse-Engineering/` |

After any source edit, rerun `build-heavyrain.bat`. Incremental builds finish in a minute or two once the shader cache is warm.

## Files this target creates or edits in the game folder

Created by `build-heavyrain.bat`:
- `HEAVY RAIN\d3d11.dll`, `HEAVY RAIN\dxgi.dll` (+ their `.pdb`)
- `HEAVY RAIN\*.dll` (NRD, DLSS, XeSS, USD, Vulkan loader, etc.)
- `HEAVY RAIN\usd\plugins\...`
- `HEAVY RAIN\rtx_shaders\*.spv`
- `HEAVY RAIN\rtx-remix\logs\`

Created manually (already done):
- `HEAVY RAIN\dxvk.conf` (copy of the template in this folder)

Untouched:
- `HEAVY RAIN\HeavyRain.exe`
- `HEAVY RAIN\user_setting.ini` (kept at 4K fullscreen per user preference)
- `HEAVY RAIN\Resources\`, `HEAVY RAIN\Videos\`, etc.

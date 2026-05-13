# rexglue fixes that got Forza Horizon 1 playable

This is the running log of fixes landed in our `rexglue` fork (the `ItsNotPaths/rexglue-sdk` `fh1` branch) that took the FH1 static recompile from "boots but silently dies" to a controllable in-game state. Each entry names the file/function, the symptom that surfaced it, the mechanism, and whether the fix is upstreamable.

Ordering is roughly the order they landed.

---

## 1. `vmaddfp` / `vnmsubfp` → real FMA3, plus `-mfma`

**File:** `rexglue-sdk/src/codegen/builders/vector.cpp` (`build_vmaddfp`, `build_vnmsubfp`)
**Also:** `rexglue-sdk/include/rex/ppc/intrinsics.h` — `#include <simde/x86/fma.h>`
**Also:** `fh1-recomp/src/CMakeLists.txt` — added `-mfma` to fh1-host and per-XEX `target_compile_options`

PPC `vmaddfp` and `vnmsubfp` are single-rounding fused multiply-adds. Rexglue's original emit was `(a*b)+c` (two roundings). Replaced with `simde_mm_fmadd_ps` / `simde_mm_fnmadd_ps`, covering `VMADDFP`, `VMADDFP128`, `VMADDCFP128`, `VNMSUBFP`, `VNMSUBFP128`.

`-mfma` is **load-bearing**: without it, the simde intrinsic silently falls back to soft `mul+add` and you lose the precision benefit silently. Verify post-build with:

```
objdump -d fh1-recompiled-release/fh1-host | grep -cE 'vfmadd|vfnmadd'
# expect ~20,000+
```

**Status:** correctness improvement. Did **not** fix FH1's geometry jitter (that was `vaddsws` — see below). Kept anyway.

**Upstreamable:** yes, issue never submitted

---

## 2. `vaddsws` per-byte blend → per-dword blend

**File:** `rexglue-sdk/src/codegen/builders/vector.cpp` (`build_vaddsws`, lines ~275–285)

PPC `vaddsws` = per-32-bit-lane signed-saturating add. The standard implementation is:

1. `sum = a + b` per dword
2. `overflow_mask = (a ^ sum) & (b ^ sum)` — only bit 31 of each dword is meaningful; lower 31 bits are noise
3. `result[lane] = overflow_mask.sign_bit ? sat_val : sum`

The blend in step 3 must look at **one bit per 32-bit lane**. Xenia uses `vblendvps`. Rexglue was using `simde_mm_blendv_epi8`, which selects per-byte on each byte's high bit — so it sampled bits 7/15/23 (arbitrary noise) in addition to bit 31. Roughly 50% of bytes got swapped to the saturated value when no real overflow occurred.

**Symptom:** persistent ~0.234 m camera snaps and rare ±8192 m teleports for weeks. The path was `CCar::TransformBodyPosToWorldPos` → `Mtx33MultVec` → `vcfpsxws` → `vaddsws` → `vcsxwfp`, with fixed-point scale 2^18. A byte-2 swap (bits 16–23) replaces ~2^16 of int magnitude → 0.25 m snap after inverse scale. Byte-1 swap → 0.001 m background wobble. Real-overflow plus garbage lower bytes → ±INT_MAX ≈ ±8192 m.

**Fix:** swap `simde_mm_blendv_epi8` for `simde_mm_blendv_ps`, wrapped with `simde_mm_castsi128_ps` / `simde_mm_castps_si128` around the int operands. Comment in source documents the spec divergence.

**Status:** RESOLVED 2026-05-10. User-confirmed cinematic camera snap gone.

**Upstreamable:** yes. github issue rejected, fix never implimented

---

## 3. `vpkd3d128` case 5 (FLOAT16_4 pack) — source-lane clobber when `dst == src`

**File:** `rexglue-sdk/src/codegen/builders/vector.cpp` (`build_vpkd3d128`, case 5)
**Diff draft:** `docs/rexglue-prs/0004-vpkd3d128-case5-pack-alias.txt`

Case 5 packs four FP32 lanes into four FP16 halfwords. The original emit issued the destination-half clear (`dst.u64[1] = 0` for shift=0, `u64[0] = 0` for shift=2) **before** the pack loop read source lanes. The clear is intended to zero the half of the destination that won't receive new data. But in the very common in-place pattern `vpkd3d128 vD, vD, 5, 2, 0`, `dst == src`, so the clear wipes the source lanes the loop is about to read. Half of every packed FP16 output came out as `0x0000` regardless of input.

**Symptom:** in-game audio silent post-splash. `FMOD::DSPConnectionI::setLevels(float*, int)` stages FP32 channel gains and converts them in-place via `vpkd3d128 v63, v63, 5, 2, 0` to FP16 coefficients at `*(this+40+i*4)`. Every coefficient packed as `0x0000`. The downstream mix multiplied real samples by zero gains → silence.

**Fix:** snapshot all 4 source `u32` lanes (`_vpks0..3`) into a local scope **before** the clear, then have the pack loop read from the snapshot. Scope block keeps the locals from colliding across multiple case-5 instances in the same function.

After fix, `*this[40] = 0x3C00` (FP16 1.0 unity) and audio is audible. Confirmed by user 2026-05-12.

**Audit notes:** Case 3 (FLOAT16_2) and cases 0/1/2/6 don't have the same aliasing risk. Case 4 (NORMSHORT4) and `build_vupkd3d128` case 20 (FP16→FP32 unpack) are worth a follow-up audit — the case 4 in-place pattern has a similar overlapping-write hazard with no observed FH1 caller.

**Status:** RESOLVED 2026-05-12.

**Upstreamable:** yes. github issue rejected, fix never implimented

## 4. SPIR-V translator: HDR sample exp_bias read from word_4 instead of word_3

**File:** `rexglue-sdk/src/graphics/pipeline/shader/spirv_translator_fetch.cpp` (around lines 1411, 2032–2037)

The Xbox 360's vertex/texture fetch constants pack `exp_adjust` (6-bit signed result exponent bias) into bits 13:18 of **word 3**. Rexglue's SPIR-V translator was reading the same bit range from **word 4**, which at that offset holds the LOD bias instead. So the sampled result was scaled by `2^LOD_bias` rather than `2^exp_adjust`. For HDR textures with non-zero exp_adjust, this crushed magnitudes.

The DXBC translator (`dxbc_translator_fetch.cpp`) didn't have the bug — only the SPIR-V path, which is why the symptom was Vulkan-only.

**Symptom:** in-game world rendered with geometry correct but everything looking like night/dim — neon road lines visible (bright sources clip past any reasonable saturation regardless of scale), mid-tones crushed to near-zero. Survived RenderDoc capture because the shader code itself is unaffected.

**Fix:** ported xenia commit `32889f51b` (2025-12-06, has207 — "Use fetch_contant_word3 for exponent bias"). Add a `fetch_constant_word_3_signed` load alongside the existing `fetch_constant_word_4` load (still needed for LOD bias and stacked-texture filtering at other call sites), then swap the call in `BitFieldSExtract(value=...)` → `Ldexp(1.0, value)` from word_4_signed to word_3_signed. Update comments.

**Status:** RESOLVED 2026-05-08. User reported "ALL LIGHTING IS PERFECT" immediately on rebuild.

**Upstreamable:** already upstream in xenia; this is a port-forward of `32889f51b` into rexglue's translator. github issue rejected, fix never implimented

**Related ports not yet applied** (probably not load-bearing for FH1 but adjacent):

- `fdd583ece` (2026-03-21) — resolve_fast_32bpp_4xmsaa sample addressing (FH1 main scene is msaa=2)
- `dbfe21675` (2026-03-03) — Vulkan f32→f16 extended-range memexport (FH1 doesn't memexport much)
- `16d2cc05c` (2026-03-05) — UNorm not signed EDRAM in k_16_16 resolve (FH1 doesn't use k_16_16 RTs)
- `2eea146b1` (2026-03-02) — k_16_16 / k_16_16_16_16 EDRAM packing clamping (ditto)

---

## 5. Empty-resolve cascade — return success no-op instead of error

**File:** `rexglue/src/graphics/util/draw.cpp` (~line 887)

The original code had:

```cpp
assert_true(x0 < x1 && y0 < y1);
if (x0 >= x1 || y0 >= y1) {
  REXGPU_ERROR("Resolve region is empty");
  return false;
}
```

Returning `false` propagates to `IssueDraw`, which then logs `PM4_DRAW_INDX_2(...): Failed in backend (...edram_mode=6)`. FH1 issues empty resolves prolifically (related to tile-bin pass behavior — see notes below), producing 700–20,000 paired errors per run during race/career loads.

**Fix:** ported the xenia post-fork patch from `xenia-canary/src/xenia/gpu/draw_util.cc:1058`:

```cpp
// Empty/inverted region after clipping (e.g. entirely outside EDRAM bounds
// due to window offset) — treat as no-op rather than error. Reduces log spam
// in Forza Horizon 1/2 which do many of these without visible impact.
if (x0 >= x1 || y0 >= y1) {
  info_out.coordinate_info.width_div_8 = 0;
  info_out.height_div_8 = 0;
  return true;
}
```

Plus removed the `assert_true` (would still fire in debug). Caller-side skip on `width_div_8 == 0 || height_div_8 == 0` is already in place at `vulkan/render_target_cache.cpp:1348` and `d3d12/render_target_cache.cpp:1186`.

The xenia comment explicitly names FH1/FH2 — this title behavior is known upstream.

**Open observation (not the fix):** rexglue tracks `BIN_MASK` / `BIN_SELECT` but does not honor the `predicated` bit on PM4 draws, so each tile pass executes the *other* tile's resolve too. The wrong-tile resolve's vertices collide with the active tile's scissor and collapse to empty. A real fix would be to honor `predicated` in the PM4 type-3 dispatcher in `command_processor.cpp`. Currently this manifests only as (now-suppressed) log noise.

**Status:** RESOLVED 2026-05-08. Verified `Resolve region is empty` string is no longer present in the linked binary.

**Upstreamable:** xenia already has it; port-forward only.

---

## 6. FunctionDispatcher race in `function_table_`

**Files:**
`rexglue/include/rex/system/function_dispatcher.h`
`rexglue/src/system/function_dispatcher.cpp`

`std::unordered_map<uint32_t, ::PPCFunc*> function_table_` was being mutated by `SetFunction` (bursty during `init_function_table()` on a dlopen thread — ~25K entries per facade `.so`) while `GetFunction` was hit per indirect dispatch from worker threads (audio worker, GPU). A concurrent insert triggers rehash; the reader's bucket pointer is invalidated; `find()` spuriously returns `end()`.

**Symptom:** sporadic `Execute(<addr>): function not in function table` errors. Ours fired at `0x830C9D78` (the audio renderer's `RegisterCommand` thunk → `DriverCallback`). In-game audio dead because the audio worker couldn't dispatch its callback. 0–3 hits per run, looked like an analyzer miss until we instrumented.

**Confirmation method:** one-line probes in `SetFunction` / `GetFunction` gated to the failing address. The decisive run showed 1 set, 1685 successful gets, and 1 transient null at the exact millisecond `SpeechFacade_default.so` grew the table from 61,477 → 85,229 entries.

**Fix:**

- Header: `mutable std::shared_mutex function_table_mutex_;`
- `SetFunction`: `std::unique_lock` around the `function_table_[guest_address] = func;` line. `Memory::SetFunction` left outside the lock (has its own serialization).
- `GetFunction`: `std::shared_lock` covering find + return.

shared_mutex is the right call given 1685:1 read:write ratio.

**Status:** RESOLVED 2026-05-08. In-game audio dispatch survived once vpkd3d128 (#3) also landed.

**Upstreamable:** yes — generic bug affecting any title that loads multiple DLL facades.

---

## 7. `UnloadUserModule` DETACH leak (workaround, not a clean fix)

**File:** `rexglue-sdk/src/system/kernel_state.cpp` (~line 737)

The original `UnloadUserModule` path tried to call the DLL's entry point with `DLL_PROCESS_DETACH`, then tear down the module's heap via `BaseHeap::Release`. The DETACH wasn't actually implemented (logged `DllMain(DLL_PROCESS_DETACH) not implemented` and skipped), but the heap teardown ran anyway and left `CSystemEventParam` mutexes in an inconsistent state. Result: T37 deadlock in `pthread_mutex` inside `CSystemEventParam::DecRef` from `CTrackModel::SetFileIOFailState`. Every load-hang log showed the `UnloadUserModule` + `BaseHeap::Release failed` preamble.

**Fix (intentional leak):** early-return on the `is_dll_module() && entry_point && call_entry` path, log `[fh1-patch] leaking DLL '<path>'` warning, skip the teardown entirely. Fires once per run on `XMediaFacade_default.xex` unload. The leak is harmless for a single-process game; the real fix is implementing DETACH-with-correct-teardown, but that's a much bigger job and FH1 doesn't need it.

**Status:** workaround landed 2026-05-11. Closed the load-hang chain that NULL-guard patches couldn't reach.

**Upstreamable:** as a workaround under a debug cvar, maybe. The proper fix (correct DETACH+teardown) is bigger.

---

## Related: cvar override (not a rexglue source change)

**File:** `fh1-recomp/src/fh1_app.cpp` — `OnPreSetup` calls `REXCVAR_SET(gpu_allow_invalid_fetch_constants, true)`.

FH1's vertex shaders reference vfetch slot 90, which the game never binds (stays `kInvalidVertex`, raw words `00000001 00000000`). With the rexglue default of `false` (matching xenia's default), every `edram_mode=4` (kColorDepth) and `edram_mode=5` (kDepthOnly) draw that touched VF90 was dropped — ~5400 color + ~2600 depth fails in 60 s. EDRAM stayed empty, the eventual resolve copied nothing, the frame went black except for the few draws whose shaders happened not to touch VF90 (neon road lines, etc.).

No rexglue source change — the cvar is `kHotReload`, so flipping it at runtime via the FH1 host app's `OnPreSetup` is enough. Listed here because it was a critical part of the same "get FH1 visually working" arc.

Other xenia compat-shim cvars worth knowing about for future GPU triage:
- `half_pixel_offset`
- `use_fuzzy_alpha_epsilon`
- `mrt_edram_used_range_clamp_to_min`
- `gamma_render_target_as_unorm16`

---

## Summary table

| # | File | Symptom fixed | Status | Upstream? |
|---|------|---------------|--------|-----------|
| 1 | `codegen/builders/vector.cpp` (vmaddfp/vnmsubfp) + `intrinsics.h` + CMake `-mfma` | (precision correctness) | Landed | yes |
| 2 | `codegen/builders/vector.cpp` (vaddsws) | 0.234 m camera snaps + ±8192 m teleports | RESOLVED | yes |
| 3 | `codegen/builders/vector.cpp` (vpkd3d128 case 5) | silent audio post-splash | RESOLVED | yes |
| 4 | `graphics/pipeline/shader/spirv_translator_fetch.cpp` | dim/dark Vulkan output | RESOLVED | already in xenia |
| 5 | `graphics/util/draw.cpp` | 700–20k spam errors/run | RESOLVED | already in xenia |
| 6 | `system/function_dispatcher.{h,cpp}` | audio worker dispatch misfire | RESOLVED | yes |
| 7 | `system/kernel_state.cpp` (UnloadUserModule) | T37 mutex deadlock on track load | RESOLVED (workaround) | maybe |

All eight + the cvar override together are what took FH1 from "boots to silent black screen" to "menu, intro, world load, controllable car, audio".

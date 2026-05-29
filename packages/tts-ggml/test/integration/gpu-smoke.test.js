'use strict'

// GPU smoke tests for both tts-ggml engines (chatterbox + supertonic).
//
// Mirrors transcription-parakeet/test/integration/gpu-smoke.test.js's
// strict-on-CPU policy on DESKTOP (darwin/linux/win32): a useGPU=true
// request that resolves to the CPU backend is treated as a regression
// because it usually means a build / linkage / kernel-init drift that CI
// must catch.  Set QVAC_TTS_GPU_SMOKE_RELAX=1 to downgrade that gate to a
// warning (e.g. for a Linux host without Vulkan SDK, or an emulator
// without Metal).
//
// On ANDROID the GPU is validated-device-only: tts-cpp engages a GPU backend
// only for Qualcomm Adreno (OpenCL) and routes every other Android GPU to CPU
// by design (ARM Mali/Xclipse abort uncatchably). The Device Farm pool mixes
// Adreno with non-Adreno devices, so a clean CPU fallback is an expected PASS
// on Android — the positive GPU path is still asserted (backendId must be
// OpenCL/Vulkan) whenever a device engages the GPU. iOS stays STRICT: it must
// engage Metal (the S3Gen scheduler is capability-gated in tts-cpp so Metal
// runs the graph natively); a CPU fallback on iOS is a regression and fails.
// See assertGpuBackend.
//
// CI runners without a real GPU (or hosted macOS where the
// Paravirtual Metal device crashes ggml's encoder) export NO_GPU=true
// to skip every smoke entry.  Real GPU runners and local dev leave
// NO_GPU unset so the strict assertions still fire there.
//
// The strict gate uses `response.stats.backendDevice` (0=CPU, 1=GPU)
// and `response.stats.backendId` (0=CPU, 1=Metal, 2=CUDA, 3=Vulkan,
// 4=OpenCL, 99=other), both surfaced by ChatterboxModel +
// SupertonicModel after Engine::backend_device() / backend_name() were
// added in tts-cpp.

const fs = require('bare-fs')
const os = require('bare-os')
const path = require('bare-path')
const proc = require('bare-process')
const test = require('brittle')

const { loadChatterboxTTS, runChatterboxTTS, resolveRefWavPath } = require('../utils/runChatterboxTTS')
const { loadSupertonicTTS, runSupertonicTTS } = require('../utils/runSupertonicTTS')
const { ensureChatterboxModels, ensureSupertonicModel } = require('../utils/downloadModel')

const platform = os.platform()
const isMobile = platform === 'ios' || platform === 'android'
const RELAX = proc.env && proc.env.QVAC_TTS_GPU_SMOKE_RELAX === '1'
const NO_GPU = proc.env && proc.env.NO_GPU === 'true'

function getBaseDir () {
  return isMobile && global.testDir ? global.testDir : '.'
}

function backendIdToName (id) {
  switch (id) {
    case 0: return 'CPU'
    case 1: return 'Metal'
    case 2: return 'CUDA'
    case 3: return 'Vulkan'
    case 4: return 'OpenCL'
    case 99: return 'other-GPU'
    default: return `unknown(${id})`
  }
}

// Which platforms wire up a GPU backend in tts-cpp's vcpkg port
// today (default-features in qvac-registry-vcpkg/ports/tts-cpp/vcpkg.json):
//   - darwin / ios:        metal
//   - linux / win32:       vulkan
//   - android:             vulkan + opencl
function expectsGpu () {
  return (
    platform === 'darwin' ||
    platform === 'ios' ||
    platform === 'linux' ||
    platform === 'win32' ||
    platform === 'android'
  )
}

function assertGpuBackend (t, engineTag, stats) {
  if (!stats) {
    t.fail(`${engineTag}/GPU: no response.stats returned (cannot verify backend)`)
    return
  }
  const dev = stats.backendDevice
  const id = stats.backendId
  const name = backendIdToName(id)
  console.log(`[${engineTag}/GPU] backendDevice=${dev} backendId=${id} (${name})`)

  if (!expectsGpu()) {
    t.is(dev, 0, `${engineTag}/${platform}: backendDevice must be 0 (CPU) on platforms with no GPU wired in`)
    return
  }

  if (dev !== 1) {
    const msg = `${engineTag}/${platform}: expected GPU backend, got ${name} (backendDevice=${dev}, backendId=${id}). ` +
                'useGPU=true was requested but the engine fell back to CPU. ' +
                'Inspect addon native logs for the load-time backend init message.'
    if (platform === 'android') {
      // Android GPU is validated-device-only: tts-cpp's init_gpu_backend engages
      // a GPU backend only for Qualcomm Adreno (OpenCL); every other Android GPU
      // (ARM Mali, Samsung Xclipse, ...) is routed to CPU by design because their
      // drivers hit GGML_ASSERT -> ggml_abort (uncatchable, kills the host app).
      // The Device Farm pool mixes Adreno with non-Adreno devices, so a clean CPU
      // fallback here is the expected, correct outcome — the engine still ran and
      // produced audio above. The positive GPU path (Adreno must use OpenCL/Vulkan)
      // is still asserted by the backendId check below whenever a device engages
      // the GPU. iOS is intentionally NOT relaxed: it must engage Metal, and the
      // strict check below is the guard against an iOS GPU regression.
      t.pass(`${engineTag}/android: CPU fallback accepted (GPU is Adreno-only on Android; other vendors route to CPU by design)`)
    } else if (RELAX) {
      t.comment(`WARNING (relaxed): ${msg}`)
      t.pass(`${engineTag}/GPU smoke completed (relaxed)`)
    } else {
      t.fail(msg)
    }
    return
  }

  if (platform === 'darwin' || platform === 'ios') {
    t.is(id, 1, `${engineTag}/${platform}: expected Metal backendId=1, got ${name}`)
  } else if (platform === 'linux' || platform === 'win32') {
    t.is(id, 3, `${engineTag}/${platform}: expected Vulkan backendId=3, got ${name}`)
  } else if (platform === 'android') {
    t.ok(id === 3 || id === 4, `${engineTag}/${platform}: expected Vulkan(3) or OpenCL(4) backendId, got ${name}`)
  }
}

// Companion to assertGpuBackend: when the caller passes useGPU=false we
// expect the engine to actually pick the CPU backend.  This is the gate
// that prevents `useGPU=false` from silently still running on GPU when
// the underlying tts-cpp library default is non-zero n_gpu_layers.
function assertCpuBackend (t, engineTag, stats) {
  if (!stats) {
    t.fail(`${engineTag}/CPU: no response.stats returned (cannot verify backend)`)
    return
  }
  const dev = stats.backendDevice
  const id = stats.backendId
  const name = backendIdToName(id)
  console.log(`[${engineTag}/CPU] backendDevice=${dev} backendId=${id} (${name})`)
  t.is(dev, 0, `${engineTag}: useGPU:false must resolve to backendDevice=0 (CPU), got ${name}`)
  t.is(id, 0, `${engineTag}: useGPU:false must resolve to backendId=0 (CPU), got ${name}`)
}

test('Chatterbox GPU smoke - useGPU=true must engage the GPU backend on GPU-capable platforms', { timeout: 600000, skip: NO_GPU }, async (t) => {
  // Android Adreno GPU smoke un-quarantined for pre-merge Device Farm validation.
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.pass('Skipped: Chatterbox GGUFs not available locally')
    return
  }

  // Mobile-aware resolution: see multiple-runs.test.js for rationale.
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) {
    t.pass('Skipped: reference audio missing')
    return
  }

  const model = await loadChatterboxTTS({
    modelDir: download.targetDir,
    refWavPath,
    language: 'en',
    useGPU: true
  })
  try {
    const result = await runChatterboxTTS(
      model,
      { text: 'GPU smoke check.' },
      { minSamples: 5000 }
    )
    console.log(result.output)
    t.ok(result.passed, 'Chatterbox/GPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Chatterbox/GPU produced audio')
    assertGpuBackend(t, 'Chatterbox', result.data.stats)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

test('Supertonic GPU smoke - useGPU=true is rejected at constructor (engine is CPU-only today)', { timeout: 60000 }, async (t) => {
  const TTSGgml = require('@qvac/tts-ggml')
  let threw = false
  try {
    /* eslint no-new: 0 */
    new TTSGgml({
      engine: TTSGgml.ENGINE_SUPERTONIC,
      files: { supertonicModel: '/dev/null' },
      voice: 'F1',
      config: { language: 'en', useGPU: true }
    })
  } catch (e) {
    threw = true
    t.ok(/CPU only today/.test(e.message),
      'rejection message references the engine docstring')
    t.ok(/Pass config:.*useGPU: false/.test(e.message),
      'rejection message tells user how to fix')
  }
  t.ok(threw, 'TTSGgml constructor should throw on Supertonic + useGPU:true')
})

// CPU smoke: useGPU:false must actually pin the engine to CPU on every
// platform (no NO_GPU skip — CPU is expected to work everywhere).  This
// is the counterpart to the GPU smoke above and exists because the
// previous tts-ggml behaviour left n_gpu_layers at the tts-cpp library
// default when useGPU:false was passed without an explicit nGpuLayers,
// which could silently fall back to GPU.  Now that ChatterboxModel /
// SupertonicModel translate explicit useGPU=false → n_gpu_layers=0,
// these tests lock that contract in.
test('Chatterbox CPU smoke - useGPU=false must run on the CPU backend', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.pass('Skipped: Chatterbox GGUFs not available locally')
    return
  }

  // Mobile-aware resolution: see multiple-runs.test.js for rationale.
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) {
    t.pass('Skipped: reference audio missing')
    return
  }

  const model = await loadChatterboxTTS({
    modelDir: download.targetDir,
    refWavPath,
    language: 'en',
    useGPU: false
  })
  try {
    const result = await runChatterboxTTS(
      model,
      { text: 'CPU smoke check.' },
      { minSamples: 5000 }
    )
    console.log(result.output)
    t.ok(result.passed, 'Chatterbox/CPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Chatterbox/CPU produced audio')
    assertCpuBackend(t, 'Chatterbox', result.data.stats)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

test('Supertonic CPU smoke - useGPU=false must run on the CPU backend', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureSupertonicModel({ targetDir: modelsDir })
  if (!download || !download.success) {
    t.pass('Skipped: Supertonic GGUF not available locally')
    return
  }

  const supertonicPath = download.path ||
    path.join(modelsDir, 'supertonic.gguf')

  const model = await loadSupertonicTTS({
    supertonicModelPath: supertonicPath,
    language: 'en',
    voice: 'F1',
    useGPU: false
  })
  try {
    const result = await runSupertonicTTS(
      model,
      { text: 'CPU smoke check.' },
      { minSamples: 5000 }
    )
    console.log(result.output)
    t.ok(result.passed, 'Supertonic/CPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Supertonic/CPU produced audio')
    assertCpuBackend(t, 'Supertonic', result.data.stats)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

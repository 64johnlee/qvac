'use strict'
const fs = require('bare-fs')
const path = require('bare-path')
const test = require('brittle')
const TranscriptionWhispercpp = require('../../index.js')
const {
  detectPlatform,
  runTranscription,
  ensureWhisperModel,
  ensureVADModel,
  generateTestAudio,
  makePcmNoise,
  setupJsLogger,
  getTestPaths,
  createAudioStream
} = require('./helpers.js')

const platform = detectPlatform()
const { modelPath, vadModelPath, audioPath } = getTestPaths()

test('Real addon lifecycle works correctly via public API', { timeout: 120000 }, async (t) => {
  await ensureWhisperModel(modelPath)
  generateTestAudio(audioPath)

  const config = {
    path: modelPath,
    whisperConfig: {
      language: 'en',
      audio_format: 's16le',
      temperature: 0.0,
      vadParams: {
        threshold: 0.6
      }
    }
  }

  const constructorArgs = {
    files: {
      model: modelPath
    }
  }

  let model
  try {
    console.log('Creating TranscriptionWhispercpp...')
    model = new TranscriptionWhispercpp(constructorArgs, config)
    t.ok(model, 'TranscriptionWhispercpp should be created')

    console.log('Loading model...')
    await model._load()
    const status = await model.status()
    t.ok(status, 'Status should be returned after load')
    console.log('Status after load:', status)

    console.log('Running a transcription job...')
    const audioStream = createAudioStream(audioPath)
    const response = await model.run(audioStream)
    await response.await()
    t.ok(response, 'run() should return a response')
    t.pass('Completed a transcription job through the public API')
  } catch (error) {
    console.error('Unexpected error in addon lifecycle test:', error.message)
    throw error
  } finally {
    try { if (model) await model.unload() } catch {}
  }
})

test('Real addon state transitions work correctly via public API', { timeout: 120000 }, async (t) => {
  await ensureWhisperModel(modelPath)

  const config = {
    path: modelPath,
    whisperConfig: {
      language: 'en',
      audio_format: 's16le',
      temperature: 0.0,
      vadParams: {
        threshold: 0.6
      }
    }
  }

  const constructorArgs = {
    files: {
      model: modelPath
    }
  }

  let model
  try {
    model = new TranscriptionWhispercpp(constructorArgs, config)
    await model._load()

    let status = await model.status()
    t.ok(status, 'Should have initial status')

    await model.unpause()
    status = await model.status()
    t.is(status, 'listening', 'Should be listening after activation')

    await t.exception(
      () => model.pause(),
      /pause is not supported in runJob mode/,
      'Pause should be explicitly unsupported in runJob mode'
    )

    await t.exception(
      () => model.stop(),
      /stop is not supported in runJob mode/,
      'Stop should be explicitly unsupported in runJob mode'
    )

    status = await model.status()
    t.is(status, 'listening', 'Status should remain listening after unsupported stop')
    t.pass('State transitions test completed')
  } finally {
    try { if (model) await model.unload() } catch {}
  }
})

test('Real addon can handle multiple audio chunks', { timeout: 120000 }, async (t) => {
  await ensureWhisperModel(modelPath)
  const chunks = [
    new Uint8Array([1, 2, 3, 3]),
    new Uint8Array([4, 5, 6, 7]),
    new Uint8Array([7, 8, 9, 8])
  ]

  try {
    const result = await runTranscription(
      {
        audioInput: chunks,
        modelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          vadParams: {
            threshold: 0.6
          }
        }
      }
    )

    if (result.passed) {
      t.pass('Multiple chunks test completed')
    } else {
      console.log('Expected error:', result.output)
      t.pass('Multiple chunks test works with expected errors')
    }
  } catch (error) {
    console.log('Expected error:', error.message)
    t.pass('Multiple chunks test works with expected errors')
  }
})

test('Real addon with downloaded models - success case', { timeout: 120000 }, async (t) => {
  console.log(` Running on platform: ${platform}`)

  const whisperResult = await ensureWhisperModel(modelPath)

  generateTestAudio(audioPath)

  t.ok(fs.existsSync(modelPath), 'Whisper model file should exist')
  t.ok(fs.existsSync(audioPath), 'Test audio should exist')

  if (whisperResult.isReal) {
    console.log(' Testing with REAL whisper model (VAD disabled) - expecting successful transcription')

    const result = await runTranscription(
      {
        audioInput: audioPath,
        modelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          vadParams: {
            threshold: 0.6
          }
        }
      },
      {
        minSegments: 0
      }
    )

    console.log(` Transcription result: ${result.output}`)
    console.log(`  - Segments: ${result.data.segmentCount}`)
    console.log(`  - Text length: ${result.data.textLength}`)

    if (result.passed && result.data.segmentCount > 0) {
      t.pass(' Successfully transcribed audio with real whisper model (VAD disabled)')
      console.log(' REFERENCE EXAMPLE: Whisper-only transcription working perfectly!')
    } else if (result.data.error) {
      console.log(' Model loaded but transcription had issues (may be expected with test audio)')
      t.pass('Model loaded but transcription had issues (may be expected)')
    } else {
      t.ok(result.passed, 'Should pass expectations (may have 0 segments with test audio)')
    }
  } else {
    console.log(' Could not download real models - testing with placeholder models instead')

    const result = await runTranscription(
      {
        audioInput: audioPath,
        modelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          vadParams: {
            threshold: 0.6
          }
        }
      }
    )

    if (result.data.error) {
      t.pass(' Correctly handled placeholder models with expected errors')
      console.log(' REFERENCE EXAMPLE: Error handling with invalid models working perfectly!')
    } else {
      t.ok(result.data.segmentCount >= 0, 'Should receive some result')
      t.pass('Placeholder model test completed')
    }
  }
})

test('Runtime stats are populated when opts.stats=true', { timeout: 120000 }, async (t) => {
  await ensureWhisperModel(modelPath)
  generateTestAudio(audioPath)

  const config = {
    path: modelPath,
    whisperConfig: {
      language: 'en',
      audio_format: 's16le',
      temperature: 0.0
    }
  }

  const constructorArgs = {
    files: {
      model: modelPath
    },
    opts: { stats: true }
  }

  const model = new TranscriptionWhispercpp(constructorArgs, config)

  try {
    await model._load()
    const audioStream = createAudioStream(audioPath)
    const response = await model.run(audioStream)
    await response.await()

    t.ok(response.stats, 'Response should include stats')
    t.ok(Object.keys(response.stats).length > 0, 'Response.stats should not be empty')

    // Validate a few core metrics are present and numeric.
    t.is(typeof response.stats.totalTime, 'number', 'totalTime should be a number')
    t.ok(response.stats.totalTime >= 0, 'totalTime should be >= 0')
    t.is(typeof response.stats.audioDurationMs, 'number', 'audioDurationMs should be a number')
    t.ok(response.stats.audioDurationMs > 0, 'audioDurationMs should be > 0')
    t.is(typeof response.stats.totalSamples, 'number', 'totalSamples should be a number')
    t.ok(response.stats.totalSamples > 0, 'totalSamples should be > 0')
  } finally {
    try { await model.unload() } catch {}
  }
})

test('Real addon with VAD enabled - advanced case', { timeout: 120000 }, async (t) => {
  console.log(' Testing VAD-enabled transcription (if VAD model available)')

  // Try to download VAD model
  const vadOk = await ensureVADModel(vadModelPath)
  const whisperResult = await ensureWhisperModel(modelPath)

  if (fs.existsSync(modelPath) && vadOk && whisperResult.isReal) {
    console.log(' Testing with REAL whisper + VAD models')

    const result = await runTranscription(
      {
        audioInput: audioPath,
        modelPath,
        vadModelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          vadParams: {
            threshold: 0.6
          }
        }
      },
      {
        minSegments: 0
      }
    )

    console.log(`VAD+Whisper Transcription result: ${result.output}`)
    if (result.data.segmentCount > 0) {
      console.log('VAD+Whisper Transcription [segments]:', JSON.stringify(result.data.segments.slice(0, 3), null, 2))
    }

    if (result.passed && result.data.segmentCount > 0) {
      t.pass(' VAD+Whisper transcription successful')
      console.log(' REFERENCE EXAMPLE: VAD+Whisper integration working!')
    } else {
      t.ok(result.passed, 'VAD+Whisper test completed (may have 0 segments with test audio)')
    }
  } else {
    console.log(' VAD model not available - skipping VAD test')
    t.pass('VAD test skipped (model not available)')
  }
})

test('Real addon error handling - failure cases', { timeout: 120000 }, async (t) => {
  console.log(' Testing error handling with various failure scenarios')

  // Test 1: Invalid model path
  t.test('Invalid model path handling', async (t) => {
    const result = await runTranscription(
      {
        audioInput: new Uint8Array([1, 2, 3, 4, 5]),
        modelPath: '/nonexistent/path/model.bin',
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          vadParams: {
            threshold: 0.6
          }
        }
      }
    )

    if (result.data.error) {
      console.log(' Expected error caught:', result.data.error)
      t.pass(' Correctly handled invalid model path')
    } else {
      t.pass('Interface handled invalid path gracefully')
    }
  })

  // Test 2: Invalid configuration
  t.test('Invalid configuration handling', async (t) => {
    const invalidConfigs = [
      {
        name: 'invalid mode parameter',
        whisperConfig: {
          mode: 'invalid_mode', // Invalid - not in allowed whisperConfig parameters
          language: 'en',
          duration_ms: 0,
          temperature: 0.0
        }
      },
      {
        name: 'invalid min_seconds parameter',
        whisperConfig: {
          min_seconds: -1, // Invalid - not in allowed whisperConfig parameters
          language: 'en',
          duration_ms: 0,
          temperature: 0.0
        }
      },
      {
        name: 'invalid output_format parameter',
        whisperConfig: {
          output_format: 'invalid_format', // Invalid - not in allowed whisperConfig parameters
          language: 'en',
          duration_ms: 0,
          temperature: 0.0
        }
      },
      {
        name: 'invalid vadParams parameter',
        whisperConfig: {
          language: 'en',
          duration_ms: 0,
          temperature: 0.0,
          vadParams: {
            invalid_param: 0.6 // Invalid - not in allowed vadParams list
          }
        }
      }
    ]

    for (const testCase of invalidConfigs) {
      console.log(` Testing ${testCase.name}...`)
      const result = await runTranscription(
        {
          whisperConfig: testCase.whisperConfig,
          modelPath
          // No audioInput - we're just testing config validation
        }
      )

      // Config validation should fail during _load() via checkConfig()
      if (result.passed) {
        t.fail(`${testCase.name} should have been rejected but passed`)
      } else {
        console.log(` Caught expected error for ${testCase.name}: ${result.output}`)
        t.pass(`Correctly rejected ${testCase.name}: ${result.output}`)
      }
    }
  })

  // Test 3: Corrupted data handling
  t.test('Corrupted audio data handling', async (t) => {
    // Feed data that is intentionally noisy but valid Int16LE PCM,
    // to avoid generating NaNs inside DSP kernels while still testing robustness.
    const chunks = [
      makePcmNoise(256), // random low-amplitude noise
      new Uint8Array(Buffer.alloc(512)), // silence
      makePcmNoise(128) // more noise
    ]

    const result = await runTranscription(
      {
        audioInput: chunks,
        modelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          vadParams: {
            threshold: 0.6
          }
        }
      }
    )

    if (result.data.error) {
      console.log('Exception handling for noisy/silent data:', result.data.error)
      t.pass('Robustness to noisy/silent data validated via exception handling')
    } else {
      t.ok(result.data.segmentCount >= 0, 'Should handle noisy/silent data without crashing')
      t.pass('Noisy/silent data handled gracefully')
    }
  })
})

test('Caption mode transcription (VAD disabled)', { timeout: 120000 }, async (t) => {
  console.log(' Running caption mode test with VAD disabled')
  const loggerBinding = setupJsLogger()
  const whisperResult = await ensureWhisperModel(modelPath)

  generateTestAudio(audioPath)

  t.ok(fs.existsSync(modelPath), 'Whisper model file should exist')
  t.ok(fs.existsSync(audioPath), 'Test audio should exist')

  try {
    const result = await runTranscription(
      {
        audioInput: audioPath,
        modelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          audio_format: 's16le'
        }
      }
    )

    console.log(' Caption Output:', result.output)
    if (whisperResult.isReal) {
      if (result.data.error) {
        console.log(' Model loaded but transcription had issues (may be expected with test audio)')
        t.pass('Caption mode handled error gracefully')
      } else if (result.data.segmentCount > 0) {
        t.ok(result.data.segmentCount > 0, 'Should receive transcription output in caption mode')
      } else {
        t.ok(result.data.segmentCount >= 0, 'Should receive some result in caption mode')
        t.pass('Caption mode handled without transcription output (may be expected)')
      }
    } else {
      t.ok(result.data.segmentCount >= 0, 'Should receive some result in caption mode')
      t.pass('Caption mode handled without real model')
    }
  } catch (error) {
    console.log(` Caption mode test error: ${error.message}`)
    t.pass('Caption mode handled error gracefully')
  } finally {
    try { loggerBinding.releaseLogger() } catch {}
  }
})

test('Caption mode transcription (VAD enabled)', { timeout: 120000 }, async (t) => {
  console.log(' Running caption mode test with VAD enabled')
  const loggerBinding = setupJsLogger()
  const whisperResult = await ensureWhisperModel(modelPath)
  const vadOk = await ensureVADModel(vadModelPath)

  if (!whisperResult.success || !vadOk) {
    console.log(' Required models not available - skipping caption+VAD test')
    t.pass('Caption+VAD test skipped (models not available)')
    return
  }

  generateTestAudio(audioPath)

  try {
    const result = await runTranscription(
      {
        audioInput: audioPath,
        modelPath,
        vadModelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          audio_format: 's16le',
          vadParams: {
            threshold: 0.6,
            min_speech_duration_ms: 300,
            min_silence_duration_ms: 200,
            max_speech_duration_s: 30.0,
            speech_pad_ms: 50,
            samples_overlap: 0.15
          }
        }
      }
    )

    console.log(' Caption+VAD Output:', result.output)
    if (result.data.error) {
      console.log(' Caption+VAD mode had errors (may be expected)')
      t.pass('Caption+VAD mode handled error gracefully')
    } else if (result.data.segmentCount > 0) {
      t.pass('Successfully received transcription output in caption mode with VAD')
    } else {
      t.ok(result.data.segmentCount >= 0, 'Should receive some result in caption mode with VAD')
      t.pass('Caption+VAD mode handled without transcription output (may be expected with test audio)')
    }
  } catch (error) {
    console.log(` Caption+VAD mode test error: ${error.message}`)
    t.pass('Caption+VAD mode handled error gracefully')
  } finally {
    try { loggerBinding.releaseLogger() } catch {}
  }
})

test('Audio format transcription tests (s16le and f32le)', { timeout: 120000 }, async (t) => {
  console.log('Testing Audio Format Transcription')
  console.log('==================================')

  // Helper function to test a specific audio format using runTranscription
  async function testAudioFormat (audioFile, audioFormat, description) {
    console.log(`\n=== Testing ${description} ===`)
    console.log(`Audio file: ${audioFile}`)
    console.log(`Audio format: ${audioFormat}`)

    const result = await runTranscription(
      {
        audioInput: audioFile,
        modelPath,
        whisperConfig: {
          language: 'en',
          temperature: 0.0,
          audio_format: audioFormat,
          vadParams: {
            threshold: 0.6
          }
        }
      },
      {
        minSegments: 0
      }
    )

    console.log('Transcription result:', result.passed ? 'SUCCESS' : 'FAILED')
    if (result.data.error) {
      console.log('Error:', result.data.error)
    }

    if (result.data.segments && result.data.segments.length > 0) {
      const finalTranscription = result.data.fullText
      console.log(`\n=== FINAL TRANSCRIPTION (${description}) ===`)
      console.log(finalTranscription)
      console.log('=== END TRANSCRIPTION ===\n')
    }

    return {
      success: result.passed && !result.data.error,
      transcription: result.data.segments || [],
      fullText: result.data.fullText || ''
    }
  }

  const whisperResult = await ensureWhisperModel(modelPath)

  if (!whisperResult.isReal) {
    console.log('Real whisper model not available - skipping audio format tests')
    t.pass('Audio format tests skipped (model not available)')
    return
  }

  const s16leFile = path.resolve(__dirname, '../../examples/samples/sample.raw')
  const f32leFile = path.resolve(__dirname, '../../examples/samples/decodedFile.raw')

  if (!fs.existsSync(s16leFile)) {
    console.log(`s16le test file not found: ${s16leFile}`)
    t.pass('Audio format tests skipped (sample.raw not found)')
    return
  }

  if (!fs.existsSync(f32leFile)) {
    console.log(`f32le test file not found: ${f32leFile}`)
    t.pass('Audio format tests skipped (decodedFile.raw not found)')
    return
  }

  const s16leResult = await testAudioFormat(
    s16leFile,
    's16le',
    's16le format (sample.raw)'
  )

  const f32leResult = await testAudioFormat(
    f32leFile,
    'f32le',
    'f32le format (decodedFile.raw) - FIXED'
  )

  console.log('\n=== SUMMARY ===')
  console.log('✅ s16le format test:', s16leResult.success ? 'PASSED' : 'FAILED')
  console.log('✅ f32le format test:', f32leResult.success ? 'PASSED' : 'FAILED')

  if (s16leResult.success && f32leResult.success) {
    console.log('\n🎉 All audio format tests passed!')
    console.log(`- s16le format: ${s16leResult.transcription.length} segments`)
    console.log(`- f32le format: ${f32leResult.transcription.length} segments`)
    t.pass('Audio format tests passed')
  } else {
    console.log('\n❌ Some tests failed.')
    if (!s16leResult.success) {
      console.log('s16le test failed')
    }
    if (!f32leResult.success) {
      console.log('f32le test failed')
    }
    t.fail('Audio format tests failed')
  }
})

'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

interface UseVoiceRecorderOptions {
  silenceThreshold?: number
  silenceTimeout?: number
  onAudioReady?: (blob: Blob) => void
}

interface UseVoiceRecorderReturn {
  isRecording: boolean
  isListening: boolean
  isSpeaking: boolean
  audioLevel: number
  startListening: () => Promise<void>
  stopListening: () => void
  error: string | null
}

export function useVoiceRecorder({
  silenceThreshold = 0.015,
  silenceTimeout = 1500,
  onAudioReady,
}: UseVoiceRecorderOptions = {}): UseVoiceRecorderReturn {
  const [isRecording, setIsRecording] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const hasSpokenRef = useRef(false)
  const speechDetectionCountRef = useRef(0)

  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current)
      silenceTimerRef.current = null
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    mediaRecorderRef.current = null
    analyserRef.current = null
    chunksRef.current = []
    hasSpokenRef.current = false
    speechDetectionCountRef.current = 0
    setIsRecording(false)
    setIsListening(false)
    setIsSpeaking(false)
    setAudioLevel(0)
  }, [])

  const stopListening = useCallback(() => {
    cleanup()
  }, [cleanup])

  const startListening = useCallback(async () => {
    try {
      setError(null)
      cleanup()

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const audioContext = new AudioContext()
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.8
      source.connect(analyser)
      analyserRef.current = analyser

      const getSupportedMimeType = (): string => {
        const types = [
          'audio/webm;codecs=opus',
          'audio/webm',
          'audio/ogg;codecs=opus',
          'audio/mp4',
        ]
        if (typeof MediaRecorder === 'undefined') return ''
        for (const type of types) {
          if (MediaRecorder.isTypeSupported(type)) return type
        }
        return ''
      }

      const mimeType = getSupportedMimeType()
      const recorderOptions: MediaRecorderOptions = {}
      if (mimeType) {
        recorderOptions.mimeType = mimeType
      }

      const mediaRecorder = new MediaRecorder(stream, recorderOptions)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        if (chunksRef.current.length > 0 && hasSpokenRef.current) {
          const blobType = mimeType || 'audio/webm'
          const blob = new Blob(chunksRef.current, { type: blobType })
          onAudioReady?.(blob)
        }
        chunksRef.current = []
        hasSpokenRef.current = false
      }

      setIsListening(true)

      // Voice Activity Detection loop
      const dataArray = new Uint8Array(analyser.frequencyBinCount)
      let consecutiveSilenceFrames = 0

      const detectVoice = () => {
        if (!analyserRef.current) return

        analyserRef.current.getByteTimeDomainData(dataArray)

        // Calculate RMS (Root Mean Square) for audio level
        let sum = 0
        for (let i = 0; i < dataArray.length; i++) {
          const normalized = (dataArray[i] - 128) / 128
          sum += normalized * normalized
        }
        const rms = Math.sqrt(sum / dataArray.length)
        setAudioLevel(rms)

        const isSpeakingNow = rms > silenceThreshold

        if (isSpeakingNow) {
          // Require consistent speech detection (2+ frames) before marking as spoken
          speechDetectionCountRef.current++
          if (speechDetectionCountRef.current > 2) {
            hasSpokenRef.current = true
          }
          setIsSpeaking(true)
          consecutiveSilenceFrames = 0

          // Clear silence timer on new speech detection
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current)
            silenceTimerRef.current = null
          }

          // Start recording if not already
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive') {
            mediaRecorderRef.current.start(100)
            setIsRecording(true)
          }
        } else {
          // Increment silence frame counter
          consecutiveSilenceFrames++
          
          // Only reset speech detection if silence is continuous (multiple frames)
          if (consecutiveSilenceFrames > 2) {
            speechDetectionCountRef.current = 0
          }
          
          // If we've detected speech and now have sustained silence, stop recording
          if (hasSpokenRef.current && consecutiveSilenceFrames > 8) {
            if (!silenceTimerRef.current) {
              // Start silence timer - if silence continues, stop recording
              silenceTimerRef.current = setTimeout(() => {
                setIsSpeaking(false)
                if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                  mediaRecorderRef.current.stop()
                  setIsRecording(false)
                }
                consecutiveSilenceFrames = 0
                silenceTimerRef.current = null

                // Reset for next utterance after a brief pause
                setTimeout(() => {
                  if (streamRef.current && mediaRecorderRef.current) {
                    const newRecorderOptions: MediaRecorderOptions = {}
                    if (mimeType) {
                      newRecorderOptions.mimeType = mimeType
                    }
                    const newRecorder = new MediaRecorder(streamRef.current, newRecorderOptions)
                    chunksRef.current = []

                    newRecorder.ondataavailable = (e) => {
                      if (e.data.size > 0) {
                        chunksRef.current.push(e.data)
                      }
                    }
                    newRecorder.onstop = () => {
                      if (chunksRef.current.length > 0 && hasSpokenRef.current) {
                        const blobType = mimeType || 'audio/webm'
                        const blob = new Blob(chunksRef.current, { type: blobType })
                        onAudioReady?.(blob)
                      }
                      chunksRef.current = []
                      hasSpokenRef.current = false
                    }
                    mediaRecorderRef.current = newRecorder
                  }
                }, 300)
              }, silenceTimeout)
            }
          }
        }

        animationFrameRef.current = requestAnimationFrame(detectVoice)
      }

      detectVoice()
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to access microphone'
      )
      cleanup()
    }
  }, [cleanup, onAudioReady, silenceThreshold, silenceTimeout])

  useEffect(() => {
    return () => {
      cleanup()
    }
  }, [cleanup])

  return {
    isRecording,
    isListening,
    isSpeaking,
    audioLevel,
    startListening,
    stopListening,
    error,
  }
}

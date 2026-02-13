'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { useVoiceRecorder } from '@/hooks/use-voice-recorder'
import { WaveformVisualizer } from '@/components/waveform-visualizer'
import { Mic, MicOff, Loader2, Square, Play, Pause } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ChatMessage } from '@/hooks/use-websocket-chat'

interface VoiceViewProps {
  messages: ChatMessage[]
  isLoading: boolean
  onSendAudio: (blob: Blob, transcribedText?: string) => void
  onSendText: (text: string) => void
  isConnected: boolean
  onTranscription?: (text: string) => void
}

export function VoiceView({
  messages,
  isLoading,
  onSendAudio,
  onSendText,
  isConnected,
  onTranscription,
}: VoiceViewProps) {
  const [status, setStatus] = useState<
    'idle' | 'listening' | 'sending' | 'thinking'
  >('idle')
  const [lastAudioBlob, setLastAudioBlob] = useState<Blob | null>(null)
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null)
  const [audioUrls, setAudioUrls] = useState<Record<string, string>>({})
  const [transcribedText, setTranscribedText] = useState<string>('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handlePlayAudio = useCallback(
    (messageId: string, audioBlob: Blob) => {
      try {
        if (playingMessageId === messageId) {
          // Stop playing
          setPlayingMessageId(null)
          return
        }

        const url = audioUrls[messageId] || URL.createObjectURL(audioBlob)
        if (!audioUrls[messageId]) {
          setAudioUrls((prev) => ({ ...prev, [messageId]: url }))
        }

        const audio = new Audio(url)
        audio.onended = () => setPlayingMessageId(null)
        audio.play()
        setPlayingMessageId(messageId)
      } catch (error) {
        console.error('Error playing audio:', error)
      }
    },
    [playingMessageId, audioUrls]
  )

  // When backend responds (isLoading goes from true -> false), go back to listening
  useEffect(() => {
    if (!isLoading && status === 'thinking') {
      setStatus(isListeningRef.current ? 'listening' : 'idle')
    }
  }, [isLoading, status])

  const isListeningRef = useRef(false)

  const handleAudioReady = useCallback(
    (blob: Blob) => {
      if (!isConnected) return
      setStatus('sending')
      setLastAudioBlob(blob)
      setTranscribedText('') // Reset transcribed text
      // Send raw audio binary over WebSocket to Django backend
      // Backend will transcribe, get AI response, and return Eleven Labs TTS
      // Use transcribedText if available, otherwise will be updated from transcription callback
      onSendAudio(blob, transcribedText || undefined)
      setStatus('thinking')
    },
    [onSendAudio, isConnected, transcribedText]
  )

  const {
    isListening,
    isSpeaking,
    audioLevel,
    startListening,
    stopListening,
    error,
  } = useVoiceRecorder({
    silenceThreshold: 0.035,
    silenceTimeout: 1500,
    onAudioReady: handleAudioReady,
  })

  // Keep ref in sync
  useEffect(() => {
    isListeningRef.current = isListening
  }, [isListening])

  useEffect(() => {
    if (isListening && !isLoading) {
      setStatus('listening')
    } else if (isLoading) {
      setStatus('thinking')
    }
  }, [isListening, isLoading])

  const handleToggleListening = async () => {
    if (isListening) {
      stopListening()
      setStatus('idle')
    } else {
      await startListening()
      setStatus('listening')
    }
  }

  const statusLabel = {
    idle: 'Tap to start',
    listening: isSpeaking ? 'Listening...' : 'Speak now...',
    sending: 'Sending audio...',
    thinking: 'Processing...',
  }

  // Filter only voice-relevant messages
  const voiceMessages = messages.filter(
    (m) => m.role !== 'system'
  )

  return (
    <div className="flex h-full flex-col">
      {/* Transcription log */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto scroll-smooth px-4 pt-4"
      >
        {voiceMessages.length === 0 && status === 'idle' ? (
          <div className="flex h-full flex-col items-center justify-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Mic className="h-8 w-8" />
            </div>
            <div className="text-center">
              <h2 className="text-xl font-semibold text-foreground">
                Voice Mode
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Tap the microphone to start. Speak naturally and pause when
                done.
              </p>
              <p className="mt-1 text-xs text-muted-foreground/70">
                Your audio is sent to the server for transcription and AI
                response with voice.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-4xl space-y-4 pb-4">
            {voiceMessages.map((entry) => (
              <div key={entry.id} className="flex flex-col gap-2 px-4">
                {/* User voice message with two-column layout */}
                {entry.role === 'user' && entry.source === 'voice' ? (
                  <div className="flex gap-4 items-start">
                    {/* Left: Audio player */}
                    {entry.audioBlob && (
                      <button
                        onClick={() =>
                          handlePlayAudio(entry.id, entry.audioBlob!)
                        }
                        className="flex-shrink-0 flex items-center gap-2 rounded-2xl bg-primary/20 px-4 py-3 transition-colors hover:bg-primary/30 self-start"
                        aria-label={
                          playingMessageId === entry.id
                            ? 'Stop audio'
                            : 'Play audio'
                        }
                      >
                        {playingMessageId === entry.id ? (
                          <Pause className="h-5 w-5" />
                        ) : (
                          <Play className="h-5 w-5" />
                        )}
                        <span className="text-xs font-medium">Audio</span>
                      </button>
                    )}

                    {/* Right: Transcribed text */}
                    <div className="flex flex-col gap-1 flex-1">
                      <span className="px-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                        Transcription
                      </span>
                      <div className="rounded-2xl rounded-tl-md border border-primary/20 bg-primary/15 px-4 py-3 text-sm leading-relaxed text-foreground">
                        <p className="whitespace-pre-wrap">{entry.text}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Regular message layout for other messages */
                  <div
                    className={cn(
                      'flex flex-col gap-1',
                      entry.role === 'user' ? 'items-end' : 'items-start'
                    )}
                  >
                    <span className="px-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                      {entry.role === 'user' ? 'You' : 'Nova'}
                    </span>
                    <div
                      className={cn(
                        'max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
                        entry.role === 'user'
                          ? 'rounded-br-md border border-primary/20 bg-primary/15 text-foreground'
                          : 'rounded-bl-md bg-secondary text-secondary-foreground'
                      )}
                    >
                      <p className="whitespace-pre-wrap">{entry.text}</p>
                    </div>
                  </div>
                )}
                <span className="px-1 text-[10px] text-muted-foreground/60">
                  {entry.timestamp.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
              </div>
            ))}
            {isLoading && (
              <div className="flex flex-col items-start gap-1 px-4">
                <span className="px-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                  Nova
                </span>
                <div className="flex items-center gap-1 rounded-2xl rounded-bl-md bg-secondary px-4 py-3">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:0ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:150ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:300ms]" />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Voice control area */}
      <div className="flex flex-col items-center gap-4 px-4 pb-6 pt-4">
        {/* Waveform */}
        <div className="h-14 w-full max-w-sm">
          {isListening && (
            <WaveformVisualizer
              audioLevel={audioLevel}
              isActive={isSpeaking}
              className="h-full"
            />
          )}
        </div>

        {/* Status text */}
        <p
          className={cn(
            'text-sm font-medium transition-colors',
            status === 'listening' && isSpeaking
              ? 'text-primary'
              : status === 'sending'
                ? 'text-foreground'
                : 'text-muted-foreground'
          )}
        >
          {!isConnected ? 'Not connected' : statusLabel[status]}
        </p>

        {/* Mic button */}
        <button
          onClick={handleToggleListening}
          disabled={!isConnected || status === 'sending'}
          className={cn(
            'relative flex h-16 w-16 items-center justify-center rounded-full transition-all',
            isListening
              ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
              : 'bg-primary text-primary-foreground hover:bg-primary/90',
            (!isConnected || status === 'sending') && 'opacity-50'
          )}
          aria-label={isListening ? 'Stop listening' : 'Start listening'}
        >
          {/* Pulse rings when listening */}
          {isListening && (
            <>
              <span
                className="absolute inset-0 rounded-full bg-primary/20"
                style={{
                  animation: 'pulse-ring 2s ease-in-out infinite',
                }}
              />
              <span
                className="absolute inset-0 rounded-full bg-primary/10"
                style={{
                  animation: 'pulse-ring 2s ease-in-out infinite 0.5s',
                }}
              />
            </>
          )}

          {status === 'sending' ? (
            <Loader2 className="h-6 w-6 animate-spin" />
          ) : isListening ? (
            <Square className="h-5 w-5 fill-current" />
          ) : (
            <Mic className="h-6 w-6" />
          )}
        </button>

        {error && (
          <div className="flex items-center gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <MicOff className="h-3 w-3" />
            {error}
          </div>
        )}
      </div>
    </div>
  )
}

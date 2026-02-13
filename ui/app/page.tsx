'use client'

import { useState, useCallback, useRef } from 'react'
import { useWebSocketChat } from '@/hooks/use-websocket-chat'
import { ChatView } from '@/components/chat-view'
import { VoiceView } from '@/components/voice-view'
import { ModeSwitcher } from '@/components/mode-switcher'
import { Sparkles, Wifi, WifiOff, Loader2, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

type Mode = 'chat' | 'voice'

export default function Home() {
  const [mode, setMode] = useState<Mode>('chat')
  const [lastAudioBlob, setLastAudioBlob] = useState<Blob | null>(null)
  const audioQueueRef = useRef<ArrayBuffer[]>([])
  const isPlayingRef = useRef(false)
  const audioContextRef = useRef<AudioContext | null>(null)

  const playNextAudio = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return
    isPlayingRef.current = true

    const buffer = audioQueueRef.current.shift()
    if (!buffer) {
      isPlayingRef.current = false
      return
    }

    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext()
      }
      const audioBuffer = await audioContextRef.current.decodeAudioData(buffer.slice(0))
      const source = audioContextRef.current.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContextRef.current.destination)
      source.onended = () => {
        isPlayingRef.current = false
        playNextAudio()
      }
      source.start()
    } catch {
      isPlayingRef.current = false
      playNextAudio()
    }
  }, [])

  const handleAudioResponse = useCallback(
    (audioData: ArrayBuffer) => {
      audioQueueRef.current.push(audioData)
      playNextAudio()
    },
    [playNextAudio]
  )

  const {
    messages,
    connectionStatus,
    isThinking,
    connect,
    sendTextMessage,
    sendAudio,
  } = useWebSocketChat({
    onAudioResponse: handleAudioResponse,
    onTranscription: (text: string) => {
      // Store transcribed text to attach to audio message
      // This will be used when sendAudio is called
    },
  })

  const isConnected = connectionStatus === 'connected'
  const isUnconfigured = connectionStatus === 'unconfigured'

  return (
    <div className="flex h-dvh flex-col bg-background">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Sparkles className="h-4 w-4" />
          </div>
          <h1 className="text-lg font-semibold text-foreground">Voice</h1>

          {/* Connection status */}
          <div className="flex items-center gap-1.5">
            <span
              className={cn(
                'h-2 w-2 rounded-full',
                isConnected
                  ? 'bg-emerald-500'
                  : connectionStatus === 'connecting'
                    ? 'bg-amber-500 animate-pulse'
                    : isUnconfigured
                      ? 'bg-muted-foreground'
                      : 'bg-destructive'
              )}
            />
            {connectionStatus === 'connecting' ? (
              <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
            ) : isConnected ? (
              <Wifi className="h-3 w-3 text-emerald-500" />
            ) : isUnconfigured ? (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <AlertCircle className="h-3 w-3" />
                <span>Set NEXT_PUBLIC_WS_URL</span>
              </span>
            ) : (
              <button
                onClick={connect}
                className="flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
              >
                <WifiOff className="h-3 w-3" />
                <span>Reconnect</span>
              </button>
            )}
          </div>
        </div>

        <ModeSwitcher mode={mode} onModeChange={setMode} />
      </header>

      {/* Unconfigured banner */}
      {isUnconfigured && (
        <div className="border-b border-border bg-muted/50 px-4 py-2.5 text-center text-sm text-muted-foreground">
          Add your Django WebSocket URL as{' '}
          <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs text-foreground">
            NEXT_PUBLIC_WS_URL
          </code>{' '}
          in your environment variables to connect.
        </div>
      )}

      {/* Content */}
      <main className="flex-1 overflow-hidden">
        {mode === 'chat' ? (
          <ChatView
            messages={messages}
            isLoading={isThinking}
            onSend={sendTextMessage}
            isConnected={isConnected}
          />
        ) : (
          <VoiceView
            messages={messages}
            isLoading={isThinking}
            onSendAudio={sendAudio}
            onSendText={sendTextMessage}
            isConnected={isConnected}
            onTranscription={(text) => {
              // Transcription callback for voice view
            }}
          />
        )}
      </main>
    </div>
  )
}

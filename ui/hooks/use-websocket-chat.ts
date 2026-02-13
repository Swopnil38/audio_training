'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  timestamp: Date
  source?: 'text' | 'voice'
  audioBlob?: Blob
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error' | 'unconfigured'

interface UseWebSocketChatOptions {
  url?: string
  onTranscription?: (text: string) => void
  onAudioResponse?: (audioData: ArrayBuffer) => void
  onError?: (error: string) => void
  reconnectAttempts?: number
  reconnectDelay?: number
}

export function useWebSocketChat(options: UseWebSocketChatOptions = {}) {
  const {
    url = process.env.NEXT_PUBLIC_WS_URL || 'wss://audio.chitrakalastudio.art/ws/chat/',
    reconnectAttempts = 5,
    reconnectDelay = 1000,
  } = options

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>(
    url ? 'disconnected' : 'unconfigured'
  )
  const [isThinking, setIsThinking] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null)
  const lastAudioMessageIdRef = useRef<string | null>(null)
  // Store callbacks in refs so the connect function stays stable
  const onTranscriptionRef = useRef(options.onTranscription)
  const onAudioResponseRef = useRef(options.onAudioResponse)
  const onErrorRef = useRef(options.onError)

  useEffect(() => {
    onTranscriptionRef.current = options.onTranscription
  }, [options.onTranscription])
  useEffect(() => {
    onAudioResponseRef.current = options.onAudioResponse
  }, [options.onAudioResponse])
  useEffect(() => {
    onErrorRef.current = options.onError
  }, [options.onError])

  const connect = useCallback(() => {
    if (!url) {
      setConnectionStatus('unconfigured')
      return
    }

    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return

    setConnectionStatus('connecting')

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setConnectionStatus('connected')
        reconnectCountRef.current = 0
      }

      ws.onmessage = (event) => {
        // Binary data = Eleven Labs TTS audio
        if (event.data instanceof Blob) {
          event.data.arrayBuffer().then((buffer) => {
            onAudioResponseRef.current?.(buffer)
          })
          return
        }

        // Text data = JSON response
        try {
          const data = JSON.parse(event.data)

          // Handle nested update structure: {type: "update", data: {type: "chat_update", ...}}
          const message = data.data || data
          const messageType = message.type || data.type

          if (messageType === 'transcription') {
            const text = message.text?.trim()
            if (text) {
              onTranscriptionRef.current?.(text)
              // Update the last voice message's text with transcription
              setMessages((prev) => {
                const lastVoiceIndex = [...prev].reverse().findIndex(
                  (m) => m.role === 'user' && m.source === 'voice'
                )
                if (lastVoiceIndex >= 0) {
                  const actualIndex = prev.length - 1 - lastVoiceIndex
                  const updated = [...prev]
                  updated[actualIndex] = {
                    ...updated[actualIndex],
                    text: text,
                  }
                  return updated
                }
                return prev
              })
            }
          } else if (messageType === 'message' || messageType === 'response') {
            const responseText =
              message.message ||
              message.data?.message ||
              message.data?.text ||
              message.text ||
              ''

            if (responseText) {
              setIsThinking(false)
              setMessages((prev) => [
                ...prev,
                {
                  id: `assistant-${Date.now()}`,
                  role: 'assistant',
                  text: responseText,
                  timestamp: new Date(),
                },
              ])
            }
          } else if (messageType === 'chat_update') {
            // Handle message updates (e.g., translations, status updates)
            const displayText = message.translated_text || message.original_text || message.text
            
            if (displayText && lastAudioMessageIdRef.current) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === lastAudioMessageIdRef.current
                    ? { ...msg, text: displayText }
                    : msg
                )
              )
            }
          } else if (data.type === 'error') {
            setIsThinking(false)
            onErrorRef.current?.(message.message || 'Server error')
          } else if (data.type === 'notification') {
            setMessages((prev) => [
              ...prev,
              {
                id: `system-${Date.now()}`,
                role: 'system',
                text: message.message,
                timestamp: new Date(),
              },
            ])
          } else if (data.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }))
          }
        } catch {
          if (typeof event.data === 'string' && event.data.trim()) {
            setIsThinking(false)
            setMessages((prev) => [
              ...prev,
              {
                id: `assistant-${Date.now()}`,
                role: 'assistant',
                text: event.data,
                timestamp: new Date(),
              },
            ])
          }
        }
      }

      ws.onerror = () => {
        setConnectionStatus('error')
      }

      ws.onclose = () => {
        setConnectionStatus('disconnected')
        wsRef.current = null

        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++
          const delay = reconnectDelay * reconnectCountRef.current
          reconnectTimerRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }
    } catch {
      setConnectionStatus('error')
    }
  }, [url, reconnectAttempts, reconnectDelay])

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
    }
    reconnectCountRef.current = reconnectAttempts
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setConnectionStatus('disconnected')
  }, [reconnectAttempts])

  const sendTextMessage = useCallback((text: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      onErrorRef.current?.('Not connected to server')
      return
    }

    setMessages((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        text,
        timestamp: new Date(),
        source: 'text',
      },
    ])

    setIsThinking(true)

    wsRef.current.send(
      JSON.stringify({
        type: 'message',
        action: 'send_message',
        data: { message: text },
        timestamp: Date.now(),
      })
    )
  }, [])

  const sendAudio = useCallback((audioBlob: Blob, transcribedText?: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      onErrorRef.current?.('Not connected to server')
      return
    }

    // Add user message with audio blob attached
    const messageText = transcribedText || '[Voice message sent]'
    const messageId = `user-${Date.now()}`
    lastAudioMessageIdRef.current = messageId
    
    setMessages((prev) => [
      ...prev,
      {
        id: messageId,
        role: 'user',
        text: messageText,
        timestamp: new Date(),
        source: 'voice',
        audioBlob,
      },
    ])

    setIsThinking(true)

    audioBlob.arrayBuffer().then((buffer) => {
      wsRef.current?.send(buffer)
    })
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  // Auto-connect on mount, clean up on unmount
  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
      }
      reconnectCountRef.current = reconnectAttempts
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect, reconnectAttempts])

  return {
    messages,
    connectionStatus,
    isThinking,
    connect,
    disconnect,
    sendTextMessage,
    sendAudio,
    clearMessages,
  }
}

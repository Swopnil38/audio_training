'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Bot, User, Mic, Play, Pause } from 'lucide-react'
import type { ChatMessage as ChatMessageType } from '@/hooks/use-websocket-chat'

export function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'
  const [isPlaying, setIsPlaying] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  // Create audio URL from blob if available
  const handlePlayAudio = async () => {
    try {
      if (!message.audioBlob) return
      
      if (!audioUrl) {
        const url = URL.createObjectURL(message.audioBlob)
        setAudioUrl(url)
        const audio = new Audio(url)
        audio.onended = () => setIsPlaying(false)
        audio.play()
        setIsPlaying(true)
      } else {
        const audio = new Audio(audioUrl)
        audio.onended = () => setIsPlaying(false)
        audio.play()
        setIsPlaying(true)
      }
    } catch (error) {
      console.error('Error playing audio:', error)
    }
  }

  if (isSystem) {
    return (
      <div className="flex justify-center px-4 py-2">
        <span className="text-xs text-muted-foreground">{message.text}</span>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'flex gap-3 px-4 py-4',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Bot className="h-4 w-4" />
        </div>
      )}
      
      {/* Message container */}
      <div className="flex flex-col gap-2 max-w-[75%]">
        {/* Audio player for voice messages */}
        {isUser && message.audioBlob && message.source === 'voice' && (
          <button
            onClick={handlePlayAudio}
            className="flex items-center gap-2 rounded-2xl rounded-br-md bg-primary/20 px-3 py-2 text-sm transition-colors hover:bg-primary/30"
            aria-label={isPlaying ? 'Stop audio' : 'Play audio'}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            <span className="text-xs">Audio</span>
          </button>
        )}

        {/* Text message */}
        <div
          className={cn(
            'rounded-2xl px-4 py-3 text-sm leading-relaxed',
            isUser
              ? 'rounded-br-md bg-primary text-primary-foreground'
              : 'rounded-bl-md bg-secondary text-secondary-foreground'
          )}
        >
          {isUser && message.source === 'voice' && (
            <div className="mb-1 flex items-center gap-1 text-[10px] uppercase tracking-wider opacity-70">
              <Mic className="h-2.5 w-2.5" />
              <span>voice</span>
            </div>
          )}
          <p className="whitespace-pre-wrap">{message.text}</p>
        </div>
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  )
}

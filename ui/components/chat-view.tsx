'use client'

import { useRef, useEffect } from 'react'
import { ChatMessage } from '@/components/chat-message'
import { ChatInput } from '@/components/chat-input'
import { Bot } from 'lucide-react'
import type { ChatMessage as ChatMessageType } from '@/hooks/use-websocket-chat'

interface ChatViewProps {
  messages: ChatMessageType[]
  isLoading: boolean
  onSend: (text: string) => void
  isConnected: boolean
}

export function ChatView({ messages, isLoading, onSend, isConnected }: ChatViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-smooth">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 px-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Bot className="h-8 w-8" />
            </div>
            <div className="text-center">
              <h2 className="text-xl font-semibold text-foreground">
                How can I help you today?
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Start typing or switch to voice mode to begin a conversation.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl py-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex gap-3 px-4 py-4">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <Bot className="h-4 w-4" />
                </div>
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

      <div className="mx-auto w-full max-w-3xl px-4 pb-4 pt-2">
        <ChatInput onSend={onSend} isLoading={isLoading} disabled={!isConnected} />
      </div>
    </div>
  )
}

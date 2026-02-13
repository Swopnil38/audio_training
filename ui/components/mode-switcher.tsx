'use client'

import { MessageSquare, AudioLines } from 'lucide-react'
import { cn } from '@/lib/utils'

type Mode = 'chat' | 'voice'

interface ModeSwitcherProps {
  mode: Mode
  onModeChange: (mode: Mode) => void
}

export function ModeSwitcher({ mode, onModeChange }: ModeSwitcherProps) {
  return (
    <div className="flex items-center rounded-xl bg-secondary p-1">
      <button
        onClick={() => onModeChange('chat')}
        className={cn(
          'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all',
          mode === 'chat'
            ? 'bg-card text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground'
        )}
        aria-label="Switch to text chat mode"
      >
        <MessageSquare className="h-4 w-4" />
        <span className="hidden sm:inline">Chat</span>
      </button>
      <button
        onClick={() => onModeChange('voice')}
        className={cn(
          'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all',
          mode === 'voice'
            ? 'bg-card text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground'
        )}
        aria-label="Switch to voice mode"
      >
        <AudioLines className="h-4 w-4" />
        <span className="hidden sm:inline">Voice</span>
      </button>
    </div>
  )
}

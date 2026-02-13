'use client'

import { cn } from '@/lib/utils'

interface WaveformVisualizerProps {
  audioLevel: number
  isActive: boolean
  className?: string
}

export function WaveformVisualizer({ audioLevel, isActive, className }: WaveformVisualizerProps) {
  const bars = 24
  const amplifiedLevel = Math.min(audioLevel * 8, 1)

  return (
    <div
      className={cn('flex items-center justify-center gap-[3px]', className)}
      role="img"
      aria-label={isActive ? 'Voice activity detected' : 'Waiting for voice input'}
    >
      {Array.from({ length: bars }).map((_, i) => {
        const distFromCenter = Math.abs(i - bars / 2) / (bars / 2)
        const baseHeight = isActive
          ? Math.max(4, (1 - distFromCenter * 0.7) * amplifiedLevel * 48 + Math.random() * 8)
          : 4

        return (
          <div
            key={i}
            className={cn(
              'w-[3px] rounded-full transition-all duration-75',
              isActive ? 'bg-primary' : 'bg-muted-foreground/30'
            )}
            style={{
              height: `${baseHeight}px`,
              opacity: isActive ? 0.5 + amplifiedLevel * 0.5 : 0.3,
            }}
          />
        )
      })}
    </div>
  )
}

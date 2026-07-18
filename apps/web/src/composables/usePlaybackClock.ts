import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

type PlaybackClockOptions = {
  duration: () => number
  hasSourceVideo: () => boolean
  sourceStart: () => number
  sourceEnd: () => number
  canonicalToSourceTime: (time: number) => number
  sourceToCanonicalTime: (time: number) => number
}

/** Owns the animation clock and keeps source-video time mapped to canonical scene time. */
export function usePlaybackClock(options: PlaybackClockOptions) {
  const currentTime = ref(0)
  const playing = ref(false)
  const playbackRate = ref(1)
  const sourceVideo = ref<HTMLVideoElement | null>(null)
  let animationFrame = 0
  let previousTime = 0

  function seek(time: number) {
    currentTime.value = Math.max(0, Math.min(options.duration() || time, time))
    if (sourceVideo.value) {
      sourceVideo.value.currentTime = options.sourceStart() + options.canonicalToSourceTime(currentTime.value)
    }
  }

  function pauseAtPlayhead() {
    playing.value = false
    sourceVideo.value?.pause()
    seek(currentTime.value)
  }

  function toggle() {
    const duration = options.duration()
    if (duration <= 0) return
    if (currentTime.value >= duration) currentTime.value = 0
    playing.value = !playing.value
    if (!options.hasSourceVideo() || !sourceVideo.value) return
    sourceVideo.value.currentTime = options.sourceStart() + options.canonicalToSourceTime(currentTime.value)
    sourceVideo.value.playbackRate = playbackRate.value
    if (playing.value) void sourceVideo.value.play()
    else sourceVideo.value.pause()
  }

  function tick(timestamp: number) {
    if (!previousTime) previousTime = timestamp
    const delta = Math.min(0.05, (timestamp - previousTime) / 1000)
    previousTime = timestamp
    const duration = options.duration()
    const video = sourceVideo.value
    if (playing.value && duration > 0) {
      if (options.hasSourceVideo() && video) {
        video.playbackRate = playbackRate.value
        currentTime.value = Math.max(0, options.sourceToCanonicalTime(video.currentTime - options.sourceStart()))
      } else {
        currentTime.value += delta * playbackRate.value
      }
      if (
        currentTime.value >= duration
        || (video && video.currentTime >= options.sourceEnd())
        || video?.ended
      ) {
        currentTime.value = 0
        playing.value = false
        video?.pause()
        if (video) video.currentTime = options.sourceStart()
      }
    }
    animationFrame = requestAnimationFrame(tick)
  }

  watch(currentTime, (time) => {
    const video = sourceVideo.value
    const desiredTime = options.sourceStart() + options.canonicalToSourceTime(time)
    if (!playing.value && video && Math.abs(video.currentTime - desiredTime) > 0.08) {
      video.currentTime = desiredTime
    }
  })

  onMounted(() => {
    animationFrame = requestAnimationFrame(tick)
  })
  onBeforeUnmount(() => {
    cancelAnimationFrame(animationFrame)
  })

  return {
    currentTime,
    playing,
    playbackRate,
    sourceVideo,
    seek,
    pauseAtPlayhead,
    toggle,
  }
}

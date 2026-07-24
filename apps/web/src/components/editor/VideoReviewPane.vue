<script setup lang="ts">
import type { ComponentPublicInstance, CSSProperties } from 'vue'
import type { SceneVideoAsset } from '../../types/scene'
import type { VideoReviewTransform } from '../../lib/videoReviewTransform'

defineProps<{
  asset: SceneVideoAsset
  transform: VideoReviewTransform
  transformStyle: CSSProperties
  zoomPercent: number
  panning: boolean
  minScale: number
  maxScale: number
  caption: string
}>()

const emit = defineEmits<{
  viewportElement: [element: HTMLDivElement | null]
  videoElement: [element: HTMLVideoElement | null]
  loadedMetadata: []
  wheel: [event: WheelEvent]
  pointerDown: [event: PointerEvent]
  pointerMove: [event: PointerEvent]
  pointerUp: [event: PointerEvent]
  pointerCancel: [event: PointerEvent]
  keydown: [event: KeyboardEvent]
  adjustZoom: [delta: number]
  reset: []
}>()

function bindViewport(element: Element | ComponentPublicInstance | null) {
  emit('viewportElement', element instanceof HTMLDivElement ? element : null)
}

function bindVideo(element: Element | ComponentPublicInstance | null) {
  emit('videoElement', element instanceof HTMLVideoElement ? element : null)
}
</script>

<template>
  <div class="reference-pane">
    <div
      :ref="bindViewport"
      class="video-review-viewport"
      :class="{ pannable: transform.scale > minScale, panning }"
      tabindex="0"
      role="region"
      aria-label="Video review frame. Use plus and minus to zoom, arrow keys to pan, and zero or Home to reset."
      aria-keyshortcuts="+ - 0 Home ArrowLeft ArrowRight ArrowUp ArrowDown"
      @wheel.prevent="emit('wheel', $event)"
      @pointerdown="emit('pointerDown', $event)"
      @pointermove="emit('pointerMove', $event)"
      @pointerup="emit('pointerUp', $event)"
      @pointercancel="emit('pointerCancel', $event)"
      @keydown="emit('keydown', $event)"
    >
      <div class="video-review-transform" :style="transformStyle">
        <video
          :ref="bindVideo"
          :src="asset.mediaUrl"
          :poster="asset.posterUrl"
          muted
          playsinline
          preload="auto"
          @loadedmetadata="emit('loadedMetadata')"
        />
        <slot />
      </div>
    </div>

    <div class="video-review-overlay-controls">
      <slot name="overlay-controls" />
    </div>
    <div class="video-review-controls" role="group" aria-label="Video review zoom controls">
      <button
        type="button"
        aria-label="Zoom out video review"
        :disabled="transform.scale <= minScale"
        @click="emit('adjustZoom', -0.25)"
      >−</button>
      <output aria-live="polite" aria-label="Video review zoom">{{ zoomPercent }}%</output>
      <button
        type="button"
        aria-label="Zoom in video review"
        :disabled="transform.scale >= maxScale"
        @click="emit('adjustZoom', 0.25)"
      >+</button>
      <button
        type="button"
        class="reset"
        aria-label="Reset video review zoom and pan"
        :disabled="transform.scale === minScale && transform.x === 0 && transform.y === 0"
        @click="emit('reset')"
      >Reset</button>
    </div>

    <slot name="floating" />
    <div class="reference-label"><i /> {{ caption }}</div>
    <div class="reference-meta">{{ asset.filename }} · {{ asset.fps.toFixed(2) }} FPS</div>
  </div>
</template>

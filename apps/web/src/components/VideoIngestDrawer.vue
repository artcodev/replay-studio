<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { mediaClient } from '../lib/api/media'
import type { VideoAsset } from '../types/media'

const props = withDefaults(defineProps<{
  open: boolean
  projectId?: string | null
  projectTitle?: string | null
}>(), {
  projectId: null,
  projectTitle: null,
})
const emit = defineEmits<{
  close: []
  ready: [asset: VideoAsset]
}>()

const fileInput = ref<HTMLInputElement | null>(null)
const selectedFile = ref<File | null>(null)
const title = ref('')
const localPreview = ref<string | null>(null)
const asset = ref<VideoAsset | null>(null)
const uploadProgress = ref(0)
const uploading = ref(false)
const dragging = ref(false)
const error = ref<string | null>(null)
let pollTimer = 0

const progress = computed(() => (asset.value ? asset.value.progress : uploadProgress.value))
const stage = computed(() => {
  if (asset.value) return asset.value.stage
  if (uploading.value) return 'Uploading original clip'
  return 'Select a continuous gameplay clip'
})

function chooseFile(file: File | undefined) {
  if (!file) return
  error.value = null
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|mkv|webm|m4v)$/i.test(file.name)) {
    error.value = 'Choose an MP4, MOV, MKV, WebM or M4V video.'
    return
  }
  if (file.size > 250 * 1024 * 1024) {
    error.value = 'The clip is larger than 250 MB.'
    return
  }
  if (localPreview.value) URL.revokeObjectURL(localPreview.value)
  selectedFile.value = file
  title.value = file.name.replace(/\.[^.]+$/, '')
  localPreview.value = URL.createObjectURL(file)
}

function onDrop(event: DragEvent) {
  dragging.value = false
  chooseFile(event.dataTransfer?.files[0])
}

async function pollAsset(id: string) {
  if (!props.projectId) return
  try {
    asset.value = await mediaClient.get(props.projectId, id)
    if (asset.value.status === 'ready') {
      uploading.value = false
      emit('ready', asset.value)
      return
    }
    if (asset.value.status === 'failed') {
      uploading.value = false
      error.value = asset.value.error || 'Video processing failed.'
      return
    }
    if (asset.value.status === 'cancelled') {
      uploading.value = false
      error.value = 'Video processing was cancelled.'
      return
    }
    pollTimer = window.setTimeout(() => pollAsset(id), 650)
  } catch (cause) {
    uploading.value = false
    error.value = cause instanceof Error ? cause.message : 'Could not read processing status.'
  }
}

async function submit() {
  if (!selectedFile.value || !props.projectId || uploading.value) return
  uploading.value = true
  uploadProgress.value = 1
  error.value = null
  try {
    asset.value = await mediaClient.upload(
      props.projectId,
      selectedFile.value,
      title.value,
      (value) => {
        uploadProgress.value = Math.max(1, Math.min(99, value))
      },
    )
    await pollAsset(asset.value.id)
  } catch (cause) {
    uploading.value = false
    error.value = cause instanceof Error ? cause.message : 'Video upload failed.'
  }
}

function reset() {
  window.clearTimeout(pollTimer)
  if (localPreview.value) URL.revokeObjectURL(localPreview.value)
  selectedFile.value = null
  localPreview.value = null
  asset.value = null
  title.value = ''
  uploadProgress.value = 0
  uploading.value = false
  error.value = null
}

watch(
  () => props.open,
  (open) => {
    if (!open && !uploading.value) reset()
  },
)

onBeforeUnmount(reset)
</script>

<template>
  <transition name="drawer">
    <div v-if="open" class="drawer-backdrop ingest-backdrop" @click.self="!uploading && emit('close')">
      <aside class="catalog-drawer ingest-drawer">
        <div class="drawer-heading">
          <div><p class="eyebrow">Video ingestion</p><h2>Build from a clip</h2></div>
          <button class="icon-button" :disabled="uploading" @click="emit('close')">×</button>
        </div>
        <p v-if="projectTitle" class="drawer-project">Project · <strong>{{ projectTitle }}</strong></p>
        <p class="drawer-copy">Upload a gameplay clip or highlight montage. We split it into continuous shots and open the strongest shot candidate for review.</p>
        <p v-if="!projectId" class="ingest-error" role="alert">Choose or create a project before uploading video.</p>

        <button
          class="drop-zone"
          :class="{ dragging, filled: selectedFile }"
          :disabled="uploading"
          @click="fileInput?.click()"
          @dragenter.prevent="dragging = true"
          @dragover.prevent
          @dragleave.prevent="dragging = false"
          @drop.prevent="onDrop"
        >
          <input ref="fileInput" type="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.m4v" hidden @change="chooseFile(($event.target as HTMLInputElement).files?.[0])" />
          <template v-if="localPreview">
            <video :src="localPreview" muted playsinline />
            <span class="file-badge">SOURCE CLIP</span>
            <div class="file-overlay"><strong>{{ selectedFile?.name }}</strong><small>{{ ((selectedFile?.size || 0) / 1048576).toFixed(1) }} MB · click to replace</small></div>
          </template>
          <template v-else>
            <span class="upload-glyph">↥</span>
            <strong>Drop football video here</strong>
            <small>MP4, MOV, MKV or WebM · max 250 MB</small>
          </template>
        </button>

        <div v-if="selectedFile" class="ingest-form">
          <label>Moment title<input v-model="title" :disabled="uploading" /></label>
          <div class="pipeline-list">
            <div :class="{ active: uploading && !asset, done: asset }"><i>01</i><span><strong>Upload</strong><small>Keep the source private</small></span></div>
            <div :class="{ active: asset?.stage.includes('proxy'), done: (asset?.progress || 0) > 58 }"><i>02</i><span><strong>Normalize</strong><small>H.264 browser proxy</small></span></div>
            <div :class="{ active: asset?.stage.includes('frames'), done: (asset?.progress || 0) > 90 }"><i>03</i><span><strong>Sample frames</strong><small>10 FPS analysis input</small></span></div>
            <div :class="{ active: asset?.stage.includes('scene') || asset?.stage.includes('moments'), done: asset?.status === 'ready' }"><i>04</i><span><strong>Prepare moments</strong><small>Rank shots for explicit reconstruction</small></span></div>
          </div>
        </div>

        <div v-if="uploading || asset" class="processing-status">
          <div><span>{{ stage }}</span><strong>{{ progress }}%</strong></div>
          <div class="processing-bar"><i :style="{ width: `${progress}%` }" /></div>
          <small v-if="asset?.duration">{{ asset.width }}×{{ asset.height }} · {{ asset.duration.toFixed(2) }}s · {{ asset.fps?.toFixed(2) }} FPS</small>
        </div>
        <p v-if="error" class="ingest-error">{{ error }}</p>

        <div class="ingest-footer">
          <p><span>LOCAL PROCESSING</span> Source video is stored in the project media volume.</p>
          <button class="button primary" :disabled="!selectedFile || !projectId || uploading" @click="submit">
            {{ uploading ? stage : 'Process clip' }}
          </button>
        </div>
      </aside>
    </div>
  </transition>
</template>

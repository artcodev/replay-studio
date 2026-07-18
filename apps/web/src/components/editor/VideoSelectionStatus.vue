<script setup lang="ts">
defineProps<{
  status: {
    state: string
    label: string
    detail: string
  }
}>()
</script>

<template>
  <div
    class="video-selection-status"
    :class="status.state"
    role="status"
    aria-live="polite"
  >
    <i aria-hidden="true" />
    <span>
      <strong>{{ status.label }}</strong>
      <small>{{ status.detail }}</small>
    </span>
  </div>
</template>

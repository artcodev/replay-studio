import {
  DEFAULT_VIDEO_OVERLAY_OPTIONS,
  VIDEO_OVERLAY_LAYER_ITEMS,
  type VideoOverlayOptions,
} from './videoOverlayOptions'

export const VIDEO_OVERLAY_PREFERENCES_KEY = 'replay-studio:video-overlay:v1'

export function parseVideoOverlayOptions(value: string | null): VideoOverlayOptions | null {
  if (!value) return null
  try {
    const parsed = JSON.parse(value) as Partial<VideoOverlayOptions> | null
    if (!parsed || typeof parsed !== 'object') return null
    // A layer added after this payload was written falls back to its default;
    // any present field must still be a real boolean so a corrupt payload is
    // rejected instead of silently disabling overlays.
    const options = { ...DEFAULT_VIDEO_OVERLAY_OPTIONS } as VideoOverlayOptions
    for (const item of VIDEO_OVERLAY_LAYER_ITEMS) {
      const stored = parsed[item.key]
      if (stored === undefined) continue
      if (typeof stored !== 'boolean') return null
      options[item.key] = stored
    }
    return options
  } catch {
    return null
  }
}

export function loadVideoOverlayOptions(storage: Storage): VideoOverlayOptions | null {
  try {
    return parseVideoOverlayOptions(storage.getItem(VIDEO_OVERLAY_PREFERENCES_KEY))
  } catch {
    return null
  }
}

export function saveVideoOverlayOptions(storage: Storage, options: VideoOverlayOptions) {
  try {
    storage.setItem(VIDEO_OVERLAY_PREFERENCES_KEY, JSON.stringify(options))
  } catch {
    // Overlay preferences are a convenience: a full or blocked storage must
    // never break the editor.
  }
}

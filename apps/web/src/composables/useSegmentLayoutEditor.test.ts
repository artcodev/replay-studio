import { describe, expect, it, vi } from 'vitest'
import { ref, type Ref } from 'vue'
import { useSegmentLayoutEditor } from './useSegmentLayoutEditor'
import type { SceneVideo } from '../features/timeline/segmentLayout'

function video(): SceneVideo {
  return {
    id: 'asset-1',
    segments: [
      {
        id: 'shot-01',
        label: 'Shot 01',
        start: 0,
        end: 4,
        layout: { group: 1, variant: 'A', label: '1-A', role: 'original', confidence: 0.9 },
      },
      {
        id: 'shot-02',
        label: 'Shot 02',
        start: 4,
        end: 10,
        layout: { group: 1, variant: 'B', label: '1-B', role: 'continuation', confidence: 0.9 },
      },
    ],
    segmentLayout: { status: 'proposed', groups: [] },
  } as unknown as SceneVideo
}

describe('segment layout autosave', () => {
  function editor(saveSegmentLayout: () => Promise<void>, sceneVideo: Ref<SceneVideo | null>) {
    return useSegmentLayoutEditor({
      scene: ref(null),
      sceneVideo,
      selectedTrackId: ref(null),
      selectedCanonicalPersonId: ref(null),
      currentTime: ref(0),
      saveState: ref(''),
      error: ref(null),
      projectId: () => 'project-1',
      saveSegmentLayout,
      seekTo: () => {},
      writeRouteTime: () => {},
      notifySceneMutation: () => {},
    })
  }

  it('persists a group edit shortly after the change, without the save button', async () => {
    vi.useFakeTimers()
    const saveSegmentLayout = vi.fn(async () => {})
    const sceneVideo = ref<SceneVideo | null>(video())
    const layout = editor(saveSegmentLayout, sceneVideo)

    // Assign shot-02 to group 2 exactly like the dropdown does.
    layout.assignGroup(sceneVideo.value!.segments![1], '2')

    expect(saveSegmentLayout).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(900)
    expect(saveSegmentLayout).toHaveBeenCalledTimes(1)
    expect(sceneVideo.value!.segments![1].layout!.group).toBe(2)

    // Rapid consecutive edits collapse into one save.
    layout.assignGroup(sceneVideo.value!.segments![0], '2')
    layout.assignGroup(sceneVideo.value!.segments![1], '3')
    await vi.advanceTimersByTimeAsync(900)
    expect(saveSegmentLayout).toHaveBeenCalledTimes(2)
    vi.useRealTimers()
  })

  it('autosaves a split so it survives a scene refresh', async () => {
    vi.useFakeTimers()
    const saveSegmentLayout = vi.fn(async () => {})
    const sceneVideo = ref<SceneVideo | null>(video())
    const layout = editor(saveSegmentLayout, sceneVideo)

    // Split shot-02 into its own event via the split control.
    layout.selection.value = [sceneVideo.value!.segments![1].id]
    layout.splitSelection()

    await vi.advanceTimersByTimeAsync(900)
    expect(saveSegmentLayout).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })
})

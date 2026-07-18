<script setup lang="ts">
import type { FrameAnnotationDraft } from '../../features/frame-annotations/frameAnnotationDraft'
import type {
  buildFrameIdentityMergeTargets,
  buildFrameIdentitySplitPreview,
} from '../../features/frame-annotations/frameAnnotationRules'
import type { FrameAnalysis, FrameAnnotationKind, FrameIdentityAction } from '../../types/analysis'

type Person = FrameAnalysis['people'][number]
type MergeTarget = ReturnType<typeof buildFrameIdentityMergeTargets>[number]
type SplitPreview = ReturnType<typeof buildFrameIdentitySplitPreview>

defineProps<{
  analysis: FrameAnalysis
  active: boolean
  annotationMode: boolean
  draft: FrameAnnotationDraft | null
  identityActions: Array<{ value: FrameIdentityAction; label: string }>
  annotationKinds: Array<{ value: FrameAnnotationKind; label: string }>
  mergeTargets: MergeTarget[]
  splitPreview: SplitPreview
  sceneDuration: number
  selectedPersonId: string | null
  saving: boolean
  reconstructing: boolean
  reconstructionStatus: string | undefined
  saveDisabled: boolean
  personLabel: (person: Person) => string
  personCanonicalId: (person: Person) => string | null
  metricBadge: (person: Person) => string | null
}>()

const emit = defineEmits<{
  'toggle-mode': []
  'action-change': [event: Event]
  delete: []
  save: []
  'select-person': [person: Person]
}>()
</script>

<template>
  <div class="frame-analysis-card" :class="{ stale: !active }">
    <div class="frame-analysis-heading">
      <div><span>Frame recognition</span><strong>{{ analysis.sceneTime.toFixed(2) }}s · #{{ analysis.frameIndex }}</strong></div>
      <div class="frame-analysis-actions">
        <i>{{ active ? 'CURRENT' : 'MOVE PLAYHEAD BACK' }}</i>
        <button :class="{ active: annotationMode }" :disabled="!active" @click="emit('toggle-mode')">
          {{ annotationMode ? 'CLOSE LABELS' : 'LABEL FRAME' }}
        </button>
      </div>
    </div>
    <div v-if="annotationMode" class="frame-annotation-editor">
      <p v-if="!draft">Click an existing box, including an unmatched detection, or drag a new box around any person in the video.</p>
      <template v-else>
        <label>
          <span>Identity correction</span>
          <select v-model="draft.action" aria-label="Frame identity correction" @change="emit('action-change', $event)">
            <option v-for="item in identityActions" :key="item.value" :value="item.value" :disabled="item.value === 'merge' && !mergeTargets.length">{{ item.label }}</option>
          </select>
        </label>
        <label v-if="draft.action === 'merge'">
          <span>Merge into</span>
          <select v-model="draft.mergeTargetId" aria-label="Identity merge target">
            <option :value="null">Choose existing identity</option>
            <option v-for="target in mergeTargets" :key="`${target.type}-${target.id}`" :value="target.id">
              {{ target.type === 'canonical' ? 'Canonical person' : target.type === 'track' ? '3D render track' : 'Manual person' }} · {{ target.label }}
            </option>
          </select>
        </label>
        <small v-if="draft.action === 'merge' && mergeTargets.length" class="split-warning">
          Targets with incompatible dedicated Bind / Unbind decisions are unavailable. Resolve roster decisions in the canonical identity inspector first.
        </small>
        <small v-else-if="draft.action === 'merge'" class="split-warning">
          No compatible merge targets. Choose another correction or resolve roster decisions in the canonical identity inspector.
        </small>
        <div v-if="draft.action === 'split'" class="identity-split-preview" role="status">
          <div>
            <strong>Split {{ splitPreview?.identityLabel }}</strong>
            <small>
              Selected observation {{ draft.targetObservationId ? 'is pinned' : 'is unavailable' }}
              <template v-if="splitPreview?.targetTime != null"> at {{ splitPreview.targetTime.toFixed(2) }}s</template>.
            </small>
          </div>
          <div class="identity-split-range">
            <label><span>Range start</span><input v-model.number="draft.rangeStart" type="number" min="0" :max="sceneDuration" step="0.01" aria-label="Identity split range start" /></label>
            <label><span>Range end · exclusive</span><input v-model.number="draft.rangeEnd" type="number" min="0" :max="sceneDuration" step="0.01" aria-label="Identity split range end" /></label>
          </div>
          <small v-if="splitPreview?.affected != null">Preview · {{ splitPreview.affected }} observation(s) become a new identity; {{ splitPreview.remaining }} stay on the current identity.</small>
          <small v-else>Preview counts require a rebuilt canonical observation graph.</small>
          <small class="split-warning">[start, end) is a cannot-link barrier. Ambiguous remapping aborts the rebuild instead of splitting a nearby player.</small>
        </div>
        <label v-if="draft.action === 'exclude'">
          <span>Exclude scope</span>
          <select v-model="draft.scope" aria-label="Identity exclusion scope">
            <option value="observation">This observation only</option>
            <option value="identity" :disabled="!draft.canonicalPersonId && !draft.sourceTrackId">Whole canonical identity</option>
          </select>
        </label>
        <label v-if="draft.action === 'confirm'">
          <span>Meaning</span>
          <select v-model="draft.kind" aria-label="Frame person meaning">
            <option v-for="item in annotationKinds" :key="item.value" :value="item.value">{{ item.label }}</option>
          </select>
        </label>
        <label v-if="draft.action === 'confirm' && draft.kind !== 'ignore'">
          <span>Label</span><input v-model="draft.label" aria-label="Frame person label" placeholder="Player A, Player B…" />
        </label>
        <small v-if="draft.action === 'confirm' && (draft.kind.startsWith('home-') || draft.kind.startsWith('away-'))">
          Confirm the observation here; bind a roster player from the canonical identity inspector.
        </small>
        <small>Box {{ Math.round(draft.bbox.x) }}, {{ Math.round(draft.bbox.y) }} · {{ Math.round(draft.bbox.width) }}×{{ Math.round(draft.bbox.height) }} px</small>
        <div class="frame-annotation-buttons">
          <button v-if="draft.annotationId" class="delete" :disabled="saving || reconstructing || reconstructionStatus === 'queued' || reconstructionStatus === 'processing'" @click="emit('delete')">Delete</button>
          <button class="save" :disabled="saveDisabled || saving || reconstructing || reconstructionStatus === 'queued' || reconstructionStatus === 'processing'" @click="emit('save')">{{ saving ? 'Saving…' : 'Save correction' }}</button>
        </div>
      </template>
      <small>Preview updates immediately. Saving queues a revision-safe tracking rebuild. Split uses an exclusive-end range and can be undone by deleting its correction.</small>
    </div>
    <div class="frame-analysis-stats">
      <span><strong>{{ analysis.people.length }}</strong> people</span>
      <span><strong>{{ analysis.matchedTracks }}</strong> matched</span>
      <span><strong>{{ analysis.ballCandidates.length }}</strong> ball candidates</span>
    </div>
    <div class="frame-detection-list">
      <button v-for="person in analysis.people" :key="person.id" :class="{ selected: person.id === selectedPersonId && active }" @click="emit('select-person', person)">
        <i :style="{ background: person.jerseyColor }" />
        <span><strong>{{ personLabel(person) }}</strong><small>{{ person.previewState === 'merged' ? `merge → ${person.mergeTargetId} · ` : person.previewState === 'split' ? `split [${person.rangeStart?.toFixed(2)}, ${person.rangeEnd?.toFixed(2)}) · ` : person.previewState === 'confirmed' ? 'confirmed · ' : '' }}{{ person.kind ? `${person.kind} · ` : '' }}{{ personCanonicalId(person) ? `${personCanonicalId(person)} · ` : '' }}x {{ person.pitch.x.toFixed(1) }} · z {{ person.pitch.z.toFixed(1) }}{{ person.matchDistance !== null ? ` · Δ${person.matchDistance.toFixed(1)}m` : '' }} <em v-if="metricBadge(person)" :class="{ uncertain: metricBadge(person) === 'UNCERTAIN' }">{{ metricBadge(person) }}</em></small></span>
        <b :class="person.previewState">{{ person.previewState === 'merged' ? 'MERGED' : person.previewState === 'split' ? 'SPLIT' : person.previewState === 'confirmed' ? 'CONFIRMED' : `${Math.round(person.confidence * 100)}%` }}</b>
      </button>
    </div>
    <div v-if="analysis.ballCandidates.length" class="frame-ball-list">
      <span v-for="ball in analysis.ballCandidates" :key="ball.id" :class="{ primary: ball.primary }">
        <i /> {{ ball.primary ? 'Selected ball' : 'Candidate' }} · {{ Math.round(ball.confidence * 100) }}% · x {{ ball.pitch.x.toFixed(1) }} · z {{ ball.pitch.z.toFixed(1) }}
      </span>
    </div>
    <small v-for="warning in analysis.warnings" :key="warning" class="frame-analysis-warning">{{ warning }}</small>
  </div>
</template>

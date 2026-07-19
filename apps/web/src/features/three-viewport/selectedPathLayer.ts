import * as THREE from 'three'
import {
  buildPathTrackingSegments,
  interpolatePathTrackingSegments,
  pathTrackingOptionsForSubject,
  pathTrackingPoints,
  type PathTrackingSegment,
} from '../../lib/pathTracking'
import type { SceneDocument } from '../../types/scene'
import type { Track } from '../../types/tracking'
import { disposeObjectResources } from './threeResources'

type SelectedPathSource = {
  subjectId: string
  keyframes: Track['keyframes']
  color: THREE.Color
  ball: boolean
}

export function resolveSelectedPathSource(
  scene: SceneDocument,
  selectedTrackId: string | null,
  ballSelected: boolean,
): SelectedPathSource | null {
  if (ballSelected) {
    return {
      subjectId: 'ball',
      keyframes: scene.payload.ball.keyframes,
      color: new THREE.Color(0x5ee7ff),
      ball: true,
    }
  }
  if (!selectedTrackId) return null
  const track = scene.payload.tracks.find((candidate) => candidate.id === selectedTrackId)
  return track ? {
    subjectId: track.id,
    keyframes: track.keyframes,
    color: new THREE.Color(track.color),
    ball: false,
  } : null
}

function pointHeight(point: PathTrackingSegment['points'][number], ball: boolean) {
  return ball ? Math.max(0.28, point.y ?? 0.22) + 0.08 : 0.115
}

function createLine(segment: PathTrackingSegment, source: SelectedPathSource) {
  const geometry = new THREE.BufferGeometry().setFromPoints(segment.points.map(
    (point) => new THREE.Vector3(point.x, pointHeight(point, source.ball), point.z),
  ))
  geometry.computeBoundingSphere()
  const observed = segment.evidence === 'observed'
  const color = source.color.clone()
  color.offsetHSL(0, observed ? -0.04 : -0.12, observed ? 0.16 : 0.28)
  const commonMaterial = {
    color,
    transparent: true,
    opacity: observed ? 0.98 : 0.62,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
  }
  const material = observed
    ? new THREE.LineBasicMaterial(commonMaterial)
    : new THREE.LineDashedMaterial({ ...commonMaterial, dashSize: 0.65, gapSize: 0.48 })
  const pathLine = new THREE.Line(geometry, material)
  pathLine.computeLineDistances()
  pathLine.renderOrder = observed ? 13 : 12
  pathLine.userData.kind = observed ? 'path-observed' : 'path-inferred'
  return pathLine
}

/**
 * Renders the full trajectory of every player track at once — a synoptic
 * overview of the whole moment. Lines reuse the selected-path styling at a
 * reduced opacity so the focused selected path stays readable on top.
 */
export class AllPathsLayer {
  private group: THREE.Group | null = null

  constructor(private readonly target: THREE.Scene) {}

  rebuild(scene: SceneDocument, visible: boolean) {
    if (this.group) {
      this.target.remove(this.group)
      disposeObjectResources([this.group])
      this.group = null
    }
    if (!visible) return
    this.group = new THREE.Group()
    this.group.userData.kind = 'all-track-paths'
    for (const track of scene.payload.tracks) {
      const normalized = pathTrackingPoints(track.keyframes).map((point) => point.keyframe)
      const segments = buildPathTrackingSegments(
        normalized,
        pathTrackingOptionsForSubject('player'),
      )
      const source: SelectedPathSource = {
        subjectId: track.id,
        keyframes: track.keyframes,
        color: new THREE.Color(track.color),
        ball: false,
      }
      for (const segment of segments) {
        const line = createLine(segment, source)
        const material = line.material as THREE.Material & { opacity: number }
        material.opacity *= 0.42
        line.renderOrder -= 4
        line.userData.kind = `all-paths-${line.userData.kind}`
        this.group.add(line)
      }
    }
    this.target.add(this.group)
  }

  dispose() {
    if (this.group) {
      this.target.remove(this.group)
      disposeObjectResources([this.group])
      this.group = null
    }
  }
}


export class SelectedPathLayer {
  private group: THREE.Group | null = null
  private cursor: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial> | null = null
  private playback: { subjectId: string; segments: PathTrackingSegment[]; ball: boolean } | null = null

  constructor(private readonly target: THREE.Scene) {}

  get replacesBallTrail() {
    return this.playback?.ball === true && this.group?.userData.hasPath === true && this.group.visible
  }

  rebuild(source: SelectedPathSource | null, visible: boolean, currentTime: number) {
    if (this.group) {
      this.target.remove(this.group)
      disposeObjectResources([this.group])
    }
    this.group = new THREE.Group()
    this.group.userData.kind = 'selected-object-path'
    this.cursor = null
    this.playback = null
    const normalized = source ? pathTrackingPoints(source.keyframes).map((point) => point.keyframe) : []
    const segments = buildPathTrackingSegments(
      normalized,
      pathTrackingOptionsForSubject(source?.ball ? 'ball' : 'player'),
    )
    if (source && segments.length) {
      this.playback = { subjectId: source.subjectId, segments, ball: source.ball }
      segments.forEach((segment) => this.group!.add(createLine(segment, source)))
      this.cursor = new THREE.Mesh(
        new THREE.RingGeometry(source.ball ? 0.34 : 0.48, source.ball ? 0.52 : 0.7, 48),
        new THREE.MeshBasicMaterial({
          color: source.ball ? 0xc7f8ff : 0xfff1b8,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.96,
          depthTest: false,
          depthWrite: false,
          toneMapped: false,
        }),
      )
      this.cursor.rotation.x = -Math.PI / 2
      this.cursor.renderOrder = 33
      this.cursor.userData.kind = 'path-current-time'
      this.group.add(this.cursor)
    }
    this.group.userData.hasPath = segments.length > 0
    this.target.add(this.group)
    this.update(source, visible, currentTime)
  }

  update(source: SelectedPathSource | null, visible: boolean, currentTime: number) {
    if (!this.group) return
    const hasPath = this.group.userData.hasPath === true
    const matches = Boolean(source && this.playback && source.subjectId === this.playback.subjectId)
    this.group.visible = visible && hasPath && matches
    if (!this.cursor || !this.playback || !hasPath || !matches) return
    const current = interpolatePathTrackingSegments(this.playback.segments, currentTime)
    this.cursor.visible = Boolean(current)
    if (current) {
      this.cursor.position.set(
        current.x,
        this.playback.ball ? Math.max(0.28, current.y ?? 0.22) + 0.08 : 0.12,
        current.z,
      )
    }
  }

  animate(elapsedSeconds: number) {
    if (!this.cursor?.visible || !this.group?.visible) return
    const pulse = (Math.sin(elapsedSeconds * 5) + 1) / 2
    this.cursor.scale.setScalar(0.92 + pulse * 0.2)
    this.cursor.material.opacity = 0.68 + pulse * 0.3
    this.cursor.rotation.z = elapsedSeconds * 0.55
  }

  dispose() {
    if (!this.group) return
    this.target.remove(this.group)
    disposeObjectResources([this.group])
    this.group.clear()
    this.group = null
    this.cursor = null
    this.playback = null
  }
}

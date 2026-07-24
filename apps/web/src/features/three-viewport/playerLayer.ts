import * as THREE from 'three'
import { shouldRenderActor, shouldRenderPlayerVisual } from '../../lib/actorVisibility'
import { interpolateKeyframes, isInferredAt } from '../../lib/interpolate'
import type { InferredPositionRenderMode } from '../../lib/threeViewOptions'
import type { Track } from '../../types/tracking'
import { disposeObjectResources } from './threeResources'

// Latent positions are faded so an identity-continuity interpolation cannot
// be mistaken for an observed player.
const INFERRED_OPACITY = 0.28

export type PlayerVisualOptions = {
  showModels: boolean
  showLabels: boolean
  shadows: boolean
}

function createLabel(track: Track, visible: boolean) {
  const canvas = document.createElement('canvas')
  canvas.width = 384
  canvas.height = 96
  const context = canvas.getContext('2d')!
  context.fillStyle = 'rgba(7, 10, 11, .84)'
  context.beginPath()
  context.roundRect(4, 4, 376, 88, 22)
  context.fill()
  context.fillStyle = track.color
  context.fillRect(18, 25, 8, 46)
  context.fillStyle = '#f3f5ef'
  context.font = '600 33px Arial, sans-serif'
  context.fillText(track.number ? `${track.number}  ${track.label}` : track.label, 42, 60)

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  }))
  sprite.userData.kind = 'player-label'
  sprite.userData.trackId = track.id
  sprite.visible = visible
  sprite.scale.set(8, 2, 1)
  sprite.position.y = 4.7
  return sprite
}

function createPlayer(track: Track, options: PlayerVisualOptions) {
  const group = new THREE.Group()
  group.userData.trackId = track.id
  const color = new THREE.Color(track.color)
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.52, 1.25, 6, 12),
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.48,
      metalness: 0.12,
      transparent: true,
    }),
  )
  body.position.y = 1.45
  body.castShadow = options.shadows
  body.userData.trackId = track.id
  body.userData.kind = 'player-model'
  body.visible = options.showModels
  group.add(body)

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 16, 12),
    new THREE.MeshStandardMaterial({
      color: 0xd4a47d,
      roughness: 0.72,
      transparent: true,
    }),
  )
  head.position.y = 2.72
  head.castShadow = options.shadows
  head.userData.trackId = track.id
  head.userData.kind = 'player-model'
  head.visible = options.showModels
  group.add(head)

  const base = new THREE.Mesh(
    new THREE.RingGeometry(0.7, 0.9, 32),
    new THREE.MeshBasicMaterial({
      color,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.72,
    }),
  )
  base.rotation.x = -Math.PI / 2
  base.position.y = 0.04
  base.userData.trackId = track.id
  base.userData.kind = 'player-model'
  base.visible = options.showModels
  group.add(base)
  group.add(createLabel(track, options.showLabels))
  return group
}

/** Owns player meshes, labels, transforms, and their GPU resources. */
export class PlayerLayer {
  private readonly groups = new Map<string, THREE.Group>()

  constructor(private readonly target: THREE.Scene) {}

  get raycastTargets() {
    return [...this.groups.values()]
  }

  rebuild(tracks: Track[], options: PlayerVisualOptions) {
    this.clear()
    tracks.forEach((track) => {
      const group = createPlayer(track, options)
      this.groups.set(track.id, group)
      this.target.add(group)
    })
  }

  update(
    tracks: Track[],
    currentTime: number,
    duration: number,
    inferredPositionMode: InferredPositionRenderMode = 'transparent',
  ) {
    tracks.forEach((track) => {
      const group = this.groups.get(track.id)
      if (!group) return
      const inferred = isInferredAt(track.keyframes, currentTime)
      if (inferred && inferredPositionMode === 'hidden') {
        group.visible = false
        return
      }
      const position = interpolateKeyframes(track.keyframes, currentTime)
      group.position.set(position.x, 0, position.z)
      const next = interpolateKeyframes(track.keyframes, Math.min(duration, currentTime + 0.2))
      group.rotation.y = Math.atan2(next.x - position.x, next.z - position.z)
      group.visible = shouldRenderActor(track, currentTime)
      // Identity confidence does not alter positional evidence. Every person is
      // solid on observed frames; only an internal inferred gap may be faded.
      this.applyInferredState(
        group,
        inferred && inferredPositionMode === 'transparent',
      )
    })
  }

  private applyInferredState(group: THREE.Group, inferred: boolean) {
    if (group.userData.inferred === inferred) return
    group.userData.inferred = inferred
    group.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        const materials = Array.isArray(object.material) ? object.material : [object.material]
        materials.forEach((material) => {
          if (
            material instanceof THREE.MeshStandardMaterial
            || material instanceof THREE.MeshBasicMaterial
          ) {
            const solid = material instanceof THREE.MeshBasicMaterial ? 0.72 : 1
            material.opacity = inferred ? INFERRED_OPACITY : solid
          }
        })
      } else if (object instanceof THREE.Sprite) {
        object.material.opacity = inferred ? 0.4 : 1
      }
    })
  }

  selectedPosition(trackId: string | null) {
    const group = trackId ? this.groups.get(trackId) : null
    return group?.visible ? group.position : null
  }

  setVisualOptions(options: Pick<PlayerVisualOptions, 'showModels' | 'showLabels'>) {
    for (const group of this.groups.values()) {
      group.traverse((object) => {
        if (object.userData.kind !== 'player-model' && object.userData.kind !== 'player-label') return
        object.visible = shouldRenderPlayerVisual(object.userData.kind, options)
      })
    }
  }

  setShadowEnabled(enabled: boolean) {
    for (const group of this.groups.values()) {
      group.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return
        if (object.userData.trackId) object.castShadow = enabled
        const materials = Array.isArray(object.material) ? object.material : [object.material]
        materials.forEach((material) => {
          if (material instanceof THREE.MeshStandardMaterial) material.needsUpdate = true
        })
      })
    }
  }

  dispose() {
    this.clear()
  }

  private clear() {
    for (const group of this.groups.values()) this.target.remove(group)
    disposeObjectResources(this.groups.values())
    this.groups.clear()
  }
}

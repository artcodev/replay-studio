import * as THREE from 'three'
import { shouldRenderBall } from '../../lib/actorVisibility'
import { interpolateKeyframes } from '../../lib/interpolate'
import type { Keyframe } from '../../types/tracking'
import { disposeObjectResources } from './threeResources'

/** Owns the reconstructed ball and its optional full-trajectory line. */
export class BallLayer {
  readonly mesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>
  private trail: THREE.Line | null = null

  constructor(private readonly target: THREE.Scene, shadows: boolean) {
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 18, 14),
      new THREE.MeshStandardMaterial({ color: 0xf8f3df, roughness: 0.54 }),
    )
    this.mesh.castShadow = shadows
    this.target.add(this.mesh)
  }

  rebuildTrail(keyframes: Keyframe[]) {
    if (this.trail) {
      this.target.remove(this.trail)
      disposeObjectResources([this.trail])
    }
    const points = keyframes.map(
      (frame) => new THREE.Vector3(frame.x, (frame.y ?? 0.22) + 0.1, frame.z),
    )
    this.trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: 0xffd36a, transparent: true, opacity: 0.82 }),
    )
    this.target.add(this.trail)
  }

  update(
    keyframes: Keyframe[],
    currentTime: number,
    showBall: boolean,
    showTrail: boolean,
    pathReplacesTrail: boolean,
  ) {
    const ball = interpolateKeyframes(keyframes, currentTime)
    this.mesh.visible = shouldRenderBall(showBall, keyframes, currentTime, ball.confidence)
    this.mesh.position.set(ball.x, Math.max(0.24, ball.y ?? 0.24), ball.z)
    if (this.trail) this.trail.visible = showTrail && !pathReplacesTrail
  }

  setShadowEnabled(enabled: boolean) {
    this.mesh.castShadow = enabled
    this.mesh.material.needsUpdate = true
  }

  dispose() {
    this.target.remove(this.mesh)
    if (this.trail) this.target.remove(this.trail)
    disposeObjectResources(this.trail ? [this.mesh, this.trail] : [this.mesh])
    this.trail = null
  }
}

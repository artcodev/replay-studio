import * as THREE from 'three'
import { interpolateKeyframes } from '../../lib/interpolate'
import type { Keyframe } from '../../types/tracking'
import { disposeObjectResources } from './threeResources'

export class SelectionLayer {
  private readonly playerRig = new THREE.Group()
  private readonly playerRing: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>
  private readonly playerMarker: THREE.Sprite
  private readonly ballRig = new THREE.Group()
  private readonly ballRing: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>

  constructor(private readonly target: THREE.Scene) {
    this.playerRig.visible = false
    this.playerRig.userData.kind = 'selection-rig'
    this.playerRing = new THREE.Mesh(
      new THREE.RingGeometry(1.03, 1.33, 64),
      new THREE.MeshBasicMaterial({
        color: 0xffd36a,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    )
    this.playerRing.rotation.x = -Math.PI / 2
    this.playerRing.position.y = 0.07
    this.playerRing.renderOrder = 30
    this.playerRig.add(this.playerRing)

    const markerCanvas = document.createElement('canvas')
    markerCanvas.width = 128
    markerCanvas.height = 160
    const context = markerCanvas.getContext('2d')!
    context.shadowColor = 'rgba(255, 211, 106, .9)'
    context.shadowBlur = 18
    context.strokeStyle = '#ffd36a'
    context.lineWidth = 8
    context.beginPath()
    context.arc(64, 58, 30, 0, Math.PI * 2)
    context.stroke()
    context.shadowBlur = 10
    context.fillStyle = '#fff2c2'
    context.beginPath()
    context.moveTo(46, 104)
    context.lineTo(82, 104)
    context.lineTo(64, 132)
    context.closePath()
    context.fill()

    const texture = new THREE.CanvasTexture(markerCanvas)
    texture.colorSpace = THREE.SRGBColorSpace
    this.playerMarker = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture,
      color: 0xffffff,
      transparent: true,
      opacity: 0.96,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    }))
    this.playerMarker.position.y = 4.45
    this.playerMarker.scale.set(2.25, 2.8, 1)
    this.playerMarker.renderOrder = 31
    this.playerRig.add(this.playerMarker)
    target.add(this.playerRig)

    this.ballRig.visible = false
    this.ballRig.userData.kind = 'ball-selection-rig'
    this.ballRing = new THREE.Mesh(
      new THREE.RingGeometry(0.48, 0.67, 48),
      new THREE.MeshBasicMaterial({
        color: 0x5ee7ff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.96,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    )
    this.ballRing.rotation.x = -Math.PI / 2
    this.ballRing.position.y = 0.09
    this.ballRing.renderOrder = 32
    this.ballRig.add(this.ballRing)
    const crosshair = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.88, 0, 0), new THREE.Vector3(-0.72, 0, 0),
        new THREE.Vector3(0.72, 0, 0), new THREE.Vector3(0.88, 0, 0),
        new THREE.Vector3(0, 0, -0.88), new THREE.Vector3(0, 0, -0.72),
        new THREE.Vector3(0, 0, 0.72), new THREE.Vector3(0, 0, 0.88),
      ]),
      new THREE.LineBasicMaterial({
        color: 0xc7f8ff,
        transparent: true,
        opacity: 0.9,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    )
    crosshair.position.y = 0.065
    crosshair.renderOrder = 32
    this.ballRig.add(crosshair)
    target.add(this.ballRig)
  }

  updatePlayer(position: THREE.Vector3 | null) {
    this.playerRig.visible = Boolean(position)
    if (position) this.playerRig.position.set(position.x, 0, position.z)
  }

  updateBall(editMode: boolean, selectedTime: number | null, keyframes: Keyframe[]) {
    const visible = editMode && selectedTime !== null && keyframes.length > 0
    this.ballRig.visible = visible
    if (!visible || selectedTime === null) return
    const ball = interpolateKeyframes(keyframes, selectedTime)
    this.ballRig.position.set(ball.x, 0, ball.z)
  }

  animate(elapsedSeconds: number) {
    if (this.playerRig.visible) {
      const pulse = (Math.sin(elapsedSeconds * 4.2) + 1) / 2
      const ringScale = 1 + pulse * 0.13
      this.playerRing.scale.setScalar(ringScale)
      this.playerRing.material.opacity = 0.58 + pulse * 0.34
      this.playerRing.rotation.z = elapsedSeconds * 0.7
      this.playerMarker.position.y = 4.35 + Math.sin(elapsedSeconds * 2.8) * 0.16
      this.playerMarker.material.opacity = 0.76 + pulse * 0.2
      const markerScale = 1 + pulse * 0.06
      this.playerMarker.scale.set(2.25 * markerScale, 2.8 * markerScale, 1)
    }
    if (this.ballRig.visible) {
      const pulse = (Math.sin(elapsedSeconds * 5.4) + 1) / 2
      this.ballRing.scale.setScalar(1 + pulse * 0.16)
      this.ballRing.material.opacity = 0.62 + pulse * 0.34
      this.ballRing.rotation.z = -elapsedSeconds * 0.9
    }
  }

  dispose() {
    this.target.remove(this.playerRig, this.ballRig)
    disposeObjectResources([this.playerRig, this.ballRig])
    this.playerRig.clear()
    this.ballRig.clear()
  }
}

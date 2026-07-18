import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import {
  renderPixelRatio,
  renderQualityProfile,
  type RenderQuality,
} from '../../lib/renderQuality'
import { disposeObjectResources } from './threeResources'
import { ViewportLighting } from './viewportLighting'

export type CameraPreset = 'broadcast' | 'orbit' | 'tactical' | 'goal'

type ShadowParticipants = {
  setShadowEnabled: (enabled: boolean) => void
}

const cameraPresets: Record<CameraPreset, {
  position: THREE.Vector3
  target: THREE.Vector3
}> = {
  broadcast: { position: new THREE.Vector3(-4, 54, 70), target: new THREE.Vector3(4, 0, 0) },
  orbit: { position: new THREE.Vector3(-28, 20, 30), target: new THREE.Vector3(8, 0, 0) },
  tactical: { position: new THREE.Vector3(0, 91, 0.1), target: new THREE.Vector3(0, 0, 0) },
  goal: { position: new THREE.Vector3(58, 7, 0), target: new THREE.Vector3(20, 0, 0) },
}

/** Owns WebGL renderer, camera, controls, resize, and their lifecycle. */
export class ViewportRenderSurface {
  readonly scene = new THREE.Scene()
  readonly camera = new THREE.PerspectiveCamera(45, 1, 0.1, 320)
  readonly renderer: THREE.WebGLRenderer
  private readonly controls: OrbitControls
  private readonly lighting: ViewportLighting
  private readonly resizeObserver: ResizeObserver | null
  private disposed = false

  constructor(private readonly host: HTMLDivElement) {
    this.scene.background = new THREE.Color(0x11191b)
    this.scene.fog = new THREE.FogExp2(0x11191b, 0.004)
    this.camera.position.copy(cameraPresets.broadcast.position)

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    })
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    host.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.07
    this.controls.maxPolarAngle = Math.PI / 2.04
    this.controls.minDistance = 8
    this.controls.maxDistance = 150
    this.controls.target.copy(cameraPresets.broadcast.target)

    this.lighting = new ViewportLighting(this.scene)
    this.resizeObserver = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(() => this.resize())
    this.resizeObserver?.observe(host)
    this.resize()
  }

  resize() {
    const { clientWidth, clientHeight } = this.host
    this.renderer.setSize(clientWidth, clientHeight, false)
    this.camera.aspect = clientWidth / Math.max(1, clientHeight)
    this.camera.updateProjectionMatrix()
  }

  applyQuality(quality: RenderQuality, participants: ShadowParticipants[]) {
    const profile = renderQualityProfile(quality)
    this.renderer.setPixelRatio(renderPixelRatio(quality, window.devicePixelRatio))
    this.renderer.toneMappingExposure = profile.toneMappingExposure
    this.renderer.shadowMap.enabled = profile.shadows
    this.renderer.shadowMap.autoUpdate = profile.shadows
    // Use the current supported percentage-closer shadow mode.
    this.renderer.shadowMap.type = THREE.PCFShadowMap
    this.lighting.apply(profile)
    participants.forEach((participant) => participant.setShadowEnabled(profile.shadows))
    this.resize()
  }

  setBallEditCursor(enabled: boolean) {
    this.renderer.domElement.style.cursor = enabled ? 'crosshair' : ''
  }

  cameraPreset(name: CameraPreset) {
    const preset = cameraPresets[name]
    this.camera.position.copy(preset.position)
    this.controls.target.copy(preset.target)
    this.controls.update()
  }

  startAnimationLoop(update: (elapsedMilliseconds: number) => void) {
    this.renderer.setAnimationLoop((time) => {
      update(time)
      this.controls.update()
      this.renderer.render(this.scene, this.camera)
    })
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.resizeObserver?.disconnect()
    // Passing null is the documented way to stop an active WebGL animation loop.
    this.renderer.setAnimationLoop(null)
    this.controls.dispose()
    this.lighting.dispose()
    disposeObjectResources([this.scene])
    this.scene.clear()
    this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}

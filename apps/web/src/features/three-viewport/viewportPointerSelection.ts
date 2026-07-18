import * as THREE from 'three'

type SelectionTargets = {
  pitch: THREE.Object3D | null
  ball: THREE.Object3D | null
  players: THREE.Object3D[]
  ballEditMode: boolean
  editMode: boolean
  selectedTrackId: string | null
}

type SelectionCommands = {
  selectTrack: (trackId: string) => void
  selectBall: () => void
  moveTrack: (position: { x: number; z: number }) => void
  moveBall: (position: { x: number; z: number }) => void
}

function isEffectivelyVisible(object: THREE.Object3D) {
  let current: THREE.Object3D | null = object
  while (current) {
    if (!current.visible) return false
    current = current.parent
  }
  return true
}

/** Owns pointer gesture arbitration and Three.js raycasting for viewport selection. */
export class ViewportPointerSelection {
  private readonly raycaster = new THREE.Raycaster()
  private readonly pointer = new THREE.Vector2()
  private readonly pointerStart = new THREE.Vector2()
  private selectionPointerId: number | null = null

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly camera: THREE.Camera,
    private readonly targets: () => SelectionTargets,
    private readonly commands: SelectionCommands,
  ) {
    renderer.domElement.addEventListener('pointerdown', this.onPointerDown)
    renderer.domElement.addEventListener('pointerup', this.onPointerUp)
    renderer.domElement.addEventListener('pointercancel', this.cancelPointerSelection)
  }

  dispose() {
    this.renderer.domElement.removeEventListener('pointerdown', this.onPointerDown)
    this.renderer.domElement.removeEventListener('pointerup', this.onPointerUp)
    this.renderer.domElement.removeEventListener('pointercancel', this.cancelPointerSelection)
    this.selectionPointerId = null
  }

  private readonly onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) {
      this.selectionPointerId = null
      return
    }
    this.selectionPointerId = event.pointerId
    this.pointerStart.set(event.clientX, event.clientY)
  }

  private readonly cancelPointerSelection = () => {
    this.selectionPointerId = null
  }

  private readonly onPointerUp = (event: PointerEvent) => {
    if (this.selectionPointerId !== event.pointerId) return
    this.selectionPointerId = null
    if (this.pointerStart.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 5) return

    const rect = this.renderer.domElement.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    )
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const targets = this.targets()

    if (targets.ballEditMode) {
      const pitchHit = targets.pitch
        ? this.raycaster.intersectObject(targets.pitch, false)[0]
        : undefined
      if (pitchHit) this.commands.moveBall({ x: pitchHit.point.x, z: pitchHit.point.z })
      return
    }

    if (
      targets.ball?.visible
      && this.raycaster.intersectObject(targets.ball, false).length
    ) {
      this.commands.selectBall()
      return
    }

    const visiblePlayerHits = this.raycaster
      .intersectObjects(targets.players, true)
      .filter((hit) => (
        typeof hit.object.userData.trackId === 'string'
        && isEffectivelyVisible(hit.object)
      ))
    const playerHit = visiblePlayerHits.find((hit) => hit.object.userData.kind === 'player-model')
      ?? visiblePlayerHits.find((hit) => hit.object.userData.kind === 'player-label')
    if (playerHit) {
      this.commands.selectTrack(playerHit.object.userData.trackId as string)
      return
    }

    if (targets.editMode && targets.selectedTrackId && targets.pitch) {
      const pitchHit = this.raycaster.intersectObject(targets.pitch, false)[0]
      if (pitchHit) this.commands.moveTrack({ x: pitchHit.point.x, z: pitchHit.point.z })
    }
  }
}

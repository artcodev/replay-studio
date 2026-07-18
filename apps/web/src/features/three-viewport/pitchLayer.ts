import * as THREE from 'three'
import { disposeObjectResources } from './threeResources'

type PitchDimensions = {
  length: number
  width: number
}

function pitchLine(
  points: Array<[number, number]>,
  material: THREE.LineBasicMaterial,
  loop = false,
) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, z]) => new THREE.Vector3(x, 0.035, z)),
  )
  return loop ? new THREE.LineLoop(geometry, material) : new THREE.Line(geometry, material)
}

/** Owns the static pitch geometry and its GPU resources. */
export class PitchLayer {
  readonly root = new THREE.Group()
  readonly surface: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshStandardMaterial>

  constructor(
    private readonly target: THREE.Scene,
    dimensions: PitchDimensions,
    shadows: boolean,
  ) {
    const { length, width } = dimensions
    this.surface = new THREE.Mesh(
      new THREE.PlaneGeometry(length, width),
      new THREE.MeshStandardMaterial({ color: 0x1c6040, roughness: 0.9, metalness: 0 }),
    )
    this.surface.rotation.x = -Math.PI / 2
    this.surface.receiveShadow = shadows
    this.surface.userData.kind = 'pitch'
    this.root.add(this.surface)

    for (let strip = 0; strip < 12; strip += 1) {
      const stripMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(length / 12, width),
        new THREE.MeshStandardMaterial({
          color: strip % 2 ? 0x245f43 : 0x164b34,
          transparent: true,
          opacity: 0.14,
          roughness: 0.92,
          metalness: 0,
          depthWrite: false,
        }),
      )
      stripMesh.rotation.x = -Math.PI / 2
      stripMesh.position.set(-length / 2 + length / 24 + (strip * length) / 12, 0.012, 0)
      this.root.add(stripMesh)
    }

    const marking = new THREE.LineBasicMaterial({
      color: 0xe9eee7,
      transparent: true,
      opacity: 0.86,
    })
    this.root.add(pitchLine([
      [-length / 2, -width / 2],
      [length / 2, -width / 2],
      [length / 2, width / 2],
      [-length / 2, width / 2],
    ], marking, true))
    this.root.add(pitchLine([[0, -width / 2], [0, width / 2]], marking))

    const circlePoints: Array<[number, number]> = []
    for (let index = 0; index < 64; index += 1) {
      const angle = (index / 64) * Math.PI * 2
      circlePoints.push([Math.cos(angle) * 9.15, Math.sin(angle) * 9.15])
    }
    this.root.add(pitchLine(circlePoints, marking, true))

    for (const side of [-1, 1]) {
      const goalX = side * (length / 2)
      const boxX = goalX - side * 16.5
      this.root.add(pitchLine([
        [goalX, -20.16],
        [boxX, -20.16],
        [boxX, 20.16],
        [goalX, 20.16],
      ], marking))
      const sixX = goalX - side * 5.5
      this.root.add(pitchLine([
        [goalX, -9.16],
        [sixX, -9.16],
        [sixX, 9.16],
        [goalX, 9.16],
      ], marking))
      const goal = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(2.4, 2.44, 7.32)),
        new THREE.LineBasicMaterial({ color: 0xcfd6d3, transparent: true, opacity: 0.8 }),
      )
      goal.position.set(goalX + side * 1.2, 1.22, 0)
      this.root.add(goal)
    }

    target.add(this.root)
  }

  setShadowEnabled(enabled: boolean) {
    this.surface.receiveShadow = enabled
    this.root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      const materials = Array.isArray(object.material) ? object.material : [object.material]
      materials.forEach((material) => {
        if (material instanceof THREE.MeshStandardMaterial) material.needsUpdate = true
      })
    })
  }

  dispose() {
    this.target.remove(this.root)
    disposeObjectResources([this.root])
    this.root.clear()
  }
}

import * as THREE from 'three'
import type { FrameAnalysis } from '../../types/analysis'
import { disposeObjectResources } from './threeResources'

/** Owns transient current-frame detection markers independently of actor meshes. */
export class AnalysisMarkerLayer {
  readonly root = new THREE.Group()

  constructor(private readonly target: THREE.Scene) {
    target.add(this.root)
  }

  setVisible(visible: boolean) {
    this.root.visible = visible
  }

  rebuild(analysis: FrameAnalysis | null) {
    const previousMarkers = [...this.root.children]
    previousMarkers.forEach((child) => this.root.remove(child))
    disposeObjectResources(previousMarkers)
    if (!analysis) return

    analysis.people.forEach((person) => {
      // Off-pitch people (bench, boards) have no honest pitch position; a
      // marker for them used to land on the centre circle.
      if (!person.pitch) return
      const marker = new THREE.Mesh(
        new THREE.RingGeometry(0.72, 1.02, 32),
        new THREE.MeshBasicMaterial({
          color: person.matchedTrackId ? 0x71e2aa : 0xff8f63,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.92,
        }),
      )
      marker.rotation.x = -Math.PI / 2
      marker.position.set(person.pitch.x, 0.065, person.pitch.z)
      this.root.add(marker)
    })
    analysis.ballCandidates.filter((ball) => ball.primary).forEach((ball) => {
      const marker = new THREE.Mesh(
        new THREE.RingGeometry(0.28, 0.58, 28),
        new THREE.MeshBasicMaterial({
          color: 0xffd36a,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 1,
        }),
      )
      marker.rotation.x = -Math.PI / 2
      marker.position.set(ball.pitch.x, 0.075, ball.pitch.z)
      this.root.add(marker)
    })
  }

  dispose() {
    this.target.remove(this.root)
    disposeObjectResources([this.root])
    this.root.clear()
  }
}

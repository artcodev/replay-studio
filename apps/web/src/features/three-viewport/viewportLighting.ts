import * as THREE from 'three'
import type { RenderQualityProfile } from '../../lib/renderQuality'

export class ViewportLighting {
  readonly hemisphere: THREE.HemisphereLight
  readonly key: THREE.DirectionalLight
  readonly fill: THREE.DirectionalLight
  readonly stadium: THREE.SpotLight[]

  constructor(target: THREE.Scene) {
    this.hemisphere = new THREE.HemisphereLight(0xe8f1ff, 0x173523, 1.45)
    target.add(this.hemisphere)

    this.key = new THREE.DirectionalLight(0xfff5e8, 1.65)
    this.key.position.set(-28, 56, 22)
    Object.assign(this.key.shadow.camera, {
      near: 1,
      far: 140,
      left: -75,
      right: 75,
      top: 65,
      bottom: -65,
    })
    this.key.shadow.camera.updateProjectionMatrix()
    target.add(this.key)

    this.fill = new THREE.DirectionalLight(0xaecbff, 0)
    this.fill.position.set(36, 24, -28)
    this.fill.castShadow = false
    this.fill.visible = false
    target.add(this.fill)

    const fixtures = [
      { position: [-58, 42, -40], aim: [-18, 0, -10], color: 0xf3f7ff },
      { position: [-58, 42, 40], aim: [-18, 0, 10], color: 0xfff8ee },
      { position: [58, 42, -40], aim: [18, 0, -10], color: 0xfff8ee },
      { position: [58, 42, 40], aim: [18, 0, 10], color: 0xf3f7ff },
    ] as const
    this.stadium = fixtures.map((fixture) => {
      const light = new THREE.SpotLight(fixture.color, 0, 180, Math.PI / 4, 0.65, 2)
      const [x, y, z] = fixture.position
      const [targetX, targetY, targetZ] = fixture.aim
      light.position.set(x, y, z)
      light.target.position.set(targetX, targetY, targetZ)
      light.castShadow = false
      target.add(light, light.target)
      return light
    })
  }

  apply(profile: RenderQualityProfile) {
    this.hemisphere.intensity = profile.hemisphereIntensity
    this.key.intensity = profile.keyLightIntensity
    this.key.castShadow = profile.shadows
    this.key.shadow.mapSize.set(profile.shadowMapSize, profile.shadowMapSize)
    this.key.shadow.bias = profile.shadows ? -0.00015 : 0
    this.key.shadow.normalBias = profile.shadows ? 0.025 : 0
    if (profile.shadows) {
      this.key.shadow.needsUpdate = true
    } else if (this.key.shadow.map || this.key.shadow.mapPass) {
      this.key.shadow.dispose()
      this.key.shadow.map = null
      this.key.shadow.mapPass = null
    }
    this.fill.intensity = profile.fillLightIntensity
    this.fill.visible = profile.fillLightIntensity > 0
    this.stadium.forEach((light) => {
      light.intensity = profile.stadiumLightIntensity
      light.visible = profile.stadiumLightIntensity > 0
    })
  }

  dispose() {
    this.key.shadow.dispose()
  }
}

import * as THREE from 'three'

type RenderObjectWithResources = THREE.Object3D & {
  geometry?: THREE.BufferGeometry
  material?: THREE.Material | THREE.Material[]
}

/** Dispose each shared Three resource at most once across an object subtree. */
export function disposeObjectResources(roots: Iterable<THREE.Object3D>) {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()

  for (const root of roots) {
    root.traverse((object) => {
      const renderObject = object as RenderObjectWithResources
      if (renderObject.geometry && !geometries.has(renderObject.geometry)) {
        geometries.add(renderObject.geometry)
        renderObject.geometry.dispose()
      }
      const objectMaterials = renderObject.material
        ? Array.isArray(renderObject.material) ? renderObject.material : [renderObject.material]
        : []
      objectMaterials.forEach((material) => {
        if (materials.has(material)) return
        materials.add(material)
        if (material instanceof THREE.SpriteMaterial && material.map && !textures.has(material.map)) {
          textures.add(material.map)
          material.map.dispose()
        }
        material.dispose()
      })
    })
  }
}

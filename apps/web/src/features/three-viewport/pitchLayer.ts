import * as THREE from 'three'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import { disposeObjectResources } from './threeResources'

type PitchDimensions = {
  length: number
  width: number
}

type ShadowRole = 'caster' | 'receiver' | 'both'

const LINE_HEIGHT = 0.035
const COLLAR_WIDTH = 2.4
const RUN_OFF_END = 8.5
const RUN_OFF_SIDE = 5.5
const OUTER_MARGIN = 26
const BOARD_HEIGHT = 1.05
const BOARD_SPAN = 4.6
const BOARD_GAP = 0.12
const BOARD_TILT = 0.13
const BOARD_CORNER_CLEARANCE = 1.8
const BOARD_DEPTH = 0.16
// Depth budget at 100+ m is a few millimetres, so keep face and cabinet far apart.
const BOARD_CABINET_FRONT = 0.06
const BOARD_FACE_OFFSET = 0.1
const GOAL_WIDTH = 7.32
const GOAL_HEIGHT = 2.44
const POST_RADIUS = 0.06
const RAIL_RADIUS = 0.035
const NET_ROOF_DEPTH = 1.1
const NET_FLOOR_DEPTH = 2.2
const NET_CELL = 0.26

/** Bottom/top colour pairs for the empty advertising panels. */
const BOARD_GRADIENTS: ReadonlyArray<readonly [number, number]> = [
  [0x061a33, 0x2b6fae],
  [0x04231f, 0x1d8c72],
  [0x1a0c2e, 0x6a4aa8],
  [0x2b1204, 0xb4741f],
  [0x0a1c2e, 0x3f8fa0],
  [0x2a0813, 0xa33f57],
]

function tagShadow<T extends THREE.Object3D>(object: T, role: ShadowRole) {
  object.userData.shadowRole = role
  return object
}

function pitchLine(
  points: Array<[number, number]>,
  material: THREE.LineBasicMaterial,
  loop = false,
) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, z]) => new THREE.Vector3(x, LINE_HEIGHT, z)),
  )
  return loop ? new THREE.LineLoop(geometry, material) : new THREE.Line(geometry, material)
}

function arcPoints(
  centerX: number,
  centerZ: number,
  radius: number,
  fromAngle: number,
  toAngle: number,
  segments: number,
): Array<[number, number]> {
  const points: Array<[number, number]> = []
  for (let index = 0; index <= segments; index += 1) {
    const angle = fromAngle + ((toAngle - fromAngle) * index) / segments
    points.push([centerX + Math.cos(angle) * radius, centerZ + Math.sin(angle) * radius])
  }
  return points
}

function spotGeometry(x: number, z: number, radius: number) {
  const geometry = new THREE.CircleGeometry(radius, 18)
  geometry.rotateX(-Math.PI / 2)
  geometry.translate(x, LINE_HEIGHT, z)
  return geometry
}

function markingsGroup(length: number, width: number) {
  const group = new THREE.Group()
  const marking = new THREE.LineBasicMaterial({
    color: 0xe9eee7,
    transparent: true,
    opacity: 0.86,
  })
  const halfLength = length / 2
  const halfWidth = width / 2

  group.add(pitchLine([
    [-halfLength, -halfWidth],
    [halfLength, -halfWidth],
    [halfLength, halfWidth],
    [-halfLength, halfWidth],
  ], marking, true))
  group.add(pitchLine([[0, -halfWidth], [0, halfWidth]], marking))

  const centreCircle = arcPoints(0, 0, 9.15, 0, Math.PI * 2, 64)
  centreCircle.pop()
  group.add(pitchLine(centreCircle, marking, true))

  const spots = [spotGeometry(0, 0, 0.14)]
  const penaltyArcHalfAngle = Math.acos(5.5 / 9.15)
  for (const side of [-1, 1]) {
    const goalX = side * halfLength
    const boxX = goalX - side * 16.5
    group.add(pitchLine([
      [goalX, -20.16],
      [boxX, -20.16],
      [boxX, 20.16],
      [goalX, 20.16],
    ], marking))
    const sixX = goalX - side * 5.5
    group.add(pitchLine([
      [goalX, -9.16],
      [sixX, -9.16],
      [sixX, 9.16],
      [goalX, 9.16],
    ], marking))

    // Penalty arc: the part of the 9.15 m circle that falls outside the box.
    const penaltyX = goalX - side * 11
    const arcCentreAngle = side > 0 ? Math.PI : 0
    group.add(pitchLine(arcPoints(
      penaltyX,
      0,
      9.15,
      arcCentreAngle - penaltyArcHalfAngle,
      arcCentreAngle + penaltyArcHalfAngle,
      24,
    ), marking))
    spots.push(spotGeometry(penaltyX, 0, 0.14))

    for (const zSide of [-1, 1]) {
      group.add(pitchLine(arcPoints(
        goalX,
        zSide * halfWidth,
        1,
        arcCentreAngle,
        arcCentreAngle + (side * zSide * Math.PI) / 2,
        8,
      ), marking))
    }
  }

  group.add(new THREE.Mesh(
    mergeGeometries(spots),
    new THREE.MeshBasicMaterial({ color: 0xe9eee7 }),
  ))
  return group
}

function mownStripes(length: number, width: number) {
  const group = new THREE.Group()
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
    group.add(stripMesh)
  }
  return group
}

/** Flat rectangular ring lying in the XZ plane, so aprons never overlap the turf. */
function surroundGeometry(
  innerLength: number,
  innerWidth: number,
  outerLength: number,
  outerWidth: number,
) {
  const shape = new THREE.Shape()
  shape.moveTo(-outerLength / 2, -outerWidth / 2)
  shape.lineTo(outerLength / 2, -outerWidth / 2)
  shape.lineTo(outerLength / 2, outerWidth / 2)
  shape.lineTo(-outerLength / 2, outerWidth / 2)
  shape.closePath()
  const hole = new THREE.Path()
  hole.moveTo(-innerLength / 2, -innerWidth / 2)
  hole.lineTo(-innerLength / 2, innerWidth / 2)
  hole.lineTo(innerLength / 2, innerWidth / 2)
  hole.lineTo(innerLength / 2, -innerWidth / 2)
  hole.closePath()
  shape.holes.push(hole)
  const geometry = new THREE.ShapeGeometry(shape)
  geometry.rotateX(-Math.PI / 2)
  return geometry
}

/** Run-off strip around the pitch plus the darker deck the boards stand on. */
function surroundGroup(
  length: number,
  width: number,
  boardHalfLength: number,
  boardHalfWidth: number,
) {
  const group = new THREE.Group()
  const collar = tagShadow(new THREE.Mesh(
    surroundGeometry(length, width, length + COLLAR_WIDTH * 2, width + COLLAR_WIDTH * 2),
    new THREE.MeshStandardMaterial({ color: 0x175338, roughness: 0.95, metalness: 0 }),
  ), 'receiver')
  const apron = tagShadow(new THREE.Mesh(
    surroundGeometry(
      length + COLLAR_WIDTH * 2,
      width + COLLAR_WIDTH * 2,
      boardHalfLength * 2,
      boardHalfWidth * 2,
    ),
    new THREE.MeshStandardMaterial({ color: 0x0f3a27, roughness: 0.97, metalness: 0 }),
  ), 'receiver')
  const deck = new THREE.Mesh(
    surroundGeometry(
      boardHalfLength * 2,
      boardHalfWidth * 2,
      boardHalfLength * 2 + OUTER_MARGIN,
      boardHalfWidth * 2 + OUTER_MARGIN,
    ),
    new THREE.MeshStandardMaterial({ color: 0x101a1d, roughness: 1, metalness: 0 }),
  )
  // The rings share edges exactly, so they stay on one plane just under the turf:
  // any height stagger would show as a hairline seam right at the board line.
  for (const ring of [collar, apron, deck]) ring.position.y = -0.004
  group.add(collar, apron, deck)
  return group
}

/** Unlit panel whose vertex colours fade from bottom to top. */
function gradientPanelGeometry(width: number, height: number, bottom: number, top: number) {
  const geometry = new THREE.PlaneGeometry(width, height, 1, 8)
  const position = geometry.getAttribute('position')
  const colors = new Float32Array(position.count * 3)
  const bottomColor = new THREE.Color(bottom)
  const topColor = new THREE.Color(top)
  const blend = new THREE.Color()
  for (let index = 0; index < position.count; index += 1) {
    const ratio = THREE.MathUtils.clamp(position.getY(index) / height + 0.5, 0, 1)
    blend.copy(bottomColor).lerp(topColor, ratio)
    blend.toArray(colors, index * 3)
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  return geometry
}

function advertBoardsGroup(boardHalfLength: number, boardHalfWidth: number) {
  const runs = [
    { centerX: 0, centerZ: -boardHalfWidth, span: boardHalfLength * 2, rotation: 0 },
    { centerX: 0, centerZ: boardHalfWidth, span: boardHalfLength * 2, rotation: Math.PI },
    { centerX: -boardHalfLength, centerZ: 0, span: boardHalfWidth * 2, rotation: Math.PI / 2 },
    { centerX: boardHalfLength, centerZ: 0, span: boardHalfWidth * 2, rotation: -Math.PI / 2 },
  ]
  const faces: THREE.BufferGeometry[] = []
  const cabinets: THREE.BufferGeometry[] = []
  let panelIndex = 0

  for (const run of runs) {
    const span = run.span - BOARD_CORNER_CLEARANCE * 2
    const count = Math.max(1, Math.round(span / (BOARD_SPAN + BOARD_GAP)))
    const step = span / count
    // Pivot at the base: lean away from the pitch, then place the run.
    const place = (geometry: THREE.BufferGeometry) => {
      geometry.rotateX(-BOARD_TILT)
      geometry.rotateY(run.rotation)
      geometry.translate(run.centerX, 0, run.centerZ)
    }
    for (let index = 0; index < count; index += 1) {
      const offset = -span / 2 + step * (index + 0.5)
      const [bottom, top] = BOARD_GRADIENTS[panelIndex % BOARD_GRADIENTS.length]
      panelIndex += 1
      const face = gradientPanelGeometry(step - BOARD_GAP, BOARD_HEIGHT, bottom, top)
      face.translate(offset, BOARD_HEIGHT / 2, BOARD_FACE_OFFSET)
      place(face)
      faces.push(face)
    }
    // One cabinet per run: panel seams stay a dark joint without coplanar overlaps.
    const cabinet = new THREE.BoxGeometry(span + 0.2, BOARD_HEIGHT + 0.07, BOARD_DEPTH)
    cabinet.translate(0, BOARD_HEIGHT / 2, BOARD_CABINET_FRONT - BOARD_DEPTH / 2)
    place(cabinet)
    cabinets.push(cabinet)
  }

  const group = new THREE.Group()
  group.add(new THREE.Mesh(
    mergeGeometries(faces),
    new THREE.MeshBasicMaterial({ vertexColors: true }),
  ))
  group.add(tagShadow(new THREE.Mesh(
    mergeGeometries(cabinets),
    new THREE.MeshStandardMaterial({ color: 0x11181d, roughness: 0.68, metalness: 0.12 }),
  ), 'receiver'))
  return group
}

function barGeometry(from: THREE.Vector3, to: THREE.Vector3, radius: number) {
  const direction = new THREE.Vector3().subVectors(to, from)
  const geometry = new THREE.CylinderGeometry(radius, radius, direction.length(), 12)
  geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize(),
  ))
  geometry.translate(
    (from.x + to.x) / 2,
    (from.y + to.y) / 2,
    (from.z + to.z) / 2,
  )
  return geometry
}

/**
 * Net strands over the classic silhouette: flat roof to `NET_ROOF_DEPTH`, then a
 * back panel slanting down to `NET_FLOOR_DEPTH` at ground level.
 */
function goalNetGeometry(goalX: number, side: number) {
  const halfWidth = GOAL_WIDTH / 2
  const positions: number[] = []
  const strand = (
    fromDepth: number,
    fromY: number,
    fromZ: number,
    toDepth: number,
    toY: number,
    toZ: number,
  ) => {
    positions.push(
      goalX + side * fromDepth, fromY, fromZ,
      goalX + side * toDepth, toY, toZ,
    )
  }
  const depthAt = (y: number) =>
    NET_FLOOR_DEPTH + (NET_ROOF_DEPTH - NET_FLOOR_DEPTH) * (y / GOAL_HEIGHT)

  const roofSteps = Math.ceil(NET_ROOF_DEPTH / NET_CELL)
  const backSteps = Math.ceil(GOAL_HEIGHT / NET_CELL)
  const profile: Array<[number, number]> = []
  for (let step = 0; step <= roofSteps; step += 1) {
    profile.push([(NET_ROOF_DEPTH * step) / roofSteps, GOAL_HEIGHT])
  }
  for (let step = 1; step <= backSteps; step += 1) {
    const y = GOAL_HEIGHT * (1 - step / backSteps)
    profile.push([depthAt(y), y])
  }

  const widthSteps = Math.ceil(GOAL_WIDTH / NET_CELL)
  for (let step = 0; step <= widthSteps; step += 1) {
    const z = -halfWidth + (GOAL_WIDTH * step) / widthSteps
    for (let index = 1; index < profile.length; index += 1) {
      const [fromDepth, fromY] = profile[index - 1]
      const [toDepth, toY] = profile[index]
      strand(fromDepth, fromY, z, toDepth, toY, z)
    }
  }
  for (const [depth, y] of profile) {
    strand(depth, y, -halfWidth, depth, y, halfWidth)
  }

  const floorSteps = Math.ceil(NET_FLOOR_DEPTH / NET_CELL)
  for (const z of [-halfWidth, halfWidth]) {
    for (let step = 0; step <= backSteps; step += 1) {
      const y = (GOAL_HEIGHT * step) / backSteps
      strand(0, y, z, depthAt(y), y, z)
    }
    for (let step = 1; step <= floorSteps; step += 1) {
      const depth = (NET_FLOOR_DEPTH * step) / floorSteps
      const top = depth <= NET_ROOF_DEPTH
        ? GOAL_HEIGHT
        : (GOAL_HEIGHT * (NET_FLOOR_DEPTH - depth)) / (NET_FLOOR_DEPTH - NET_ROOF_DEPTH)
      strand(depth, 0, z, depth, top, z)
    }
  }

  return new THREE.BufferGeometry().setAttribute(
    'position',
    new THREE.Float32BufferAttribute(positions, 3),
  )
}

function goalsGroup(length: number) {
  const posts: THREE.BufferGeometry[] = []
  const rails: THREE.BufferGeometry[] = []
  const nets: THREE.BufferGeometry[] = []
  const postZ = GOAL_WIDTH / 2 + POST_RADIUS
  const crossbarY = GOAL_HEIGHT + POST_RADIUS

  for (const side of [-1, 1]) {
    const goalX = side * (length / 2)
    const at = (depth: number, y: number, z: number) =>
      new THREE.Vector3(goalX + side * depth, y, z)
    posts.push(barGeometry(at(0, 0, -postZ), at(0, crossbarY, -postZ), POST_RADIUS))
    posts.push(barGeometry(at(0, 0, postZ), at(0, crossbarY, postZ), POST_RADIUS))
    posts.push(barGeometry(at(0, crossbarY, -postZ), at(0, crossbarY, postZ), POST_RADIUS))
    rails.push(barGeometry(
      at(NET_ROOF_DEPTH, GOAL_HEIGHT, -postZ),
      at(NET_ROOF_DEPTH, GOAL_HEIGHT, postZ),
      RAIL_RADIUS,
    ))
    rails.push(barGeometry(
      at(NET_FLOOR_DEPTH, RAIL_RADIUS, -postZ),
      at(NET_FLOOR_DEPTH, RAIL_RADIUS, postZ),
      RAIL_RADIUS,
    ))
    for (const zSide of [-1, 1]) {
      const z = zSide * postZ
      rails.push(barGeometry(
        at(0, crossbarY, z),
        at(NET_ROOF_DEPTH, GOAL_HEIGHT, z),
        RAIL_RADIUS,
      ))
      rails.push(barGeometry(
        at(NET_ROOF_DEPTH, GOAL_HEIGHT, z),
        at(NET_FLOOR_DEPTH, RAIL_RADIUS, z),
        RAIL_RADIUS,
      ))
    }
    nets.push(goalNetGeometry(goalX, side))
  }

  const group = new THREE.Group()
  group.add(tagShadow(new THREE.Mesh(
    mergeGeometries(posts),
    new THREE.MeshStandardMaterial({ color: 0xf4f7f5, roughness: 0.42, metalness: 0.06 }),
  ), 'caster'))
  group.add(tagShadow(new THREE.Mesh(
    mergeGeometries(rails),
    new THREE.MeshStandardMaterial({ color: 0xbcc6c2, roughness: 0.6, metalness: 0.08 }),
  ), 'caster'))
  group.add(new THREE.LineSegments(
    mergeGeometries(nets),
    new THREE.LineBasicMaterial({
      color: 0xdfe8e4,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    }),
  ))
  return group
}

function cornerFlagsGroup(halfLength: number, halfWidth: number) {
  const poles: THREE.BufferGeometry[] = []
  const flags: THREE.BufferGeometry[] = []
  for (const xSide of [-1, 1]) {
    for (const zSide of [-1, 1]) {
      const x = xSide * halfLength
      const z = zSide * halfWidth
      const pole = new THREE.CylinderGeometry(0.028, 0.028, 1.5, 8)
      pole.translate(x, 0.75, z)
      poles.push(pole)
      const flag = new THREE.PlaneGeometry(0.42, 0.3)
      flag.translate(0.21, 0, 0)
      flag.rotateY(Math.atan2(zSide, -xSide))
      flag.translate(x, 1.32, z)
      flags.push(flag)
    }
  }
  const group = new THREE.Group()
  group.add(tagShadow(new THREE.Mesh(
    mergeGeometries(poles),
    new THREE.MeshStandardMaterial({ color: 0xf1f4f2, roughness: 0.5, metalness: 0.05 }),
  ), 'caster'))
  group.add(new THREE.Mesh(
    mergeGeometries(flags),
    new THREE.MeshStandardMaterial({
      color: 0xf2c541,
      roughness: 0.85,
      metalness: 0,
      side: THREE.DoubleSide,
    }),
  ))
  return group
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
    const boardHalfLength = length / 2 + RUN_OFF_END
    const boardHalfWidth = width / 2 + RUN_OFF_SIDE

    this.surface = new THREE.Mesh(
      new THREE.PlaneGeometry(length, width),
      new THREE.MeshStandardMaterial({ color: 0x1c6040, roughness: 0.9, metalness: 0 }),
    )
    this.surface.rotation.x = -Math.PI / 2
    this.surface.userData.kind = 'pitch'
    tagShadow(this.surface, 'receiver')
    this.root.add(this.surface)

    this.root.add(
      mownStripes(length, width),
      surroundGroup(length, width, boardHalfLength, boardHalfWidth),
      markingsGroup(length, width),
      advertBoardsGroup(boardHalfLength, boardHalfWidth),
      goalsGroup(length),
      cornerFlagsGroup(length / 2, width / 2),
    )

    this.setShadowEnabled(shadows)
    target.add(this.root)
  }

  setShadowEnabled(enabled: boolean) {
    this.root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      const role = object.userData.shadowRole as ShadowRole | undefined
      object.castShadow = enabled && (role === 'caster' || role === 'both')
      object.receiveShadow = enabled && (role === 'receiver' || role === 'both')
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

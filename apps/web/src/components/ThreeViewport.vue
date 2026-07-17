<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { FrameAnalysis, SceneDocument, Track } from '../types'
import { shouldRenderActor, shouldRenderBall, shouldRenderPlayerVisual } from '../lib/actorVisibility'
import { interpolateKeyframes } from '../lib/interpolate'
import {
  playerActionColor,
  playerActionLabel,
  type PlayerActionPlaybackState,
} from '../lib/playerActions'
import {
  buildPathTrackingSegments,
  interpolatePathTrackingSegments,
  pathTrackingOptionsForSubject,
  pathTrackingPoints,
  type PathTrackingSegment,
} from '../lib/pathTracking'
import { renderPixelRatio, renderQualityProfile, type RenderQuality } from '../lib/renderQuality'

const props = withDefaults(defineProps<{
  scene: SceneDocument
  currentTime: number
  selectedTrackId: string | null
  editMode: boolean
  ballEditMode?: boolean
  selectedBallKeyframeTime?: number | null
  showTrails: boolean
  showLabels: boolean
  showModels?: boolean
  showBall?: boolean
  showAnalysisMarkers?: boolean
  showPathTracking?: boolean
  ballSelected?: boolean
  activePlayerAction?: PlayerActionPlaybackState | null
  frameAnalysis: FrameAnalysis | null
  renderQuality?: RenderQuality
}>(), {
  showModels: true,
  showBall: true,
  showAnalysisMarkers: true,
  showPathTracking: false,
  ballSelected: false,
  activePlayerAction: null,
  ballEditMode: false,
  selectedBallKeyframeTime: null,
  renderQuality: 'basic',
})

const emit = defineEmits<{
  select: [trackId: string]
  selectBall: []
  moveTrack: [position: { x: number; z: number }]
  moveBall: [position: { x: number; z: number }]
}>()

const host = ref<HTMLDivElement | null>(null)
const status = ref('Broadcast camera · drag to inspect')

const actionKeypointLabels = {
  'wind-up': 'Wind-up',
  contact: 'Contact',
  release: 'Release',
  apex: 'Apex',
  impact: 'Impact',
  recovery: 'Recovery',
} as const

const actionPreview = computed(() => {
  const state = props.activePlayerAction
  if (!state) return null
  const phase = Math.max(0, Math.min(1, state.phase))
  const keypoint = state.nearestKeypoint
  let keypointTiming: string | null = null
  if (keypoint) {
    const distance = Math.abs(keypoint.offsetSeconds)
    keypointTiming = distance < 0.005
      ? 'now'
      : keypoint.offsetSeconds > 0
        ? `${distance.toFixed(2)}s ago`
        : `in ${distance.toFixed(2)}s`
  }
  return {
    label: playerActionLabel(state.action.type),
    type: state.action.type,
    color: playerActionColor(state.action.type),
    phasePercent: Math.round(phase * 100),
    keypointLabel: keypoint ? actionKeypointLabels[keypoint.kind] : null,
    keypointTiming,
  }
})
let renderer: THREE.WebGLRenderer | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let scene3d: THREE.Scene | null = null
let resizeObserver: ResizeObserver | null = null
let pitchMesh: THREE.Mesh | null = null
let ballMesh: THREE.Mesh | null = null
let ballTrail: THREE.Line | null = null
let selectedPathGroup: THREE.Group | null = null
let selectedPathCursor: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial> | null = null
let selectedPathPlayback: { subjectId: string; segments: PathTrackingSegment[]; ball: boolean } | null = null
let selectionRig: THREE.Group | null = null
let selectionRing: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial> | null = null
let selectionMarker: THREE.Sprite | null = null
let ballSelectionRig: THREE.Group | null = null
let ballSelectionRing: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial> | null = null
let hemisphereLight: THREE.HemisphereLight | null = null
let keyLight: THREE.DirectionalLight | null = null
let fillLight: THREE.DirectionalLight | null = null
let stadiumLights: THREE.SpotLight[] = []
let pointerStart = new THREE.Vector2()
let selectionPointerId: number | null = null
const playerGroups = new Map<string, THREE.Group>()
const analysisMarkerGroup = new THREE.Group()
const raycaster = new THREE.Raycaster()
const pointer = new THREE.Vector2()

type RenderObjectWithResources = THREE.Object3D & {
  geometry?: THREE.BufferGeometry
  material?: THREE.Material | THREE.Material[]
}

function disposeObjectResources(roots: Iterable<THREE.Object3D>) {
  const disposedGeometries = new Set<THREE.BufferGeometry>()
  const disposedMaterials = new Set<THREE.Material>()
  const disposedTextures = new Set<THREE.Texture>()

  for (const root of roots) {
    root.traverse((object) => {
      const renderObject = object as RenderObjectWithResources
      const geometry = renderObject.geometry
      if (geometry && !disposedGeometries.has(geometry)) {
        disposedGeometries.add(geometry)
        geometry.dispose()
      }

      const materials = renderObject.material
        ? Array.isArray(renderObject.material) ? renderObject.material : [renderObject.material]
        : []
      materials.forEach((material) => {
        if (disposedMaterials.has(material)) return
        disposedMaterials.add(material)
        if (material instanceof THREE.SpriteMaterial && material.map && !disposedTextures.has(material.map)) {
          disposedTextures.add(material.map)
          material.map.dispose()
        }
        material.dispose()
      })
    })
  }
}

function resizeRenderer() {
  if (!host.value || !renderer || !camera) return
  const { clientWidth, clientHeight } = host.value
  renderer.setSize(clientWidth, clientHeight, false)
  camera.aspect = clientWidth / Math.max(1, clientHeight)
  camera.updateProjectionMatrix()
}

function updateShadowParticipants(enabled: boolean) {
  if (pitchMesh) pitchMesh.receiveShadow = enabled
  if (ballMesh) ballMesh.castShadow = enabled
  scene3d?.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const materials = Array.isArray(object.material) ? object.material : [object.material]
    materials.forEach((material) => {
      if (material instanceof THREE.MeshStandardMaterial) material.needsUpdate = true
    })
  })
  for (const group of playerGroups.values()) {
    group.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      if (object.userData.trackId) object.castShadow = enabled
    })
  }
}

function applyRenderQuality() {
  if (!renderer) return
  const profile = renderQualityProfile(props.renderQuality)
  renderer.setPixelRatio(renderPixelRatio(props.renderQuality, window.devicePixelRatio))
  renderer.toneMappingExposure = profile.toneMappingExposure
  renderer.shadowMap.enabled = profile.shadows
  renderer.shadowMap.autoUpdate = profile.shadows
  renderer.shadowMap.type = profile.softShadows ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap

  if (hemisphereLight) hemisphereLight.intensity = profile.hemisphereIntensity
  if (keyLight) {
    keyLight.intensity = profile.keyLightIntensity
    keyLight.castShadow = profile.shadows
    keyLight.shadow.mapSize.set(profile.shadowMapSize, profile.shadowMapSize)
    keyLight.shadow.bias = profile.shadows ? -0.00015 : 0
    keyLight.shadow.normalBias = profile.shadows ? 0.025 : 0
    if (profile.shadows) {
      keyLight.shadow.needsUpdate = true
    } else if (keyLight.shadow.map || keyLight.shadow.mapPass) {
      keyLight.shadow.dispose()
      keyLight.shadow.map = null
      keyLight.shadow.mapPass = null
    }
  }
  if (fillLight) {
    fillLight.intensity = profile.fillLightIntensity
    fillLight.visible = profile.fillLightIntensity > 0
  }
  stadiumLights.forEach((light) => {
    light.intensity = profile.stadiumLightIntensity
    light.visible = profile.stadiumLightIntensity > 0
  })
  updateShadowParticipants(profile.shadows)
  resizeRenderer()
}

function createStadiumFloodlights(target: THREE.Scene) {
  const fixtures = [
    { position: [-58, 42, -40], aim: [-18, 0, -10], color: 0xf3f7ff },
    { position: [-58, 42, 40], aim: [-18, 0, 10], color: 0xfff8ee },
    { position: [58, 42, -40], aim: [18, 0, -10], color: 0xfff8ee },
    { position: [58, 42, 40], aim: [18, 0, 10], color: 0xf3f7ff },
  ] as const

  stadiumLights = fixtures.map((fixture) => {
    const light = new THREE.SpotLight(
      fixture.color,
      0,
      180,
      Math.PI / 4,
      0.65,
      2,
    )
    const [x, y, z] = fixture.position
    const [targetX, targetY, targetZ] = fixture.aim
    light.position.set(x, y, z)
    light.target.position.set(targetX, targetY, targetZ)
    light.castShadow = false
    target.add(light, light.target)
    return light
  })
}

function line(points: Array<[number, number]>, material: THREE.LineBasicMaterial, loop = false) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, z]) => new THREE.Vector3(x, 0.035, z)),
  )
  return loop ? new THREE.LineLoop(geometry, material) : new THREE.Line(geometry, material)
}

function createPitch(target: THREE.Scene) {
  const { length, width } = props.scene.payload.pitch
  const pitch = new THREE.Group()
  const fieldMaterial = new THREE.MeshStandardMaterial({ color: 0x1c6040, roughness: 0.9, metalness: 0 })
  pitchMesh = new THREE.Mesh(new THREE.PlaneGeometry(length, width), fieldMaterial)
  pitchMesh.rotation.x = -Math.PI / 2
  pitchMesh.receiveShadow = renderQualityProfile(props.renderQuality).shadows
  pitchMesh.userData.kind = 'pitch'
  pitch.add(pitchMesh)

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
    pitch.add(stripMesh)
  }

  const marking = new THREE.LineBasicMaterial({ color: 0xe9eee7, transparent: true, opacity: 0.86 })
  pitch.add(line([[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]], marking, true))
  pitch.add(line([[0, -width / 2], [0, width / 2]], marking))
  const circlePoints: Array<[number, number]> = []
  for (let i = 0; i < 64; i += 1) {
    const angle = (i / 64) * Math.PI * 2
    circlePoints.push([Math.cos(angle) * 9.15, Math.sin(angle) * 9.15])
  }
  pitch.add(line(circlePoints, marking, true))

  for (const side of [-1, 1]) {
    const goalX = side * (length / 2)
    const boxX = goalX - side * 16.5
    pitch.add(line([[goalX, -20.16], [boxX, -20.16], [boxX, 20.16], [goalX, 20.16]], marking))
    const sixX = goalX - side * 5.5
    pitch.add(line([[goalX, -9.16], [sixX, -9.16], [sixX, 9.16], [goalX, 9.16]], marking))
    const goal = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2.4, 2.44, 7.32)),
      new THREE.LineBasicMaterial({ color: 0xcfd6d3, transparent: true, opacity: 0.8 }),
    )
    goal.position.set(goalX + side * 1.2, 1.22, 0)
    pitch.add(goal)
  }

  target.add(pitch)
}

function makeLabel(track: Track) {
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
  context.fillText(`${track.number}  ${track.label}`, 42, 60)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }))
  sprite.userData.kind = 'player-label'
  sprite.userData.trackId = track.id
  sprite.visible = props.showLabels
  sprite.scale.set(8, 2, 1)
  sprite.position.y = 4.7
  return sprite
}

function createPlayer(track: Track) {
  const group = new THREE.Group()
  group.userData.trackId = track.id
  const color = new THREE.Color(track.color)
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.52, 1.25, 6, 12),
    new THREE.MeshStandardMaterial({ color, roughness: 0.48, metalness: 0.12 }),
  )
  body.position.y = 1.45
  body.castShadow = renderQualityProfile(props.renderQuality).shadows
  body.userData.trackId = track.id
  body.userData.kind = 'player-model'
  body.visible = props.showModels
  group.add(body)

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0xd4a47d, roughness: 0.72 }),
  )
  head.position.y = 2.72
  head.castShadow = renderQualityProfile(props.renderQuality).shadows
  head.userData.trackId = track.id
  head.userData.kind = 'player-model'
  head.visible = props.showModels
  group.add(head)

  const base = new THREE.Mesh(
    new THREE.RingGeometry(0.7, 0.9, 32),
    new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.72 }),
  )
  base.rotation.x = -Math.PI / 2
  base.position.y = 0.04
  base.userData.trackId = track.id
  base.userData.kind = 'player-model'
  base.visible = props.showModels
  group.add(base)
  group.add(makeLabel(track))
  return group
}

function rebuildPlayers() {
  if (!scene3d) return
  for (const group of playerGroups.values()) {
    scene3d.remove(group)
  }
  disposeObjectResources(playerGroups.values())
  playerGroups.clear()
  props.scene.payload.tracks.forEach((track) => {
    const group = createPlayer(track)
    playerGroups.set(track.id, group)
    scene3d!.add(group)
  })
}

function updatePlayerVisualVisibility() {
  for (const group of playerGroups.values()) {
    group.traverse((object) => {
      if (object.userData.kind !== 'player-model' && object.userData.kind !== 'player-label') return
      object.visible = shouldRenderPlayerVisual(object.userData.kind, {
        showModels: props.showModels,
        showLabels: props.showLabels,
      })
    })
  }
}

function updateAnalysisMarkerVisibility() {
  analysisMarkerGroup.visible = props.showAnalysisMarkers
}

function rebuildTrail() {
  if (!scene3d) return
  if (ballTrail) {
    scene3d.remove(ballTrail)
    disposeObjectResources([ballTrail])
  }
  const points = props.scene.payload.ball.keyframes.map(
    (frame) => new THREE.Vector3(frame.x, (frame.y ?? 0.22) + 0.1, frame.z),
  )
  const material = new THREE.LineBasicMaterial({ color: 0xffd36a, transparent: true, opacity: 0.82 })
  ballTrail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material)
  scene3d.add(ballTrail)
}

type SelectedPathSource = {
  subjectId: string
  keyframes: Track['keyframes']
  color: THREE.Color
  ball: boolean
}

function selectedPathSource(): SelectedPathSource | null {
  if (props.ballSelected) {
    return {
      subjectId: 'ball',
      keyframes: props.scene.payload.ball.keyframes,
      color: new THREE.Color(0x5ee7ff),
      ball: true,
    }
  }
  if (!props.selectedTrackId) return null
  const track = props.scene.payload.tracks.find((candidate) => candidate.id === props.selectedTrackId)
  if (!track) return null
  return {
    subjectId: track.id,
    keyframes: track.keyframes,
    color: new THREE.Color(track.color),
    ball: false,
  }
}

function selectedPathPointHeight(point: PathTrackingSegment['points'][number], ball: boolean) {
  return ball ? Math.max(0.28, point.y ?? 0.22) + 0.08 : 0.115
}

function createTrackedPathLine(
  segment: PathTrackingSegment,
  source: SelectedPathSource,
) {
  const geometry = new THREE.BufferGeometry().setFromPoints(segment.points.map(
    (point) => new THREE.Vector3(
      point.x,
      selectedPathPointHeight(point, source.ball),
      point.z,
    ),
  ))
  geometry.computeBoundingSphere()

  const observed = segment.evidence === 'observed'
  const color = source.color.clone()
  color.offsetHSL(0, observed ? -0.04 : -0.12, observed ? 0.16 : 0.28)
  const commonMaterial = {
    color,
    transparent: true,
    opacity: observed ? 0.98 : 0.62,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
  }
  const material = observed
    ? new THREE.LineBasicMaterial(commonMaterial)
    : new THREE.LineDashedMaterial({
      ...commonMaterial,
      dashSize: 0.65,
      gapSize: 0.48,
    })
  const pathLine = new THREE.Line(geometry, material)
  pathLine.computeLineDistances()
  pathLine.renderOrder = observed ? 13 : 12
  pathLine.userData.kind = observed ? 'path-observed' : 'path-inferred'
  return pathLine
}

function rebuildSelectedPath() {
  if (!scene3d) return
  if (selectedPathGroup) {
    scene3d.remove(selectedPathGroup)
    disposeObjectResources([selectedPathGroup])
  }
  selectedPathGroup = new THREE.Group()
  selectedPathGroup.userData.kind = 'selected-object-path'
  selectedPathCursor = null
  selectedPathPlayback = null

  const source = selectedPathSource()
  const normalizedKeyframes = source
    ? pathTrackingPoints(source.keyframes).map((point) => point.keyframe)
    : []
  const segments = buildPathTrackingSegments(
    normalizedKeyframes,
    pathTrackingOptionsForSubject(source?.ball ? 'ball' : 'player'),
  )
  if (source && segments.length) {
    selectedPathPlayback = {
      subjectId: source.subjectId,
      segments,
      ball: source.ball,
    }
    segments.forEach((segment) => selectedPathGroup!.add(createTrackedPathLine(segment, source)))
    selectedPathCursor = new THREE.Mesh(
      new THREE.RingGeometry(source.ball ? 0.34 : 0.48, source.ball ? 0.52 : 0.7, 48),
      new THREE.MeshBasicMaterial({
        color: source.ball ? 0xc7f8ff : 0xfff1b8,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.96,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    )
    selectedPathCursor.rotation.x = -Math.PI / 2
    selectedPathCursor.renderOrder = 33
    selectedPathCursor.userData.kind = 'path-current-time'
    selectedPathGroup.add(selectedPathCursor)
  }

  selectedPathGroup.visible = props.showPathTracking && segments.length > 0
  selectedPathGroup.userData.hasPath = segments.length > 0
  scene3d.add(selectedPathGroup)
  updateObjects()
}

function rebuildAnalysisMarkers() {
  const previousMarkers = [...analysisMarkerGroup.children]
  for (const child of previousMarkers) {
    analysisMarkerGroup.remove(child)
  }
  disposeObjectResources(previousMarkers)
  const analysis = props.frameAnalysis
  if (!analysis) return
  analysis.people.forEach((person) => {
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
    analysisMarkerGroup.add(marker)
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
    analysisMarkerGroup.add(marker)
  })
}

function createSelectionRig(target: THREE.Scene) {
  selectionRig = new THREE.Group()
  selectionRig.visible = false
  selectionRig.userData.kind = 'selection-rig'

  selectionRing = new THREE.Mesh(
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
  selectionRing.rotation.x = -Math.PI / 2
  selectionRing.position.y = 0.07
  selectionRing.renderOrder = 30
  selectionRig.add(selectionRing)

  const markerCanvas = document.createElement('canvas')
  markerCanvas.width = 128
  markerCanvas.height = 160
  const markerContext = markerCanvas.getContext('2d')!
  markerContext.shadowColor = 'rgba(255, 211, 106, .9)'
  markerContext.shadowBlur = 18
  markerContext.strokeStyle = '#ffd36a'
  markerContext.lineWidth = 8
  markerContext.beginPath()
  markerContext.arc(64, 58, 30, 0, Math.PI * 2)
  markerContext.stroke()
  markerContext.shadowBlur = 10
  markerContext.fillStyle = '#fff2c2'
  markerContext.beginPath()
  markerContext.moveTo(46, 104)
  markerContext.lineTo(82, 104)
  markerContext.lineTo(64, 132)
  markerContext.closePath()
  markerContext.fill()

  const markerTexture = new THREE.CanvasTexture(markerCanvas)
  markerTexture.colorSpace = THREE.SRGBColorSpace
  const markerMaterial = new THREE.SpriteMaterial({
    map: markerTexture,
    color: 0xffffff,
    transparent: true,
    opacity: 0.96,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  })
  selectionMarker = new THREE.Sprite(markerMaterial)
  selectionMarker.position.y = 4.45
  selectionMarker.scale.set(2.25, 2.8, 1)
  selectionMarker.renderOrder = 31
  selectionRig.add(selectionMarker)

  target.add(selectionRig)
}

function createBallSelectionRig(target: THREE.Scene) {
  ballSelectionRig = new THREE.Group()
  ballSelectionRig.visible = false
  ballSelectionRig.userData.kind = 'ball-selection-rig'

  ballSelectionRing = new THREE.Mesh(
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
  ballSelectionRing.rotation.x = -Math.PI / 2
  ballSelectionRing.position.y = 0.09
  ballSelectionRing.renderOrder = 32
  ballSelectionRig.add(ballSelectionRing)

  const crosshairMaterial = new THREE.LineBasicMaterial({
    color: 0xc7f8ff,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  })
  const crosshair = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.88, 0, 0), new THREE.Vector3(-0.72, 0, 0),
      new THREE.Vector3(0.72, 0, 0), new THREE.Vector3(0.88, 0, 0),
      new THREE.Vector3(0, 0, -0.88), new THREE.Vector3(0, 0, -0.72),
      new THREE.Vector3(0, 0, 0.72), new THREE.Vector3(0, 0, 0.88),
    ]),
    crosshairMaterial,
  )
  crosshair.position.y = 0.065
  crosshair.renderOrder = 32
  ballSelectionRig.add(crosshair)

  target.add(ballSelectionRig)
}

function animateSelectionRig(elapsedSeconds: number) {
  if (!selectionRig?.visible || !selectionRing || !selectionMarker) return
  const pulse = (Math.sin(elapsedSeconds * 4.2) + 1) / 2
  const ringScale = 1 + pulse * 0.13
  selectionRing.scale.setScalar(ringScale)
  selectionRing.material.opacity = 0.58 + pulse * 0.34
  selectionRing.rotation.z = elapsedSeconds * 0.7

  selectionMarker.position.y = 4.35 + Math.sin(elapsedSeconds * 2.8) * 0.16
  selectionMarker.material.opacity = 0.76 + pulse * 0.2
  const markerScale = 1 + pulse * 0.06
  selectionMarker.scale.set(2.25 * markerScale, 2.8 * markerScale, 1)
}

function animateBallSelectionRig(elapsedSeconds: number) {
  if (!ballSelectionRig?.visible || !ballSelectionRing) return
  const pulse = (Math.sin(elapsedSeconds * 5.4) + 1) / 2
  const scale = 1 + pulse * 0.16
  ballSelectionRing.scale.setScalar(scale)
  ballSelectionRing.material.opacity = 0.62 + pulse * 0.34
  ballSelectionRing.rotation.z = -elapsedSeconds * 0.9
}

function animateSelectedPathCursor(elapsedSeconds: number) {
  if (!selectedPathCursor?.visible || !selectedPathGroup?.visible) return
  const pulse = (Math.sin(elapsedSeconds * 5) + 1) / 2
  selectedPathCursor.scale.setScalar(0.92 + pulse * 0.2)
  selectedPathCursor.material.opacity = 0.68 + pulse * 0.3
  selectedPathCursor.rotation.z = elapsedSeconds * 0.55
}

function updateObjects() {
  props.scene.payload.tracks.forEach((track) => {
    const group = playerGroups.get(track.id)
    if (!group) return
    const position = interpolateKeyframes(track.keyframes, props.currentTime)
    group.position.set(position.x, 0, position.z)
    const next = interpolateKeyframes(track.keyframes, Math.min(props.scene.duration, props.currentTime + 0.2))
    group.rotation.y = Math.atan2(next.x - position.x, next.z - position.z)
    // Accepted actors have a continuous latent state for the whole moment.
    // Confidence controls evidence/QA, never whether a player pops in or out.
    group.visible = shouldRenderActor(track)
  })
  if (ballMesh) {
    const frames = props.scene.payload.ball.keyframes
    const ball = interpolateKeyframes(frames, props.currentTime)
    ballMesh.visible = shouldRenderBall(props.showBall, frames, props.currentTime, ball.confidence)
    ballMesh.position.set(ball.x, Math.max(0.24, ball.y ?? 0.24), ball.z)

    if (ballSelectionRig) {
      const selectedTime = props.selectedBallKeyframeTime
      const hasSelectedKeyframe = props.ballEditMode && selectedTime !== null && frames.length > 0
      ballSelectionRig.visible = hasSelectedKeyframe
      if (hasSelectedKeyframe) {
        const selectedBall = interpolateKeyframes(frames, selectedTime)
        ballSelectionRig.position.set(selectedBall.x, 0, selectedBall.z)
      }
    }
  }
  if (ballTrail) {
    const selectedBallPathReplacesTrail = props.showPathTracking
      && props.ballSelected
      && selectedPathPlayback?.ball === true
      && selectedPathGroup?.userData.hasPath === true
    ballTrail.visible = props.showTrails && !selectedBallPathReplacesTrail
  }
  if (selectionRig) {
    const selected = props.selectedTrackId ? playerGroups.get(props.selectedTrackId) : null
    selectionRig.visible = Boolean(selected?.visible)
    if (selected?.visible) selectionRig.position.set(selected.position.x, 0, selected.position.z)
  }
  if (selectedPathGroup) {
    const source = selectedPathSource()
    const playback = selectedPathPlayback
    const hasPath = selectedPathGroup.userData.hasPath === true
    const selectionMatchesPath = Boolean(source && playback && source.subjectId === playback.subjectId)
    selectedPathGroup.visible = props.showPathTracking && hasPath && selectionMatchesPath
    if (selectedPathCursor && playback && hasPath && selectionMatchesPath) {
      const current = interpolatePathTrackingSegments(playback.segments, props.currentTime)
      selectedPathCursor.visible = Boolean(current)
      if (current) {
        selectedPathCursor.position.set(
          current.x,
          playback.ball ? Math.max(0.28, current.y ?? 0.22) + 0.08 : 0.12,
          current.z,
        )
      }
    }
  }
}

function cameraPreset(name: 'broadcast' | 'orbit' | 'tactical' | 'goal') {
  if (!camera || !controls) return
  const presets = {
    broadcast: { position: new THREE.Vector3(-4, 54, 70), target: new THREE.Vector3(4, 0, 0) },
    orbit: { position: new THREE.Vector3(-28, 20, 30), target: new THREE.Vector3(8, 0, 0) },
    tactical: { position: new THREE.Vector3(0, 91, 0.1), target: new THREE.Vector3(0, 0, 0) },
    goal: { position: new THREE.Vector3(58, 7, 0), target: new THREE.Vector3(20, 0, 0) },
  }
  camera.position.copy(presets[name].position)
  controls.target.copy(presets[name].target)
  controls.update()
  status.value = `${name[0].toUpperCase()}${name.slice(1)} camera`
}

function normalizedPointer(event: PointerEvent) {
  if (!renderer) return
  const rect = renderer.domElement.getBoundingClientRect()
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
}

function onPointerDown(event: PointerEvent) {
  if (event.button !== 0) {
    selectionPointerId = null
    return
  }
  selectionPointerId = event.pointerId
  pointerStart.set(event.clientX, event.clientY)
}

function isEffectivelyVisible(object: THREE.Object3D) {
  let current: THREE.Object3D | null = object
  while (current) {
    if (!current.visible) return false
    current = current.parent
  }
  return true
}

function cancelPointerSelection() {
  selectionPointerId = null
}

function onPointerUp(event: PointerEvent) {
  if (selectionPointerId !== event.pointerId) return
  selectionPointerId = null
  if (!camera || !renderer) return
  if (pointerStart.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 5) return
  normalizedPointer(event)
  raycaster.setFromCamera(pointer, camera)
  if (props.ballEditMode) {
    const pitchHit = pitchMesh ? raycaster.intersectObject(pitchMesh, false)[0] : undefined
    if (pitchHit) emit('moveBall', { x: pitchHit.point.x, z: pitchHit.point.z })
    return
  }
  if (ballMesh?.visible && raycaster.intersectObject(ballMesh, false).length) {
    emit('selectBall')
    return
  }
  const hits = raycaster.intersectObjects([...playerGroups.values()], true)
  const visiblePlayerHits = hits.filter((hit) => (
    typeof hit.object.userData.trackId === 'string'
    && isEffectivelyVisible(hit.object)
  ))
  const playerHit = visiblePlayerHits.find((hit) => hit.object.userData.kind === 'player-model')
    ?? visiblePlayerHits.find((hit) => hit.object.userData.kind === 'player-label')
  if (playerHit) {
    emit('select', playerHit.object.userData.trackId as string)
    return
  }
  if (props.editMode && props.selectedTrackId && pitchMesh) {
    const pitchHit = raycaster.intersectObject(pitchMesh, false)[0]
    if (pitchHit) emit('moveTrack', { x: pitchHit.point.x, z: pitchHit.point.z })
  }
}

onMounted(() => {
  if (!host.value) return
  scene3d = new THREE.Scene()
  scene3d.background = new THREE.Color(0x11191b)
  scene3d.fog = new THREE.FogExp2(0x11191b, 0.004)

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 320)
  camera.position.set(-4, 54, 70)
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.domElement.style.cursor = props.ballEditMode ? 'crosshair' : ''
  host.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.07
  controls.maxPolarAngle = Math.PI / 2.04
  controls.minDistance = 8
  controls.maxDistance = 150
  controls.target.set(4, 0, 0)

  hemisphereLight = new THREE.HemisphereLight(0xe8f1ff, 0x173523, 1.45)
  scene3d.add(hemisphereLight)
  keyLight = new THREE.DirectionalLight(0xfff5e8, 1.65)
  keyLight.position.set(-28, 56, 22)
  keyLight.shadow.camera.near = 1
  keyLight.shadow.camera.far = 140
  keyLight.shadow.camera.left = -75
  keyLight.shadow.camera.right = 75
  keyLight.shadow.camera.top = 65
  keyLight.shadow.camera.bottom = -65
  keyLight.shadow.camera.updateProjectionMatrix()
  scene3d.add(keyLight)
  fillLight = new THREE.DirectionalLight(0xaecbff, 0)
  fillLight.position.set(36, 24, -28)
  fillLight.castShadow = false
  fillLight.visible = false
  scene3d.add(fillLight)
  createStadiumFloodlights(scene3d)

  createPitch(scene3d)
  scene3d.add(analysisMarkerGroup)
  updateAnalysisMarkerVisibility()
  rebuildPlayers()
  rebuildTrail()
  rebuildSelectedPath()
  rebuildAnalysisMarkers()

  ballMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 18, 14),
    new THREE.MeshStandardMaterial({ color: 0xf8f3df, roughness: 0.54 }),
  )
  ballMesh.castShadow = renderQualityProfile(props.renderQuality).shadows
  scene3d.add(ballMesh)

  createSelectionRig(scene3d)
  createBallSelectionRig(scene3d)

  resizeObserver = new ResizeObserver(resizeRenderer)
  resizeObserver.observe(host.value)
  applyRenderQuality()

  renderer.domElement.addEventListener('pointerdown', onPointerDown)
  renderer.domElement.addEventListener('pointerup', onPointerUp)
  renderer.domElement.addEventListener('pointercancel', cancelPointerSelection)
  renderer.setAnimationLoop((time) => {
    updateObjects()
    controls?.update()
    animateSelectionRig(time / 1000)
    animateBallSelectionRig(time / 1000)
    animateSelectedPathCursor(time / 1000)
    renderer?.render(scene3d!, camera!)
  })
})

watch(
  () => props.scene.payload.tracks.map((track) => `${track.id}:${track.number}:${track.label}:${track.color}`).join('|'),
  rebuildPlayers,
)
watch(() => JSON.stringify(props.scene.payload.ball.keyframes), rebuildTrail)
watch(
  () => [
    props.selectedTrackId,
    props.ballSelected,
    props.ballSelected
      ? JSON.stringify(props.scene.payload.ball.keyframes)
      : JSON.stringify(props.scene.payload.tracks.find((track) => track.id === props.selectedTrackId)?.keyframes ?? []),
    props.scene.payload.tracks.find((track) => track.id === props.selectedTrackId)?.color ?? '',
  ],
  rebuildSelectedPath,
)
watch(() => JSON.stringify(props.frameAnalysis), rebuildAnalysisMarkers)
watch(() => [props.showModels, props.showLabels], updatePlayerVisualVisibility)
watch(() => props.showBall, updateObjects)
watch(() => props.ballEditMode, (enabled) => {
  if (renderer) renderer.domElement.style.cursor = enabled ? 'crosshair' : ''
  updateObjects()
})
watch(() => props.selectedBallKeyframeTime, updateObjects)
watch(() => props.showAnalysisMarkers, updateAnalysisMarkerVisibility)
watch(() => props.showPathTracking, updateObjects)
watch(() => props.renderQuality, applyRenderQuality)

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  if (renderer) {
    renderer.domElement.removeEventListener('pointerdown', onPointerDown)
    renderer.domElement.removeEventListener('pointerup', onPointerUp)
    renderer.domElement.removeEventListener('pointercancel', cancelPointerSelection)
    renderer.setAnimationLoop(null)
  }
  controls?.dispose()
  if (scene3d) {
    disposeObjectResources([scene3d])
    scene3d.clear()
    analysisMarkerGroup.clear()
    playerGroups.clear()
    stadiumLights = []
  }
  keyLight?.shadow.dispose()
  if (renderer) {
    renderer.dispose()
    renderer.domElement.remove()
  }
})

defineExpose({ cameraPreset })
</script>

<template>
  <div ref="host" class="three-host">
    <div class="viewport-hud">
      <span>{{ status }}</span>
    </div>
    <div v-if="ballEditMode" class="edit-hint">
      Ball keypoint<span v-if="selectedBallKeyframeTime !== null"> · {{ selectedBallKeyframeTime.toFixed(2) }}s</span> — click the pitch to place it
    </div>
    <div v-else-if="editMode" class="edit-hint">Click the pitch to place the selected player</div>
    <aside
      v-if="actionPreview"
      class="action-preview-hud"
      :style="{ '--action-color': actionPreview.color }"
      role="status"
      aria-live="polite"
      aria-label="Active player action preview"
    >
      <header>
        <span>Active action</span>
        <code>{{ actionPreview.type }}</code>
      </header>
      <strong>{{ actionPreview.label }}</strong>
      <div class="action-phase-heading">
        <span>Normalized phase</span>
        <b>{{ actionPreview.phasePercent }}%</b>
      </div>
      <div
        class="action-phase-track"
        role="progressbar"
        aria-label="Action phase"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuenow="actionPreview.phasePercent"
      >
        <i :style="{ width: `${actionPreview.phasePercent}%` }" />
      </div>
      <div v-if="actionPreview.keypointLabel" class="action-keypoint">
        <span>Nearest keypoint</span>
        <b>{{ actionPreview.keypointLabel }}<small v-if="actionPreview.keypointTiming"> · {{ actionPreview.keypointTiming }}</small></b>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.action-preview-hud {
  --action-color: #ffd36a;
  position: absolute;
  z-index: 6;
  right: 12px;
  bottom: 12px;
  width: min(228px, calc(100% - 24px));
  padding: 10px;
  border: 1px solid color-mix(in srgb, var(--action-color) 48%, transparent);
  border-radius: 3px;
  background: rgba(7, 10, 10, .88);
  box-shadow: 0 10px 28px rgba(0, 0, 0, .34);
  backdrop-filter: blur(11px);
  pointer-events: none;
}

.action-preview-hud header,
.action-phase-heading,
.action-keypoint {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.action-preview-hud header > span,
.action-phase-heading > span,
.action-keypoint > span {
  color: #78837d;
  font: 500 8px/1.2 'DM Mono', monospace;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.action-preview-hud header code {
  color: color-mix(in srgb, var(--action-color) 84%, white);
  font: 600 8px/1.2 'DM Mono', monospace;
}

.action-preview-hud > strong {
  display: block;
  margin-top: 4px;
  overflow: hidden;
  color: #f0f4f0;
  font: 650 13px/1.3 Inter, sans-serif;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.action-phase-heading {
  margin-top: 9px;
}

.action-phase-heading b {
  color: var(--action-color);
  font: 650 9px/1.2 'DM Mono', monospace;
}

.action-phase-track {
  height: 3px;
  margin-top: 5px;
  overflow: hidden;
  background: rgba(255, 255, 255, .1);
}

.action-phase-track i {
  display: block;
  height: 100%;
  background: var(--action-color);
  box-shadow: 0 0 9px color-mix(in srgb, var(--action-color) 65%, transparent);
}

.action-keypoint {
  margin-top: 9px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, .08);
}

.action-keypoint b {
  color: #e3e9e4;
  font: 600 8px/1.2 'DM Mono', monospace;
  white-space: nowrap;
}

.action-keypoint small {
  color: #8c9791;
  font: inherit;
  font-weight: 500;
}

@media (max-width: 760px) {
  .action-preview-hud {
    right: 8px;
    bottom: 8px;
    padding: 8px;
  }
}
</style>

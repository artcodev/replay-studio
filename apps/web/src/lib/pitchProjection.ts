import type { CalibrationEvidenceMarking } from '../types'

type Point = [number, number]
type Row3 = [number, number, number]
export type Matrix3 = [Row3, Row3, Row3]

const pitchLines: Array<[string, Point, Point]> = [
  ['touch-top', [-52.5, -34], [52.5, -34]],
  ['touch-bottom', [-52.5, 34], [52.5, 34]],
  ['goal-left', [-52.5, -34], [-52.5, 34]],
  ['goal-right', [52.5, -34], [52.5, 34]],
  ['halfway', [0, -34], [0, 34]],
  ['penalty-left-main', [-36, -20.16], [-36, 20.16]],
  ['penalty-left-top', [-52.5, -20.16], [-36, -20.16]],
  ['penalty-left-bottom', [-52.5, 20.16], [-36, 20.16]],
  ['penalty-right-main', [36, -20.16], [36, 20.16]],
  ['penalty-right-top', [36, -20.16], [52.5, -20.16]],
  ['penalty-right-bottom', [36, 20.16], [52.5, 20.16]],
  ['goal-area-left-main', [-47, -9.16], [-47, 9.16]],
  ['goal-area-left-top', [-52.5, -9.16], [-47, -9.16]],
  ['goal-area-left-bottom', [-52.5, 9.16], [-47, 9.16]],
  ['goal-area-right-main', [47, -9.16], [47, 9.16]],
  ['goal-area-right-top', [47, -9.16], [52.5, -9.16]],
  ['goal-area-right-bottom', [47, 9.16], [52.5, 9.16]],
]

function curve(centerX: number, centerZ: number, radius: number, side?: 'left' | 'right') {
  return Array.from({ length: 180 }, (_, index): Point => {
    const angle = index / 179 * Math.PI * 2
    return [centerX + Math.cos(angle) * radius, centerZ + Math.sin(angle) * radius]
  }).filter(([x]) => side === 'left' ? x >= -36 : side === 'right' ? x <= 36 : true)
}

const pitchCurves: Array<[string, Point[]]> = [
  ['center-circle', curve(0, 0, 9.15)],
  ['penalty-arc-left', curve(-41.5, 0, 9.15, 'left')],
  ['penalty-arc-right', curve(41.5, 0, 9.15, 'right')],
]

function matrix3(value: number[][]): Matrix3 | null {
  if (value.length !== 3 || value.some((row) => row.length !== 3 || row.some((item) => !Number.isFinite(item)))) return null
  return value as Matrix3
}

export function invertHomography(value: number[][]): Matrix3 | null {
  const matrix = matrix3(value)
  if (!matrix) return null
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null
  return [
    [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
    [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
    [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
  ]
}

/** Apply a finite homography without silently accepting a point at infinity. */
export function projectHomographyPoint(point: Point, matrix: Matrix3) {
  const denominator = matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2]
  if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-8) return null
  const x = (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2]) / denominator
  const y = (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2]) / denominator
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null
}

function linePoints(start: Point, end: Point) {
  return Array.from({ length: 90 }, (_, index): Point => {
    const alpha = index / 89
    return [start[0] + (end[0] - start[0]) * alpha, start[1] + (end[1] - start[1]) * alpha]
  })
}

export function projectPitchMarkings(imageToPitch: number[][] | null | undefined, width: number, height: number) {
  if (!imageToPitch || width <= 0 || height <= 0) return []
  const pitchToImage = invertHomography(imageToPitch)
  if (!pitchToImage) return []
  const sources: Array<[string, 'line' | 'curve', Point[]]> = [
    ...pitchLines.map(([id, start, end]): [string, 'line', Point[]] => [id, 'line', linePoints(start, end)]),
    ...pitchCurves.map(([id, points]): [string, 'curve', Point[]] => [id, 'curve', points]),
  ]
  return sources.flatMap(([id, kind, pitchPoints]): CalibrationEvidenceMarking[] => {
    const points = pitchPoints
      .map((point) => projectHomographyPoint(point, pitchToImage))
      .filter((point): point is { x: number; y: number } => Boolean(
        point
        && point.x > -width * .2
        && point.x < width * 1.2
        && point.y > -height * .2
        && point.y < height * 1.2,
      ))
      .map((point) => ({ x: Math.round(point.x * 100) / 100, y: Math.round(point.y * 100) / 100 }))
    if (points.length < (kind === 'curve' ? 8 : 2)) return []
    return [{ id, kind, points }]
  })
}

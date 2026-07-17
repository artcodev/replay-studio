export type RenderQuality = 'basic' | 'enhanced'

export interface RenderQualityProfile {
  pixelRatioCap: number
  shadows: boolean
  softShadows: boolean
  shadowMapSize: number
  hemisphereIntensity: number
  keyLightIntensity: number
  fillLightIntensity: number
  stadiumLightIntensity: number
  toneMappingExposure: number
}

export const RENDER_QUALITY_PROFILES: Readonly<Record<RenderQuality, RenderQualityProfile>> = {
  basic: {
    pixelRatioCap: 1.25,
    shadows: false,
    softShadows: false,
    shadowMapSize: 1024,
    hemisphereIntensity: 1.45,
    keyLightIntensity: 1.65,
    fillLightIntensity: 0.3,
    stadiumLightIntensity: 1800,
    toneMappingExposure: 1.14,
  },
  enhanced: {
    pixelRatioCap: 2,
    shadows: true,
    softShadows: true,
    shadowMapSize: 2048,
    hemisphereIntensity: 1.25,
    keyLightIntensity: 1.85,
    fillLightIntensity: 0.5,
    stadiumLightIntensity: 2600,
    toneMappingExposure: 1.18,
  },
}

export function renderQualityProfile(quality: RenderQuality): RenderQualityProfile {
  return RENDER_QUALITY_PROFILES[quality]
}

export function renderPixelRatio(quality: RenderQuality, devicePixelRatio: number): number {
  const safeDeviceRatio = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0
    ? Math.max(1, devicePixelRatio)
    : 1
  return Math.min(safeDeviceRatio, renderQualityProfile(quality).pixelRatioCap)
}

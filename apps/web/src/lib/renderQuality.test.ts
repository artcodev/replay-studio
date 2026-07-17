import { describe, expect, it } from 'vitest'
import { renderPixelRatio, renderQualityProfile } from './renderQuality'

describe('render quality profiles', () => {
  it('keeps the basic renderer lightweight', () => {
    expect(renderQualityProfile('basic')).toMatchObject({
      shadows: false,
      pixelRatioCap: 1.25,
      stadiumLightIntensity: 1800,
    })
    expect(renderPixelRatio('basic', 3)).toBe(1.25)
  })

  it('enables the enhanced lighting and shadow budget', () => {
    expect(renderQualityProfile('enhanced')).toMatchObject({
      shadows: true,
      softShadows: true,
      shadowMapSize: 2048,
      stadiumLightIntensity: 2600,
    })
    expect(renderPixelRatio('enhanced', 3)).toBe(2)
  })

  it('uses a brighter stadium-lighting budget in enhanced mode', () => {
    const basic = renderQualityProfile('basic')
    const enhanced = renderQualityProfile('enhanced')
    expect(basic.stadiumLightIntensity).toBeGreaterThan(0)
    expect(enhanced.stadiumLightIntensity).toBeGreaterThan(basic.stadiumLightIntensity)
    expect(enhanced.toneMappingExposure).toBeGreaterThanOrEqual(basic.toneMappingExposure)
  })

  it('sanitizes invalid device pixel ratios', () => {
    expect(renderPixelRatio('basic', Number.NaN)).toBe(1)
    expect(renderPixelRatio('enhanced', 0)).toBe(1)
  })
})

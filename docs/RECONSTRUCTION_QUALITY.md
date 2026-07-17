# Reconstruction quality and calibration QA

Этот документ является каноническим контрактом качества для video-to-game-state
контура Replay Studio. Его задача — не создать ещё один субъективный `confidence`,
а отделить факт завершения вычисления от доказанного качества результата.

## 1. Что именно реконструируется

Текущий продукт строит **2D game state на плоскости поля**, который визуализируется
в Three.js:

- координата игрока — точка контакта с газоном (`x`, `z`) в поле 105 x 68 м;
- рост, поза и положение частей тела не реконструируются;
- single-view homography не восстанавливает высоту мяча;
- игрок вне кадра остаётся неизвестным, если нет синхронизированного второго
  ракурса или лицензированного tracking-источника; визуальный слой сохраняет его
  как низкоуверенный `presence-inferred`, но не объявляет эту позицию измеренной;
- match API даёт личности, состав и события, но не должен придумывать координаты.

Поэтому корректное название результата — `ground-plane game state`, а не
полная 3D-реконструкция матча. Three.js отвечает за представление этих данных,
но не превращает 2D-наблюдение в измеренную 3D-геометрию.

## 2. Два независимых статуса

```json
{
  "processingStatus": "ready",
  "quality": {
    "verdict": "review"
  }
}
```

`processingStatus = ready` означает только то, что job завершён и артефакты
сохранены. `quality.verdict` отвечает на другой вопрос: достаточно ли измерений,
чтобы показывать реконструкцию как достоверную.

Вердикты:

| Verdict | Значение |
|---|---|
| `pass` | Все обязательные runtime-gates измерены и прошли |
| `review` | Есть пограничная метрика или отсутствует обязательное доказательство |
| `reject` | Хотя бы один обязательный gate явно нарушен |

Неизвестная метрика имеет status `unknown`. Для обязательного gate это приводит
к общему `review`, а не к `pass`. Таким образом старые результаты без provenance
не объявляются корректными задним числом.

## 3. Evidence-first архитектура

Итоговая сцена не должна быть единственным рабочим состоянием CV pipeline.
Рекомендуемая цепочка артефактов:

```text
FrameManifest (точные PTS, shot/cut)
  -> CalibrationCandidate[] на каждый sampled frame
  -> RawDetection[] в исходном разрешении
  -> Tracklet[] и association edges
  -> Role/team/identity evidence
  -> ProjectedObservation[] с uncertainty и provenance
  -> ReconstructionQualityReport
  -> принятый ReconstructionRun
```

Артефакты считаются immutable и имеют model/config version. Это позволяет:

- заменить калибратор без повторной детекции;
- заменить tracker без повторного декодирования видео;
- сравнить текущий run с предыдущим accepted run;
- воспроизвести причину каждой позиции на карте;
- не публиковать частично записанный результат при падении worker.

Минимальный provenance каждой координаты:

```json
{
  "projection": {
    "source": "direct",
    "calibrationFrameIndex": 142,
    "uncertaintyMetres": 0.74,
    "clamped": false
  },
  "projectionSource": "direct",
  "positionUncertaintyMetres": 0.74
}
```

Разрешённые источники:

- `direct` — матрица оценена и проверена на этом source frame;
- `manual-propagated` — ручная матрица перенесена во времени и проверена;
- `screen-approximate` — координата не метрическая, должна быть визуально
  отличима и не может скрыто входить в metric run.

Если проекцию нельзя доказать, предпочтительно сохранить `pitchPosition = null`,
а не правдоподобную экранную эвристику.

## 4. Покадровый контракт калибровки

`videoAsset.reconstruction.calibration.frameEvidence` хранит фактическое решение,
использованное для каждого sampled frame:

```json
{
  "sourceFrameIndex": 142,
  "sampleIndex": 18,
  "sceneTime": 3.6,
  "sourceTime": 184.12,
  "status": "accepted",
  "source": "pnlcalib",
  "projectionSource": "direct",
  "backend": "pnlcalib-points-lines",
  "confidence": 0.88,
  "imageToPitch": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "keypointCount": 19,
  "inlierCount": 17,
  "inlierRatio": 0.895,
  "reprojectionError": 3.4,
  "reprojectionP95": 6.1,
  "visiblePitchSide": "left",
  "rejectionReasons": [],
  "personSupport": {"supported": 13, "total": 14, "ratio": 0.929},
  "cameraMotion": {"status": "estimated"}
}
```

`status` принимает `accepted`, `rejected` или `missing`. `source` описывает
backend, а `projectionSource` — откуда взялась именно применённая матрица. Эти
поля нельзя заменять общим confidence.

Пока идёт миграция, тот же массив может находиться в compatibility-поле
`reconstruction.calibrationFrames`. QA evaluator читает оба варианта.

## 5. Реализованные runtime-метрики

Вычисление находится в `apps/api/app/quality_metrics.py`. Оно не зависит от
YOLO/PyTorch и не мутирует сцену.

| Metric | Как считается | Что обнаруживает |
|---|---|---|
| `calibrationCoverage` | accepted frames / sampled frames | Калибровка есть только на малой части момента |
| `directCalibrationCoverage` | direct frames / sampled frames | Результат почти полностью держится на propagation |
| `maxCalibrationGap` | Самая длинная непрерывная серия missing/rejected с учётом cadence | Длительный слепой участок камеры |
| `calibrationResidualP50/P95` | Распределение покадрового reprojection error | Средняя ошибка и плохой хвост, который скрывает median |
| `calibrationInlierRatioP10` | Десятый процентиль отношения inliers | Нестабильные худшие кадры |
| `semanticAlignmentF1P10` | Десятый процентиль bidirectional line F1 | Белые пиксели совпали случайно или не покрывают модель поля |
| `visiblePitchSideAgreement` | Доля left/right votes за доминирующую сторону | Mirror flips внутри одного shot |
| `projectionFallbackRatio` | screen-relative observations / observations с provenance | Скрытое смешивание метров и экранных координат |
| `boundaryClampRatio` | Явные clamp flags; для legacy — точные попадания в границу как proxy | Скопление ошибочных точек на краях поля |
| `playerSpeedViolationRatio` | Сегменты игрока быстрее 14 м/с | ID switch, скачок homography, неверная ассоциация |
| `ballSpeedViolationRatio` | Сегменты мяча быстрее 50 м/с | Перескок между ложными ball candidates |
| `trackContinuity` | observations / ожидаемые samples между первым и последним наблюдением | Пропуски внутри трека |
| `trackFragmentationRatio` | Доля треков с gap больше max(0.6 s, 2.5 cadence) | Потеря идентичности и повторное появление как новый tracklet |

Нулевые-confidence endpoints, добавленные только для интерполяции UI, исключаются
из speed и continuity. Они не являются CV-наблюдениями.

### Начальные gates

Пороги ниже — инженерная стартовая точка. Их нужно переоценить после создания
gold set; они не являются научно подтверждённым стандартом.

| Gate | Pass | Review | Reject |
|---|---:|---:|---:|
| Accepted calibration coverage | >= 90% | 75–90% | < 75% |
| Longest calibration gap | <= 0.60 s | 0.60–1.20 s | > 1.20 s |
| Reprojection p50 | <= 4 px | 4–8 px | > 8 px |
| Reprojection p95 | <= 8 px | 8–15 px | > 15 px |
| Inlier ratio p10 | >= 0.70 | 0.50–0.70 | < 0.50 |
| Semantic-line F1 p10 | >= 0.15 | 0.08–0.15 | < 0.08 |
| Visible-side agreement | >= 90% | 80–90% | < 80% |
| Screen fallback ratio | 0% | 0–5% | > 5% |
| Boundary clamp/contact ratio | <= 0.5% | 0.5–2% | > 2% |
| Player speed violations | <= 1% | 1–5% | > 5% |
| Ball speed violations | <= 1% | 1–5% | > 5% |
| Median track completeness | >= 90% | 75–90% | < 75% |
| Fragmented tracks | <= 10% | 10–30% | > 30% |

Ball-speed и inlier-ratio gates пока информационные (`required = false`): отсутствие
мяча или backend без inlier contract не должно само по себе отклонять все сцены.
При этом известное нарушение всё равно отображается в отчёте.

Manual anchors не получают автоматический `pass`: один хороший anchor frame не
доказывает корректность всей панорамы/зума. Без per-frame validation coverage и
gap остаются `unknown`; большой alignment error способен дать `reject` сразу.

## 6. Как смотреть калибровку и понимать, что она корректна

Один overlay на выбранном кадре недостаточен. Calibration QA panel должна
показывать именно сохранённый `frameEvidence`, который произвёл координаты.

### Timeline

- зелёный — accepted direct;
- жёлтый — accepted propagated/manual;
- красный — rejected;
- серый — missing или screen fallback;
- клик открывает конкретный source frame;
- отдельные переходы обозначают camera cut, replay, pan и zoom;
- есть быстрые переходы к worst p95 frames и самому длинному gap.

### Overlay выбранного кадра

- source image без resize-неоднозначности;
- спроецированные **семантически именованные** линии поля;
- detected keypoints/curves с именами;
- inliers зелёным, outliers красным;
- residual vectors от наблюдения до reprojection;
- foot points людей и источник каждой проекции;
- horizon/видимый полигон камеры;
- параллельная minimap без temporal smoothing;
- переключатель кандидатов PnLCalib / local keypoints / manual;
- независимые поля `visiblePitchSide` и `attackingGoal`.

### Проверка, которой можно доверять

1. На всём timeline нет неожиданных mirror flips.
2. Overlay не «плавает» относительно разметки поля при playback.
3. Неиспользованные при fit контрольные точки также имеют малую ошибку.
4. p95 остаётся приемлемым, а не только representative/median frame.
5. Игроки не прижимаются к границе и не совершают физически невозможные скачки.
6. Ручная проверка worst frames подтверждает численные gates.

Даже выполнение этих условий означает «runtime sanity passed», а не измеренную
точность модели. Последняя требует held-out gold set.

## 7. Gold-set workflow

### 7.1 Выбор данных

Начальный набор: 300–500 source frames, сгруппированных по непрерывным shots.
Выбор должен покрывать:

- левые/правые ворота и центральную часть;
- wide, medium, close-up;
- pan, tilt, zoom и статичную камеру;
- дальних маленьких игроков;
- occlusion, crowd/bench false positives;
- goalkeeper/referee;
- мяч на газоне, в воздухе, motion blur;
- live shot и replay;
- разные комплекты формы и освещение.

Split делается **по матчам и shots**, а не случайно по соседним кадрам. Иначе
почти одинаковые кадры утекут одновременно в train и test. Test split после
фиксации не используется для подбора порогов.

### 7.2 Разметка

Для каждого кадра сохраняются:

- точный video PTS и shot id;
- semantic pitch keypoints/curves и visible side;
- bbox/mask и foot point каждого человека;
- persistent track id, team и role;
- goalkeeper/referee отдельно от обычного player;
- мяч: visible/occluded, image centre; pitch point только когда он измерим;
- причины `ignore` для crowd, graphics и phantom.

Не менее 10% test frames размечаются двумя людьми. Расхождения foot point,
семантики линий и track identity проходят adjudication; это одновременно даёт
оценку естественной ошибки разметки.

### 7.3 Offline-метрики

| Слой | Метрики |
|---|---|
| Calibration | [JaC@5 + completeness](https://github.com/SoccerNet/sn-calibration), reprojection p50/p95, held-out landmark error, temporal jitter, mirror-flip rate |
| Person detection | AP50/AP75, precision/recall, foot-point error; отдельно по bbox height/дальности |
| Ball detection | recall, image-centre error, false candidates/frame, longest miss |
| Tracking | HOTA/DetA/AssA, IDF1, ID switches/min, fragments, track completeness |
| Team/role | macro-F1, confusion matrix для player/goalkeeper/referee/other |
| End-to-end game state | [GS-HOTA](https://github.com/SoccerNet/sn-gamestate), pitch error p50/p95 и доля наблюдений <= 1/2/5 м |

Runtime metrics не заменяют эти показатели. Например, низкая скорость не
доказывает правильный track ID, а высокий `person-in-pitch` может быть получен
неверной homography, которая аккуратно укладывает всех в поле.

### 7.4 Regression policy

Каждый model/config run сохраняет:

- commit/config/model/dataset version;
- метрики по test split и по каждому stratum;
- worst-N кадров с overlay;
- latency, peak RAM/VRAM и размер artifacts;
- сравнение с последним accepted baseline.

Изменение не принимается только по среднему score. Не допускается улучшить общий
recall ценой резкого ухудшения goalkeeper, far-side или mirror-side strata.

## 8. CLI и использование в API

Из Python:

```python
from app.quality_metrics import evaluate_reconstruction_quality

report = evaluate_reconstruction_quality(scene)
scene["payload"]["videoAsset"]["reconstruction"]["quality"] = report
```

Evaluator не мутирует `scene` и может получать evidence отдельно:

```python
report = evaluate_reconstruction_quality(scene, frame_evidence)
```

Локальный JSON/CI:

```bash
cd apps/api
../../.venv/bin/python -m app.quality_metrics scene.json
../../.venv/bin/python -m app.quality_metrics scene.json --fail-on reject
```

`--fail-on review` делает отсутствующие обязательные доказательства ошибкой CI;
`--fail-on reject` пропускает ручную review-очередь, но блокирует явный reject.

## 9. Следующий уровень точности

После введения измерений приоритет компонентов становится проверяемым:

1. sequence-level camera calibration и сравнение PnLCalib/TVCalib-кандидатов;
2. detector на исходном разрешении и 10+ FPS с far-side strata;
3. segmentation/pose foot point вместо bottom-centre bbox;
4. ByteTrack/BoT-SORT/TrackLab + ReID и offline association;
5. track-level team/role classification;
6. специализированный temporal ball pipeline;
7. синхронизация и extrinsics до multi-view triangulation.

Качество каждого шага принимается только после A/B на фиксированном gold set.

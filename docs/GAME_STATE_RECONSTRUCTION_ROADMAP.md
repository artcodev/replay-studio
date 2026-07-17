# Game-state reconstruction roadmap

## Решение для проекта

Нам не нужно выбирать только один из трёх предложенных источников. Они занимают разные уровни:

- [SoccerNet sn-gamestate](https://github.com/SoccerNet/sn-gamestate) — эталон задачи, контракт результата, модульный baseline и GS-HOTA evaluation.
- [Roboflow Sports](https://github.com/roboflow/sports) — лёгкий практический reference для детекции, трекинга, keypoint calibration, team clustering и радара.
- [From Broadcast to Minimap](https://arxiv.org/abs/2504.06357) — архитектурный reference для качества уровня challenge winner: YOLOv5m + SegFormer camera estimation + DeepSORT, ReID, orientation и jersey recognition.

Оптимальная архитектура Replay Studio: быстрый интерактивный preview остаётся в текущем FastAPI-процессе, а точный Game State pipeline становится отдельным GPU worker с нейтральным JSON-контрактом. Редактор и Vue-проигрыватель не должны зависеть от TrackLab, PyTorch или конкретной модели.

## 1. SoccerNet sn-gamestate

### Что берём прямо сейчас

1. Семантический контракт: `role = player | goalkeeper | referee | other`, team, nullable jersey number, pitch x/y, track identity. Ручная разметка Replay Studio уже приведена к этому набору ролей.
2. Разделение pipeline на pitch calibration, detection, ReID/tracking, team affiliation и jersey OCR. Каждый модуль должен сохранять собственный confidence и provenance.
3. Tracker State как сохраняемый промежуточный артефакт. Это позволит менять калибровку или классификацию ролей без повторной тяжёлой детекции.
4. GS-HOTA-подобную офлайн-оценку: локализация, association, role/team/jersey — отдельные измерения, а не один субъективный «quality score».
5. Набор калибраторов как benchmark. Репозиторий позволяет сравнивать TVCalib, PnLCalib и NBJW вместо жёсткого выбора одного метода.

### Плюсы

- Это точное определение нашей задачи: все люди на поле, их реальные координаты, роли, команды и номера.
- Есть dataset v1.3, визуализация, tracker-state artifacts и официальная метрика.
- Репозиторий обновлялся 1 мая 2026 года до TrackLab 1.3.24 с исправлением GS-HOTA.
- Человеческие исправления в нашем редакторе можно превратить в небольшой evaluation set на пользовательском видео.

### Минусы и ограничения

- GPL-3.0 и набор зависимостей требуют лицензионного решения перед коммерческим выпуском.
- Reference environment использует Python 3.9/PyTorch/CUDA/mmcv; встраивать его в текущий Python 3.14 API-процесс рискованно.
- Baseline тяжёлый, требует весов/GPU и не гарантирует realtime.
- Dataset и его видео имеют собственные условия использования; кодовая лицензия не заменяет data license.

### Где применить

Отдельный сервис `reconstruction-worker`:

```text
API -> reconstruction job -> GPU worker/TrackLab adapter
                              -> tracker-state artifact
                              -> neutral game-state JSON
API <- job result ------------+
Vue editor reads only neutral JSON + provenance
```

## 2. Roboflow Sports

### Что берём прямо сейчас

- ByteTrack/track association patterns вместо нашей простой nearest-neighbour связи.
- Pitch keypoint detection и homography/radar projection как второй calibration candidate.
- Team classification/embedding пример как прозрачную замену одного цветового histogram.
- Отдельные football player, ball и pitch-keypoint datasets для быстрого experiment.
- Визуальные overlays и компактный radar для диагностики пользователя.

### Плюсы

- MIT license.
- Python >= 3.8 и понятные examples; ниже порог входа, чем у полного TrackLab.
- Репозиторий прямо перечисляет те же трудные места, которые видны в нашем клипе: маленький быстрый мяч, occlusion, ReID, jersey OCR и camera calibration.
- Хорош для быстрого A/B эксперимента на конкретном 1-A.

### Минусы

- Это toolbox/examples, а не единый production pipeline или гарантированный benchmark winner.
- Сам README отмечает, что Python package пока нет и установка идёт из source.
- ByteTrack сам по себе не решает повторный вход игрока, смену камеры, номер и длительные occlusion.
- Качество зависит от внешних весов/datasets и их лицензий.

### Где применить

Первый практический spike: прогнать один непрерывный момент и сравнить recall, false positives, ID switches и pitch error с текущим pipeline. Не заменять весь backend до измерения.

## 3. From Broadcast to Minimap

### Что берём

Статья показывает, что высокий результат возникает не от «более крупной YOLO» в одиночку. Победивший pipeline сочетает:

- fine-tuned detector;
- отдельный SegFormer camera parameter estimator;
- DeepSORT с ReID;
- orientation prediction;
- jersey number recognition;
- временную согласованность на всём матче.

Это подтверждает наш следующий архитектурный шаг: детекция, калибровка и идентичность должны оптимизироваться совместно, но храниться как отдельные evidence layers.

### Где применить

- проектирование точного worker;
- список ablation experiments;
- multi-pass fusion разных ракурсов;
- восстановление identity после выхода/возврата в кадр;
- OCR номера только на лучших видимых кадрах трека, а не на каждом кадре.

Статья — reference, не готовая dependency. Она не заменяет открытый код SoccerNet/Roboflow и не даёт нам права на тренировочное видео.

## Целевая схема pipeline

```text
Video ingest
  -> shot/replay grouping
  -> sampled frame store
  -> detector observations
  -> pitch/camera candidates
  -> short-term tracker
  -> ReID + team/role + jersey evidence
  -> API roster/event priors
  -> human frame anchors
  -> multi-pass alignment and fusion
  -> game-state tracks + uncertainty
  -> Vue review/edit
  -> 3D replay
```

Приоритет доказательств:

1. явная ручная метка пользователя;
2. согласованные наблюдения нескольких ракурсов;
3. визуальная модель и temporal association;
4. roster/event prior из API;
5. fallback эвристика.

API не должен телепортировать игрока в координату события. Он только повышает/понижает вероятность гипотезы.

## Ближайшие измеримые этапы

### R1 — выполнено в текущем MVP

- Исправить far-side filter, из-за которого терялся игрок на 0:00.
- Добавить frame annotations: home/away player, goalkeeper, referee, other, ignore.
- Сделать ручную bbox источником detection и tracking anchor при следующем rebuild.
- Сохранять последний успешный результат, если rebuild падает.

### R2 — выполнено: semantic-keypoint calibration

- Добавлена открытая 32-keypoint pitch-модель Roboflow Sports как локальный backend/fallback.
- Гомография строится по семантическим точкам с RANSAC, поэтому left/right больше не выбирается по безымянному прямоугольнику.
- Сохраняются inliers, reprojection error, frame index и provenance.
- ByteTrack/ReID и полноценный validation-set benchmark остаются отдельным этапом.

### R3 — выполнено: SoccerNet PnLCalib worker

- Добавлен отдельный контейнер с официальным PnLCalib points+lines backend и закреплённым commit `sn-gamestate`.
- API общается с ним через neutral JSON и не импортирует TrackLab/PnLCalib.
- Координаты игроков и мяча рассчитываются по homography каждого кадра; пропуски добирает локальная semantic-keypoint модель.
- На исходном видео проверены реальные кадры и геометрический overlay.
- Следующий измеримый долг: 100–300 вручную проверенных кадров и GS-HOTA-подобные decomposition metrics.

### R4 — identity and multi-pass

- Выполнено: confirm/exclude/merge/split, scope одного observation, `[start,end)`
  range или всей identity, защита merge-графа/cannot-link, preview и
  fingerprint-guarded CAS rebuild после сохранения или удаления.
- Split привязан к immutable observation snapshot и при detector reorder
  применяется только к единственному геометрически совместимому кандидату;
  ambiguity/missing/recycled ID завершаются fail closed.
- Propagate ручной bbox вперёд/назад optical-flow/tracker проходом с preview diff.
- Выбирать лучший кадр номера/лица/формы из всех ракурсов.
- Fusion только после временного alignment и геометрической проверки.

## Критерий выбора

Не выбирать модель по числу найденных bbox. Модель принимается, если на нашем validation set она улучшает far-side recall и уменьшает ложных людей, ID switches и метрическую ошибку без неприемлемого роста времени. В интерфейсе должны оставаться confidence, source и возможность исправления любого результата.

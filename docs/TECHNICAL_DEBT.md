# Технический долг Replay Studio

Актуально на 19 июля 2026 года после clean-cut рефакторинга project-centric
архитектуры. Это канонический реестр **активного** технического долга. Закрытые
работы приведены ниже только как baseline и набор регрессионных инвариантов.

Git/история публикации и `.env`/секреты не входят в скоуп этого документа.
Подробные доказательства и дискуссионные формулировки находятся в
[`PROJECT_AUDIT.md`](PROJECT_AUDIT.md), нормативные границы — в
[`ARCHITECTURE.md`](ARCHITECTURE.md), а обязательные pipeline-инварианты — в
[`PIPELINE_EDGE_CASES.md`](PIPELINE_EDGE_CASES.md).

## Правила ведения долга

- **P0** — пользовательская правка теряется, возвращается неверная revision или
  система публикует противоречивое состояние. Закрывается до следующего
  расширения функциональности.
- **P1** — достижимый reliability/data/accuracy риск текущего локального MVP.
- **P2** — производительность, наблюдаемость, эксплуатация и поддерживаемость;
  выполняется после измерения либо после закрытия зависимого P1.
- **P3 / R&D** — следующий продуктовый или deployment-этап, для которого нужен
  датасет, лицензия, отдельная модель или решение о способе развёртывания.

Пункт считается закрытым только когда удалён второй источник истины, реализован
canonical path, добавлен регрессионный тест и обновлена связанная документация.
Наличие маршрута, таблицы, fake-worker теста или runtime confidence не является
доказательством продуктовой точности.

Legacy-прототипы не являются поддерживаемым продуктом. Глобальные routes,
embedded dense series, startup backfills, dual writes и compatibility fallback
не сохраняются «на всякий случай». Временный мост допустим только с явным
условием удаления и не может стать вторым writable source of truth.

## Состояние архитектуры после рефакторинга

| Граница | Текущий canonical baseline | Остаточный долг |
| --- | --- | --- |
| Project и навигация | Нормализованы `Project`, `Match`, `VideoAsset`, `Segment`, `AnalysisRun`; editor/media/jobs project-scoped; Vue Router владеет навигацией; автоматического выбора проекта/сегмента нет | `TD-MULTI-01`, `TD-OPS-01` |
| Backend composition | `app.main` — composition root; прежние god-модули reconstruction/video/ball/identity/OCR/ReID/player-actions распилены по capability и защищены architecture-тестами | `TD-CODE-01` — только конкретные остатки, не новый generic manager |
| Persistence | Project store разделён на resource/match/identity/analysis repositories; canonical Match, snapshot, references и project pointer публикуются одной транзакцией | `TD-PG-01`, `TD-SEG-01`, `TD-MATCH-01` |
| Execution control plane | API только ставит compact jobs; pipeline/reconstruction runners владеют claim/lease/heartbeat/recovery/cancellation; `AnalysisRun` — telemetry, а не scheduler | `TD-CI-01`, `TD-OBS-01`, `TD-DIST-01` |
| Reconstruction data plane | Dense identity/track/ball/calibration series вынесены из PostgreSQL/Scene в immutable SHA-256 artifacts; чтение выполняется bounded windows | `TD-STORAGE-01`, `TD-STORAGE-02` |
| Match boundary | Browser и project model используют provider-neutral internal IDs и immutable snapshot reference; provider provenance хранится отдельно | `TD-PROV-01`, `TD-PROV-02`, `TD-PROV-03`, `TD-MATCH-01`, `TD-STATS-01` |
| Calibration и identity | Семантическая PnLCalib-калибровка, temporal hypothesis graph, canonical people, correction graph, ReID и jersey-OCR worker contracts реализованы | `TD-QA-01`–`TD-QA-04`, `TD-IDENT-01`, `TD-CAL-02` |
| Ручное редактирование | Реализованы frame-person annotations, confirm/ignore/merge/split/bind, manual ball timeline, selected-object path и player-action timeline | Manual player trajectory остаётся `TD-TRACK-01`; automatic actions — `TD-ACT-01` |

Последняя подтверждённая clean-cut проверка: **809 API-тестов**, **284 web-теста**,
worker suites — ball 13, calibration 13, identity 19, jersey-OCR 11,
model-validation 25 passed / 2 skipped; `vue-tsc` чистый. Эти проверки доказывают
контракты и safety, но не точность CV-моделей на реальном видео.

## P0 — correctness

### TD-CAS-01 — player-action endpoint возвращает устаревшую revision

`player_action_commands.py` выбрасывает результат `scenes.put`, а
`scene_analysis_routes.py` возвращает pre-save Scene. В БД revision уже
увеличена, но клиент получает старую и следующий save может завершиться 409.

**Закрыто, когда:** add/update/delete возвращают persisted Scene; API-тест
сравнивает response revision с БД и проверяет следующий последовательный save.

### TD-TRACK-01 — ручное перемещение игрока не является durable correction

Vue изменяет `track.keyframes`, но compact Scene PUT удаляет dense keyframes, а
следующая гидрация восстанавливает их из `identityTimeline`. UI поэтому может
сообщить о сохранении правки, которой нет.

**Canonical решение:** dedicated player-trajectory command с CAS и отдельным
sparse immutable correction artifact. Anchor должен ссылаться на стабильную
canonical person/observation lineage, а не на временный renderer track ID.
Rebuild накладывает ручные anchors на derived trajectory; merge/split выполняет
детерминированный remap либо fail-closed. Dense keyframes через общий Scene PUT
не сохраняются. Пока команда не реализована, UI не должен подтверждать такой
drag как сохранённый.

### TD-CAL-01 — calibration preview скрыто изменяет Scene revision

`persist_frame_calibration_preview()` вызывает `scenes.put(scene)`, выбрасывает
persisted result и возвращает только draft. Даже Cancel оставляет клиент со
старой revision.

**Закрыто, когда:** preview либо полностью ephemeral до Apply, либо API явно
возвращает новую Scene revision; последовательность Preview → Cancel → Save
покрыта регрессией без 409.

## P1 — reliability, данные и доказуемая точность

### Транзакции и runtime

| ID | Долг | Критерий закрытия |
| --- | --- | --- |
| **TD-SEG-01** | Materialization segment Scene публикует child Scene, ownership, Segment и parent Scene несколькими транзакциями; возможен частичный граф | Одна application transaction и failure-injection тест, после которого не остаётся ни одного частичного ресурса |
| **TD-PG-01** | `SceneRepository.put()` имеет select-then-insert race; CI проверяет PostgreSQL DDL, но не реальные queue/claim/lease/CAS гонки | Narrow duplicate-key → 409 или atomic insert/on-conflict; PostgreSQL lane для Scene CAS, queue/lease/recovery/cancellation и concurrency tests |
| **TD-VALID-01** | Scene title не ограничен API-схемой при `VARCHAR(240)`; общий Scene JSON не имеет явного бюджета | `max_length=240`, согласованная UI-валидация, bounded payload policy и тесты 422 вместо PostgreSQL 500 |
| **TD-MEDIA-01** | `ffmpeg`/`ffprobe` запускаются без operation-specific timeout; зависший child занимает pipeline/reconstruction slot | Короткий probe budget, decode/transcode budgets по duration/realtime factor, стадийная ошибка, гарантированный terminate/reap и retry после cancel |
| **TD-CI-01** | Poison-head fix покрыт SQLite/unit путём, но отсутствует monitor → child → invalid head → next job интеграция на PostgreSQL | Детерминированный lightweight clip/worker smoke в PR; real models остаются nightly |
| **TD-HEALTH-01** | Container health смешивает дешёвый liveness с несколькими model-readiness запросами | Отдельный быстрый liveness и cached/parallel readiness с per-dependency status |
| **TD-SEC-01** | CORS запрещает чтение ответа, но malicious origin всё ещё может отправить простой mutation request на localhost | Same-origin/custom-header или локальный bearer для mutation routes; TLS и multi-user auth остаются deployment gate |

### Artifact и project lifecycle

| ID | Долг | Критерий закрытия |
| --- | --- | --- |
| **TD-STORAGE-01** | `FilesystemArtifactStore` умеет только `put/get`; superseded artifacts и неподтверждённые video generations накапливаются | Reachability mark-and-sweep с dry-run и grace period; защита current/in-flight refs; per-project quota; DB/media/artifact size metrics; тест, что PostgreSQL растёт как metadata, а не как число кадров |
| **TD-MULTI-01** | Нормализованная мультипроектность не имеет сквозного browser regression suite | E2E с двумя проектами/матчами/assets/segments: никакой auto-selection, cross-project 404, независимые jobs/annotations/artifacts и корректное переключение маршрутов |
| **TD-MATCH-01** | Manual-import provenance строится, но выбрасывается; источник, reference, capturedAt и notes теряются | Provenance хранится рядом с external references/snapshot и возвращается в integration diagnostics; импорт и rollback покрыты тестом |

### Quality gate — инфраструктура есть, реальной приёмки ещё нет

| ID | Долг | Критерий закрытия |
| --- | --- | --- |
| **TD-QA-01** | Spain–Belgium manifest остаётся `draft` без gold labels | Reviewed/frozen набор 100–300 разнообразных кадров: обе стороны поля, partial pitch, pan/zoom/cuts, far-side people, crossing, goalkeeper/referee, blur/occlusion и hard-negative ball; второй reviewer и video SHA-256 |
| **TD-QA-02** | Evaluator принимает predictions JSON, но accepted reconstruction run нельзя штатно экспортировать в этот контракт | Versioned run/artifact → `predictions-v1` exporter с pipeline/model/checkpoint provenance и immutable output |
| **TD-QA-03** | Нет принятого baseline и regression thresholds | Baseline report по frozen set; пороги coverage/precision/recall/error/IDF1 и reviewable delta policy в CI; prediction/report artifacts сохраняются для диагностики |

Синтетические evaluator-тесты, readiness workers и три удачных calibration кадра
подтверждают интеграцию, но не закрывают `TD-QA-01`–`TD-QA-03`.

### Provider correctness, затем статистика

| ID | Долг | Критерий закрытия |
| --- | --- | --- |
| **TD-PROV-01** | Public DTO смешивает lifecycle sync и полноту данных; runtime EventBundle invariants не централизованы | Раздельные `syncing/succeeded/failed` и `ready/partial/unavailable`; core graph fail-closed validator; invalid optional section отбрасывается и получает явный retryable/failed status |
| **TD-PROV-02** | Retryable section failure может выглядеть как пустой успешный roster; `addedTime` теряется; synthetic IDs могут коллидировать; один матч дублируется между providers/import | Per-section status, added time, provider+event namespace и provider-neutral match identity с миграцией project bindings |
| **TD-STATS-01** | Canonical team/player match statistics и вызовы API-Football statistics endpoints отсутствуют | Versioned team/player stats + coverage descriptor; embedded fixture blocks читаются первыми; отдельные provider calls учитывают quota/cache |
| **TD-PART-01** | Production resolver не строит participation intervals из lineup/substitutions/events | Participation evidence строится из canonical match clock и применяется только как eligibility/negative constraint или prior, но никогда как самостоятельное доказательство личности |

`TD-STATS-01` выполняется только после `TD-PROV-01/02`: статистика поверх
частичного snapshot не должна выглядеть достоверной.

## P2 — измерение, эксплуатация и поддерживаемость

| ID | Долг | Критерий закрытия |
| --- | --- | --- |
| **TD-OBS-01** | AnalysisRun хранит текущую фазу и общий elapsed, но не историю фаз; stale/invalid worker outcomes плохо различимы оператором | Durable phase start/end/elapsed, queue/model-load/cache counters, structured terminal outcome, bounded run history и request/run IDs в логах |
| **TD-PERF-01** | Нет воспроизводимого latency/RAM/VRAM baseline на заданном hardware/clip | Benchmark harness фиксирует hardware, image/model versions, cold/warm cache, per-phase latency, peak RAM/VRAM и допустимый realtime factor |
| **TD-PERF-02** | Person inference остаётся per-frame; calibration cache process-local; возможны повторные cold-cache вычисления | После `TD-PERF-01`: batch person/multi-frame ball, disk cache по frame/model SHA; overlap фаз только при доказанном выигрыше без resource contention |
| **TD-FE-01** | `ThreeViewport` обновляет объекты и в rAF, и watcher'ом `currentTime`; interpolation/path делает линейные сканы и allocations | Один clock-driven update path; бинарный поиск/cursor после профилирования; stable artifact signatures вместо deep JSON watchers |
| **TD-PROV-03** | Provider transports создают client на запрос, имеют unbounded process cache и неполный retry/Redis lifecycle; manual schema version отсутствует | Shared низкоуровневые HTTP/cache primitives без объединения provider mapping; bounded cache, retry/backoff, client close/timeouts, transport contract suite и versioned manual schema |
| **TD-OPS-01** | Нет проверенного project export/import и регулярного backup/restore | Версионированный project bundle с annotations/manifests и опциональным media; backup policy, restore drill и документированный RPO/RTO — независимо от внешнего доступа |
| **TD-STORAGE-02** | Локальный shared volume не готов к multi-host data plane | Backend-neutral object store, staging для FFmpeg, retention/privacy policy и восстановление manifest → object; только после `TD-STORAGE-01` |
| **TD-INFRA-01** | Model weights provision неоднороден; identity-worker ещё использует MD5; lock strategy и non-root images не унифицированы | Pinned URL + SHA-256 для всех weights; vendored licensed/compatible runtime code; per-image lockfiles либо доказанно совместимый shared lock; Docker `USER` и quickstart |
| **TD-CODE-01** | Остались прямые `BEGIN IMMEDIATE`, status-from-message, полные route prefixes, side-effect ORM registration и `kind=demo` по title | Миграция по capability с транзакционными тестами; typed error mapping; явная ORM/router registration; demo только через fixture/factory |
| **TD-TEST-01** | Нет coverage floor, gradual mypy/ruff gate и поведенческих тестов части video/calibration/3D UI | Зафиксированный baseline coverage без снижения; ruff + постепенно расширяемый mypy; component tests на позиции/видимость/counters, а не string assertions |
| **TD-QA-04** | Evaluator не измеряет role/team/occlusion strata, semantic-line/mirror/temporal calibration, pitch-space ball/player error, ball continuity и official HOTA/GS-HOTA | Метрики добавляются только вместе с размеченными данными и frozen contract; official SoccerNet evaluator интегрируется без самодельной аппроксимации |
| **TD-DOC-01** | `PIPELINE_EDGE_CASES.md` смешивает реализованные и будущие меры; часть docs не связана с README | Каждая строка имеет status и стабильный `TD-*`; active items ссылаются на этот реестр, закрытые — на regression test/doc |

## P3 / R&D и deployment gate

| ID | Следующий этап | Условие начала/закрытия |
| --- | --- | --- |
| **TD-IDENT-01** | Измерить short-horizon tracker + PRTReID/OCR против StrongSORT/TrackLab; повысить useful real-player identification coverage | Только на `TD-QA-01` gold set; thresholds и abstention сохраняют one-to-one/unknown invariants |
| **TD-CAL-02** | Extreme zoom, replay, blur, partial pitch без direct anchors, rolling shutter и нестандартные размеры поля | Отдельные strata в benchmark; manual anchors остаются наблюдаемым продуктовым fallback, а не legacy |
| **TD-BALL-01** | Automatic ball accuracy, candidate confirm/reject evidence и реальная airborne height | Hard-negative/hidden/blur labels, image+pitch metrics и отдельная 2.5D/pose модель; manual trajectory остаётся authoritative correction |
| **TD-QA-05** | Автоматизированный внешний бенчмарк на SoccerNet GSR (экспорт GSR predictions, клиповый оркестратор, GS-HOTA официальным evaluator'ом из `.references/sn-gamestate`) | **Решено 2026-07-19 (владелец): в обозримом будущем не выполняется.** Проверка качества через внешний сервис/датасет не ведётся; приёмка изменений качества — визуальная, на реальных сегментах проекта: по одному изменению за явным флагом/настройкой, сравнение до/после в редакторе и офлайн-артефактами (2D-радар, JSONL-журнал прогона), с сохранённой возможностью отката на прежний дефолт. Правило 7 реестра (accuracy только по frozen labelled set) на этот период заменено визуальной приёмкой; материалы `.references/sn-gamestate` сохраняются на случай пересмотра решения |
| **TD-ACT-01** | Automatic action suggestions и review workflow | `source=automatic,status=suggested`, Accept/Reject/convert-to-manual, отдельный fingerprint/selective invalidation и labelled action set; см. [`PLAYER_ACTIONS.md`](PLAYER_ACTIONS.md) |
| **TD-ANIM-01** | Синхронизация подтверждённых действий с лицензированными UCS rigs/clips | Retargeting, root motion, ball-contact IK, blending и deterministic scrubbing не влияют на identity/action truth; см. [`PLAYER_ACTIONS.md`](PLAYER_ACTIONS.md) |
| **TD-DIST-01** | Multi-host backpressure, fair claiming, GPU scheduling и SLO | Capacity benchmark, `SKIP LOCKED`/retry policy, per-model resource limits и owner-fencing сохраняются |
| **TD-EXT-01** | Внешний или multi-user deployment | Auth/tenant authorization, TLS, external CORS, quotas, privacy и лицензионный аудит обязательны до первого внешнего bind |

## Порядок выполнения

1. Закрыть `TD-CAS-01`, `TD-TRACK-01`, `TD-CAL-01` и временно не показывать
   пользователю ложное «saved» для неподдерживаемого player drag.
2. Параллельно закрыть транзакционные `TD-SEG-01`, `TD-PG-01` и жизненный цикл
   данных `TD-STORAGE-01`.
3. Создать измеримую точку правды: `TD-QA-01` → `TD-QA-02` → `TD-QA-03`.
4. Выполнить provider Slice A (`TD-PROV-01/02`, `TD-MATCH-01`), затем
   `TD-STATS-01` и `TD-PART-01`.
5. Добавить `TD-OBS-01`/`TD-PERF-01`; только после замера оптимизировать
   inference, cache и frontend hot paths.
6. R&D и внешний deployment не вытесняют correctness, gold quality и storage
   lifecycle текущего локального MVP.

## Закрытый baseline и регрессионные инварианты

Следующие направления закрыты как **архитектурная реализация**, но не являются
заявлением о точности реальных моделей:

- god-файлы reconstruction/video/ball/identity/OCR/ReID/player-actions удалены
  либо распилены по независимым capabilities; import graph и composition roots
  защищены architecture-тестами;
- глобальные Scene/video/match mutation routes и implicit project creation
  удалены; browser работает через project-scoped router;
- Match snapshot, references и project pointer публикуются атомарно;
- durable jobs, leases, recovery, fencing и last-good publication заменили
  API background work и payload-proportional recovery polling;
- dense reconstruction data вынесены из Scene/PostgreSQL: live cutover уменьшил
  Shot 2 `4,468,127 → 332,876 B`, а финальные восемь Scene — до 81,565 B
  суммарно; подробности в [`PERFORMANCE.md`](PERFORMANCE.md);
- семантическая PnLCalib-калибровка, temporal propagation/cut barriers и
  side/orientation separation реализованы; подробности в
  [`CALIBRATION.md`](CALIBRATION.md);
- canonical people, immutable observations, confirm/exclude/merge/split/bind,
  correction lineage, ReID/OCR contracts/caches и explicit abstention
  реализованы; подробности в [`IDENTITY_RESOLUTION.md`](IDENTITY_RESOLUTION.md);
- manual ball timeline, selected-object path и manual player-action timeline
  реализованы; подробности в [`BALL_TRACKING.md`](BALL_TRACKING.md),
  [`PATH_TRACKING.md`](PATH_TRACKING.md) и [`PLAYER_ACTIONS.md`](PLAYER_ACTIONS.md);
- versioned benchmark schemas/evaluators реализованы, но остаются scaffolding,
  пока `TD-QA-01`–`TD-QA-03` не создадут frozen product evidence.

Неотменяемые правила текущей архитектуры:

1. PostgreSQL — compact control plane; frame series и model evidence — data
   plane artifacts.
2. Job discovery/heartbeat никогда не читает `Scene.payload`.
3. Реконструкция запускается только явно для выбранного segment/composition;
   открытие проекта, refresh Match и навигация не ставят jobs.
4. Новый run не уничтожает last-good до fenced atomic publication.
5. Missing/partial/corrupt data отображаются явно; silent fallback запрещён.
6. Ручная правка привязывается к стабильной domain identity и переживает rebuild
   либо отклоняется fail-closed.
7. Accuracy принимается только по независимо размеченному frozen set, а не по
   runtime confidence, smooth trajectory или readiness модели.
8. Первый canonical replacement удаляет legacy path в том же cutover; dual
   write и embedded dense fallback не возвращаются.
9. Обратная совместимость со старыми локальными данными не поддерживается
   (решение владельца, 2026-07-19): прогоны, кэши, артефакты и БД
   пересобираемы; контрактные изменения делаются bump'ом версии, данные-
   миграции и конверсии старых форматов не пишутся. Инварианты внутри
   одного прогона (fencing, атомарность, digest-нормализация) остаются.

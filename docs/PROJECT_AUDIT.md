# Аудит проекта — 17–18 июля 2026

> **Обновление 18 июля 2026 — пере-аудит после волны модификаций.**
> Все 149 находок (60 основных + SRV + PROV) перепроверены против нового
> дерева 18 агентами; 8 агентов свежим взглядом отаудировали новые
> подсистемы, их high/critical находки прошли адверсариальную верификацию.
> Итог см. в разделе 11; статусы в разделах 1–10 отражают состояние на
> 17 июля и скорректированы пометками раздела 11.
>
> **Обновление 18 июля 2026 (вечер) — аудит большой реструктуризации.**
> Актуальные статусы всех накопленных находок и оценка новой модульной
> архитектуры — в разделе 12; он имеет приоритет над статусами разделов
> 1–11.

> **Статус документа.** Это исторический снимок состояния до project-centric
> этапа того же дня, а не текущий backlog. Номера строк и сводка `53/5/2` ниже
> относятся к исходному снимку. Актуальный долг ведётся в
> [`TECHNICAL_DEBT.md`](TECHNICAL_DEBT.md).

## Дельта после аудита

| Исходная находка | Текущий статус | Реализовано / что осталось |
| --- | --- | --- |
| CI-1, CI-2 | Закрыто для текущего scope | GitHub Actions запускает API, web, worker-contract suites и `docker compose config` |
| BE-2 / PERF-1 | Закрыто для normalized reconstruction | Phase progress вынесен в compact `AnalysisRun`; legacy-only fixtures сохраняют compatibility fallback |
| BE-8 | Закрыто | API только атомарно сохраняет compact reconstruction job; отдельный `reconstruction-runner` является единственным execution path локально и в Compose |
| SRV-1 | Закрыто | Alembic принимает legacy/create-all DB, добавляет project/identity tables; API startup выполняет upgrade, а Compose gates API/runner одноразовым `migrate`; CI проверяет fresh PostgreSQL migration |
| SRV-5/6 | В основном закрыто | Нормализованные projects и уникальные ownership links, project-scoped public listings, idempotent backfill. Глобальные legacy routes ещё существуют |
| SRV-7 | Backend foundation закрыт | Один project владеет несколькими videos; composition принимает project segments из разных assets. Полный composition UX и alignment benchmark остаются |
| SRV-9 | Частично | Video-processing публикует compact `AnalysisRun` и поддерживает cancellation, но всё ещё запускается через API `BackgroundTasks` без crash recovery |
| SRV-10 | Частично | Multi-pass имеет compact progress/cancellation и cross-asset input, но durable lease/recovery runner ещё не подключён |
| SRV-13 | Закрыто | Nginx принимает 260 MB, отключает request buffering и имеет upload/analysis timeouts |
| SRV-14 | Частично | Host ports ограничены loopback; TLS, auth и tenant authorization не реализованы |
| SRV-15 | Частично | `.env.example` не содержит API-Football secret; историческую ротацию ключа должен подтвердить владелец |
| SRV-16 | Частично | Сервисы имеют restart/health policies; backup/restore и off-host media policy отсутствуют |
| Provider split-brain | Закрыт для public project API | Canonical Match snapshot и internal IDs отделены от integration diagnostics; legacy catalog/scene routes остаются compatibility API |
| Отсутствие quality gate | Foundation закрыт | Versioned schemas/evaluator готовы; реальный manifest остаётся draft без gold labels, поэтому accuracy claim отсутствует |

Не закрыты исходные BE-1/BE-3 (`reconstruction.py`), FE-1/FE-2 (`App.vue`),
Postgres contention и запуск concurrent CAS/lease suites на Postgres, durable
video/multi-pass queue, object storage, auth, TLS, backups и production
observability. Детали и приоритеты перечислены в
актуальном technical-debt документе.

Полный аудит кодовой базы Replay Studio. Все находки получены из чтения
исходного кода рабочего дерева (commit `e0f2009` + незакоммиченные правки),
а не из документации; каждая ссылка `file:line` перепроверена против текущего
состояния файлов повторным проходом верификации. Документация использовалась
только для фиксации расхождений «док ↔ код» и помечена явно.

Методология: два прохода. Первый — 8 независимых аналитиков по направлениям
(архитектура бэкенда, фронтенд, сервисы/инфраструктура, тесты/CI, гигиена
репозитория, соответствие документации, безопасность, производительность),
каждая находка уровня high/critical прошла адверсариальную проверку отдельным
агентом с задачей её опровергнуть. Второй — повторная верификация всех 60
находок против текущего дерева после `git init`, обновления `.gitignore` и
появления пакета `apps/api/app/providers/`. Слой провайдеров матчевых данных
вынесен в отдельный раздел, так как он переделывается прямо сейчас.

## Сводка

| Статус | Кол-во |
|---|---|
| Актуальны | 53 |
| Устранены с момента первого прохода | 5 |
| Устранены частично | 2 |
| Опровергнуты при перепроверке | 0 |

Устранено: инициализация git + push на приватный remote; три пункта по
`.gitignore` (веса моделей, битые gitlink из `.references/`, правило `data/`);
утечка ключа TheSportsDB через детали 502-ошибок — закрыта новой иерархией
санитизированных ошибок в `providers/base.py`.

Частично: от «нет VCS и CI» осталась половина про CI (`.github/` отсутствует);
игнорирование `.references/` убрало риск gitlink, но усугубило runtime-связку
ball-worker (см. INFRA-3).

## Сильные стороны (не трогать при рефакторинге)

- Конкурентная модель бэкенда: ревизионный full-document CAS
  (`store.py:267-325` в терминах первого прохода, ныне `store.py` после
  сдвига строк), fenced-leases c heartbeat вне ревизионируемого документа,
  атомарный перехват устаревших run'ов, crash-recovery монитор
  (`apps/api/app/reconstruction_recovery.py`). Любой рефакторинг обязан
  сохранить эти инварианты.
- 623 собираемых API-теста (быстрые, изолированные: ffmpeg/сеть замоканы),
  181 vitest, регрессии на CAS/lease/split-merge конкуренцию.
- Низкоуровневая гигиена безопасности: subprocess только списками аргументов,
  чистый ORM без сырых f-string SQL, защита от path traversal в выдаче медиа,
  worker-URL только из конфигурации оператора (SSRF нет).
- Изоляция ML-воркеров: пиновые коммиты/чексуммы, health-контракты,
  политика «никаких тихих подмен эмбеддингов».
- Честная документация: выборочные проверки «док ↔ код» первого прохода
  расхождений почти не нашли (совпадают счётчики тестов, SHA чекпоинтов,
  пины коммитов).

---

## 1. Архитектура бэкенда (`apps/api`)

### BE-1. `reconstruction.py` — god-модуль на 14 205 строк — high, актуально

203 top-level `def`/`class`, ≥10 несвязанных доменов в одном файле.
Швы подтверждены по текущим строкам: очередь/прогресс job'ов
(`ReconstructionProgress`:639, `queue_reconstruction`:1031), калибровка
(1155–3862), camera motion (3914–4156), плотная проекция мяча (4157–4600),
jersey OCR (4793–5492), трекинг (5493–5972), кластеризация команд
(5973–6028), canonical identity + roster (6343–7672), `reconstruct_scene`
(8111–9434), интерактивная правка идентичностей
(`upsert_frame_person_annotation`:11888, `delete_frame_person_annotation`:12184),
roster binding (`set_canonical_roster_binding`:12990,
`clear_canonical_roster_binding`:13337), `analyze_scene_frame`:13550,
lease heartbeat/runner (14028–14205). `main.py:38-54` импортирует из него
13 функций и 3 символа — request-path и батч-пайплайн живут в одном
пространстве имён.

**Рекомендация.** Разнести в пакет `reconstruction/` по перечисленным швам.
Первый срез с максимальной отдачей: интерактивная правка + roster binding +
`analyze_scene_frame` (~4 600 строк, строки 9416–14008) в `corrections.py`
и `roster_binding.py` — это request-path код, разделяющий с пайплайном лишь
мелкие хелперы `_identity_annotations`/`_annotation_*`. Затем `runner.py`,
`calibration.py`, `ball.py`, `tracking.py`, `identity.py`, оставив
`pipeline.py` с `reconstruct_scene`.

### BE-2. Каждый тик прогресса переписывает весь scene-JSON под эксклюзивной блокировкой — high, актуально

`ReconstructionProgress.update` → `_persist_reconstruction_state`
(`reconstruction.py:704-709`, `980-999`) → `put_if_reconstruction_run`
(`store.py:457`), который делает deepcopy всего payload (`store.py:225`)
внутри `BEGIN IMMEDIATE` (`store.py:210-220`). Вызов — на каждый
проанализированный кадр (`reconstruction.py:8468`) и на каждый батч
калибровки/identity (8231, 8505–8517). Дублируется с находкой PERF-1 —
это одна и та же проблема, видимая с двух сторон.

**Рекомендация.** Вынести прогресс из ревизионируемого документа — точно так
же, как уже сделано для leases (`ReconstructionLeaseRow` существует именно
для этого): маленькая строка прогресса на scene_id без инкремента ревизии.
Либо минимальный вариант: троттлинг записи до 1 раза в 1–2 с при сохранении
покадрового in-memory listener. Убирает ~99 % полных перезаписей.

### BE-3. `reconstruct_scene` — оркестратор на ~1 324 строки — high, актуально

`reconstruction.py:8111-9434`: один `try` на 8174 с обработчиками на
9387–9391 (охват 1 200+ строк), вложенные замыкания `calibration_progress`
(8214) и `identity_progress` (8516), сквозные мутируемые локалы
(`frame_calibrations`, `calibration_warnings`, `identity_warnings`,
`identity_worker_diagnostics`, 8207–8213).

**Рекомендация.** Датакласс `PipelineContext` со сквозным состоянием и
функция уровня модуля на каждую фазу (`prepare_frames`, `calibrate`,
`detect_people`, `embed_identities`, `run_jersey_ocr`, `track`,
`resolve_identities`, `detect_ball`, `publish`). Границы фаз уже явные —
список `RECONSTRUCTION_PHASES` (`reconstruction.py:571`) и точки
`progress.update`.

### BE-4. Recovery-монитор сканирует всю таблицу scenes под write-lock каждые 5 секунд — medium, актуально

`reconstruction_recovery.py:66-96`, интервал `config.py:33`;
`fail_unrecoverable_reconstruction_runs` (`store.py:545`) выполняет
`select(SceneRow).with_for_update()).all()` (`store.py:570`) внутри
`BEGIN IMMEDIATE`; `list_recoverable_reconstruction_runs` (`store.py:624`)
повторно грузит все строки (647–650) и фильтрует статус в Python;
`find_segment_scene` (`store.py:866-873`) — полный скан. Индексируемых
колонок статуса/run_id у `SceneRow` нет.

**Рекомендация.** Продвинуть нужные монитору поля в индексируемые колонки
`SceneRow` (`reconstruction_status`, `run_id`, `input_fingerprint`,
`parent_scene_id`, `selected_segment_id`), поддерживать их в `_next_payload`
при записи; кандидатов фильтровать в SQL и блокировать только совпавшие
строки. Пустой тик становится index lookup.

### BE-5. Доменная модель — нетипизированный dict; 31k строк ходят по `.get()`-цепочкам — medium, актуально

`schemas.py:125` — `payload: dict[str, Any]`; 80 вхождений
`.get("videoAsset"` по `apps/api/app` (выросло с 73); 56 вхождений
`scene.get("payload", {})` в `reconstruction.py`; ни mypy, ни ruff не
сконфигурированы (`apps/api/pyproject.toml`).

**Рекомендация.** Не пытаться пайдантизировать payload целиком. Сначала
маленький модуль типизированных аксессоров (`scene_doc.py`:
`video_asset(scene)`, `reconstruction_of(scene)`, `tracks_of(scene)`) и
механическая замена цепочек; затем TypedDict для двух самых горячих
под-объектов (`reconstruction`, `videoAsset`) и mypy с узким allowlist.

### BE-6. Параллельные дубли семантики identity-correction/merge для двух представлений трека — medium, актуально

`_apply_track_identity_corrections` (`reconstruction.py:11541`) против
`_apply_scene_track_identity_corrections` (11756), оба через
`_terminal_identity_target` (9650); `_merge_raw_track_states` (10622) против
`_merge_scene_track_documents` (11689). Инфраструктурные дубли:
`_canonical_json`/`_json_value` в `person_detection_cache.py:34/44` и
`ball_detection_cache.py:34/44`; валидаторы в `identity_worker.py:84-137` и
`jersey_ocr_worker.py:125-160`; одинаковый readiness-probe в четырёх
клиентах (`calibration_worker.py:35`, `identity_worker.py:39`,
`jersey_ocr_worker.py:75`, `ball_worker.py:20`).

**Рекомендация.** Для corrections: приводить scene-треки к `TrackState`
(или общему протоколу) один раз и гонять единственную реализацию — дрейф
семантики merge/exclude между пайплайном и публикацией напрямую портит
опубликованные идентичности. Для инфраструктуры: выделить
`detection_cache_common.py` и общий `worker_client.py`.

### BE-7. `main.py` (1 745 строк): 34 inline-роута + ~250 строк доменной логики match-import — medium, актуально

`_manual_match_bundle` — `main.py:241-487`, `_persist_match_binding_bundle` —
`main.py:1475-1560`; подключены лишь два роутера (`main.py:113-114`), при
этом request-path сервисы импортируются из `reconstruction.py`
(`main.py:38-54`). Новый пакет `providers/` логику из `main.py` не забрал.
С первого прохода файл вырос на ~350 строк.

**Рекомендация.** Повторить уже существующий паттерн
`identity_review_routes.py`: `videos_routes.py`, `scenes_routes.py`,
`calibration_routes.py`, `catalog_routes.py`, `match_binding_routes.py`;
`_manual_match_bundle`/`_persist_match_binding_bundle` — в сервисный модуль
`match_binding.py` рядом с `providers/`. Низкий риск, создаёт модули-адресаты
для распила BE-1.

### BE-8. CPU-bound ML-реконструкция исполняется в процессе API-сервера — medium, актуально

`POST /api/scenes/{id}/reconstruct` планирует job через FastAPI
`BackgroundTasks` (`main.py:647-686`, `add_task`:680); `_load_model`
импортирует ultralytics in-process (`reconstruction.py:753-762`);
recovery-монитор стартует в lifespan API (`main.py:95`) и держит до 2
daemon-потоков (`config.py:34`).

**Рекомендация.** Lease/claim-механика в `store.py` уже поддерживает
многопроцессное владение, поэтому фикс дешёвый: entrypoint
`python -m app.reconstruction_runner` с `ReconstructionRecoveryMonitor` как
отдельный процесс (Docker-сервис рядом с `services/*`), а HTTP-endpoint
оставить только `queue_reconstruction` — монитор заберёт job в пределах
одного 5-секундного полла.

---

## 2. Фронтенд (`apps/web`)

### FE-1. `App.vue` — god-компонент на 4 614 строк без слоя состояния — high, актуально

Один `<script setup>`: ~73 refs, ~65 computeds, 8 watch, ~140 функций,
33 вызова `api.*`. `defineStore`/`provide`/`inject` в `src/` отсутствуют,
pinia не установлена, каталогов `composables/`/`stores/` нет. Переплетённые
контуры: playback `tick()` (1206–1224), state-machine сохранения
(1295–1317), поллинг реконструкции (2352–2402), race-guard identity review
(1242–1260); шаблон — 3303–4614.

**Рекомендация.** Извлекать по-контурные composables в порядке отдачи:
(1) `useReconstructionPolling` — единственный владелец таймера;
(2) `usePlaybackClock` — currentTime/playing/seek + синхронизация видео;
(3) `useFrameAnnotation`; (4) `usePitchCalibration`; (5) `useIdentityReview`;
(6) `useVideoReviewViewport`; (7) `useMatchCatalog`; (8) `useSceneWorkspace`.
Шаблон разнести на панельные компоненты, питаемые composables.

### FE-2. Воспроизведение перерендеривает весь App на частоте кадров — high, актуально

`currentTime` — top-level ref (`App.vue:120`), мутируется каждый rAF в
`tick()` (1206–1224). Привязки по всему шаблону: playhead (4048), range
v-model (4092), таймкод (4157/4212), два прямых вызова
`interpolateKeyframes` в шаблоне (4214–4215); цепные computeds
`activePlayerActionPlayback` (285) и `videoPathProjectionContext` (562–564)
зависят от currentTime → все динамические привязки App пересчитываются
~60 раз/с.

**Рекомендация.** Сильнейший аргумент за FE-1: `currentTime` в
`usePlaybackClock`, передавать только листовым компонентам (транспорт,
playhead, viewport). Два шаблонных `interpolateKeyframes` — в один computed;
playhead двигать CSS-transform из маленького выделенного компонента.

### FE-3. SceneDocument глубоко реактивен; render-loop ThreeViewport читает его через Vue-прокси каждый кадр — medium, актуально

`App.vue:118` — обычный deep `ref`; `shallowRef`/`markRaw`/`toRaw` в
`src/` — ноль вхождений. `ThreeViewport.vue:911-918` —
`renderer.setAnimationLoop` → `updateObjects()` (710–733) итерирует
`props.scene.payload.tracks` с двумя `interpolateKeyframes` на трек;
`lib/interpolate.ts:8` — линейный `findIndex` на каждый вызов.

**Рекомендация.** `shallowRef<SceneDocument>` (все мутации и так заменяют
документ целиком) либо `markRaw` payload + явный счётчик ревизии для
watcher'ов. В ThreeViewport — снапшот `toRaw(...)` при смене сцены и
кэшируемый курсор на трек: интерполяция O(1) при монотонном воспроизведении.

### FE-4. Поллинг реконструкции: один общий id таймера на 11 точек старта; каждые 700 мс `scene.value` заменяется целиком при активных инпутах — medium, актуально

`App.vue:205` — единственный `reconstructionTimer`; `pollReconstruction`
(2352) самоперепланируется (2366) и заменяет `scene.value` (2358); 11 точек
старта (1999, 2033, 2086, 2154, 2216, 2256, 2343, 2414, 2429, 2444, 3138);
таймер предварительно чистят только 3 из них. Guard (2356) проверяет лишь
смену сцены, не эпоху. `v-model="selectedTrack.label"` и
`v-model.number="selectedTrack.number"` (4207–4208) не блокируются на время
run'а.

**Рекомендация.** `useReconstructionPolling` c API `start(sceneId)`/`stop()`
и счётчиком эпох, проверяемым после каждого `await` (паттерн requestId уже
есть в `loadIdentityReview`, 1242–1260). Инпуты прямой мутации блокировать
(или буферизовать) при `reconstructionRunning`.

### FE-5. Watcher'ы ThreeViewport сериализуют массивы keyframes и frameAnalysis через `JSON.stringify` на каждый reactive flush — medium, актуально

`ThreeViewport.vue:925` — `watch(() => JSON.stringify(...ball.keyframes))`;
926–936 — то же для выбранного трека; 937 — для `frameAnalysis`. Поллинг
подменяет документ каждые 700 мс, заставляя всё это пересериализовываться.

**Рекомендация.** Смотреть дешёвые сигналы идентичности/версии: после FE-3 —
`watch(() => props.scene)` по ссылке или кортежи
`(keyframes.length, first.t, last.t)`. `JSON.stringify` из watcher-геттеров
убрать полностью.

### FE-6. Самая рискованная оркестрация не тестируема и не тестируется — medium, актуально

`VideoIngestDrawer.vue` (валидация файла ~31–45, самоперепланирующийся
650 мс `pollAsset` ~52–70) и `CalibrationQaPanel.vue` (630 строк,
14-проповый контракт, `App.vue:4437-4455`) — единственные компоненты без
тестов (у остальных 10 тесты есть). Race-guard логика живёт в `App.vue`.
`ThreeViewport.test.ts` рендерит только SSR-строку.

**Рекомендация.** Извлечение composables из FE-1 — это и есть стратегия
тестирования: `useReconstructionPolling`/`useIdentityReview` тестируются
юнитом с замоканным api. Краткосрочно: тесты VideoIngestDrawer
(отклонение >250 МБ и не-видео, переходы поллинга через
`vi.useFakeTimers`), GateView-деривацию CalibrationQaPanel вынести в `lib/`
по образцу `calibrationDiagnostics.ts`.

### FE-7. Тяжёлый prop-drilling — low, актуально

ThreeViewport: 16 props + 4 события (3975–3997); CalibrationQaPanel: 14 + 3
(4437–4455); PlayerActionTimeline: 8 + 5; ManualBallTimeline: 6 + 5;
IdentityReviewPanel: `:disabled` собирается инлайн из шести булевых refs
(4188).

**Рекомендация.** После FE-1 передавать объекты composables (или
provide/inject readonly-store) для сквозной тройки scene/currentTime/selection;
агрегатные флаги типа `mutationsLocked` считать один раз в workspace-слое.

### FE-8. Гигиена зависимостей — low, актуально

`three` 0.185.0 против `@types/three` 0.184.1; `vite` и
`@vitejs/plugin-vue` в `dependencies`. `lib/api.ts` непоследователен в
кодировании URL: identity/roster/catalog используют `encodeURIComponent`
(51–78, 125–140, 186–223), а `getScene` (37), `saveScene` (39),
`reconstructScene` (110), `analyzeFrame` (144) и pitch-calibration
(229–252) интерполируют id сырыми.

**Рекомендация.** Поднять `@types/three` до 0.185.x, tooling — в
devDependencies, добавить единый хелпер `scenePath(id, ...segments)` с
обязательным кодированием сегментов.

---

## 3. Сервисы и инфраструктура

### INFRA-1. Web-образ копирует хостовые macOS `node_modules`; `.dockerignore` нет нигде — high, актуально

`apps/web/Dockerfile:3-5`: `COPY package.json` → `npm install` → `COPY . .`
поверх свежих Linux-модулей ложатся darwin-arm64 бинарники (esbuild,
rollup) с хоста. `.dockerignore` в проекте — ноль файлов.
`apps/web/package-lock.json` не существует (lockfile только в корне
workspace и в контекст не попадает) → `npm install` плавающий. Контекст
apps/api тащит `replay-studio.db` и тесты.

**Рекомендация.** `.dockerignore` для apps/web (node_modules, dist) и
apps/api (tests, `*.db`, `*.egg-info`, `__pycache__`); `npm ci` + lockfile в
контексте (сгенерировать `apps/web/package-lock.json` или собирать из корня
workspace).

### INFRA-2. Ни у одного из 9 compose-сервисов нет restart-политики — high, актуально

`grep -c restart docker-compose.yml` → 0. Воркеры гоняют CPU-инференс
больших моделей одним uvicorn-worker'ом; OOM-kill оставляет сервис лежать
до ручного вмешательства, и из-за деградационной политики
(`BALL_DETECTION_FAILURE_POLICY: fallback`, compose:176; identity/jersey —
`service_started`) простой молчалив: реконструкции тихо теряют
identity/OCR/ball evidence.

**Рекомендация.** `restart: unless-stopped` всем долгоживущим сервисам —
самое дешёвое улучшение доступности; в пару — вывести readiness воркеров в
web-UI, чтобы деградация была видима.

### INFRA-3. Runtime-связка ball-worker с неверсионируемым `.references/` — high, частично (стало хуже)

Gitlink-риск снят (`.gitignore:41`), но `docker-compose.yml:127` по-прежнему
бинд-маунтит `./.references/WASB-SBDT/src/models/hrnet.py` — каталог,
которого на свежем клоне гарантированно нет. Вендоренного `hrnet.py` в
`services/ball-worker/` не появилось.

**Рекомендация.** Завендорить один MIT-файл в
`services/ball-worker/vendor/hrnet.py` (с upstream LICENSE и хэшем коммита),
обновить volume; либо fetch-скрипт, клонирующий WASB-SBDT на пиновом
коммите в `.references/`.

### INFRA-4. Все контейнеры под root; purge-слой identity-worker не уменьшает образ — medium, актуально

**Рекомендация.** Non-root user в Dockerfile'ах; purge build-tools объединить
в один RUN-слой или перейти на multi-stage.

### INFRA-5. MinIO — мёртвый груз — medium, актуально

`minio/minio:latest` (без пина), персистентный volume, hardcoded
credentials `replay`/`replay-studio`, консоль опубликована на 9001 — при
нуле ссылок из кода (grep по apps/api, apps/web, services). Новый провайдер
`api_football.py` стал вторым потребителем Redis, контраст «Redis нужен —
MinIO нет» усилился.

**Рекомендация.** Удалить сервис и volume из compose (и упоминание консоли в
README) до реальной реализации object storage; при возврате — пиновый тег,
healthcheck, credentials через `.env`.

### INFRA-6. Распространение весов моделей неконсистентно между воркерами — medium, актуально

Четыре механизма одновременно: identity-worker — fetch-скрипт с чексуммами
(эталон); jersey-ocr — `cache_models.py`; calibration —
`COPY models` + `RUN test -s` без какого-либо скрипта скачивания
(`services/calibration-worker/Dockerfile:24-26`); ball-worker — бинд-маунт
весов и кода из `.references/`.

**Рекомендация.** Стандартизировать на паттерне identity-worker: fetch-скрипт
с пиновыми URL + SHA-256 за build-arg `DOWNLOAD_MODELS`. Для calibration
URL уже задокументированы в `docs/CALIBRATION.md`.

### INFRA-7. HTTP-контракт воркеров реализован ~4 раза с обеих сторон — medium, актуально

Health-роуты в каждом сервисе; идентичный readiness-probe
`httpx.get(endpoint, timeout=...)` в четырёх клиентах API (см. BE-6).

**Рекомендация.** Общий мини-пакет `services/worker-contract/`
(health-route factory, TTL-LRU, manifest parse/validate) + единый
`WorkerClient` в apps/api. Провайдерскую логику не трогать — общий только
транспорт/контракт.

### INFRA-8. Старт API жёстко завязан на health calibration-worker; у api/web нет своих healthcheck — medium, актуально

`docker-compose.yml:182-183` — `condition: service_healthy` для calibration
при `service_started` для identity/jersey (противоречивая политика); у api
нет healthcheck при существующем `/api/health` (`main.py:488`).

**Рекомендация.** Ослабить calibration до `service_started` (API уже
корректно деградирует), добавить healthcheck api на `/api/health`, web —
`depends_on: api: condition: service_healthy`.

### INFRA-9. Конфигурация размазана по трём несинхронизированным источникам — low, актуально

`pitch_keypoint_model`/`pitch_keypoint_image_size` (`config.py:79,85`) нет
ни в `.env.example`, ни в compose; `.env.example` перемешивает worker-only и
API-переменные без секций (провайдерные ключи добавлены, но структура не
исправлена); Postgres-креды hardcoded и уже закоммичены.

**Рекомендация.** env_file для тюнингов API; `.env.example` посекционно по
процессам-потребителям; `POSTGRES_PASSWORD` через `${...}`.

---

## 4. Тесты и CI

### CI-1. CI нет вообще — critical, частично (git есть, CI нет)

Git инициализирован и запушен (`origin git@github-art:artcodev/replay-studio.git`,
251 файл). Но `.github/`, Makefile, pre-commit — ничего нет: 623 API-теста +
~40 сервисных + 181 vitest запускаются только вручную.

**Рекомендация.** `.github/workflows/ci.yml` с четырьмя джобами:
(a) api: `pip install -e 'apps/api[test]' && pytest apps/api/tests`;
(b) матрица по сервисам; (c) web: `npm ci && npm run typecheck && npm test
&& npm run build`; (d) `docker compose config` lint. Суммарно < 3 минут.

### CI-2. `pytest` из корня падает на сборе; тесты calibration-worker незапускаемы ниоткуда — high, актуально

Воспроизведено: `623 tests collected, 1 error … Interrupted`.
`services/calibration-worker/tests/test_worker_cache.py:7` делает
`from app.main import …`, а `pytest.ini:3` ставит `pythonpath = apps/api` →
`app.main` резолвится в API. Остальные три воркера избегают коллизии через
`tests/conftest.py` и неколлизионные имена пакетов.

**Рекомендация.** Переименовать `services/calibration-worker/app` в
`calibration_worker_service` (конвенция соседей) + такой же conftest. Даёт
единую команду `pytest` из корня как локальный gate.

### CI-3. Video-ingest — слепая зона покрытия — high, актуально

`test_video_processing.py` — 15 строк, 1 тест на чистый хелпер при
251-строчном ffmpeg-модуле; HTTP-тестов на `POST /api/videos`,
pitch-calibration и multi-pass роуты нет (grep по тестам — ноль вхождений).
Замер первого прохода: video_processing 22 %, ball_worker 25 %,
segment_layout 35 %, video_store 39 %, multi_pass 45 %.

**Рекомендация.** По отдаче: (1) HTTP-тесты `POST /api/videos` — отклонение
по лимитам 250 МБ/60 с и падение ffprobe (паттерн ASGITransport уже есть,
subprocess замокать); (2) границы сегментации `segment_layout`;
(3) частичные отказы `multi_pass`.

### CI-4. Замера покрытия нет — medium, актуально. 
pytest-cov не установлен; порогов нет ни в Python, ни в vitest.
**Рекомендация:** `--cov=app --cov-fail-under=78` в CI (замеренный базлайн
80 %), `@vitest/coverage-v8` для `src/lib`.

### CI-5. Ноль статического анализа Python на 31k строк — medium, актуально.
Ни ruff, ни mypy; новый пакет `providers/` тоже без проверки.
**Рекомендация:** ruff (lint+format) + mypy в нестрогом режиме в api-джобу CI.

### CI-6. Ни один тест не пересекает реальную границу процессов — medium, актуально.
`test_real_workers.py` скипается без opt-in; клиенты тестируются на моках.
**Рекомендация:** nightly-джоба `docker compose up --wait` + минимальный
смоук (маленький клип → poll → форма документа); model-validation harness —
по расписанию на раннере с весами.

### CI-7. Фронтенд-пробелы — low, актуально.
VideoIngestDrawer и CalibrationQaPanel без тестов; ThreeViewport — только
SSR-строка (см. FE-6).

---

## 5. Гигиена репозитория

### RH-1. Нет VCS — critical, УСТРАНЕНО.
Git инициализирован, закоммичен, запушен на приватный remote.

### RH-2. ~1.15 ГБ бинарников в первом коммите — high, УСТРАНЕНО.
Переписанный `.gitignore` покрывает всё, включая бесрасширенные чекпоинты
PnLCalib через правило на каталог; в индексе 251 файл, максимум 576 КБ.

### RH-3. Битые gitlink из `.references/` — high, частично.
Gitlink-риск снят; runtime-связка ball-worker осталась и обострилась —
см. INFRA-3.

### RH-4. Правило `data/` прятало фикстуру — medium, УСТРАНЕНО.
Заменено на `data/media/`; `data/matches/spain-belgium-2026-qf.json`
закоммичен.

### RH-5. Дрейф пинов Python-зависимостей; lockfile отсутствует — medium, актуально

ball-worker: `fastapi==0.115.8`, `uvicorn 0.34.0`, `Pillow 11.1.0`,
`opencv 4.10.0.84` — против `0.115.14`/`0.34.3`/`11.3.0`/`4.11.0.86` у трёх
остальных; `apps/api/pyproject.toml` — открытые диапазоны; uv.lock/poetry.lock
нет нигде.

**Рекомендация.** uv-workspace: минимальный `pyproject.toml` каждому сервису,
намеренные пины torch 1.13.1+cpu сохранить, `uv.lock` закоммитить; либо
минимально — pip-compile с общим `constraints.txt`.

### RH-6. Quickstart README падает на свежей машине — medium, актуально (обострилось)

`apps/api/models/` и `calibration-worker/models/` теперь в gitignore, так
что на свежем клоне весов гарантированно нет, а скриптового пути их получить
по-прежнему нет (у calibration-worker только `benchmark_frame.py`).

**Рекомендация.** Секция «Model weights» в README (чекпоинт → URL → SHA-256 →
путь) + fetch-скрипты из INFRA-6.

### RH-7. Нет task-runner'а и входной точки для 9-командного сетапа — low, актуально

Makefile/justfile/CLAUDE.md/CONTRIBUTING нет. Ловушка двух БД жива:
`sqlite:///./replay-studio.db` относителен cwd (`config.py:9`) → в корне
10 МБ базы, в `apps/api/` — вторая на 24 КБ.

**Рекомендация.** Makefile/justfile (`setup`, `fetch-models`, `dev-api`,
`dev-web`, `workers-up`, `verify`) + короткий CLAUDE.md; путь SQLite
заякорить на корень репозитория.

### RH-8. `.env.example` неполон и не размечен — low, актуально.
См. INFRA-9: PITCH_KEYPOINT_* отсутствуют, worker-переменные вперемешку.

---

## 6. Документация

Сильнейшее направление: выборочные проверки расхождений почти не нашли.
Остались среднего/низкого приоритета:

- **DOC-1 (medium).** Нет обзорного `docs/ARCHITECTURE.md` для новичка
  (карта репо, дата-флоу video → ingestion → detector → workers → scene
  JSON → Vue editor, модель revision/CAS/lease). Новый слой `providers/`
  добавил 4 недокументированных модуля.
- **DOC-2 (medium).** Док-сет расколот между русским (TECHNICAL_DEBT,
  RECONSTRUCTION_QUALITY, ROADMAP, FOOTBALL_DATA_APIS) и английским
  (остальное). Выбрать канонический язык или добавить резюме.
- **DOC-3 (low).** Замеры PERFORMANCE.md без даты/железа — добавить
  provenance-строку (дата, CPU, версия PyTorch, коммит воркера).
- **DOC-4 (low).** R4 в roadmap мешает сделанное с несделанным без
  маркеров; пункт про alignment-gated fusion устарел (реализовано).
- **DOC-5 (low).** PIPELINE_EDGE_CASES.md и UI_INFORMATION_ARCHITECTURE.md
  не слинкованы из README (первый описывает неотменяемые инварианты).
- **DOC-6 (low).** У 9 из 11 признанных ограничений в TECHNICAL_DEBT.md нет
  стабильных ID (только TD-ACT-01/TD-ANIM-01) — присвоить TD-XXX.

---

## 7. Безопасность

Низкоуровневая гигиена хорошая (см. «Сильные стороны»); экспозиция — на
периметре.

### SEC-1. Весь мутирующий API без аутентификации; compose публикует API и воркеры на всех интерфейсах — high, актуально

Depends/Security/APIKeyHeader/HTTPBearer — ноль вхождений; только
CORSMiddleware (`main.py:106-112`). Публикации: api `8000:8000` (compose:193),
воркеры 8090–8093, web `8080:80`, консоль MinIO `9001:9001` — bind
0.0.0.0 по умолчанию. Любой в LAN может залить 250 МБ
(`main.py:503`), запустить CPU-тяжёлую реконструкцию (`main.py:646`),
править/удалять сцены.

**Рекомендация.** Локально: `127.0.0.1:8000:8000` и убрать публикацию портов
воркеров вовсе (API ходит к ним по compose-сети). При любом развёртывании
дальше одной машины — хотя бы shared bearer через FastAPI dependency + квоты
на upload.

### SEC-2. Утечка ключа TheSportsDB через 502 — medium, УСТРАНЕНО.
Новая иерархия ошибок: `providers/base.py:8-13` (мандат санитизации),
`providers/thesportsdb.py:85-99` — `str(exc)` с URL-ключом больше не
попадает в сообщение; `api_football.py` следует тому же контракту.

### SEC-3. ffmpeg/ffprobe без таймаутов — medium, актуально

`video_processing.py:27-28` — `subprocess.run(...)` без timeout (все пути:
probe, транскод, poster, кадры); `ball_frames.py:158-163` — то же для
плотной выборки. Один битый файл вешает обработку навсегда.

**Рекомендация.** timeout (120–300 с, от `max_video_duration`), ловить
`TimeoutExpired`, помечать asset failed с явным сообщением стадии.

### SEC-4. Hardcoded-креды MinIO/Postgres — low, актуально (и уже в истории git).
См. INFRA-5/INFRA-9. После параметризации — сменить значения, поскольку
старые закоммичены в приватный remote.

### SEC-5. CSRF на мутирующих POST — low, актуально.
Только CORSMiddleware; multipart upload и body-less POST — preflight-free.
**Рекомендация:** требовать кастомный заголовок (X-Requested-With) на
мутирующих роутах — его наличие форсирует preflight.

### SEC-6. `PUT /api/scenes` принимает неограниченный JSON; title переполняет колонку — low, актуально.
`SceneDocument.title` без max_length при `String(240)` в БД
(`database.py:15`); лимита тела запроса нет.
**Рекомендация:** `max_length=240` + ограничение размера тела (middleware
или uvicorn limits).

---

## 8. Производительность

Ядро алгоритмики здоровое (контентно-адресуемые кэши, QA-gated temporal
calibration, Hungarian). Затраты — в оркестрации.

### PERF-1. ~2 000+ полных CAS-перезаписей документа за прогон — critical, актуально

То же, что BE-2, со стороной масштаба: тики на каждый сэмплированный кадр
(10 fps, `reconstruction.py:8468-8481`) и на каждый плотный кадр мяча
(25 fps, `ball_progress`:8635–8646). Документ ~4.5 МБ.
**Рекомендация:** см. BE-2. Убирает ~99 % записей.

### PERF-2. PnLCalib на каждом сэмплированном кадре при готовом temporal-графе — high, актуально

`reconstruction.py:8244-8247` шлёт все кадры; замер — 9.32 с/кадр CPU;
полный клип 600 кадров ≈ 90+ минут холодного прогона. Temporal solver
(`solve_calibration_sequence`, разрешение на 8578–8581) уже умеет заполнять
промежутки, но anchor-subsampling не включён; `camera_motion_edges`
(8352–8362) уже классифицирует монтажные склейки.

**Рекомендация.** Калибровать только якоря (первый/последний кадр шота,
кадры после cut/unreliable, периодические max-gap), интерьер — temporal
solver с fallback на прямой инференс при отказе QA-gates. 5–20× на холодном
прогоне.

### PERF-3. LRU-кэш calibration-worker меньше рабочего сета — high, актуально

512 записей (`services/calibration-worker/app/main.py:115`) при клипе в 600
кадров: последовательный повтор в LRU меньше рабочего сета → 0 % попаданий,
тёплый путь (замеренный 31.7×) деградирует к холодному на клипах >~51 с;
кэш процесс-локальный, теряется при рестарте; клиент повторно грузит байты
кадров (`calibration_worker.py:81`).

**Рекомендация.** Максимум записей → 4096; персист результата на диск по
SHA-256 кадра + версии модели/конфига (паттерн
`person_detection_cache.py`/`ball_detection_cache.py` готов); опционально —
hashes-only precheck endpoint, чтобы тёплый прогон не гонял байты вовсе.

### PERF-4. Поллинг полного 4.5 МБ документа каждые 700 мс — high, актуально

`App.vue:2352-2366` + `GET /api/scenes/{id}` возвращает полный документ
(`main.py:1152-1153`) с deepcopy payload на каждый запрос
(`store.py:175, 273-279`). За прогон 431 с — ~615 поллов ≈ 2.8 ГБ
сериализации/передачи/парсинга; каждый полл дёргает JSON.stringify-watcher'ы
ThreeViewport.

**Рекомендация.** `GET /api/scenes/{id}/reconstruction-progress` на несколько
сотен байт (объект `reconstruction['progress']` самодостаточен); полный
документ — один раз на терминальном статусе. deepcopy в `_with_live_lease`
заменить точечной копией мутируемой цепочки.

### PERF-5. Этапы пайплайна строго последовательны, инференс покадровый — high, актуально

Калибровка (удалённый процесс, 8244) → покадровый локальный YOLO
(`_predict_frame` batch=1, 2363–2374; keypoint-fallback уже батчит 4) →
identity (8500–8529) → плотный мяч по одному кадру (7929+). Все worker-HTTP
циклы — последовательные, один запрос в полёте.

**Рекомендация.** Запрос калибровки параллельно локальной детекции
(состояние не пересекается до merge на 8364); батчить Ultralytics 4–8
кадров; для мяча — многокадровые тайл-батчи через существующий
`_predict_batch` (`ball_detection.py:426`). Экономия: длительность более
короткой из двух стадий + 20–40 % пропускной способности детектора.

### PERF-6. O(n²) линковка кандидатов мяча — medium, актуально

`_ball_keyframes` (`reconstruction.py:7692-7712`): на каждую детекцию каждый
плотный кадр итерируется весь накопленный список кандидатов, при этом gap-тест
(7704–7705) допускает только треки возрастом 1–2 кадра.

**Рекомендация.** Индекс кандидатов по последнему кадру
(`dict[int, list]`), смотреть только бакеты `frame-1`/`frame-2`, старше —
на покой. O(detections) при идентичном выводе.

### PERF-7. Recovery-монитор — эксклюзивная транзакция + полная загрузка каждые 5 с — medium, актуально.
Сторона производительности BE-4; там же рекомендация (дешёвый read-only
probe перед `BEGIN IMMEDIATE`).

### PERF-8. Линейные сканы keyframes и аллокации в render loop — medium, актуально

`interpolate.ts:8` — `findIndex` O(K) со свежим замыканием на вызов; мяч —
до ~1 500 плотных keyframes на каждый рендер-кадр; `selectedPathSource`
аллоцирует `new THREE.Color` на вызов (`ThreeViewport.vue:421-436`).

**Рекомендация.** Бинарный поиск + кэшируемый курсор на субъект; лукап
источника и Color — из render loop в существующий watcher
`rebuildSelectedPath`; JSON.stringify-источники → дешёвые фингерпринты.

---

## 9. Слой провайдеров матчевых данных (`apps/api/app/providers/`)

Отдельный проход перед переделкой провайдеров статистики: 6 аспектов
(контракт/схемы, обе реализации, реестр+роуты, реальные потребители, тесты),
17 high/critical находок подтверждены верификацией, 0 опровергнуто.
Провайдерские тесты прогнаны фактически: 19 тестов, все зелёные, 1.3 с.

### Что слой уже делает правильно

- Санитизированная провайдер-нейтральная таксономия ошибок с
  `provider`/`code`/`retryable` (`base.py:8-53`), консистентно маппится в
  HTTP-статусы и `X-Match-Data-*` заголовки (`main.py:174-194`).
- Persisted-provenance: снапшот записывает породившего провайдера, refresh
  переигрывает его явно, без тихого fallback.
- `ContextVar`-override реестра реализован без утечек между запросами
  (`registry.py:110-120`).
- Разделение дедуплицированных players и сырых lineup-строк для аудита;
  quality-проверки API-Football богаче (starters, GK, дубли, согласованность
  замен, `api_football.py:458-512`) — сохранить как эталон.
- `rosterQuality.automaticIdentityEligible` честно гейтит автоматическую
  identity вниз по конвейеру (`reconstruction.py:6807-6820`,
  `reconstructionUi.ts:190-191`).

### Контракт и схемы

- **PROV-1 (critical).** Контракт вообще не представляет статистику матча и
  игроков. Полный набор полей EventBundle — event/players/lineup/timeline/
  substitutions/roster_quality/fetched_at/warnings (`schemas.py:103-114`);
  матч — только финальный счёт (`schemas.py:98-99`) и свободная строка
  статуса; игрок — только identity/позиция/номер (`schemas.py:13-22`) — ни
  минут, ни голов, ни карточек, ни рейтинга. Карточки/голы восстановимы лишь
  подсчётом timeline-строк с несвязанным строковым `type`, при этом оба
  провайдера предупреждают об усечении таймлайна (кап 5 событий,
  `thesportsdb.py:346-349`). **Рекомендация:** опциональные версионируемые
  блоки `ExternalTeamMatchStats` (по team_id: владение, удары, xG, угловые,
  фолы, сейвы, пасы) и `ExternalPlayerMatchStats` (по player_id: минуты,
  голы, ассисты, удары, карточки, рейтинг) + дескриптор покрытия по образцу
  `ExternalRosterQuality`. Статистику из подсчёта таймлайна не выводить.
- **PROV-2 (high).** `capabilities` — мёртвые метаданные: одинаковые
  константы у обоих провайдеров (`thesportsdb.py:46`, `api_football.py:54`),
  сериализуются (`registry.py:75`), но нигде не ветвятся; реальные различия
  (двух-командный поиск API-Football) всплывают как runtime-ошибки.
  **Рекомендация:** либо удалить, либо сделать несущими: Literal-словарь
  (`team-statistics`, `player-statistics`, `free-text-search`, …), гейтить
  опциональные endpoint'ы и UI по нему.
- **PROV-3 (high).** Roster-quality реализована трижды с расходящимися
  правилами, включая провайдер-специфичную ветку
  `if bundle.source == "thesportsdb" and len(players) == 5` в нейтральном
  слое (`main.py:152`; `thesportsdb.py:231-260`; `api_football.py:458-512`).
  **Рекомендация:** один нейтральный модуль качества с декларируемыми
  provider limits вместо сравнения строк source.
- **PROV-4 (high).** `EventBundle.source` имеет default `"thesportsdb"`
  (`schemas.py:106`), и адаптер TheSportsDB на него полагается
  (`thesportsdb.py:350-359` не передаёт source) — ловушка тихой
  мисатрибуции для любого нового провайдера, скопировавшего паттерн.
  **Рекомендация:** сделать поле обязательным (или штамповать
  `bundle.source = provider.id` в реестре после каждого вызова); legacy
  снапшоты трактовать как thesportsdb при чтении.
- **PROV-5 (high).** `type` таймлайна — свободная строка с разной
  нормализацией по адаптерам (`thesportsdb.py:302` vs `api_football.py:391`)
  и substring-эвристиками потребителей — кросс-провайдерная статистика из
  него невыводима. **Рекомендация:** закрытый нормализованный словарь
  (goal / own-goal / penalty-goal / yellow-card / red-card / substitution /
  var / other) рядом с сырой строкой; нормализация — в одном общем модуле.
- Medium: модель минут схлопывает добавленное время (45+3 → 48);
  валидация путь-зависима (Manual* строгие, провайдерные External* — нет,
  sentinel-id попадают в join-ключи); resolver-шов participation-evidence
  существует, но таймлайн снапшота в него не заведён — естественная точка
  посадки player-stats сейчас мертва.

### Реализация TheSportsDB (`thesportsdb.py`)

Утечка ключа через 502 подтверждённо устранена (все сообщения SportsDbError
статичны, 85–102; `main.py:190-193` рендерит только их).

- **PROV-6 (high).** Форма upstream-JSON не валидируется: тело `null`
  проходит `response.json()` (`:84`) без guard'а, кэшируется локально
  (`:103`) и в Redis (`:106`), затем все вызыватели зовут `.get` → 500 на
  весь TTL (кэш отравлен). У api_football guard есть (`:158-162`).
  **Рекомендация:** `isinstance(data, dict)` в `_get` до кэширования,
  код `invalid-provider-response`; кэшировать только валидное.
- **PROV-7 (high).** `event_bundle` — всё-или-ничего: 3 параллельных запроса
  через `asyncio.gather` БЕЗ `return_exceptions` (`:263-267`) — один 429 на
  бесплатном ключе валит весь bundle. **Рекомендация:** паттерн
  api_football — деградация по блокам с warnings (фатален только lookup
  события) + коалесцирование запросов и клиентский backoff.
- Medium: сырые upstream-id с индексными fallback'ами без неймспейса
  провайдера; Redis-клиент без socket-таймаутов и без close, in-memory кэш
  неограничен; новый `httpx.AsyncClient` на каждый запрос, retry нет даже
  для retryable; ключи Redis не скоупятся кредами (апгрейд с ключа `123`
  продолжит отдавать усечённые данные); у фикса утечки ключа нет
  регрессионного теста; дублирование с api_football уже дрейфует
  (кэш-слой, парсинг скаляров, quality, минуты).

### Реализация API-Football (`api_football.py`)

- **PROV-8 (high).** Статистика не запрашивается вовсе: единственные
  endpoint'ы — fixtures?date (`:244`), teams?search (`:253`),
  headtohead (`:283-291`), fixtures?id (`:515`), lineups (`:533`),
  events (`:540`). Данных, ради которых затевается переделка, в пайплайне
  нет. **Рекомендация:** добавить team/player stats как новые методы
  провайдера + блоки EventBundle; сперва читать embedded-блоки ответа
  fixture-by-id (паттерн `:521+` уже бережёт квоту).
- **PROV-9 (high).** Retryable-отказы под-запросов (429) тихо
  деградируются в строки warnings (`:544-548`, `:565-574`) с потерей
  code/retryable — и персистятся как снапшоты с пустым ростером.
  **Рекомендация:** единый контракт partial-bundle: либо пробрасывать
  retryable (bind падает, клиент ретраит — как thesportsdb), либо
  структурированная деградация (per-section status с кодом), чтобы
  «квота» не превращалась в «состава нет».
- Medium: ~половина файла — копипаста с thesportsdb (нужен общий базовый
  класс); голые числовые id без неймспейса → коллизии между провайдерами в
  сохранённых снапшотах; quota-ошибка в теле HTTP-200 маппится в 502 вместо
  503; in-process кэш неограничен; free-text `search_events` жёстко падает
  при одинаковой заявленной capability `search`; новый AsyncClient на запрос.

### Реестр и роуты

- **PROV-10 (high).** Конфигурация заморожена на import-time:
  `sports_provider = MatchDataProviderRegistry()` на уровне модуля
  (`registry.py:123`), settings через `@lru_cache` (`config.py:94-96`),
  `configured` — снапшот ключа (`api_football.py:56-74`). Смена ключа
  требует рестарта процесса; кривой `MATCH_DATA_PROVIDER` роняет импорт.
  **Рекомендация:** конструировать реестр в lifespan/dependency, валидация
  настройки со внятной startup-ошибкой, `configured` — от живых settings.
- **PROV-11 (high).** Тихий кросс-провайдерный fallback
  (`registry.py:37-48`: первый configured в порядке словаря — api-football)
  плюс опциональный `provider` в трёх每 catalog-роутах (`main.py:1117,
  1128, 1139`) и в `create_scene` — провайдер-специфичный event id может
  резолвиться не тем провайдером. **Рекомендация:** сделать fallback
  громким; требовать явный provider для event-id-роутов либо отклонять id,
  чья provenance расходится с резолвящим провайдером.
- Medium: `create_scene` глотает любые MatchDataError и персистит
  неверифицированный stub matchBinding (включая несуществующих провайдеров
  и события); маппинг ошибок сваливает мисконфиг сервера и клиентский ввод
  в общий 502; параметры `date`/`event_id` не валидируются по формату и
  утекают в upstream-запросы и Redis-ключи; тестируемость держится на
  monkeypatch связанных методов синглтона.

### Реальный контракт потребителей (что переделка обязана сохранить)

Потребляемое подмножество много уже схем: identity/review читают только
player {id, name, number, team_id, position}, event/team ids+names, timeline
{id, minute, label, type}, rosterQuality и warnings; lineup[],
substitutions[], thumbnails, lineup_role/order — персистятся,
фингерпринтятся, ранжируются, но никем не читаются.

- **PROV-12 (high).** Фронтенд пинит `schemaVersion` строгим равенством 2
  (`reconstructionUi.ts:161`) при захардкоженном 2 на сервере
  (`main.py:211`): любое повышение версии обнуляет карточку матча и
  таймлайн. Расширение статистикой должно быть либо аддитивным при
  version 2, либо менять проверку на `>= 2` синхронно (плюс аудит
  `_legacy_binding_rank`, `main.py:1163-1181`, который ранжирует снапшоты
  длинами списков — новые stats-списки должны войти туда осознанно).
- **PROV-13 (high).** Провайдер-специфика захардкожена в потребителях:
  five-player-cap ветка (`main.py:152`), default source thesportsdb в
  legacy-путях (`main.py:1637, 1653`), матч по английской прозе warning'а
  (`identity_review.py`), замороженный фронтенд-каталог
  `LEGACY_MATCH_DATA_PROVIDERS` (`matchDataProviders.ts:11-35`).
- **PROV-14 (high).** Жёсткий инвариант: `players[].team_id` обязан быть
  строго равен `event.home.id`/`event.away.id` (`main.py:143-148`), иначе
  автоматическая identity тихо выключается и roster binding блокируется
  (`reconstruction.py:6807, 6771-6779`). Оба адаптера сейчас соблюдают —
  зафиксировать как контракт провайдера bundle-валидацией.
- Ещё сохранить: `event_bundle`-шов реестра для 11 тестов-потребителей
  (`registry.py:113-120` — осознанная индирекция); `source` — открытая
  строка, `manual` — валидный source; `roster_quality` опционален, у
  потребителей есть None-fallback; Redis-ключи по образцу api_football
  (`match-data:{id}:{credential_scope}:{key}`), не thesportsdb.

### Тесты слоя

- **PROV-15 (high).** Общего контрактного сьюта нет: два расходящихся
  набора (api-football проверяет `bundle.source`, thesportsdb — нет;
  parametrize по провайдерам отсутствует). **Рекомендация:** до переделки —
  один параметризованный сьют, обязательный для каждого зарегистрированного
  провайдера (source==provider.id, строковые id, связность
  lineup/substitutions, словарь reasons, warnings на пустых блоках, коды
  MatchDataError).
- **PROV-16 (high).** Транспорт `_get` thesportsdb не исполняется ни одним
  тестом (все мокают `provider._get`): маппинг 429/5xx/сетевых ошибок,
  TTL-кэш, Redis-fallback — нулевое покрытие ровно того, что переделка
  тронет первым. **Рекомендация:** httpx.MockTransport-тесты на
  200/429/5xx/network/invalid-JSON с проверкой кода и retryable.
- **PROV-17 (high).** 11 потребительских тестов зависят от monkeypatch-шва
  `app.main.sports_provider.event_bundle`. Либо заморозить шов на время
  переделки, либо сперва мигрировать на явную инжекцию (FastAPI dependency)
  отдельным коммитом с полным прогоном сьюта.
- Medium: route-маппинг ошибок (404/502/503/Retry-After) не покрыт; ветки
  ошибок `_get` api_football не покрыты; Redis-down не покрыт нигде;
  конкуренция ContextVar-override не покрыта.

---

## 10. Готовность к server-side: серверная БД и независимые проекты (матчи)

Отдельная оценка (4 направления, 22 агента, каждый блокер адверсариально
верифицирован; 1 из 17 заявленных блокеров отсеян по severity). Вердикт:
**частично готово**. Ядро CAS/lease спроектировано мультипроцессно и в
основном переносимо на Postgres, но вокруг него — конкретные блокеры.

### Что уже готово к серверу (проверено по коду)

- Все CAS-пути берут `SELECT … FOR UPDATE` на строку сцены в non-sqlite
  ветке (`store.py:348-352, 410-414, 476-479, 570, 703-707, 826-835`).
- Порядок блокировок scene→lease консистентен во всех путях — ordering
  deadlock между claim/heartbeat/publish отсутствует.
- Lease-времена — портируемые epoch-float (`database.py:37-39`); гонки
  сканов recovery обезврежены атомарным claim — две реплики API не запустят
  один run дважды.
- psycopg3 объявлен в зависимостях (`pyproject.toml:11`), соответствует
  compose `DATABASE_URL`; `pool_pre_ping` включён (`database.py:67`).
- ML-воркеры БД не трогают вовсе — переносимость сосредоточена в apps/api.

### Блокеры: серверная БД (Postgres)

- **SRV-1 (days).** Миграций нет: `init_database` — только `create_all`
  (`database.py:71-72`), alembic отсутствует в дереве и зависимостях.
  `create_all` не умеет ALTER: первое же добавление типизированной колонки
  на живой серверной БД даёт `UndefinedColumn` на каждом запросе без пути
  апгрейда. Вся существующая эволюция — только внутри JSON payload
  (`store.py:32-43`, `main.py:1185`). **Фикс:** alembic c baseline-ревизией,
  `alembic upgrade head` вместо `create_all`.
- **SRV-2 (days).** Recovery-монитор каждые 5 с в каждом процессе берёт
  неупорядоченный full-table `FOR UPDATE` (`store.py:570`), держа локи всех
  сцен всех проектов на время десериализации каждого payload; в паре с
  неупорядоченным IN-list `FOR UPDATE` в `put_many` (`store.py:348-352`) —
  экспозиция на 40P01-дедлоки, которые всплывают как необработанные
  исключения. **Фикс:** скан кандидатов без локов → индивидуальный lock
  (`ORDER BY id` / `FOR UPDATE SKIP LOCKED`), retry на
  deadlock/serialization вокруг CAS-транзакций.
- **SRV-3 (days).** Postgres-покрытие тестами — ноль: все конкурентные
  тесты строят sqlite-движки (`test_reconstruction_leases.py:59-63` и
  соседние), ветка `store.py:221-222` не исполняется ни одним тестом.
  **Фикс:** Postgres-lane (testcontainers/CI-сервис), параметризовать
  store/lease/CAS-сьюты по `DATABASE_URL`, добавить двухсессионный тест
  контеншена FOR UPDATE.
- **SRV-4 (hours).** Insert-путь `put()`: `FOR UPDATE` по отсутствующей
  строке на Postgres ничего не лочит (gap-локов нет) — гонка двух create
  одного id даёт неперехваченный `IntegrityError` → 500 вместо конфликта
  ревизии (`store.py:410-424, 449-451`). **Фикс:** ловить IntegrityError на
  insert-пути → `SceneRevisionConflict` (или `INSERT … ON CONFLICT`).
- Отсеяно верификацией: гонка `seed()` при одновременном старте реплик —
  ошибка подтверждена, но это одноразовый стартовый крэш демо-сида, не
  блокер эксплуатации.
- Gaps: payload-колонка — generic JSON, не JSONB, при full-table
  payload-сканах в горячих путях; пул — дефолтные 5+10 без конфигурации,
  isolation level не пинован; записи `video_assets` — last-write-wins без
  контроля конкуренции; `updated_at` различается по timezone/precision между
  бекендами.

### Блокеры: «независимые проекты (матчи)»

Сущности проекта/матча в БД нет. Единица работы — корневая video-сцена
(`video_processing.py:212-230`); «проект» существует только как runtime-обход
указателей `parentSceneId` по ВСЕЙ таблице сцен (`store.py:281-326`),
«матч» — это per-scene JSON-снапшот matchBinding, продублированный на каждую
сцену проекта (`main.py:210-229`, `project_match.py:60-83`).

- **SRV-5 (days).** Ни project/match-сущности, ни тенантности: `GET
  /api/scenes` и `GET /api/videos` — глобальные списки без фильтра
  (`main.py:1147-1149, 545-547`); колонок `project_id`/owner нет
  (`database.py:11-19, 42-62`). **Фикс:** таблица `projects` (id, title,
  match source/eventId) + FK `project_id` на scenes и video_assets;
  скоупить листинги; backfill — существующим обходом родительских
  указателей + `matchBinding.eventId`.
- **SRV-6 (days).** `project_scenes()` на каждый bind/import/refresh грузит
  всю таблицу сцен с полными payload в память (`store.py:289-297`,
  вызов `main.py:1339`). **Фикс:** тот же `project_id` → `WHERE project_id
  = ?`; обход указателей оставить только для одноразового backfill.
- **SRV-7 (weeks).** Один матч не может владеть несколькими видео/ракурсами:
  каждый upload безусловно создаёт новый корень (`video_processing.py:
  212-230`), multi-pass собирается только из сегментов одного родителя
  (`main.py:626-632`, `multi_pass.py:643-664`) — второй ролик того же матча
  становится несвязным проектом. **Фикс:** проект владеет многими asset'ами;
  выбор пассов multi-pass поверх asset'ов проекта (выравнивание уже на
  motion-signature DTW, `multi_pass.py:183-228` — кросс-asset правдоподобен);
  upload в существующий проект через `project_id` в форме.
- Gaps: эвристика demo-сцен по тексту заголовка прячет реальные проекты;
  legacy-миграция биндингов обходит все проекты на каждом старте; снапшот
  matchBinding дублируется на каждую сцену вместо хранения один раз.

### Блокеры: серверный рантайм

- **SRV-8 (days).** Все медиа-артефакты — на локальном диске принявшей
  реплики (`video_processing.py:23-24`, `main.py:513-524, 562-575`,
  `ball_frames.py:196-213`): вторая реплика отвечает 409/`unavailable` на
  чужие asset'ы. **Фикс:** краткосрочно — общий сетевой том MEDIA_ROOT
  (кэши контентно-адресуемы и публикуются атомарно — шаринг переживут);
  долгосрочно — object storage (MinIO уже в compose) со staging ffmpeg во
  временный локальный каталог.
- **SRV-9 (days).** Обработка загрузки видео — in-process BackgroundTasks
  без lease и без recovery (`main.py:541`, `video_processing.py:153-251`):
  рестарт/крэш навсегда оставляет asset в `processing` (recovery-монитор
  сканирует только реконструкции, `reconstruction_recovery.py:14-35`).
  **Фикс:** та же схема, что у реконструкции: lease/claim по asset id,
  recovery-скан, идемпотентные стадии ffmpeg.
- **SRV-10 (days).** Multi-pass без fencing-lease и исключён из recovery
  (`main.py:642`, `multi_pass.py:722-726, 772`; явные исключения в
  `store.py:582-585, 653-654, 712-714`): крэш вешает композит в
  `processing`, возможен дубль вычислений. **Фикс:** распространить lease
  на multi-pass, реконструкцию детей вести через
  `queue_reconstruction`/claim, включить в монитор.
- **SRV-11 (weeks).** Auth/тенантности нет ни на одном роуте — общий
  глобальный неймспейс для любого вызывающего (совпадает с SEC-1; для
  multi-user это блокер, а не hardening).
- Gaps: YOLO-инференс в процессе API (см. BE-8); один hardcoded URL на
  воркер без балансировки/очереди/backpressure; 250 МБ медиа через
  FileResponse без offload (нет X-Accel-Redirect); flock-локи кэша слабее
  на NFS; относительный `media_root` и localhost-CORS как дефолты.

### Блокеры: эксплуатация

- **SRV-12 (days).** Аутентификации нет нигде (`main.py:104-114`, ни одного
  route-level Depends/Security; таблицы пользователей нет). Дешёвый
  промежуточный шаг: глобальная dependency с shared bearer против
  `AUTH_TOKEN` в Settings — часы работы, закрывает открытую запись для
  доверенной команды. Полноценный multi-user (users, sessions/OIDC,
  owner-скоупинг) — недели.
- **SRV-13 (hours).** Продовый nginx ломает основной сценарий:
  в `apps/web/nginx.conf:7-11` нет `client_max_body_size` (дефолт 1 МБ) при
  серверном лимите 250 МБ (`config.py:18`) — загрузка видео через web-тир
  умирает на nginx. **Фикс:** `client_max_body_size 260m;` +
  `proxy_request_buffering off` + `proxy_read_timeout`.
- **SRV-14 (hours).** TLS нет; неаутентифицированный API опубликован прямо
  на хост (`docker-compose.yml:192-193`), воркеры и консоль MinIO — тоже.
  **Фикс:** один TLS-терминирующий reverse proxy (Caddy/Traefik + ACME),
  убрать `ports:` у api/воркеров/консоли, `CORS_ORIGINS` на реальный origin.
- **SRV-15 (hours).** Hardcoded-креды в compose и закоммиченный
  третьесторонний ключ: `.env.example:4` содержит похожий на живой
  `API_FOOTBALL_API_KEY=4ed3…c315`. **Фикс:** ротировать ключ немедленно
  (он в истории git и на remote), пустое значение в `.env.example`,
  креды через `${VAR}`/secrets.
- **SRV-16 (days).** Бэкапов/DR нет: четыре именованных тома без
  какого-либо бэкап-механизма, restart-политик нет, восстановление после
  инцидента не определено. **Фикс:** pg_dump-sidecar на off-host +
  restic/rsync медиа-тома; `restart: unless-stopped`; alembic до первого
  изменения схемы.
- Gaps: логирование — один module-logger на весь API, без конфигурации и
  структуры; request-id/метрик/трейсинга нет; квот на пользователя нет,
  concurrency ffmpeg не ограничена (капнута только реконструкция);
  single-process uvicorn без истории воркеров/реплик.

### Минимальный путь к «серверная БД + проекты» (один хост, доверенная команда)

1. alembic + baseline (SRV-1) и Postgres-lane в тестах (SRV-3) — фундамент,
   часы→дни.
2. `projects` + `project_id` на scenes/video_assets, скоупленные листинги,
   backfill (SRV-5/6) — дни.
3. Lease для video-processing и multi-pass (SRV-9/10) — дни.
4. Shared bearer-token + TLS-proxy + nginx body-size + ротация ключа +
   pg_dump (SRV-12..16) — часы→дни.
5. Фиксы Postgres-конкуренции: SKIP LOCKED в мониторе, IntegrityError→409
   (SRV-2/4) — часы→дни.

Мульти-реплика/мульти-хост дополнительно требует общего MEDIA_ROOT (SRV-8)
и выноса реконструкции из процесса API (BE-8). Несколько ракурсов одного
матча (SRV-7) — самый крупный продуктовый кусок, недели.

---

## 11. Пере-аудит после модификаций — 18 июля 2026

Дерево изменилось радикально: +5,5k/−1,4k строк в 58 файлах плюс ~60 новых
файлов. Проверка своими руками: корневой `pytest` — **701 passed, 2 skipped
за 9.4 с** (сбор больше не падает), фронтенд — **238 тестов зелёные**,
`vue-tsc` чистый.

### Статусы 149 прежних находок

| Статус | Кол-во |
|---|---|
| Актуальны | 101 |
| Частично устранены | 32 |
| Полностью устранены | 16 |

**Полностью закрыто этой волной:** CI-2 (пакет calibration-worker
переименован в `calibration_worker_service` + conftest — корневой pytest
работает), INFRA-1 (.dockerignore везде, `npm ci`), INFRA-2 (restart-политики
на всех сервисах), INFRA-8 (healthchecks api/web, ослаблен гейт на
calibration-worker), SRV-1 (alembic: `init_database` теперь
`upgrade_database`, compose-сервис `migrate` через
`service_completed_successfully`), SRV-7 (проект владеет несколькими видео),
SRV-13 (nginx `client_max_body_size 260m`), PERF-2 (**якорная калибровка
реализована**: `_select_calibration_anchor_frames`,
`reconstruction.py:2728-2770`, интерьер заполняет temporal solver),
PROV-6 (валидация формы upstream-JSON в thesportsdb), DOC-1 частично→
`docs/ARCHITECTURE.md` создан (агент оценил как partial — см. ниже).

**Существенный прогресс (partial):** BE-2/PERF-1 — прогресс project-сцен
пишется в компактную строку `AnalysisRunRow`
(`analysis_runtime.py:86-129`, `project_store.py:1373-1427`), full-document
CAS остался только как fallback для legacy-сцен без проекта; BE-8/SRV-9/10 —
выделенный сервис `reconstruction-runner` в compose
(`docker-compose.yml:241-265`, `RECONSTRUCTION_RECOVERY_IN_API: 0`) с
killable-подпроцессом на job, но локальный dev-дефолт остаётся in-process,
а upload и multi-pass — см. новые находки ниже; SEC-1 — loopback-биндинги
портов сделаны, auth нет; SRV-5/6 — слой проектов появился
(нормализованные таблицы, UNIQUE/CASCADE-владение), но старые глобальные
роуты живут нескоупленными рядом; CI-1 — `.github/workflows/ci.yml`
существует, но матрица воркеров сломана (см. NEW-11).

### Новые находки в новом коде (подтверждены верификацией)

Свежий аудит 8 подсистем: 2 critical + 10 high подтверждено, 3 заявленных
high отсеяно верификаторами.

#### Backfill и canonical match (`project_backfill.py`, `canonical_match.py`)

- **NEW-1 (critical).** Петля «миграция → backfill» на КАЖДОМ рестарте
  чеканит дубликат Match, снапшота и перехэшированных roster-id: миграция
  читает сцены через `project_scenes` с гидрированным overlay
  (`source="canonical"`, `storageSource="project-snapshot"`,
  `store.py:190`, `canonical_match.py:387-388, 473`), пересобирает биндинг
  с потерей `storageSource` (`main.py:1272, 1281-1290`) → «изменение»
  каждый старт; backfill затем видит новый биндинг и мintит новый Match.
  **Фикс:** backfill пропускает канонические/проектные биндинги (или
  проекты с `current_match_snapshot_id`); миграция сравнивает биндинги
  семантически. Обязателен тест двойного прогона
  миграция+backfill (идемпотентность).
- **NEW-2 (high).** Backfill на каждом старте перезатирает живые
  `AnalysisRun` lossy-проекцией из scene-JSON — стирает identitySync-
  диагностику (`project_store.py:1321-1333` безусловная замена
  status/diagnostics/таймстемпов). **Фикс:** создавать только отсутствующие
  строки, не перезаписывать более свежие.
- **NEW-3 (high).** Дедупликации одного реального матча между провайдерами
  нет: `bundle.source` зашит в хэш canonical id
  (`canonical_match.py:56-64`), а backfill использует ДРУГУЮ схему
  деривации id (`project_backfill.py:127-142`, длина 32 против 24) — даже
  одно и то же событие одного провайдера до/после миграции не сходится в
  один Match. **Фикс:** провайдер-нейтральная идентичность матча + единая
  функция деривации + миграция roster-биндингов через external references.

#### Слой проектов (`project_store.py`, `project_routes.py`)

- **NEW-4 (high).** Новые `PUT /projects/{id}/match` и `/match/refresh`
  обходят обе защиты legacy-пути: не проверяют
  queued/processing-реконструкции и durable roster-решения
  (`project_routes.py:532-591` зовут `persist_canonical_match` без
  прекондиций; legacy-гарды в `main.py:1383-1499`). **Фикс:** переиспользовать
  те же гарды (вынести их из main.py в project-слой).
- **NEW-5 (high).** Legacy bind-путь персистит canonical-снапшот ДО
  валидации и scene-CAS (`main.py:1522-1527` → `1537-1538` → `1589`):
  отклонённая смена матча всё равно становится current, т.к. сцены читают
  matchBinding из проектного снапшота. **Фикс:** валидация до любого
  коммита project-слоя; в идеале один транзакционный scope с `put_many`.
- **NEW-6 (high).** `ProjectStore` не перенял атомарную дисциплину
  SceneStore: `_begin_atomic_write` определён (`project_store.py:152-166`),
  но используется только в одном методе из ~7 read-modify-write; CAS
  `update_project` (`:442-465`) на SQLite подвержен lost-update.
  **Фикс:** `_begin_atomic_write` в начале каждого RMW-метода, как в
  SceneStore.

#### Runner и восстановление (`reconstruction_runner/dispatch`, store)

- **NEW-7 (critical).** Дети multi-pass гоняются с новыми мониторами
  восстановления: `multi_pass.py:872` вызывает `reconstruct_scene(child)`
  без run-токена и lease → ребёнок в `processing` без lease подпадает под
  «брошенную работу» (`store.py:813-822`; исключён только родитель,
  `store.py:805-806, 864-866`) — дублирование вычислений, ложные отказы,
  навечно застрявший композит. **Фикс:** дать детям настоящий
  claim (через `queue_reconstruction`/`reconstruct_scene_by_id`) либо
  исключить их из recovery-скана по `parentCompositeRunId`.
- **NEW-8 (high).** Legacy-сцены без `runId` вечно в списке recoverable, но
  `reconstruct_scene_by_id` всегда их отвергает (`reconstruction.py:14777`
  сравнение до claim-а, который бы обновил документ, `store.py:913-914`) —
  бесконечный churn подпроцессов. **Фикс:** принимать `legacy-…`-токен при
  пустом stored runId.
- **NEW-9 (high).** SRV-10 закрыт наполовину: multi-pass по-прежнему
  выполняет весь CPU-объём в контейнере API (`main.py:707`,
  `project_routes.py:789` — plain BackgroundTask) без lease и recovery.
- **NEW-10 (high).** SRV-9 закрыт наполовину: у video-upload появились
  AnalysisRun-зеркало и cancel, но нет lease/heartbeat/восстановления после
  рестарта: `processing` навсегда; cancel-гонка в обработчике ошибок может
  оставить `cancelling` навечно.

#### Миграции и CI

- **NEW-11 (high).** Матрица worker-contracts в CI не ставит
  тестовые зависимости: `ci.yml:75` — только `pip install pytest -r
  requirements.txt`, а тесты трёх сервисов импортируют httpx —
  3 из 5 ног матрицы падают на сборе. **Фикс:** ставить
  `requirements-test.txt` (identity-worker'у — создать его).
- **NEW-12 (high, воспроизведено).** Percent-encoded `DATABASE_URL` роняет
  alembic на старте: URL идёт в `Config.set_main_option` через
  configparser-интерполяцию (`schema_migrations.py:17`, `alembic/env.py:16`)
  — пароль с `@`/`%` валит API, runner и migrate-сервис до коннекта.
  **Фикс:** `url.replace("%", "%%")` либо передача через
  `config.attributes`.

#### Фронтенд

- **NEW-13 (high).** Watcher терминальных job'ов срабатывает на исторических
  run'ах при каждом открытии сцены (`App.vue:3784-3792` — ключ
  `scene:job:status` без проверки наблюдённого перехода) → лишние
  перезагрузки сцены, риск затирания несохранённых правок. **Фикс:**
  реагировать только на переход из активного статуса.
- Замечание тренда: `App.vue` вырос до 5 207 строк — проектный workspace
  прикручен к god-компоненту; composables так и не появились (FE-1
  усугубился). `reconstruction.py` — 14 847 строк (BE-1 усугубился).

#### Прочее заметное (medium, без верификации не требовалось)

- Anchor-калибровка: нет прямого fallback при отказе интерьерного
  заполнения; `calibration_anchor_max_gap_seconds` не согласован с
  хардкодом 2.0 s temporal solver'а; ложная причина отказа
  `no-automatic-calibration` для пропущенных кадров.
- Providers-delta: появился аккуратный `events_by_date_with_fallback` с
  курируемым списком кодов, но fallback молчалив (нет телеметрии и
  provenance-заголовка), thesportsdb `event_bundle` — всё ещё
  всё-или-ничего, stats-моделей в схемах так и нет (PROV-1 актуален).
- CI не собирает ни одного Docker-образа; api-джоба ставит CUDA-torch
  вместо CPU-пинов Dockerfile; свежий клон не поднимет compose (веса).
- Query-амплификация в project-слое: полный ProjectDocument грузится
  дважды на каждое чтение сцены; поиск матчей пишет транзакцию на
  каждый результат.

### Обновлённые приоритеты после пере-аудита

| Приоритет | Действия | Находки |
|---|---|---|
| P0 | Идемпотентность миграция+backfill (тест двойного прогона) | NEW-1, NEW-2 |
| P0 | Lease/исключение для детей multi-pass | NEW-7 |
| P0 | Экранирование DATABASE_URL в alembic; httpx в CI-матрице | NEW-12, NEW-11 |
| P1 | Гарды новых match-роутов; порядок validate→persist; атомарность ProjectStore | NEW-4..6 |
| P1 | Recovery для upload/multi-pass; legacy-runId churn | NEW-8..10 |
| P1 | Terminal-job watcher | NEW-13 |
| P1 | Единая идентичность матча между провайдерами | NEW-3 |
| P2 | Остальные medium + всё актуальное из плана ниже | — |

## 12. Аудит большой реструктуризации — 18 июля 2026 (вечер)

Дерево реструктурировано радикально: 621 изменённый файл, 445 новых, 32
удалённых. `reconstruction.py` 14 847 → 304 строки (оркестратор с
`__all__ = ("reconstruct_scene",)`), `main.py` 1 799 → 78 (composition root
c 10 роутерами, ноль inline-роутов), `App.vue` 5 207 → 15 строк. В
`apps/api/app` теперь 330 модулей с дисциплиной
contract/repository/phase/command/routes; largest — 613 строк. Появился
`AGENTS.md` с политикой «каноническая архитектура важнее обратной
совместимости; вытесненные пути удаляются». Проверка тестов своими руками:
**бэкенд 875 passed / 2 skipped за 16.8 с, фронтенд 269 passed, vue-tsc
чистый**.

### Актуальные статусы всех 162 накопленных находок

| Статус | Кол-во |
|---|---|
| Полностью устранены | 65 |
| Частично устранены | 40 |
| Актуальны | 54 |
| Неактуальны (код удалён по политике AGENTS.md) | 3 |

**Реструктуризация закрыла все структурные находки аудита:**

- BE-1/BE-3 — распил по рекомендованным швам: 304-строчный fencing-wrapper,
  ~130 `reconstruction_*`-модулей, типизированные frozen-dataclass
  результаты фаз вместо сквозных мутируемых локалов.
- BE-2/PERF-1 — прогресс теперь компактный lease-fenced upsert телеметрии,
  «never reads or writes SceneRow»
  (`reconstruction_run_repository.py:481-543`); полная публикация документа —
  ровно один терминальный CAS. Legacy-fallback удалён вместе со `store.py`.
- BE-4/PERF-7 — recovery-монитор переведён на индексированную
  `ReconstructionJobRow` (`database.py:61-81`) + промоутнутые индексные
  колонки SceneRow c `sync_scene_index`; full-table `with_for_update`-скан
  больше не существует; локи — только по-scene на claim.
- BE-7 — main.py как composition root, match-import в
  `manual_match_import.py` за 49-строчным роутером.
- FE-1/FE-2 — App.vue разложен на 2 lazy-страницы, 5 editor-контекстов и
  20 по-контурных composables (включая рекомендованные `usePlaybackClock`
  с rAF-часами); `appArchitecture.test.ts` защищает топологию AST-тестами.
- NEW-1/NEW-2 — сами backfill/миграция удалены по политике AGENTS.md
  (obsolete); NEW-4..13 — **все 10 разрешены** (гарды match-роутов,
  validate→persist, атомарность ProjectStore, lease детей multi-pass,
  legacy-runId churn, recovery upload/multi-pass, alembic URL-escaping,
  httpx в CI-матрице, terminal-job watcher).
- Инварианты конкуренции пережили распил и местами усилились: dialect-ветка
  централизована в `database_transaction.py`, порядок scene→lease сохранён,
  fingerprints fail-closed. Проверено отдельным агентом с фокусом именно
  на этом.

### Новые находки в реструктурированном коде (верифицированы)

Свежий аудит нашёл всего 2 high (для рефакторинга такого масштаба —
исключительный результат):

- **R-1 (high).** Мутационные роуты player-action возвращают документ с
  устаревшей ревизией: `scene_analysis_routes.py:116-141` возвращает
  локальный dict, а `player_action_commands.py:14,20` выбрасывает результат
  `scenes.put` (ревизию бампит только возвращаемый deepcopy —
  `scene_repository.py:211-222`). Воспроизведено: в БД revision 2, в ответе
  revision 1 → следующий полный save падает 409. **Фикс:** возвращать
  результат `scenes.put` (как это делает ball-trajectory путь) + тест на
  равенство ревизий.
- **R-2 (high).** Ручное перемещение игрока на поле тихо теряется:
  `useEditorCompositionContext.ts:151-189` пишет keyframes в scene и ставит
  «Unsaved changes», но `compactSceneWrite` (`lib/api/scenes.ts:24-27`)
  удаляет keyframes треков из PUT-тела, бэкенд сохраняет compact-документ
  как есть, а гидрация перестраивает keyframes из артефактов — правка
  исчезает после save. **Фикс:** выделенный endpoint track-keyframes
  (зеркально `PUT /ball-trajectory`) либо удалить UI 3D-перемещения по
  политике AGENTS.md.
- **R-3 (medium, организационное).** Весь cutover существует только в
  незакоммиченном рабочем дереве: HEAD всё ещё несёт полное legacy-дерево,
  `git checkout -- .` воскресит монолиты, CI ни разу не прогонялся на новой
  архитектуре. **Закоммитить и запушить реструктуризацию немедленно.**
- Прочее medium: ORM-registry, разнесённый по side-effect-импортам двух
  семейств; дословный дубль транзакционного примитива между
  scene/project-персистенсом; 11 позиционных аргументов на швах фаз;
  `/api/health` до ~8 с последовательных проб при 5-секундном
  container-healthcheck; статус-коды из строк сообщений в пяти
  роут-модулях; inline-объекты в props инспектора (остаток FE-2);
  in-place мутация сцены не пробивает prop-границы; линейная интерполяция
  в render loop (PERF-8 жив); инференс всё ещё последовательный
  однокадровый (PERF-5 жив).

### Что остаётся актуальным (54 находки, главные кластеры)

1. **Провайдеры/статистика — цель переделки так и не реализована**:
   PROV-1 (нет моделей статистики матча/игрока), PROV-15 (нет общего
   контрактного сьюта), транспорты — копипаст-близнецы с неограниченным
   кэшем, registry — import-time singleton (PROV-10), thesportsdb
   всё-или-ничего (PROV-7), retryable→warnings у api-football (PROV-9),
   минуты 45+3→48, participation-шов резолвера не подключён.
2. **Auth/тенантность** — по-прежнему ноль Depends во всём приложении
   (SRV-11/12, SEC-1).
3. **Инструментальная дисциплина** — нет mypy/ruff (CI-5), нет coverage
   (CI-4), нет Python-lockfile (RH-5), СceneDocument остаётся untyped dict
   (BE-5 partial: 163 `scene: dict` аннотаций).
4. **Эксплуатация** — логирование/метрики/request-id (gaps), ffmpeg без
   таймаутов (SEC-3), FileResponse для 250 МБ, воркеры под root.
5. **Производительность хвостом** — PERF-5 (последовательный инференс),
   PERF-8 (render loop), JSON.stringify-watcher'ы (FE-5 residual).

### Приоритеты после реструктуризации

| Приоритет | Действия | Находки |
|---|---|---|
| P0 | **Закоммитить и запушить cutover**; прогнать CI на новой архитектуре | R-3 |
| P0 | Ревизия в ответах player-action; судьба ручного перемещения игрока | R-1, R-2 |
| P1 | Переделка провайдеров: stats-модели + контрактный сьют + общий транспорт (см. раздел 9) | PROV-1, 15, 6/7, 10 |
| P1 | ffmpeg-таймауты; bearer-auth dependency | SEC-3, SRV-12 |
| P2 | mypy/ruff + coverage в CI; типизация SceneDocument поверх scene_document.py | CI-4/5, BE-5 |
| P2 | Батчинг инференса; interpolate-курсоры; prop-границы инспектора | PERF-5/8, FE-2 residual |
| P3 | Остальное из кластеров выше | — |

---

## Приоритетный план (исторический, 17 июля)

| Приоритет | Действия | Находки |
|---|---|---|
| P0 | CI-workflow (git уже есть); вынести прогресс из scene-документа; ротация закоммиченного API-Football ключа | CI-1, BE-2/PERF-1, SRV-15 |
| P0 (перед переделкой провайдеров) | Контрактный сьют по провайдерам; HTTP-тесты транспорта; stats-блоки EventBundle проектировать с учётом PROV-12..14 | PROV-15..17, PROV-1 |
| P1 | Якорная калибровка + кэш воркера + progress-endpoint; параллелизация стадий | PERF-2..5 |
| P1 | Завендорить hrnet.py; restart-политики; .dockerignore | INFRA-1..3 |
| P1 | Loopback-биндинг портов; таймауты ffmpeg | SEC-1, SEC-3 |
| P1 | Начать распил: corrections/roster из reconstruction.py; useReconstructionPolling + usePlaybackClock из App.vue | BE-1/3, FE-1/2 |
| P1 (server-side) | alembic; projects/project_id; lease на upload/multi-pass; bearer+TLS+nginx | SRV-1, 5, 6, 9, 10, 12–14 |
| P2 | Переименовать пакет calibration-worker (чинит root pytest); тесты video-ingest; pytest-cov + ruff/mypy; Postgres-lane | CI-2..5, SRV-3 |
| P2 | fetch-скрипты весов + секция README; uv-lockfile; Makefile | RH-5..7, INFRA-6 |
| P3 | Остальные medium/low по разделам | — |

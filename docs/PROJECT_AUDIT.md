# Аудит проекта — рабочий backlog

Актуально на 19 июля 2026. Ревью владельца проекта интегрировано в текст:
скоуп-исключения применены, приоритеты пересобраны, фактические поправки
внесены (каждая перепроверена по коду), добавлены пункты, найденные в ходе
ревью (PROV-16..18, REC-1). Пункты сверены с текущим рабочим деревом
(341 модуль в `apps/api/app`).

**Скоуп.** Git/история публикации и `.env`/секреты вне скоупа — аудит
оценивает correctness, архитектуру, runtime, тесты и инфраструктуру
исполнения. Рабочее допущение: ближайший milestone — loopback-only
single-user; всё, что нужно только для внешнего доступа, вынесено в
отдельный gate-раздел и не конкурирует с correctness-приоритетами.

**Замер на 19 июля** (сьюты раздельно): API — 809 passed (31.7 c);
воркеры — ball 13, calibration 13, identity 19, jersey-ocr 11,
model-validation 25 passed / 2 skipped; frontend — 284 passed;
`vue-tsc` чистый.

## P0 — correctness

- **AUD-2. Ревизия в ответах player-action.**
  `scene_analysis_routes.py:116-141` возвращает локальный dict, а
  `player_action_commands.py:14,20` выбрасывает результат `scenes.put`
  (ревизию бампит только возвращаемый deepcopy,
  `scene_repository.py:211-222`). Воспроизведено: в БД revision 2, в ответе
  1 → следующий полный save падает 409. Команда обязана возвращать
  persisted Scene с новой ревизией (как ball-trajectory путь); API-тест
  обязан сравнивать response revision с ревизией в БД.
- **AUD-3. Ручное перемещение игрока тихо теряется.**
  `useEditorCompositionContext.ts:151-189` пишет `track.keyframes` и ставит
  «Unsaved changes», но `compactSceneWrite` (`lib/api/scenes.ts:24-27`)
  вырезает keyframes из PUT-тела, а гидрация перестраивает их из
  артефактов — правка исчезает после сохранения. **Решение принято:**
  функция нужна продукту → dedicated track-trajectory command/endpoint с
  публикацией артефакта и CAS, зеркально уже обкатанному пути
  `PUT /ball-trajectory` → `publish_ball_trajectory_artifact` →
  `scenes.put`. Сохранять dense keyframes через общий Scene PUT нельзя.
- **SEC-3. ffmpeg/ffprobe без таймаутов.** `video_ffmpeg.py:17-18`
  (`run_media_command`: probe/shots/proxy/poster/frames) и плотная выборка
  ball-frames — один битый файл вешает pipeline-runner навсегда. Единого
  `timeout=max_video_duration` недостаточно: нужны operation-specific
  бюджеты — короткий probe-таймаут и отдельные decode/transcode-бюджеты
  от длительности, операции и допустимого realtime factor;
  `TimeoutExpired` → asset failed с явной стадией.

## 1. Провайдеры и статистика матчей/игроков

**Принятый порядок: два последовательных slice.** Сначала correctness
provider boundary, затем статистика. Stats-числа поверх частичных
снапшотов, невалидных ID и потерянного sync status дали бы уверенно
выглядящие, но неверные данные — против политики проекта «никаких тихих
подмен».

### Slice A — provider correctness (P1)

- **PROV-17. Runtime EventBundle validator (fail-closed).** Проверять
  `source == event.provider`, принадлежность player/team, ссылки
  lineup/substitutions, согласованность `roster_quality` с фактическими
  строками. Отдельные provider-тесты покрывают часть инвариантов, но
  тест ловит только известного провайдера в CI — production-валидатор
  ловит любого будущего в рантайме. Сюда же — инвариант
  `players[].team_id == event.home.id/event.away.id`: нарушение сейчас
  не полностью тихое (часть строк отбрасывается с warning), но граф
  остаётся частичным, поэтому fail-closed валидация обязательна.
- **PROV-16. Public sync status теряется.** Canonical snapshot вычисляет
  `ready/partial/unavailable`, но public view любого external snapshot
  возвращает `synced`; `rosterQuality` не входит в public DTO — UI может
  показать неполный/пустой состав как успешно синхронизированный.
- **Section-status для partial bundle (PROV-7 + PROV-9).** Сейчас два
  противоположных дефекта: thesportsdb — всё-или-ничего (один 429 из трёх
  параллельных запросов валит bundle, `thesportsdb_provider.py:80-88`,
  gather без `return_exceptions`); api-football — retryable-отказы
  тихо деградируются в строки warnings без code/retryable
  (`match_contracts.py:100-109`) и персистятся как снапшоты с пустым
  ростером. Свести оба к общей per-section модели статуса, отличающей
  «квоту/retryable» от «данных нет».
- **Минуты: `addedTime` не заполняется.** Поле доведено до
  public-контрактов и фронтенда, но `canonicalize_event_bundle` его
  никогда не пишет — 45+3 становится 48-й минутой. P1 correctness: влияет
  на привязку событий к видео и participation evidence, а не только на
  подпись в UI. Прокинуть из маппингов провайдеров.
- **ID-неймспейс synthetic-идентификаторов.** Проблема не ограничена
  TheSportsDB: index-based fallback-id (`unknown`, `home/away`,
  `unknown-{order}`) потенциально коллидируют и у API-Football. Runtime
  namespace обязан включать provider + event/match identity.
- **Дедуп матча (NEW-3 остаток).** Один реальный матч из api-football,
  thesportsdb-fallback или ручного импорта — три несвязанных MatchRow;
  смена провайдера перехэширует roster-id и отвязывает project people.
  Провайдер-нейтральная идентичность (home/away/date либо резолюция через
  external references) с миграцией биндингов.

### Slice B — статистика (P1, после Slice A)

- **PROV-1.** Моделей статистики нет нигде (grep по
  `possession/team_statistics/player_statistics/MatchStats` — ноль;
  `match_contracts.py`, `canonical_match.py:227-230`,
  `types/match.ts:148-152`). Нужны версионируемые блоки
  `ExternalTeamMatchStats` (владение, удары, xG, угловые, фолы, сейвы) и
  `ExternalPlayerMatchStats` (минуты, голы, ассисты, карточки, рейтинг) +
  дескриптор покрытия по образцу roster_quality.
- **PROV-8.** Добавить вызовы `fixtures/statistics` и `fixtures/players`
  API-Football; сначала читать embedded-блоки fixture-by-id (паттерн уже
  есть — бережёт free-plan квоту).
- **Participation evidence.** `ParticipationEvidence` принимается и
  скорится (`roster_identity_contract.py:94-219`,
  `roster_identity_scoring.py:237-312`), но production-код его не строит.
  Таймлайн снапшота (карточки, замены) — готовая точка посадки
  player-stats в резолвер.

### P2 провайдерского слоя

- **Транспорты.** Подтверждено для обоих: unbounded in-process кэш
  (докстринги заявляют «bounded» — bounded только Redis через setex),
  новый `httpx.AsyncClient` на каждый запрос без retry/backoff, нет
  явного lifecycle Redis-клиента (без socket-таймаутов, без close).
  Credential scope в кэш-ключах уже есть у обоих
  (`thesportsdb_transport.py:29-30,85`); расхождение схем ключей
  (`sportsdb:` vs `match-data:api-football:`) — косметика, уйдёт при
  выделении общих примитивов. Границы выделения: общие только HTTP/cache
  primitives; provider-specific mapping и ошибки не обобщать.
- **PROV-15. Контрактный сьют.** Один параметризованный сьют для каждого
  зарегистрированного провайдера (source==provider.id, строковые id,
  связность lineup/substitutions, словарь reasons, warnings на пустых
  блоках, коды ошибок) — дополнение к PROV-17, не замена.
- **PROV-10. Registry DI.** Import-time singleton (`registry.py:146`) с
  замороженным снимком settings; DI-конструктор используют только тесты.
  Testability/lifecycle-долг — не опережает ошибки данных Slice A.
- **PROV-18. Search candidate persistence.** Кандидаты сохраняются
  по-одному в отдельных транзакциях, expiration отсутствует, хотя 404
  говорит «candidate expired». Batch-upsert + явная retention/expiry
  policy либо убрать вводящую в заблуждение семантику.
- **Timeline type.** Закрытый словарь есть на canonical-слое
  (`canonical_match.py:31-47`), на транспортном — свободная строка с
  тройной нормализацией. Единый модуль нормализации + закрытое поле рядом
  с сырым.
- **Roster-quality.** Политика качества считается в 3–4 местах с разными
  правилами (provider-ветка из нейтрального слоя уже убрана). Один
  нейтральный модуль с декларируемыми provider limits.
- **Search UX / capabilities.** Free-text q-путь без fallback, hardcoded
  «Spain vs Belgium» в UI; `capabilities` — мёртвые байт-идентичные
  кортежи: либо удалить, либо сделать несущими (team-pair-search vs
  free-text) и гейтить UI.
- **Manual-import версия.** Optional server-поля старые payload не ломают
  (исходная формулировка была жёстче факта), но явная schema version нужна
  для миграции семантики и обязательных полей.
- **Тесты транспортов.** Не исполняются: 429→rate-limit+retryable, 5xx,
  сетевые ошибки, Redis-паттерны (все тесты строят `redis_url=None`);
  у thesportsdb нет регрессии на утечку ключа (ключ в base_url — самый
  рискованный путь); у api-football непокрыты 401/403-ветки.

## 2. PostgreSQL и транзакции (P1)

- **SRV-4. Insert-гонка `put()`.** `FOR UPDATE` по отсутствующей строке
  на Postgres не лочит; `IntegrityError` проигравшего re-raise'ится как
  500 (`scene_repository.py:184-198, 217-219`). В 409 преобразовывать
  только duplicate-key race, не любой `IntegrityError`: предпочтителен
  atomic insert/on-conflict либо узкая проверка constraint + concurrency
  test на PostgreSQL.
- **SRV-3. PostgreSQL-lane в pytest.** CI гоняет pytest только на SQLite
  (alembic-джоба на Postgres проверяет лишь DDL). В lane в первую очередь:
  queue/claim/lease/recovery, Scene CAS, cancellation, insert race —
  транзакционные сьюты важнее общего прогона всех domain-тестов на двух
  СУБД.
- **REC-1 остаток (fix уже в дереве).** Poison-head recovery закрыт:
  dense fence проверяет только атомарный `claim_reconstruction_run`,
  poison job → `invalid`, lease удаляется, следующий queued остаётся
  единственным recoverable. Остаток: (a) регрессия пока SQLite и зовёт
  worker напрямую — нужен PostgreSQL monitor → child → invalid head →
  next job integration test; (b) `reconstruction_job.main()` всегда
  возвращает 0 (проверено: результат `reconstruct_scene_by_id`
  выбрасывается) — supervisor не различает stale skip и invalid claim.
- **`updated_at`.** SQLite отдаёт naive-строки, Postgres — с офсетом
  (`scene_repository.py:86-90`, `video_store.py:32`) — нормализовать в
  UTC при сериализации.

## 3. Runtime и наблюдаемость

- **Health-probe split (P1).** Контейнерный healthcheck (5 c) не должен
  зависеть от четырёх последовательных model-readiness проб (~8 c
  worst-case). Разделить дешёвый liveness и cached/parallel dependency
  readiness.
- **kind=demo эвристика (P1/P2 correctness).** `scene_document.py:124`
  матчит «smoke» в заголовке ДО классификации multiPass/segment —
  активная ошибка классификации пользовательских данных. Удалить
  текстовую эвристику; demo создавать только явной fixture/factory
  командой.
- **Reconstruction smoke в CI (P1).** Poison-head прошёл unit-сьюты
  именно из-за отсутствия monitor/process/PostgreSQL сценария. В каждый
  PR — маленький детерминированный клип + fake/lightweight worker mode;
  real models — nightly.
- **Логирование/метрики (P2).** Три module-logger'а на всё приложение,
  без конфигурации и структуры; request-id/метрик/трейсинга нет.
- **Поллинг runs (P2).** Hot-path опрос analysis-runs возвращает всю
  историю без limit/pruning — ограничить активными + N последних.
- **Диск (P2).** Суммарное потребление диска загрузками не ограничено
  (конкуренция ffmpeg уже ограничена очередью pipeline-runner).

## 4. Производительность (P2, сначала измерение)

- **Целевой бенчмарк — предусловие.** До любой параллелизации
  зафиксировать hardware + эталонный клип + latency budget (допустимый
  realtime factor). Фазовые тайминги снимать существующей телеметрией
  runs (`analysis_run_telemetry.py`, `quality_benchmark_*`), не отдельным
  профилировщиком.
- **PERF-5 как гипотеза.** Calibration keypoints и ball tiles уже
  батчатся; per-frame остаётся person inference. Порядок: измерить →
  батчить person/multi-frame ball → только затем перекрывать независимые
  фазы (overlap может ухудшить latency из-за contention).
- **PERF-3.** Кэш calibration-worker процесс-локален, не переживает
  рестарт (TTL 1 ч), тёплые прогоны заново гоняют байты кадров.
  Диск-персист по SHA кадра + hashes-only precheck.
- **PERF-8.** `interpolateKeyframes` — линейный скан, дважды на трек за
  кадр; ball-трек до ~1 500 точек. Сначала бинарный поиск
  (детерминированный fix); stateful cursor — только если profiler покажет
  остаточную цену и lifecycle курсора однозначен.
- **FE-5.** Watcher сериализует frameAnalysis, selected path — dense
  keyframes. Передавать revision/hash артефакта либо stable computed
  signature; новых deep-watcher'ов не вводить.
- **Inline-props инспектора.** `selection/analysis/matchView/controllers/
  commands` создаются инлайн — стабильные typed context objects/computed;
  при этом `currentTime` должен обновлять только реально зависящие от
  него части инспектора.

## 5. Тесты и инструменты (P2)

- **Coverage.** pytest-cov (`--cov-fail-under` от базлайна) +
  `@vitest/coverage-v8` в CI.
- **mypy/ruff.** После распила на 341 модуль межмодульные контракты —
  главный риск; AST-тесты архитектуры типы не проверяют. ruff + mypy
  (нестрого) в api-джобу.
- **Фронтенд-гэпы.** VideoIngestDrawer и CalibrationQaPanel без тестов;
  three-viewport слои проверяются только как строки в
  `appArchitecture.test.ts` — нужны поведенческие юниты на счётчики/
  позиции объектов.

## 6. Инфраструктура (P2)

- **Веса моделей + quickstart.** Свежий клон падает на сборке
  calibration-worker (нет pnl_SV_kp/lines) и на ball-detection (нет .pt);
  ball-worker бинд-маунтит `hrnet.py` из git-игнорируемого
  `.references/`. Стандартизировать на паттерне identity-worker
  (fetch + SHA-256 за build-arg), `hrnet.py` завендорить (MIT), секция
  «Model weights» в README.
- **Worker-транспорт — без generic manager.** Validators семантически
  различны и остаются capability-specific. Допустим только общий
  низкоуровневый HTTP multipart/readiness transport после стабилизации
  одинакового lifecycle; versioned model validation и candidate policy в
  общий helper не переносить.
- **Транзакционный helper.** `begin_write_transaction`
  (`database_transaction.py`) уже используют scene/analysis_run/
  reconstruction_run/cancellation; прямые копии `BEGIN IMMEDIATE` остались
  в шести модулях (external_reference, pipeline_domain, project_match,
  project_identity, project_store, project_resource — проверено grep'ом).
  Мигрировать по одному с транзакционными тестами, не механической
  заменой без проверки lock order.
- **ORM-registry.** Модели в `database.py` и `project_models.py` склеены
  side-effect-импортами — сделать регистрацию явной.
- **Python-зависимости.** Дрейф пинов ball-worker против трёх остальных;
  lockfile отсутствует. uv-workspace + `uv.lock` (пины torch 1.13.1+cpu
  сохранить).
- **Task-runner.** Makefile/justfile (`setup`/`fetch-models`/`dev-api`/
  `dev-web`/`workers-up`/`verify`); AGENTS.md закрыл только
  policy-половину.
- **Keyword-only на швах фаз.** 11 позиционных аргументов с соседними
  одинаково типизированными параметрами — прежде всего identity/temporal
  calibration seams. Keyword-only — минимальный шаг; cohesive immutable
  DTO допустим, только если описывает одну capability, а не новый god
  context.
- **Строковые статус-коды.** Пять роут-модулей выводят HTTP-статус из
  текста сообщения — маппить по типам/кодам исключений.
- **Prefix-дисциплина.** `identity_review_routes.py` и
  `identity_decision_routes.py` хардкодят полные пути вместо префикса
  роутера.
- **Root-контейнеры.** Ни один Dockerfile не объявляет USER; purge-слой
  identity-worker в отдельном RUN не уменьшает образ.
- **SEC-6 (уточнено).** Отсутствует только title-constraint
  (`scene_contracts.py:12` при `String(240)` в БД) и лимит на общий
  Scene JSON; video upload уже ограничен nginx и streaming-проверкой API.

## 7. Документация (P3)

- Языковой раскол: 5 документов на русском, остальные и README — на
  английском; выбрать канонический язык или добавить резюме.
- Roadmap R4: маркеры done/pending у 4 из 5 пунктов; alignment-gated
  fusion давно реализован — перенести в «выполнено».
- PIPELINE_EDGE_CASES.md (неотменяемые инварианты!) и
  UI_INFORMATION_ARCHITECTURE.md не слинкованы из README.
- 12 из 14 признанных ограничений в TECHNICAL_DEBT.md без TD-XXX
  идентификаторов.

## Gate перед внешним доступом

Не входит в P0/P1, пока порты loopback-only; становится обязательным до
первого внешнего bind или недоверенного браузера:

- **Auth.** Ноль auth-dependency во всех 15 роут-модулях; ресурсы
  скоуплены проектом, но проекты глобальны. Минимум — глобальный
  shared-bearer через `FastAPI(dependencies=[...])`; полноценный
  multi-user — отдельный этап.
- **Cross-site mutation.** CORS запрещает чтение ответа, но не отправку
  simple request; multipart и body-less POST'ы preflight-free — защитный
  header или bearer.
- **TLS.** Терминатора в репо нет — Caddy/Traefik перед web.
- **CORS/квоты.** Внешние origin через env + документация; per-user
  квоты.

## Не планировать без evidence

Отклонено/отложено до конкретного паттерна и бенчмарка:

- **JSON→JSONB** — JSON-path запросов нет, индексируемые значения
  вынесены в колонки; возвращаться только с query pattern + benchmark.
- **Pool tuning** — QueuePool defaults; явные размеры — capacity tuning
  после измерения concurrency и лимитов Postgres.
- **NFS/symlink-протокол ball-frames** — риск реален только для будущего
  multi-host; текущий named volume его не требует.
- **Generic worker manager** — см. раздел 6.
- **MEDIA_ROOT startup-валидация** — превентивный hardening; Compose уже
  согласован на `/data/media`, воспроизведённых ошибок нет.
- **Бэкапы томов** — важно (payload — один JSON-blob, потеря тома
  невосстановима), но план зависит от решения по внешнему
  доступу/деплою; связать с gate-этапом.
- **Roboflow-путь калибровки как основной** — отклонено решением
  владельца (2026-07-19). PnLCalib сильнее по всем осям, кроме веса
  модели: опора на точки+линии+дуги вместо ≤32 вершин, полная модель
  камеры вместо прямой `findHomography` без отбраковки, temporal-граф
  и QA-гейты вместо покадровой гомографии. 32-keypoint детектор
  (`football-pitch-detection.pt`, метод `roboflow-field-keypoints`)
  остаётся только локальным fallback при недоступности воркера, с
  явным provenance; см. также CALIBRATION.md «Why not the other
  references as the primary backend».

## Решения и открытые вопросы

**Принято:**
1. Track trajectory — dedicated endpoint + immutable artifact + CAS
   (зеркало ball-trajectory пути), drag-UI не отключаем.
2. Статистика — два последовательных slice: сначала provider
   correctness, затем stats.
3. Обратная совместимость со старыми локальными данными не
   поддерживается — решение владельца (2026-07-19). Проект локальный:
   завершённые прогоны, дисковые кэши, артефакты и содержимое БД можно
   терять и пересобирать новым прогоном. Контрактные изменения — через
   bump schemaVersion/algorithm (старые записи кэшей становятся
   промахами), схема БД — разрушающими ревизиями без переноса данных,
   протоколы воркеров — заменой без поддержки старой версии. Инварианты
   одного прогона (fingerprint-fence, атомарная публикация,
   digest-нормализация queue↔publish) — это корректность, не
   совместимость; они остаются.

**Открыто:**
4. Внешний доступ: остаётся ли ближайший milestone loopback-only?
   Если да — gate-раздел не вытесняет correctness-приоритеты; если нет —
   gate обязателен до следующего запуска.
5. Performance target: зафиксировать hardware + эталонный клип + latency
   budget до параллелизации inference (см. раздел 4).

## Сводный порядок работ

| Приоритет | Что | Пункты |
|---|---|---|
| P0 | Persisted revision в player-action; dedicated track-trajectory endpoint+artifact; operation-specific ffmpeg/ffprobe бюджеты | AUD-2, AUD-3, SEC-3 |
| P1 | PostgreSQL транзакционные риски: insert race (narrow on-conflict), queue/claim/lease/recovery+CAS+cancellation lane, monitor→child→next-job smoke, exit-code observability | SRV-4, SRV-3, REC-1, CI-6 |
| P1 | Provider correctness (Slice A): EventBundle validator, честный sync/roster quality, section-status, `addedTime`, namespaced IDs, neutral match identity | PROV-16, PROV-17, PROV-7+9, ID, дедуп |
| P1 | Статистика (Slice B, после A): stats-модели, API-Football statistics endpoints, participation evidence | PROV-1, PROV-8 |
| P1 | Liveness/readiness split; убрать `smoke`-эвристику | раздел 3 |
| P2 | Transport lifecycle/retry/cache bounds; contract suite; registry DI; manual schema version; timeline type; roster-quality модуль | раздел 1 (P2) |
| P2 | Измерение → батчинг person/ball → overlap фаз; interpolation binary search; stable signatures/props; calibration disk cache | раздел 4 |
| P2 | Coverage, ruff/mypy, поведенческие фронтенд-тесты; веса моделей/quickstart; миграция транзакционного helper'а | разделы 5–6 |
| Gate | Auth, cross-site protection, TLS, внешние CORS/квоты, бэкапы | gate-раздел |
| Без evidence | JSONB, pool tuning, NFS protocol, generic worker manager | отдельный раздел |

# Football data sources for Replay Studio

Актуальность исследования: 17 июля 2026 года. Цены ниже — публичные цены без налогов; перед коммерческим запуском их нужно перепроверить у поставщика. Этот документ разделяет три разных класса источников: match metadata API, tracking/event datasets и video discovery. Они решают разные задачи и не заменяют друг друга.

## Краткое решение

Для текущего некоммерческого прототипа основной источник состава —
API-Football, а интегрированный TheSportsDB остаётся альтернативным серверным
адаптером. API-Football даёт полный match-day lineup и события на бесплатном
тарифе, тогда как TheSportsDB free может возвращать усечённые данные. Public
project UI не передаёт provider и не видит upstream IDs: он получает opaque
match-candidate ID, а сервер разрешает его внутри provider namespace и сохраняет
provider-neutral snapshot. Для проверки координат и алгоритмов используем
открытые StatsBomb, SkillCorner и Metrica офлайн. Sportmonks имеет наибольшую
потенциальную ценность среди доступных коммерческих API благодаря
`ballCoordinates`, но это данные только для выбранных матчей/соревнований и не
замена реконструкции из нашего видео.

Приоритеты:

1. API-Football — использовать настроенным по умолчанию адаптером для полного состава и таймлайна.
2. TheSportsDB — оставить серверной альтернативой за тем же provider-neutral match contract, не протаскивая его ID в project UI.
3. StatsBomb Open Data — использовать для нормализации типов событий и тестовых сценариев.
4. SkillCorner Open Data + Metrica Sample Data — использовать как ground truth для офлайн-метрик координат и трекинга.
5. Sportmonks — проверить trial на одном поддерживаемом матче, когда понадобится привязка мяча/события по координатам.
6. ScoreBat — только для обнаружения и легального embed хайлайтов; не подавать embed как источник кадров для CV.
7. Sportradar/Opta — рассматривать только при коммерциализации и согласованном бюджете.

## Как получить полный состав конкретного матча

Под «полным составом» здесь понимаются обе команды, 11 стартовых игроков, скамейка, номера, позиции и устойчивые provider player IDs. Сезонный squad не заменяет match-day lineup: в нём нет достоверного признака, кто стартовал или находился на скамейке именно в этом матче.

| Приоритет | Источник и endpoint | Что реально получаем | Цена / ограничение | Решение для Replay Studio |
|---|---|---|---|---|
| 1 | [API-Football `/fixtures/lineups`](https://api-sports.io/documentation/football/v3) | `startXI`, `substitutes`, player ID, номер, позиция, formation и grid для обеих команд; `/fixtures/events` даёт замены, голы, карточки и ассисты | Free: 100 запросов/день, все endpoints, но доступные сезоны ограничены. [World Cup 2026 поддерживается](https://www.api-football.com/news/post/fifa-world-cup-2026-guide-to-using-data-with-api-sports); полноту конкретного fixture всё равно проверяем по фактическому ответу | Рекомендуемый primary provider для MVP |
| 2 | [TheSportsDB `lookuplineup`](https://www.thesportsdb.com/documentation) | Тот же формат, который уже нормализует backend; Premium повышает лимит ответа с 5 до 100 строк | $90/год или $9/месяц для Single Developer. Crowd-sourced покрытие конкретного матча не гарантировано | Явно выбираемая альтернатива; нельзя использовать с API-Football fixture ID |
| 3 | [Sportmonks `include=lineups`](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/lineups) | Стартеры (`type_id=11`), скамейка (`12`), номера, позиции и formation field; богатые связанные события | От €29/месяц за 5 лиг, 14-дневный trial; история старше трёх сезонов — add-on | Сильный production-кандидат после MVP |
| 4 | [football-data.org Match Resource](https://docs.football-data.org/general/v4/match.html) | Отдельные `lineup` и `bench`, IDs, позиции, номера, formation, substitutions | Line-ups & Subs начинаются с Free + Deep Data за €29/месяц; обычный free их не включает | Простой контракт, но меньше деталей, чем у API-Football |
| Offline | [StatsBomb Open Data](https://github.com/hudl/open-data) | Полные lineup JSON и детальные events для выбранных опубликованных матчей | Бесплатно для исследования с атрибуцией; не гарантирует текущий World Cup | Бенчмарки, тесты адаптера и action schema, не live provider |
| Enterprise | [Sportradar](https://developer.sportradar.com/soccer/reference/soccer-extended-sport-event-lineups), Stats Perform / Opta | Подтверждённые lineups, глубокий timeline; у отдельных продуктов — готовые XY/player tracking данные | Trial у Sportradar, production по индивидуальному контракту; Opta — custom quote | Имеет смысл при коммерциализации или отказе от собственного tracking |

Практическая схема: API-Football является configured primary, а TheSportsDB —
отдельным адаптером. В `match_snapshots` сохраняются нормализованные event,
teams, roster, lineup, timeline, substitutions, roster quality, warnings и
`fetchedAt`; `external_references` и private snapshot columns хранят provider
mapping. Scene routes не являются источником match writes. Raw upstream JSON
целиком пока не сохраняется; это отдельный audit/diagnostics
этап. Повторная синхронизация создаёт новый snapshot и не должна перетирать
ручные назначения пользователя. Перед допуском состава к автоматической
идентификации реализация оценивает фактическую полноту обеих команд и сохраняет
причины неполноты.

## Выбор провайдера и ключи

Backend регистрирует оба адаптера одновременно. `MATCH_DATA_PROVIDER` задаёт
адаптер для project-scoped `/api/projects/{projectId}/match/search`; ключи
остаются на сервере и никогда не передаются во Vue:

```dotenv
MATCH_DATA_PROVIDER=api-football
API_FOOTBALL_API_KEY=your-dashboard-key
API_FOOTBALL_BASE_URL=https://v3.football.api-sports.io

# Доступны одновременно с API-Football
SPORTSDB_API_KEY=123
SPORTSDB_BASE_URL=https://www.thesportsdb.com/api/v1/json
```

Допустимые значения `MATCH_DATA_PROVIDER`: `api-football` и `thesportsdb`.
Неизвестное значение — ошибка конфигурации. Пустой `API_FOOTBALL_API_KEY`
делает API-Football недоступным. `GET /api/health` возвращает состояние
адаптеров без ключей. Match search сохраняет временный private candidate
reference; select возвращает его opaque ID, а server-side refresh использует
provider/external event ID текущего snapshot. Никакой ID не пробуется у другого
provider автоматически.

### Контракт API-Football v3

| Операция Replay Studio | Официальный endpoint | Основные поля для нормализации |
|---|---|---|
| Ручная проверка доступа | `GET /status` | `subscription.plan/active`, `requests.current/limit_day`; запрос не расходует дневную квоту. Текущий registry этот endpoint не вызывает |
| Планируемая проверка покрытия | `GET /leagues?id={league}&season={season}` | `seasons[].coverage.fixtures.events/lineups`; текущий adapter оценивает полноту фактического fixture/lineup, но не вызывает `leagues` |
| Матчи дня | `GET /fixtures?date=YYYY-MM-DD` | `fixture.id/date/status`, `league`, `teams`, `goals`, `score` |
| Поиск пары команд | найти team IDs, затем `GET /fixtures/headtohead?h2h={a}-{b}` и фильтровать локально | те же fixture/team IDs; текстовые названия сами по себе не являются устойчивыми ключами |
| Полный match-day состав | `GET /fixtures/lineups?fixture={id}` | Upstream также даёт цвета формы и coach; текущий normalized contract сохраняет team, formation, `startXI`/`substitutes` и `id/name/number/pos/grid` игрока |
| Таймлайн | `GET /fixtures/events?fixture={id}` | `time.elapsed/extra`, `team`, `player`, `assist`, `type`, `detail`, `comments`; в `subst` поле `player` — уходящий, `assist` — входящий игрок |
| Обогащённый матч | `GET /fixtures?id={id}` | adapter использует встроенные `events` и `lineups` одним вызовом; отдельные endpoints вызываются только если соответствующего ключа нет в fixture response |

API-Football использует base URL
`https://v3.football.api-sports.io`, принимает GET-запросы и ожидает ключ в
заголовке `x-apisports-key`. Lineups обычно появляются за 20–40 минут до
матча, но для некоторых соревнований могут прийти только после игры. Endpoint
lineups обновляется примерно раз в 15 минут, events — раз в 15 секунд.

Free-план даёт 100 запросов в день и 10 в минуту. Текущий adapter кеширует
успешные ответы, явно преобразует HTTP 429 в retryable provider error и
возвращает `Retry-After: 60`, но пока не считывает и не сохраняет
`x-ratelimit-requests-remaining` и `X-RateLimit-Remaining`; это остаётся
диагностическим техдолгом. `configured`/`available` в
`GET /api/health` сейчас означают наличие серверного ключа, а не
успешную live-проверку `/status`. Для World Cup 2026 официальные
идентификаторы — `league=1`, `season=2026`; заявленное coverage включает
lineups и events, но adapter принимает решение по фактически полученному
lineup и сохраняет причины неполноты в `rosterQuality` и `warnings`.

Источники: [API-Football v3 documentation](https://api-sports.io/documentation/football/v3),
[официальный World Cup 2026 guide](https://www.api-football.com/news/post/fifa-world-cup-2026-guide-to-using-data-with-api-sports),
[rate-limit policy](https://www.api-football.com/news/post/how-ratelimit-works),
[pricing](https://www.api-football.com/pricing) и
[TheSportsDB documentation](https://www.thesportsdb.com/documentation).

## Какие данные улучшают наш продукт

| Данные | Как используются | Какую проблему решают | Ограничение |
|---|---|---|---|
| Match ID, дата, турнир, команды | Привязка загруженного видео к матчу | Убирает ручной ввод и неоднозначность названий | Не подтверждает, что конкретный кадр относится к событию |
| Состав и стартовые позиции | Список допустимых личностей для ручной метки | Сужает Player A/B до 22–30 реальных кандидатов | Не идентифицирует человека по изображению |
| Замены | Ограничение кандидатов по игровому времени | Не даёт назначить ушедшего игрока кадру после замены | Нужен корректный clock alignment |
| Голы, карточки, замены, VAR | Якоря для группировки 1-A/1-B/повторов | Помогает назвать и упорядочить моменты | API-время обычно минутное и недостаточно точно для кадра |
| События с timestamp до секунды | Поиск события в broadcast clock | Уменьшает область видео для CV-анализа | Не все API дают секунды/добавленное время одинаково |
| События с x/y | Проверка стороны поля и приблизительной зоны | Отсекает зеркальную или неверную калибровку | Координаты события — не непрерывный tracking |
| Formation/lineup position | Семантическая проверка треков | Помогает различить защитников, форвардов, вратаря | Формация не даёт точное положение в моменте |
| Ball coordinates | Якоря траектории мяча и события | Снижает число ложных кандидатов мяча | Редкая премиальная функция, ограниченное покрытие |
| Player tracking x/y | Ground truth/validation, иногда готовый replay | Проверяет реконструкцию и multi-pass fusion | Почти всегда лицензируется отдельно и не совпадает с нашим видео |
| Referee/officials | Разрешённая ручная роль и role classifier | Судья больше не загрязняет командные кластеры | Состав судей не помогает визуально различать их |
| Video highlight URL/embed | Обнаружение момента и preview | Упрощает пользователю поиск клипа | Embed обычно нельзя скачивать и анализировать как raw video |

## Сравнение сервисов

| Источник | Полезные данные | Плюсы | Минусы | Стоимость и free | Где применить у нас |
|---|---|---|---|---|---|
| [TheSportsDB](https://www.thesportsdb.com/documentation) | Матчи, команды, игроки, lineup, timeline, иногда YouTube highlights | Уже интегрирован; простой API; есть стабильный free key | На free ответы могут быть усечены; неоднородное покрытие; нет tracking | [Free](https://www.thesportsdb.com/docs_pricing.php?billing=annual): $0, до 30 req/min, limited data. Single Developer: $90/год, до 100 req/min; Small Business: $200/год | Альтернативный каталог и ручная привязка матча; сохранять нормализованный snapshot в проекте |
| [API-Football](https://api-sports.io/documentation/football/v3) | Fixtures, lineups, players, injuries, events, statistics, odds | Много лиг; все endpoints доступны на free для недавних сезонов; полный match-day lineup | 100 запросов/день быстро расходуются; нет полного player tracking; условия redistribution нужно проверять | [Free](https://www.api-football.com/pricing): 100 req/day. Pro $19/мес — 7,500/day; Ultra $29 — 75,000/day; Mega $39 — 150,000/day | Основной provider adapter для состава и таймлайна; сравнение с явно выбранным альтернативным источником |
| [football-data.org](https://www.football-data.org/pricing) | Fixtures, live scores, standings; на Deep Data — lineups, substitutions, scorers, cards, squads | Чистая модель данных; понятные лимиты; хороший календарь | Free даёт мало соревнований и задержанные данные; глубокие события платные; нет tracking | Free €0: 12 competitions, 10 calls/min. Live €12/мес; Deep Data €29/мес; Standard €49; Advanced €99; Pro €199 | Надёжный календарь/результат и проверка match identity; не основной источник реконструкции |
| [Sportmonks Football API](https://www.sportmonks.com/football-api/plans-pricing/) | Составы, события, статистика, referee, formations; для части матчей — нормализованные [`ballCoordinates`](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/ballcoordinates) с timestamp | Самый прямой коммерческий сигнал для проверки мяча и события | Координаты доступны не для всех лиг/матчей; trial требует карту; цена растёт с лигами; не raw tracking всех игроков | 14-day trial. Starter €29/мес (около €24 при annual) за 5 leagues; Growth €99; Pro €249; Enterprise — quote | Экспериментальная привязка мяча и секунд события; проверка зеркальности/зоны поля; не заменять CV |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | События, lineups и для выбранных игр 360 freeze-frame | Богатая event schema; бесплатные размеченные координаты; хороший offline fixture | Только выбранные соревнования; обязательная атрибуция; 360 — снимки вокруг события, не непрерывный broadcast tracking | Бесплатный open dataset; коммерческий продукт — [по запросу](https://statsbomb.com/what-we-do/soccer-data/360-2/) | Тесты нормализации событий; синтетические сцены; проверка расположения игроков в момент удара/паса |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | Broadcast tracking 10 FPS, player/ball x/y, identities, фазы и динамические события для 10 матчей A-League 2024/25 | Ближе всего к нашему single-camera кейсу; MIT; годится для численных метрик | Очень маленькое открытое покрытие; коммерческий полный продукт — quote; исходные broadcast-права отдельно | Open dataset бесплатно; коммерческий доступ — contact sales | Ground truth для ошибки позиции, пропусков, ID switches и multi-pass; regression benchmark |
| [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data) | Синхронизированные tracking + events, нормализованные координаты, поле 105×68 | Простой эталон формата; удобно тестировать проигрыватель и интерполяцию | 2–3 анонимизированных игры; не наш broadcast video; коммерческий доступ — quote | Sample бесплатно с атрибуцией | Контракт импорта tracking; тесты временной синхронизации и 3D player |
| [ScoreBat Video API](https://www.scorebat.com/video-api/docs/) | Официальные highlight embeds, thumbnails и match metadata | Быстрый легальный discovery; простой каталог | Даёт embeds, а не гарантированно скачиваемый raw video; watermark/ads на free; каждый view расходует credits | [Free/paid](https://www.scorebat.com/video-api/): free limited; Starter $69/мес — 5k credits; Standard $139 — 20k; Advanced $299 — 100k | «Найти хайлайт» и preview. Не использовать как CV input без отдельного права на видео |
| [Sportradar Soccer](https://developer.sportradar.com/soccer/docs/soccer-ig-api-basics) | Глубокие timelines, lineups, formations, статистика; Extended может включать event x/y | Широкое профессиональное покрытие и SLA | Production price не публична; строгая лицензия/redistribution; tracking может быть отдельным продуктом | [Trial](https://developer.sportradar.com/getting-started/docs/your-account): 30 дней, 1,000 requests за rolling 30 days, 1 QPS. Production — quote | Коммерческий fallback, live-event alignment, enterprise launch |
| [Stats Perform / Opta](https://www.statsperform.com/faqs/stats-perform-faqs-pricing-licensing/) | Профессиональные event, live, player/team stats, часть advanced/tracking продуктов | Отраслевой стандарт и очень глубокая семантика | Индивидуальная лицензия и цена; тяжёлая интеграция; не подходит для бесплатного MVP | Custom quote; поставщик отдельно обсуждает стартапы | Только коммерческий этап: high-confidence event binding и аналитика |
| [OpenLigaDB](https://openligadb.de/) | Community scores, fixtures, tables | Бесплатно, без auth, ODbL | Неполное/неоднородное покрытие; нет составов, точных событий и tracking | Бесплатно | Резервный поиск результата/матча для популярных европейских лиг |
| [SoccerNet](https://www.soccer-net.org/data) | Research datasets для action spotting и Game State Reconstruction | Бенчмарк, labels, метрики и baseline | Это dataset/research kit, не live API; видео скачивается по правилам SoccerNet/NDA; лицензии компонентов разные | Бесплатно для исследовательского использования по условиям датасета | Обучение/оценка CV, а не пользовательский match-data provider |

## Рекомендуемый provider-neutral контракт

Поля конкретного API нельзя протаскивать напрямую в UI и реконструкцию.
Нормальный project response использует только внутренние Replay Studio IDs:

```json
{
  "id": "match_opaque_internal_id",
  "revision": 3,
  "name": "Spain vs Belgium",
  "competition": "World Cup",
  "kickoffAt": "2026-07-14 19:00",
  "homeTeam": {
    "id": "team_internal_home",
    "name": "Spain"
  },
  "awayTeam": {
    "id": "team_internal_away",
    "name": "Belgium"
  },
  "roster": [
    {
      "id": "player_internal_id",
      "teamId": "team_internal_home",
      "name": "Player name",
      "number": "8",
      "position": "Midfielder",
      "role": "starter"
    }
  ],
  "events": [
    {
      "id": "event_internal_id",
      "kind": "goal",
      "minute": 69,
      "teamId": "team_internal_home",
      "playerId": "player_internal_id",
      "label": "Goal"
    }
  ],
  "sync": {
    "state": "synced",
    "stale": false,
    "warnings": []
  }
}
```

Обязательные правила:

- хранить provider, исходные ID и `fetchedAt` только в
  `match_snapshots`/`external_references` и explicit integration diagnostics;
- возвращать Vue только internal match/team/player/event IDs;
- нормализовать игровое время с периодом, добавленным временем и признаком точности;
- не путать `event position` с непрерывной позицией игрока/мяча;
- нормализованный immutable snapshot хранится только на уровне проекта и
  передаётся reconstruction как версионированный input;
- raw upstream response целиком пока не сохраняется и остаётся отдельным
  диагностическим/retention решением;
- объединять два провайдера только после entity resolution по командам, kickoff и игрокам;
- ручная метка пользователя всегда имеет больший приоритет, чем API и модель;
- не менять реконструированный трек и не запускать rebuild только из-за refresh
  имени или состава.

## Как API должен помогать CV

### Перед анализом

- Из состава строим допустимые роли и кандидатов личности.
- Из формы команд задаём prior для кластеризации цветов, но не жёсткую метку.
- Вратари получают отдельный role prior, чтобы третий цвет формы не стал третьей командой.
- По минуте события ищем окно broadcast video, но оставляем допуск на montage/replay/clock drift.

### Во время анализа

- Событие с координатой задаёт мягкую область интереса, а не принудительную позицию.
- Замены исключают невозможные личности только после надёжной синхронизации часов.
- Многопроходный анализ разных ракурсов объединяется по визуальному времени/движению; API event ID служит меткой группы, а не алгоритмом синхронизации кадров.

### После анализа

- Сравниваем сторону/зону события и траекторию; конфликт показываем пользователю.
- Предлагаем имена только для треков с достаточным сочетанием roster prior, номера, цвета и временной доступности.
- Экспортируем provenance: что пришло из видео, что из API, а что исправил пользователь.

## План интеграции

### Этап 1 — текущий MVP

- Общий `MatchDataProvider` и adapters API-Football/TheSportsDB реализованы;
  public project UI работает только с canonical IDs и configured adapter.
- API-Football key остаётся серверной переменной окружения; полный lineup и
  timeline нормализуются в immutable project snapshot.
- Успешные upstream ответы кешируются в memory/Redis; canonical snapshot,
  external references и current pointer сохраняются в БД. Public UI показывает
  roster completeness/warnings, но не provider IDs.
- Provider provenance доступна через отдельный integration-diagnostics route;
  raw upstream JSON retention и provenance каждого identity suggestion остаются
  отдельными решениями.
- Refresh не запускает реконструкцию автоматически и не превращает canonical
  match в второй scene-owned source of truth.

### Этап 2 — оценка реконструкции

- Импортировать открытые SkillCorner/Metrica матчи в тот же tracking contract.
- Считать positional error, recall, false positives, ID switches и role/team accuracy.
- Использовать StatsBomb 360 как набор проверок «состав игроков вокруг события».

### Этап 3 — координатные данные

- Через Sportmonks trial проверить реальные `ballCoordinates` и точность timestamp.
- Реализовать мягкую fusion-проверку, не подменяющую визуальный результат.
- Сравнить стоимость с коммерческими Sportradar/Opta только после появления требований к лигам и SLA.

## Юридические и продуктовые ограничения

- Право получить JSON не означает право публично перераспространять его целиком.
- Права на match data и права на видео — разные лицензии.
- YouTube/ScoreBat embed нельзя автоматически считать разрешением скачать ролик для обработки.
- Открытые datasets требуют соблюдения своей лицензии и атрибуции; их нельзя автоматически смешивать с пользовательским видео как публичный датасет.
- Для текущего enthusiast-проекта достаточно бесплатных источников, но перед монетизацией нужно отдельно проверить Ultralytics AGPL, SoccerNet GPL/data terms и каждую provider license.

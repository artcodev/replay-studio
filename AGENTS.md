# Replay Studio project instructions

## Architecture and legacy policy

The canonical, simplest, and most efficient architecture wins over backward
compatibility with earlier prototypes.

- Legacy database layouts, global scene/video routes, embedded
  `SceneDocument` result fields, startup backfills, compatibility projections,
  dual writes, and legacy-only tests are not product requirements unless the
  user explicitly says otherwise.
- When a canonical replacement exists, migrate the current first-party
  consumers and delete the superseded path. Do not keep dormant adapters or
  fallback branches "just in case".
- Breaking internal API and schema changes are acceptable. A development
  database may be reset or data may be re-imported instead of maintaining a
  permanent migration chain. Never perform a destructive reset without the
  user's explicit approval.
- Transitional code is allowed only when required to complete the active
  cutover safely. It must have a named removal condition and must not become a
  second writable source of truth.
- Do not confuse legacy compatibility fallbacks with intentional product
  fallbacks. A current fallback is allowed only when it solves a real product
  edge case, is observable, preserves data/quality invariants, and is tested.
  Silent accuracy degradation is not an acceptable fallback.
- Tests should protect the target architecture and current product behavior,
  not preserve obsolete implementation contracts.

The normative architecture and ownership boundaries are documented in
`docs/ARCHITECTURE.md`.

## Code structure policy

- Framework composition roots (`app.main`, Vue `App.vue`) assemble modules;
  they do not own feature workflows or persistence logic.
- Split modules by independent reasons to change, not by arbitrary line count.
  A cohesive numerical algorithm may be long; a file mixing HTTP, storage,
  orchestration, and domain rules is a god file even when it is shorter.
- HTTP routers own transport translation only. Repositories own data access and
  atomic writes. Application services coordinate repositories. Domain modules
  must not import HTTP composition roots.
- Do not replace one god file with one equally broad `utils`, `helpers`,
  `manager`, global store, or mega-composable. New module names must describe
  the capability they own.
- Transport DTOs are strict and capability-specific. Do not recreate a generic
  `schemas` barrel, silently accept unknown fields, or put runtime services in
  contract modules.
- Third-party providers keep transport/cache, pure normalization, and provider
  orchestration in separate owners. Provider choice is an explicit argument;
  task-local or global override state is not a routing mechanism.
- Model integrations expose a small provider-neutral contract and a dedicated
  factory. Local inference, remote/subprocess transports, wire codecs, and
  candidate-selection algorithms do not live in one aggregate detector file.
- Image/video payloads crossing a service boundary use multipart/binary media
  or immutable artifact references; never embed frame bytes as base64 in JSON
  or scheduler rows. A canonical versioned worker endpoint replaces and
  deletes its compatibility route in the same cutover.
- Worker responses are validated as strict versioned contracts. Unknown wire
  fields fail closed, and aggregate diagnostics live in an explicit result
  DTO rather than attributes attached to a `dict`/`list` subclass.
- Vue route pages compose a small number of explicit feature contexts.
  Cross-feature dependencies are typed arguments, never mutable callback
  registries, lifecycle-hook buses, or hidden service locators.
- Once a one-shot prototype cutover has completed, remove its runtime module
  and tests. Current Alembic migrations remain schema authority; retired data
  repair commands are not permanent product features.
- Tests patch the capability boundary they exercise, never symbols re-exported
  from a composition root.
- A domain edit and a persisted command are different capabilities. Do not add
  `persist`, `commit`, `dry_run`, or similar boolean mode switches to make one
  function serve both paths; expose an explicit draft/planner and an
  always-persist command instead.

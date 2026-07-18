# Project identity persistence

`ProjectPerson` is the durable identity boundary across segments and camera
angles. A reconstructed scene still owns its evidence-rich local
`canonicalPeople`; `ProjectPersonMembership` maps each
`(project, scene, canonicalPersonId)` onto one project identity.

This separation is important: detector/track IDs are only local labels. Equal
`person-1` values in different scenes are not evidence that the observations
belong to the same footballer.

## Persistence model

`project_people` stores:

- an internal project-person ID;
- an optional `rosterPersonId` from the provider-neutral canonical match;
- display name, team, role, jersey number and active/excluded state;
- the strongest retained identity confidence.

`project_person_memberships` stores:

- the project and project-person owner;
- scene ID and scene-local canonical-person ID;
- assignment source: `scene-local`, `accepted-roster`, or `explicit`;
- scene-local status, confidence and observation count.

Both tables use Replay Studio IDs. Provider names, fixture IDs and upstream
player IDs remain inside integration snapshots/references and never appear in
the public project-identity DTO.

## Automatic synchronization

After a reconstruction is atomically published as `ready`, terminal handling
calls `sync_project_identities_from_scene()`. The operation is idempotent and
records a compact result under `AnalysisRun.diagnostics.identitySync`.

A sync failure is observable but does not invalidate the accepted 3D scene or
rerun the expensive vision pipeline. Delivering the same terminal result again
can safely retry synchronization.

Automatic synchronization is deliberately conservative:

- a confirmed scene roster binding merges across scenes only when its value is
  present in the project's current provider-neutral canonical match snapshot;
- an internal reconstruction `externalPlayerId` that is absent from the
  canonical snapshot is counted in diagnostics and is not persisted as a
  roster binding;
- an existing membership is preserved across reconstruction rebuilds;
- an explicitly reassigned membership is authoritative and is never silently
  moved by automatic synchronization;
- equal local `canonicalPersonId` values in different scenes create separate
  anonymous project people unless stronger evidence exists;
- the only automatic cross-scene merge key is an accepted canonical
  `rosterPersonId`;
- a conflict between explicit ownership and a different roster binding fails
  closed instead of moving observations silently.

Anonymous project-person IDs are stable for the project/scene/local-person
tuple. Roster-backed project-person IDs are stable for the project and
canonical roster identity.

## Public API and UI

List the current identity graph:

```http
GET /api/projects/{projectId}/identities
```

The response includes each project person and all scene memberships. The
project **Identities** tab displays totals, roster linkage, team/role/status,
confidence and membership provenance. Each membership has a target
project-person selector and an explicit **Assign** action; no-op submission is
disabled and request failures stay visible without optimistic ownership drift.

Explicitly reassign one scene identity to an existing project person:

```http
POST /api/projects/{projectId}/identities/{projectPersonId}/memberships
Content-Type: application/json

{
  "sceneId": "scene-id",
  "scenePersonId": "canonical-person-id"
}
```

The server verifies that the target person and scene belong to the project,
sets `assignmentSource` to `explicit`, and removes the abandoned anonymous
project person only when it has no remaining memberships or roster binding.
Concurrent conflicting writes fail with a conflict response. The UI reloads the
identity graph after a successful assignment. This operation merges/reassigns
one scene membership; it does not split the underlying scene-local identity or
rewrite detector evidence.

## What this does not prove

Project persistence does not make ReID accurate. It records decisions and
prevents ID scope errors; cross-segment accuracy still requires a labelled
benchmark. The current automatic merge rule intentionally abstains unless a
canonical roster identity has been accepted. Appearance, jersey and alignment
evidence remain review inputs until their product thresholds are validated.

Multi-angle composition may add namespaced cross-view evidence only after
accepted time alignment and explicit roster or reliable jersey agreement.
Proximity alone never creates a project-person merge.

## Remaining work

- preview and explicit undo/history for project-membership reassignment;
- project-level split flow (scene-local identity split still belongs in the
  editor because it changes reconstruction evidence);
- stale-membership presentation when a source scene is superseded or deleted;
- benchmark IDF1, switches, fragmentation and abstention coverage across
  segments and camera angles;
- optional non-roster cross-segment proposals with strict review and no silent
  auto-merge.

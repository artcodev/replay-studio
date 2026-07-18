from __future__ import annotations

"""Passive in-memory state for one reconstructed person track."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackState:
    id: int
    points: list[dict] = field(default_factory=list)
    feature_sum: np.ndarray | None = None
    feature_count: int = 0
    last_frame: int = 0
    last_height: float = 0.0
    role: str | None = None
    annotation_ids: set[str] = field(default_factory=set)
    identity_tombstone_ids: set[str] = field(default_factory=set)
    roster_binding_annotation_ids: set[str] = field(default_factory=set)
    roster_binding_state: str | None = None
    manual_identity_owner_ids: set[str] = field(default_factory=set)
    manual_kind: str | None = None
    manual_label: str | None = None
    manual_semantic_key: tuple[int, str, int, str] | None = None
    manual_external_player_id: str | None = None
    source_tracklet_ids: set[str] = field(default_factory=set)
    reid_feature_sum: np.ndarray | None = None
    reid_feature_count: int = 0
    reid_observation_count: int = 0
    reid_observation_ids: set[str] = field(default_factory=set)
    reid_evidence_fingerprints: set[str] = field(default_factory=set)
    reid_duplicate_evidence_count: int = 0
    reid_samples: list[np.ndarray] = field(default_factory=list)
    reid_sample_candidates: list[dict] = field(default_factory=list)
    reid_selected_metadata: list[dict] = field(default_factory=list)
    canonical_person_id: str | None = None
    identity_status: str = "unresolved"
    identity_confidence: float | None = None
    identity_evidence: list[dict] = field(default_factory=list)
    identity_conflicts: list[dict] = field(default_factory=list)
    identity_group_id: str | None = None
    reid_role_votes: dict[str, float] = field(default_factory=dict)
    # Each manual split partition carries an explicit barrier value so a later
    # scene-document merge cannot silently join the partitions again.
    identity_split_partitions: dict[str, str] = field(default_factory=dict)

    @property
    def feature(self) -> np.ndarray:
        assert self.feature_sum is not None
        return self.feature_sum / max(1, self.feature_count)

    @property
    def local_tracklet_id(self) -> str:
        return f"tracklet-{self.id:04d}"

    @property
    def positive_annotation_ids(self) -> set[str]:
        """Manual annotations that positively assert this identity."""

        return self.annotation_ids - self.identity_tombstone_ids

    @property
    def reid_feature(self) -> np.ndarray | None:
        if self.reid_feature_sum is None or self.reid_feature_count <= 0:
            return None
        value = self.reid_feature_sum / self.reid_feature_count
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-8 else None


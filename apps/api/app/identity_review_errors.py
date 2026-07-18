"""Errors shared by identity-review query capabilities."""


class IdentityReviewError(RuntimeError):
    """Persisted review evidence cannot be resolved safely."""


class IdentityReviewArtifactUnavailableError(IdentityReviewError):
    """A referenced immutable artifact cannot satisfy the review contract."""


__all__ = ("IdentityReviewArtifactUnavailableError", "IdentityReviewError")

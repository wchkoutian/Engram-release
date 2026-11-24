"""The memory fusion policy, shared by training, inference and the checkpoint contract.

The conditioning volume a side branch learns to read is a function of *how* the memory was
fused, not just of what was written into it. Bounding the confidence a cell can report,
attenuating generated evidence written over a real observation, or refusing to average
geometrically disagreeing observations all change the distribution of the packed channels.
A checkpoint trained under one policy and evaluated under another is therefore reading
out-of-distribution conditioning, and nothing in the tensor shapes would catch it.

So the policy lives in one dataclass, is read from one set of config keys by both the
trainer and the eval runners, and is recorded in checkpoint metadata under a contract
name. ``validate_against`` turns a silent distribution mismatch into a startup error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


LEGACY_FUSION_CONTRACT = "unbounded_fusion_v0"
BOUNDED_PROVENANCE_FUSION_CONTRACT = "bounded_provenance_fusion_v1"
FUSION_CONTRACTS = (LEGACY_FUSION_CONTRACT, BOUNDED_PROVENANCE_FUSION_CONTRACT)


def _optional_float(value: Any) -> Optional[float]:
    """Parse a bound that may legitimately be absent.

    ``inf`` is accepted as a spelling of "unbounded" so a ``w_max`` sweep can name its
    control arm in the same units as the rest of the sweep.
    """
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "null", "inf", "+inf", "infinity"}:
            return None
        return float(token)
    value = float(value)
    return None if value == float("inf") else value


@dataclass(frozen=True)
class MemoryFusionPolicy:
    """How observations are combined in the voxel store.

    The defaults are the historical unbounded inverse-variance fusion, so an untouched
    config keeps producing the numbers it produced before.
    """

    w_max: Optional[float] = None
    conf_report_max: Optional[float] = None
    real_weight_scale: float = 1.0
    gen_into_real_scale: float = 1.0
    gen_into_real_budget: Optional[float] = None
    tau_geom: Optional[float] = None
    fuse_position: bool = True
    drop_on_layer_budget: bool = False
    immutable_real_memory: bool = False
    reserve_real_in_topk: bool = False

    @property
    def is_legacy(self) -> bool:
        default = MemoryFusionPolicy()
        return all(
            getattr(self, field.name) == getattr(default, field.name)
            for field in fields(self)
        )

    @property
    def contract(self) -> str:
        """Contract name implied by the settings, so it can never disagree with them."""
        return (
            LEGACY_FUSION_CONTRACT
            if self.is_legacy
            else BOUNDED_PROVENANCE_FUSION_CONTRACT
        )

    def memory_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for :class:`~engram.memory.EngramMemory`."""
        return asdict(self)

    def runner_kwargs(self) -> Dict[str, Any]:
        """Return policy values using the runner-compatible ``memory_*`` names."""
        renamed = {
            "immutable_real_memory": "memory_immutable_real",
            "reserve_real_in_topk": "memory_reserve_real_in_topk",
        }
        return {
            renamed.get(key, f"memory_{key}"): value
            for key, value in asdict(self).items()
        }

    def to_metadata(self) -> Dict[str, Any]:
        return {"contract": self.contract, **asdict(self)}

    @classmethod
    def from_metadata(cls, data: Optional[Dict[str, Any]]) -> "MemoryFusionPolicy":
        """Rebuild a policy from checkpoint metadata.

        Metadata predating this contract has no entry at all, which unambiguously means
        the legacy policy.
        """
        if not data:
            return cls()
        declared = data.get("contract", LEGACY_FUSION_CONTRACT)
        if declared not in FUSION_CONTRACTS:
            raise ValueError(
                f"unknown memory fusion contract {declared!r}; expected one of "
                f"{FUSION_CONTRACTS}"
            )
        known = {field.name for field in fields(cls)}
        policy = cls(**{key: value for key, value in data.items() if key in known})
        if policy.contract != declared:
            raise ValueError(
                f"memory fusion metadata declares {declared!r} but its settings imply "
                f"{policy.contract!r}"
            )
        return policy

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MemoryFusionPolicy":
        """Read the shared ``memory_*`` config keys used by training and evaluation."""
        return cls(
            w_max=_optional_float(cfg.get("memory_w_max")),
            conf_report_max=_optional_float(cfg.get("memory_conf_report_max")),
            real_weight_scale=float(cfg.get("memory_real_weight_scale", 1.0)),
            gen_into_real_scale=float(cfg.get("memory_gen_into_real_scale", 1.0)),
            gen_into_real_budget=_optional_float(cfg.get("memory_gen_into_real_budget")),
            tau_geom=_optional_float(cfg.get("memory_tau_geom")),
            fuse_position=bool(cfg.get("memory_fuse_position", True)),
            drop_on_layer_budget=bool(cfg.get("memory_drop_on_layer_budget", False)),
            immutable_real_memory=bool(cfg.get("memory_immutable_real", False)),
            reserve_real_in_topk=bool(cfg.get("memory_reserve_real_in_topk", False)),
        )

    def validate_against(self, other: "MemoryFusionPolicy", source: str) -> None:
        """Raise if ``other`` (usually the checkpoint's) differs from this policy."""
        if self == other:
            return
        differences = {
            field.name: (getattr(other, field.name), getattr(self, field.name))
            for field in fields(self)
            if getattr(other, field.name) != getattr(self, field.name)
        }
        raise ValueError(
            f"memory fusion policy differs from {source}: the side branch was trained to "
            f"read conditioning fused under {other.contract!r} but this run uses "
            f"{self.contract!r}. Differing fields (checkpoint -> active): {differences}"
        )

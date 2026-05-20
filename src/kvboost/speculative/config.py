# src/kvboost/speculative/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kvboost.streaming.config import StreamingConfig


SpeculativeMode = Literal["greedy", "sampling"]


@dataclass(slots=True)
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Speculative decoding uses a small "draft" model to propose ``draft_k``
    tokens per round, then verifies them in one batched forward pass on the
    target model. When acceptance rate is high (typical 0.3-0.5 on aligned
    model families) effective tok/s scales by ~1 + draft_k * acceptance_rate.

    The draft model loads through the same ``StreamingCausalLM`` pipeline as
    the target, but defaults to ``residency_mode="full_resident"`` because
    draft models are small enough to fit fully in VRAM.

    Modes
    -----
    greedy:
        Accept token i iff target.argmax(logits[i]) == draft_ids[i]. On first
        reject, the correction token is target.argmax at that position. With
        ``temperature=0`` baselines this produces bit-identical output.

    sampling:
        Reject sampling per Leviathan et al. 2023.
        accept_prob[i] = min(1, target_P[draft_id_i] / draft_P[draft_id_i]).
        On reject, sample from normalize(max(target_P - draft_P, 0)).
    """

    # Draft model
    draft_model_id: str
    draft_streaming_config: Optional["StreamingConfig"] = None
    draft_device: Optional[str] = None

    # Decoding
    draft_k: int = 5
    mode: SpeculativeMode = "greedy"
    temperature: float = 1.0

    # Debug toggles
    enable_kv_rollback: bool = True

    def validate(self) -> None:
        if not self.draft_model_id:
            raise ValueError("SpeculativeConfig.draft_model_id is required")
        if self.draft_k < 1:
            raise ValueError(
                f"SpeculativeConfig.draft_k must be >= 1, got {self.draft_k}"
            )
        if self.mode not in ("greedy", "sampling"):
            raise ValueError(
                f"SpeculativeConfig.mode must be 'greedy' or 'sampling', "
                f"got {self.mode!r}"
            )
        if self.mode == "sampling" and self.temperature <= 0:
            raise ValueError(
                "sampling mode requires temperature > 0, "
                f"got {self.temperature}"
            )

    def summary(self) -> str:
        return (
            f"SpeculativeConfig(draft={self.draft_model_id}, "
            f"k={self.draft_k}, mode={self.mode}, "
            f"temperature={self.temperature})"
        )

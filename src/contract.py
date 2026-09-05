"""Single source of truth for the DistilBERT inference contract."""

from dataclasses import dataclass

PROFILE_MIN = (1, 1)
PROFILE_OPT = (4, 128)
PROFILE_MAX = (8, 512)

INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "hidden_states"


@dataclass(frozen=True)
class WorkloadCase:
    case_id: str
    mode: str
    batch: int
    sequence: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.batch, self.sequence


SHORT_CASES = (
    WorkloadCase("short-b1-s1", "short", 1, 1),
    WorkloadCase("short-b4-s1", "short", 4, 1),
    WorkloadCase("short-b8-s1", "short", 8, 1),
)

SEQUENCE_CASES = (
    WorkloadCase("seq-b1-s8", "sequence", 1, 8),
    WorkloadCase("seq-b4-s128", "sequence", 4, 128),
    WorkloadCase("seq-b8-s512", "sequence", 8, 512),
)

WORKLOAD_CASES = SHORT_CASES + SEQUENCE_CASES
SUPPORTED_WORKLOAD_SHAPES = tuple(case.shape for case in WORKLOAD_CASES)

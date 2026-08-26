"""Single source of truth for the TinyTransformerEncoder input/output contract."""

from dataclasses import dataclass

VOCAB_SIZE = 32000
HIDDEN = 256

# (batch, sequence) TensorRT optimization profile bounds.
# Sequence length 1 is required for the single-token decode path.
PROFILE_MIN = (1, 1)
PROFILE_OPT = (4, 128)
PROFILE_MAX = (8, 512)

INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "hidden_states"


@dataclass(frozen=True)
class WorkloadCase:
    """A named inference workload, separate from the generic operator matrix."""

    case_id: str
    mode: str
    batch: int
    sequence: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.batch, self.sequence


DECODE_CASES = (
    WorkloadCase("decode-b1-s1", "decode", 1, 1),
    WorkloadCase("decode-b4-s1", "decode", 4, 1),
    WorkloadCase("decode-b8-s1", "decode", 8, 1),
)

PREFILL_CASES = (
    WorkloadCase("prefill-b1-s8", "prefill", 1, 8),
    WorkloadCase("prefill-b4-s128", "prefill", 4, 128),
    WorkloadCase("prefill-b8-s512", "prefill", 8, 512),
)

WORKLOAD_CASES = DECODE_CASES + PREFILL_CASES
SUPPORTED_WORKLOAD_SHAPES = tuple(case.shape for case in WORKLOAD_CASES)

"""Single source of truth for the TinyTransformerEncoder input/output contract (see test.md)."""

VOCAB_SIZE = 32000
HIDDEN = 256

# (batch, sequence) TensorRT optimization profile bounds.
PROFILE_MIN = (1, 8)
PROFILE_OPT = (4, 128)
PROFILE_MAX = (8, 512)

INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "hidden_states"

"""TensorRT execution-context runner: zero-copy over PyTorch CUDA tensor addresses."""
import tensorrt as trt
import torch

from src.contract import HIDDEN, INPUT_NAMES, OUTPUT_NAME

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TrtRunner:
    def __init__(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def run_async(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Enqueue inference without synchronizing the runner stream."""
        if not input_ids.is_cuda or not attention_mask.is_cuda:
            raise ValueError("TensorRT inputs must be CUDA tensors")
        if input_ids.dtype != torch.int32 or attention_mask.dtype != torch.int32:
            raise TypeError("TensorRT inputs must have dtype torch.int32")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("inputs must be matching 2D tensors")
        batch, seq = input_ids.shape

        self.context.set_input_shape(INPUT_NAMES[0], (batch, seq))
        self.context.set_input_shape(INPUT_NAMES[1], (batch, seq))

        output = torch.empty((batch, seq, HIDDEN), device="cuda", dtype=torch.float16)

        self.context.set_tensor_address(INPUT_NAMES[0], input_ids.data_ptr())
        self.context.set_tensor_address(INPUT_NAMES[1], attention_mask.data_ptr())
        self.context.set_tensor_address(OUTPUT_NAME, output.data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)
        # input_ids/attention_mask/output must stay alive until here: the engine
        # only holds their raw device addresses, not Python references to the tensors.
        return output

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.run_async(input_ids, attention_mask)
        self.stream.synchronize()
        return output

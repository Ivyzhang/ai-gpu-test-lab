"""TensorRT execution-context runner for the DistilBERT contract."""
import tensorrt as trt
import torch

from src.contract import INPUT_NAMES, OUTPUT_NAME
from src.workloads import WORKLOADS, WORKLOAD_NAMES

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TrtRunner:
    def __init__(self, engine_path: str, workload_name: str = "distilbert-base-uncased"):
        if workload_name not in WORKLOADS:
            raise ValueError(f"unknown workload: {workload_name}; choose from {WORKLOAD_NAMES}")
        self.spec = WORKLOADS[workload_name]
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        for name in INPUT_NAMES:
            if self.engine.get_tensor_dtype(name) != trt.DataType.INT32:
                raise TypeError(f"TensorRT input {name} must have INT32 dtype")
        if self.engine.get_tensor_dtype(OUTPUT_NAME) != trt.DataType.FLOAT16:
            raise TypeError("TensorRT hidden_states output must have FLOAT16 dtype")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def _validate_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if not input_ids.is_cuda or not attention_mask.is_cuda:
            raise ValueError("TensorRT inputs must be CUDA tensors")
        if input_ids.dtype != torch.int32 or attention_mask.dtype != torch.int32:
            raise TypeError("TensorRT inputs must have dtype torch.int32")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("inputs must be matching 2D tensors")
        if input_ids.numel() and (
            input_ids.min() < 0 or input_ids.max() >= self.spec.vocab_size
        ):
            raise ValueError(f"input_ids must be in [0, {self.spec.vocab_size})")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask values must be 0 or 1")

    def run_async(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Enqueue inference on the runner stream without synchronizing it."""
        self._validate_inputs(input_ids, attention_mask)
        batch, sequence = input_ids.shape
        shape = (batch, sequence)

        for name in INPUT_NAMES:
            accepted = self.context.set_input_shape(name, shape)
            if accepted is False:
                raise ValueError(f"TensorRT rejected {name} shape {shape}")

        expected_output = (batch, sequence, self.spec.hidden)
        actual_output = tuple(self.context.get_tensor_shape(OUTPUT_NAME))
        if actual_output != expected_output:
            raise RuntimeError(f"unexpected output shape: {actual_output}")

        caller_stream = torch.cuda.current_stream(input_ids.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            output = torch.empty(expected_output, device=input_ids.device, dtype=torch.float16)
            output.record_stream(self.stream)

        self.context.set_tensor_address(INPUT_NAMES[0], input_ids.data_ptr())
        self.context.set_tensor_address(INPUT_NAMES[1], attention_mask.data_ptr())
        self.context.set_tensor_address(OUTPUT_NAME, output.data_ptr())

        accepted = self.context.execute_async_v3(self.stream.cuda_stream)
        if accepted is False:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        return output

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.run_async(input_ids, attention_mask)
        self.stream.synchronize()
        return output

    def close(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        self.context = None
        self.engine = None
        self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

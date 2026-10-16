"""DistilBERT HTTP encoder, serving original weights in FP32/FP16/INT8/NF4."""
import argparse
from importlib.metadata import version
import threading
import time
import uuid

from .download_models import model_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['fp32', 'fp16', 'int8', 'nf4'], required=True)
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    import torch
    import bitsandbytes as bnb
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    options = {'local_files_only': True, 'torch_dtype': torch.float32 if args.mode == 'fp32' else torch.float16,
               'attn_implementation': 'eager'}
    if args.mode == 'int8':
        options.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    elif args.mode == 'nf4':
        options.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False))
    model = AutoModel.from_pretrained(args.model_dir, **options).eval()
    if args.mode in ('fp32', 'fp16'):
        model.to('cuda:0')
    counts = {'linear_int8': sum(isinstance(x, bnb.nn.Linear8bitLt) for x in model.modules()),
              'linear_4bit': sum(isinstance(x, bnb.nn.Linear4bit) for x in model.modules())}
    if args.mode == 'int8' and counts['linear_int8'] == 0 or args.mode == 'nf4' and counts['linear_4bit'] == 0:
        raise RuntimeError('requested quantization did not replace any linear modules')
    identity, _ = model_hash(args.model_dir)
    with torch.inference_mode():
        warm = tokenizer('Readiness inference.', return_tensors='pt').to('cuda:0')
        y = model(**warm).last_hidden_state
        if not torch.isfinite(y).all():
            raise RuntimeError('readiness output not finite')
        torch.cuda.synchronize()
    del warm, y
    info = {'model_hash': identity, 'mode': args.mode, 'quantized_modules': counts,
            'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
            'parameter_footprint_bytes': model.get_memory_footprint(),
            'versions': {name: version(name) for name in ('torch', 'transformers', 'bitsandbytes')},
            'gpu': torch.cuda.get_device_name(0), 'attention': 'eager', 'tf32': False,
            'execution': 'one model, serialized GPU execution; concurrent HTTP requests queue'}
    print(info, flush=True)
    app = FastAPI()
    lock = threading.Lock()

    @app.get('/health')
    def health():
        return info

    @app.post('/encode')
    def encode(payload: dict):
        texts = payload.get('texts')
        if not isinstance(texts, list) or not 1 <= len(texts) <= 8 or any(not isinstance(t, str) or not t.strip() or len(t) > 100000 for t in texts):
            raise HTTPException(400, 'texts must contain 1..8 nonempty strings (max 100000 characters each)')
        inputs = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        if inputs['input_ids'].shape[1] > 512:
            raise HTTPException(400, 'input exceeds 512 tokens; truncation is disabled')
        with lock, torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            inputs = inputs.to('cuda:0')
            output = model(**inputs).last_hidden_state
            hidden = output.float().cpu()
            if not torch.isfinite(hidden).all():
                raise HTTPException(500, 'non-finite encoder output')
            result = {'request_id': str(uuid.uuid4()), 'model_hash': identity, 'mode': args.mode,
                      'input_ids': inputs['input_ids'].cpu().tolist(), 'attention_mask': inputs['attention_mask'].cpu().tolist(),
                      'hidden_states': hidden.tolist(),
                      'forward_and_copy_ms': (time.perf_counter() - began) * 1000,
                      'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        return result

    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()

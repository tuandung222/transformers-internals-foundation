---
title: Compile và CUDA graph
---

# `torch.compile`, CUDA graph và `generate`

Chương cuối Phần 4: bật compile cho generate. Đây là cách hiệu quả nhất để tăng tốc decode 2-4x cho GPU NVIDIA Ampere/Hopper.

## Vì sao compile?

`torch.compile` (PyTorch 2.0+) wrap model forward thành compile graph. Bên trong:

1. **Dynamo** trace Python code thành FX graph.
2. **Inductor** lower FX graph thành kernel code (Triton, C++).
3. **CUDA graph**: gom các kernel call thành một graph được launch một phát thay vì launch từng kernel.

Lợi ích:

- **Kernel fusion**: matmul + bias + activation -> một kernel.
- **CUDA graph eliminate launch overhead**: mỗi kernel launch CUDA cost ~10us. Decode 1 token có thể có 200+ kernel call, tổng overhead 2ms (= 500 tok/s upper bound từ launch alone). CUDA graph gom thành 1 launch.

Net: decode latency giảm 2-4x.

## Auto-compile trong `_sample`

```python
model_forward = (
    self.get_compiled_call(generation_config.compile_config)
    if self._valid_auto_compile_criteria(model_kwargs, generation_config)
    else self.__call__
)
```

Điều kiện `_valid_auto_compile_criteria`:

- `generation_config.compile_config` không phải None.
- Cache là `StaticCache` (compile-friendly).
- Backend attention compile-friendly (sdpa, flex; flash hỗ trợ qua wrapper).
- Không có dynamic feature (output_attentions, beam search, ...).

Nếu thoả, `get_compiled_call` return wrapped function: `torch.compile(self.__call__, **compile_config)`.

User cũng có thể manually compile:

```python
model = AutoModelForCausalLM.from_pretrained(...)
model.compile()  # hoặc model.forward = torch.compile(model.forward)
out = model.generate(...)
```

## CompileConfig

```python
from transformers import GenerationConfig, CompileConfig

gen_config = GenerationConfig(
    cache_implementation="static",
    max_new_tokens=128,
    compile_config=CompileConfig(
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
    ),
)
out = model.generate(input_ids, generation_config=gen_config)
```

- `fullgraph=True`: trace toàn bộ forward, không break ở Python control flow. Yêu cầu code compile-friendly.
- `dynamic=False`: shape tĩnh. Recompile nếu shape thay đổi.
- `mode="reduce-overhead"`: dùng CUDA graph aggressive.

## Compile-friendly invariants

Code muốn compile cần thoả:

1. **Tensor shape tĩnh** trong loop: input_ids `(B, 1)`, cache slot `(B, max_len, ...)`, position `(B, 1)`.
2. **Không Python `if/for` data-dependent**: `if x.sum() > 0: ...` sẽ break graph. Dùng `torch.where(...)`.
3. **Không gọi numpy, list ops trên tensor data**: `[x for x in tensor]` break.
4. **In-place mutation chỉ vào tensor có địa chỉ tĩnh**: như `StaticCache.keys` đã `mark_static_address`.

Đây là lý do HF refactor:

- Cache: `cumulative_length` dùng tensor không int (chương 5 Phần 3).
- Mask: precompute `(B, max_cache_len)`, không cat mỗi step.
- Attention backend: `is_causal` flag thay vì check `attention_mask is None`.

## Prefill vs decode

```python
prefill_consumed = False
outputs = self._prefill(input_ids, ...)
while not done:
    if prefill_consumed:
        outputs = model_forward(**inputs, return_dict=True)  # compiled
    prefill_consumed = True
    ...
```

Prefill **không compile**. Lý do: prefill shape `(B, T_prompt)` thay đổi mỗi prompt -> recompile. Tốn 30-60s compile cho mỗi shape mới. Decode shape `(B, 1)` cố định -> compile một lần, reuse.

`_prefill` chạy eager forward (fast một lần). `model_forward` (compiled) chạy decode.

## CUDA graph mode

`mode="reduce-overhead"` bật CUDA graph. Inductor sinh code mà toàn bộ kernel của decode được capture vào CUDA graph. Mỗi `model_forward(...)` chỉ launch 1 graph, không launch từng kernel.

Yêu cầu:

- Buffer tensor có địa chỉ tĩnh (`mark_static_address`).
- Input shape tĩnh.
- Không có alloc trong graph (free + reuse buffer).

HF đã chuẩn bị cache + mask để thoả các yêu cầu này. User chỉ cần `cache_implementation="static"` + `compile_config=CompileConfig(mode="reduce-overhead")`.

## Compile time vs runtime

Lần đầu gọi `generate` với compile: ~30-60s warmup (compile + cache graph). Lần thứ hai: ms-level start (cache hit). Để tránh user surprise, HF có flag warm-up:

```python
# Warm-up compile
_ = model.generate(input_ids, max_new_tokens=2)
# Now subsequent calls are fast
out = model.generate(input_ids, max_new_tokens=200)
```

Production tip: pre-warm với một forward dummy trước khi serve user.

## Benchmark numbers (chỉ tham khảo, tuỳ GPU)

Llama-3 8B, A100, batch=1, max_new_tokens=512:

| Setup | Latency (s) | Tokens/s |
|---|---|---|
| Dynamic cache, eager | 12.5 | 41 |
| Static cache, eager | 11.8 | 43 |
| Static cache, compile (default) | 5.4 | 95 |
| Static cache, compile reduce-overhead | 3.9 | 131 |

~3.2x speedup từ baseline. Speedup càng cao với model nhỏ (launch overhead chiếm tỉ trọng lớn hơn).

## Pitfall

**1. Recompile vì batch size thay đổi**: nếu generate với `B=1` rồi `B=4`, recompile. Workaround: dùng `dynamic=True` nhưng chậm hơn.

**2. Recompile vì `max_cache_len` thay đổi**: cache shape phụ thuộc max_len. User mỗi request prompt khác nhau -> max_cache_len khác -> recompile. Workaround: dùng `max_cache_len` cố định lớn.

**3. CUDA graph fail với output_attentions**: dynamic output không capture. Tắt output_attentions khi compile.

**4. Sliding window không hỗ trợ một số attention backend khi compile**: HF có integration nhưng vài backend edge case fail. Test trước.

**5. Streamer block CUDA graph**: `streamer.put(token.cpu())` sync CPU. Graph mode prefer async. Dùng `TextIteratorStreamer` để giải nén qua thread.

## Tổng kết Phần 4

Phần 4 đã đi qua:

- Top-level generate: parse config, validate, choose mode.
- `_sample`: greedy + multinomial, structure of loop.
- LogitsProcessor: composable transform trên logit.
- StoppingCriteria: OR-combined check stop.
- Beam search: multi-candidate, cache reorder.
- Assisted decoding: speculative với draft model.
- Compile + CUDA graph: 2-4x speedup decode.

Phần 5 (cuối) đi vào các convention chung của HF transformers: `PreTrainedModel`, `Config`, `Auto*` class, `_keys_to_ignore_on_load_missing`, output dataclass. Đây là kiến thức "meta-level" giúp đọc code nhanh và tin tưởng vào pattern.

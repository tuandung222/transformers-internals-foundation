---
title: _sample chi tiết
---

# `_sample`: greedy và multinomial sampling

Hai mode quan trọng nhất: greedy (`do_sample=False`) và sample (`do_sample=True`). Cả hai cùng method `_sample`. Phân biệt chỉ ở dòng pick token.

## Skeleton

```python
def _sample(self, input_ids, logits_processor, stopping_criteria, generation_config, ...,**model_kwargs):
    pad_token_id = generation_config._pad_token_tensor
    do_sample = generation_config.do_sample
    has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)

    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if self._valid_auto_compile_criteria(model_kwargs, generation_config)
        else self.__call__
    )

    prefill_consumed = False
    outputs = self._prefill(input_ids, generation_config, model_kwargs, ...)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed:
            next_sequence_length = 1 if model_kwargs["use_cache"] else None
            model_inputs = self.prepare_inputs_for_generation(input_ids, next_sequence_length=next_sequence_length, **model_kwargs)
            with self._optimize_model_for_decode():
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True

        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=...)

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        del outputs

    return input_ids
```

(đã rút gọn cho dễ đọc, bỏ phần `return_dict_in_generate`, streamer, encoder_outputs cleanup)

## Phân tích từng phần

### Setup state

```python
batch_size = input_ids.shape[0]
unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
```

`unfinished_sequences`: tensor `(B,)` đánh dấu sequence nào trong batch chưa stop. Khi một sequence stop (gặp EOS hoặc max_length), bit của nó về 0. Loop dừng khi mọi bit = 0.

Vì sao cần dùng mask thay vì break? Vì batch: trong cùng batch, sequence A có thể stop sớm (gặp EOS sau 10 token), sequence B vẫn còn generate (chưa EOS sau 100 token). Ta phải tiếp tục forward cho B, nhưng đầu ra của A bỏ. Mask cho phép GPU vẫn batched, không cần dynamic batch size.

### Auto-compile dispatch

```python
model_forward = (
    self.get_compiled_call(generation_config.compile_config)
    if self._valid_auto_compile_criteria(model_kwargs, generation_config)
    else self.__call__
)
```

Nếu user enable compile (qua `generation_config.compile_config` hoặc auto-detect static cache + GPU), `model_forward` là phiên bản compiled. Mỗi `model_forward(**inputs)` chạy compile graph, cuda graph, ... HF có heuristic `_valid_auto_compile_criteria` để tự bật khi an toàn.

Khi không an toàn (dynamic shape, sliding window không hỗ trợ, ...), fallback về `self.__call__` (eager).

### Prefill

```python
prefill_consumed = False
outputs = self._prefill(input_ids, generation_config, model_kwargs, ...)
```

`_prefill` chạy model forward với toàn bộ prompt (`input_ids` shape `(B, T_prompt)`). Cache được fill từ rỗng thành `T_prompt` token mỗi layer. Output có `logits` shape `(B, T_prompt, vocab_size)`; ta sẽ dùng `logits[:, -1, :]` để sample token đầu tiên.

`prefill_consumed = False` ban đầu. Sau iter đầu tiên, set `True` để mọi iter sau gọi `model_forward(...)` thay vì `_prefill`. Tách prefill ra method riêng cho phép HF có path optimize riêng (ví dụ không compile prefill).

### Decode loop

```python
while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
    if prefill_consumed:
        next_sequence_length = 1 if model_kwargs["use_cache"] else None
        model_inputs = self.prepare_inputs_for_generation(input_ids, next_sequence_length=next_sequence_length, **model_kwargs)
        with self._optimize_model_for_decode():
            outputs = model_forward(**model_inputs, return_dict=True)
    prefill_consumed = True
    ...
```

`prepare_inputs_for_generation` là method **model-specific** (override mỗi model). Nó:

1. Chỉ lấy token cuối của `input_ids` (token mới sample) làm `input_ids` mới shape `(B, 1)`. Vì cache đã giữ history, ta chỉ cần đưa token mới.
2. Compute `position_ids` cho token mới (= length cache).
3. Compute `attention_mask` extend thêm 1 vị trí.
4. Compute `cache_position` cho cache slot mới.

`_optimize_model_for_decode` là context manager (no-op default, model có thể override). Whisper dùng để pause encoder.

### Logit process

```python
next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
next_token_scores = logits_processor(input_ids, next_token_logits)
```

`outputs.logits` shape `(B, q_len, vocab)`. Sample lấy logit của token cuối `[:, -1, :]`. Cast về float32 cho stability khi process.

`logits_processor` là `LogitsProcessorList`, applied chuỗi:

```python
class LogitsProcessorList(list):
    def __call__(self, input_ids, scores, **kwargs):
        for processor in self:
            scores = processor(input_ids, scores)
        return scores
```

Mỗi processor là một transform `(input_ids, scores) -> scores`. Chương 4 ta đi chi tiết.

### Pick token

```python
if do_sample:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

Greedy: `argmax`, deterministic. Sample: softmax, multinomial sampling.

`torch.multinomial` chấp nhận tensor probabilities `(B, vocab)`, return `(B, num_samples)`. `squeeze(1)` ra `(B,)`.

Khi `temperature=0`, sample thoái hoá thành greedy (nhưng numerically khác: softmax với scaled scores). User thường nên dùng `do_sample=False` cho greedy thay vì `do_sample=True, temperature=0`.

### EOS padding

```python
if has_eos_stopping_criteria:
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
```

Cho sequence đã stop (`unfinished=0`), force next_token = pad_token. Sequence đã stop tiếp tục được forward nhưng output bị ignore (token chỉ pad).

### Append và stop check

```python
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
this_peer_finished = unfinished_sequences.max() == 0
del outputs
```

`stopping_criteria(input_ids, scores)` return mask `(B,)`: True nếu sequence cần stop. `~` (logical not) đảo lại, `&` (logical and) update mask.

`del outputs` quan trọng: giữ ref `outputs.logits` tới iter sau khiến tensor logit (rất lớn, `(B, T_prompt, vocab)` ở iter đầu) không được free. Explicit `del` để giải phóng.

## Compile-friendly invariants

Để compile path hiệu quả, mọi tensor shape **trong loop** phải tĩnh:

- `input_ids` truyền vào `model_forward` luôn `(B, 1)` (qua `prepare_inputs_for_generation`).
- `attention_mask` shape `(B, max_cache_len)`.
- `cache_position` shape `(B, 1)`.
- `past_key_values` là `StaticCache` (mark_static_address).

Nếu shape thay đổi (ví dụ dùng `DynamicCache`), recompile mỗi iter. Compile vô dụng.

## Pitfall

**1. `output_attentions=True` không hoạt động với compile**: compile path không thể capture intermediate. Pass `output_attentions=True` với compile dẫn tới fallback eager hoặc warning.

**2. Streamer block GPU**: `streamer.put(next_tokens.cpu())` chuyển tensor sang CPU. CPU-GPU sync. Nếu tokenizer.decode chậm, generate bị slow down. Dùng `TextIteratorStreamer` thay vì `TextStreamer` để giải nén CPU work trong thread khác.

**3. `unfinished_sequences` không stop khi `synced_gpus=True`**: với FSDP/DeepSpeed Zero3, ta phải chạy tới `max_length` để mọi rank sync, không break sớm. `synced_gpus` flag được set tự động bởi HF nếu detect distributed env.

Chương sau ta đi vào `LogitsProcessor`.

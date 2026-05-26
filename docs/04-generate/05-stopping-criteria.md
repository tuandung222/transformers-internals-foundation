---
title: StoppingCriteria
---

# `StoppingCriteria` chi tiết

`StoppingCriteria` quyết định khi nào dừng generate. Tương tự `LogitsProcessor`, là list các criterion, OR-combined: sequence dừng nếu **bất kỳ** criterion nào trigger.

File: `src/transformers/generation/stopping_criteria.py`.

## Interface

```python
class StoppingCriteria:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        raise NotImplementedError
```

Return tensor `(B,)` bool: True cho sequence cần stop.

## `StoppingCriteriaList`

```python
class StoppingCriteriaList(list):
    def __call__(self, input_ids, scores, **kwargs):
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done
```

OR-combine: nếu bất kỳ criterion nào return True cho seq `i`, `is_done[i] = True`.

## `MaxLengthCriteria`

```python
class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int, max_position_embeddings: int | None = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids, scores, **kwargs):
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(...)
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
```

Đơn giản: nếu `cur_len >= max_length`, stop toàn batch. `max_length` = `prompt + generated`.

`max_position_embeddings` là cap cứng (position embed của model chỉ trained tới đó). Cảnh báo nếu vượt.

## `MaxNewTokensCriteria` (deprecated)

Đã merge vào `MaxLengthCriteria` qua việc compute `max_length = start_length + max_new_tokens` ở `_get_stopping_criteria`. Code có legacy nhưng không khuyến khích dùng trực tiếp.

## `EosTokenCriteria`

```python
class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: torch.Tensor):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        is_done = torch.isin(input_ids[:, -1], self.eos_token_id)
        return is_done
```

Check token cuối có phải EOS. `torch.isin` cho phép `eos_token_id` là list (model như Llama-3 có nhiều EOS: `<|eot_id|>`, `<|end_of_text|>`).

Quan trọng: chỉ check token cuối, không check toàn sequence. Hợp lý vì generate append từng token một; EOS chỉ relevant nếu nó vừa được sample.

## `StopStringCriteria`

Phức tạp nhất. User truyền `stop_strings=["</s>", "User:", ...]`. Criterion decode `input_ids` về string, check substring match.

```python
class StopStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings if isinstance(stop_strings, list) else [stop_strings]
        # Precompute token-string match table
        self.embedding_vec = self._stop_string_create_embedding_vec(...)

    def __call__(self, input_ids, scores, **kwargs):
        # Use embedding_vec to compute match efficiently on GPU
        ...
```

Implementation tinh tế: decode mỗi step là chậm. HF precompute "embedding vec" map mỗi token id sang vector flag (match end-position, length-token). Mỗi step, gather các vector cho token mới và token gần đây, scan match. Hoàn toàn vectorized trên GPU.

Lý do quan trọng: với batch lớn (32, 64) và max_new_tokens=1024, decode mỗi step sang string là bottleneck. Embedding approach cho phép O(B * max_token_per_string) thay vì O(B * T) decode.

## `MaxTimeCriteria`

```python
class MaxTimeCriteria(StoppingCriteria):
    def __init__(self, max_time, initial_time=None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_time is None else initial_time

    def __call__(self, input_ids, scores, **kwargs):
        is_done = time.time() - self.initial_timestamp > self.max_time
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
```

Wall-clock limit. Cho generation interactive nơi latency budget cố định.

## `_get_stopping_criteria` build từ config

Trong `_get_stopping_criteria`:

```python
def _get_stopping_criteria(self, generation_config, stopping_criteria, tokenizer):
    criteria = StoppingCriteriaList()

    if generation_config.max_length is not None:
        criteria.append(MaxLengthCriteria(
            max_length=generation_config.max_length,
            max_position_embeddings=self.config.max_position_embeddings,
        ))

    if generation_config.max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))

    if generation_config.stop_strings is not None:
        if tokenizer is None:
            raise ValueError("Need tokenizer for stop_strings")
        criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))

    if generation_config._eos_token_tensor is not None:
        criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))

    if stopping_criteria is not None:
        criteria.extend(stopping_criteria)  # user-provided

    return criteria
```

User-provided `stopping_criteria` được merge vào cuối. Cho phép user extend.

## Pitfall

**1. `max_length` nhỏ hơn prompt**: `generate` raise. Nhưng nếu vừa bằng prompt length, generate ra 0 token mới. Edge case dễ miss.

**2. EOS không trong vocab tokenizer**: model trained với EOS đặc biệt (ví dụ Llama-3 có `<|eot_id|>`), nhưng tokenizer custom không có. `generation_config._eos_token_tensor` rỗng -> generation không dừng -> generate tới max_length.

**3. Stop string không bao đủ**: stop_strings=["User:"]. Nếu model output "User :" (có space), không match. User phải pass nhiều variant.

**4. Wall-clock + multi-batch**: `MaxTimeCriteria` áp dụng cho cả batch. Nếu một sequence trong batch chậm, mọi sequence stop. Không có per-sequence time limit.

## Order matter?

Không như LogitsProcessor (order ảnh hưởng output), StoppingCriteria là OR. Order không matter cho correctness. Nhưng order ảnh hưởng performance: nếu `MaxLengthCriteria` cheapest và `StopStringCriteria` đắt, đặt `MaxLength` trước có thể short-circuit. Trong HF, `StoppingCriteriaList` luôn evaluate hết, không short-circuit. Có thể tối ưu, nhưng overhead nhỏ.

Chương sau ta đi vào beam search.

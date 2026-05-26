---
title: Top-level generate
---

# Top-level `generate`

File: `src/transformers/generation/utils.py`. Method `GenerationMixin.generate` ~ line 2172. Đây là entry point. Cấu trúc bên trong dài (~600 dòng) nhưng chia thành các block rõ.

## Skeleton

```python
@torch.no_grad()
def generate(
    self,
    inputs=None,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus=None,
    assistant_model=None,
    streamer=None,
    negative_prompt_ids=None,
    negative_prompt_attention_mask=None,
    custom_generate=None,
    use_model_defaults=None,
    trust_remote_code=False,
    **kwargs,
):
    # 1. Handle kwargs, generation_config, validate, obtain mode
    generation_mode_kwargs = self._extract_generation_mode_kwargs(...)
    generation_config, model_kwargs = self._prepare_generation_config(generation_config, ...)
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if custom_generate is not None:
        decoding_method = custom_generate
    elif deprecated_mode_repo is None:
        decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])

    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

    # 2. Prepare encoder outputs (encoder-decoder)
    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 3. Prepare input_ids for decode loop
    input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(...)

    # 4. Prepare logits processor and stopping criteria
    prepared_logits_processor = self._get_logits_processor(...)
    prepared_stopping_criteria = self._get_stopping_criteria(...)

    # 5. Setup cache
    model_kwargs["past_key_values"] = self._prepare_cache_for_generation(...)
    model_kwargs["use_cache"] = True

    # 6. Dispatch to decoding method
    result = decoding_method(
        self,
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        **generation_mode_kwargs,
        **model_kwargs,
    )

    return result
```

Sáu block. Ta đi từng cái.

## Block 1: Config và mode

```python
generation_mode_kwargs = self._extract_generation_mode_kwargs(
    custom_generate, kwargs, synced_gpus, assistant_model, streamer
)
generation_config, model_kwargs = self._prepare_generation_config(generation_config, ...)
generation_mode = generation_config.get_generation_mode(assistant_model)
```

`generation_config` là `GenerationConfig` instance. Nguồn ưu tiên:

1. Argument truyền vào `generate(generation_config=...)`.
2. `model.generation_config` (load từ `generation_config.json` của checkpoint).
3. Default `GenerationConfig()`.

`get_generation_mode` quyết định mode dựa vào:

- `do_sample=False, num_beams=1` -> GREEDY_SEARCH
- `do_sample=True, num_beams=1` -> SAMPLE
- `num_beams > 1, do_sample=False` -> BEAM_SEARCH
- `num_beams > 1, do_sample=True` -> BEAM_SAMPLE
- `assistant_model is not None` -> ASSISTED_GENERATION

Một số mode khác (DOLA, contrastive search) trước đây là first-class, hiện ở Hub repo `transformers-community/dola`, ... Truy cập qua `custom_generate="transformers-community/dola"`.

## Block 2: Encoder forward

```python
if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
    model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(...)
```

Chỉ cho encoder-decoder model. Encoder chạy **một lần**, output cache vào `model_kwargs["encoder_outputs"]`. Mỗi decode step, `prepare_inputs_for_generation` lấy `encoder_outputs` này pass vào decoder. Đây là chỗ "encoder K/V cross-attention được tính một lần" như Phần 2 đã nói.

## Block 3: decoder_input_ids

```python
input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(...)
```

Trường hợp decoder-only: `input_ids` = prompt user truyền vào (đã tokenize).

Trường hợp encoder-decoder: `input_ids` thường bắt đầu bằng `decoder_start_token_id` (T5: `pad_token`, BART: `eos_token`). Đây là token "kickstart" decoder.

## Block 4: processors

```python
prepared_logits_processor = self._get_logits_processor(
    generation_config=generation_config,
    input_ids_seq_length=input_ids_length,
    encoder_input_ids=inputs_tensor,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    logits_processor=logits_processor,
    device=inputs_tensor.device,
    model_kwargs=model_kwargs,
    negative_prompt_ids=negative_prompt_ids,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
)

prepared_stopping_criteria = self._get_stopping_criteria(
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,
    tokenizer=generation_mode_kwargs.get("tokenizer"),
)
```

`_get_logits_processor` build một `LogitsProcessorList` chứa các processor mà generation_config kích hoạt:

- `temperature != 1.0` -> `TemperatureLogitsWarper`
- `top_k > 0` -> `TopKLogitsWarper`
- `top_p < 1.0` -> `TopPLogitsWarper`
- `repetition_penalty != 1.0` -> `RepetitionPenaltyLogitsProcessor`
- `no_repeat_ngram_size > 0` -> `NoRepeatNGramLogitsProcessor`
- ... và nhiều processor khác.

`_get_stopping_criteria` build `StoppingCriteriaList`:

- `max_length` hoặc `max_new_tokens` -> `MaxLengthCriteria`
- `stop_strings` -> `StopStringCriteria`
- `eos_token_id` -> `EosTokenCriteria`

Phần 4 ta sẽ tách hai chương riêng cho hai loại này.

## Block 5: Cache setup

```python
model_kwargs["past_key_values"] = self._prepare_cache_for_generation(
    generation_config, model_kwargs, ...
)
model_kwargs["use_cache"] = True
```

Tuỳ `generation_config.cache_implementation`:

- `None`/`"dynamic"` (default) -> `DynamicCache`
- `"static"` -> `StaticCache(max_cache_len=...)`
- `"quantized"` -> `QuantizedCache(backend="hqq"/"quanto", nbits=4/2)`
- `"offloaded"` -> `OffloadedCache`

User truyền `model.generation_config.cache_implementation = "static"` để bật static cache.

## Block 6: Dispatch

```python
result = decoding_method(
    self, input_ids,
    logits_processor=prepared_logits_processor,
    stopping_criteria=prepared_stopping_criteria,
    generation_config=generation_config,
    **generation_mode_kwargs,
    **model_kwargs,
)
```

Quan sát: `decoding_method` được lấy từ `type(self)` bằng `getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])`. Tức là unbound class method. Gọi với `self` làm tham số đầu để bind. Pattern này lạ một chút (tại sao không gọi `self._sample(...)` luôn?), lý do là để hỗ trợ `custom_generate` (function bên ngoài).

## Pitfall thường gặp

**1. `generation_config` không pickup**: nếu truyền `temperature` qua `kwargs`, không qua `generation_config`, một số case không pickup. Khuyến nghị: tạo `GenerationConfig(temperature=..., ...)` rồi pass.

**2. `max_length` vs `max_new_tokens`**: `max_length` là tổng (prompt + generated), `max_new_tokens` là chỉ generated. Lẫn lộn dẫn tới truncate sớm hoặc gen quá dài.

**3. Encoder forward chạy lại nếu truyền `encoder_outputs` sai**: nếu pass `encoder_outputs=None` (truyền nhưng giá trị None), HF không chạy lại; pass `del model_kwargs["encoder_outputs"]` hoặc không pass.

**4. `do_sample=False` nhưng `temperature > 0`**: greedy không sample, temperature bị bỏ qua. HF có warning nhưng không error.

Chương kế tiếp đi vào `_sample` chi tiết.

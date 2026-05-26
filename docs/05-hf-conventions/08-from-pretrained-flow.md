---
title: from_pretrained flow
---

# `from_pretrained` flow end-to-end

Method magic nhất của transformers. User gọi:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
```

Và 5 giây sau có model sẵn sàng inference. Bên trong, hàng chục bước. Chương cuối Phần 5 đi từng bước.

## Skeleton

`PreTrainedModel.from_pretrained` ~ line 3700 của `modeling_utils.py`, dài ~1000 dòng. Ta đọc structure:

```python
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path,
    *model_args,
    config=None,
    cache_dir=None,
    ignore_mismatched_sizes=False,
    force_download=False,
    local_files_only=False,
    token=None,
    revision="main",
    use_safetensors=None,
    weights_only=True,
    torch_dtype=None,
    device_map=None,
    tp_plan=None,
    distributed_config=None,
    attn_implementation=None,
    **kwargs,
):
    # Step 1: resolve config (auto + load)
    if config is None:
        config, model_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, ...)

    # Step 2: validate attn_implementation
    config._attn_implementation = cls._check_and_select_attn_implementation(config, attn_implementation, ...)

    # Step 3: resolve cache dir + download files
    resolved_archive_file = cached_file(pretrained_model_name_or_path, "model.safetensors", ...)

    # Step 4: instantiate model with empty/meta weights
    with init_empty_weights() if device_map is not None else nullcontext():
        model = cls(config, *model_args, **model_kwargs)

    # Step 5: load state dict
    state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)

    # Step 6: validate state dict (missing/unexpected keys, tied weights)
    missing_keys, unexpected_keys, mismatched_keys = _load_state_dict_into_model(model, state_dict, ...)

    # Step 7: tie weights
    model.tie_weights()

    # Step 8: device placement (device_map, dtype)
    model = dispatch_model(model, device_map, ...)

    # Step 9: TP/FSDP apply
    if tp_plan:
        apply_tp_plan(model, tp_plan, mesh)

    # Step 10: post-load hooks (quantizer, peft, ...)
    if hf_quantizer is not None:
        hf_quantizer.postprocess_model(model)

    # Step 11: warning về missing/unexpected
    _log_missing_unexpected(missing_keys, unexpected_keys)

    # Step 12: set generation_config nếu có
    if model.can_generate():
        model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, ...)

    model.eval()
    return model
```

12 bước. Ta đi từng cái.

## Step 1: Resolve config

```python
if config is None:
    config, model_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
```

Nếu user không pass config, load `config.json` từ path/hub. `cls.config_class` là `LlamaConfig` (đã set trên `LlamaPreTrainedModel`).

`config_class.from_pretrained` chia thành:

1. Resolve `config.json`: nếu là path local, đọc trực tiếp. Nếu là hub repo (`"meta-llama/Llama-3.2-1B"`), gọi `huggingface_hub.hf_hub_download`.
2. Parse JSON, instantiate `LlamaConfig(**dict)`.

`kwargs` được dùng để override config attribute (`config.torch_dtype = torch.bfloat16` qua `from_pretrained(torch_dtype=torch.bfloat16)`).

## Step 2: Validate attention

```python
config._attn_implementation = cls._check_and_select_attn_implementation(config, attn_implementation, ...)
```

User pass `attn_implementation="sdpa"`. Class check `_supports_sdpa`. Nếu False, raise. Nếu True, set `config._attn_implementation = "sdpa"`. Mọi layer attention sau đó đọc field này.

Nếu user không pass, HF auto-select. Logic:

1. Nếu GPU NVIDIA Ampere+ và flash_attn lib installed: pick `flash_attention_2`.
2. Else nếu PyTorch >= 2.0: pick `sdpa`.
3. Else: `eager`.

## Step 3: Download weights

```python
resolved_archive_file = cached_file(
    pretrained_model_name_or_path,
    "model.safetensors",  # hoặc "pytorch_model.bin"
    cache_dir=cache_dir,
    force_download=force_download,
    revision=revision,
    token=token,
)
```

`cached_file` (huggingface_hub helper) handle:

1. Check cache local (`~/.cache/huggingface/hub`). Nếu hit, return path.
2. Else download từ hub. Resume nếu interrupted.
3. Verify checksum.

Ưu tiên `.safetensors` (an toàn) hơn `.bin` (pickle, có thể chứa code độc).

Model lớn shard nhiều file: `model-00001-of-00010.safetensors`, ..., có `model.safetensors.index.json` chỉ key nào ở file nào.

## Step 4: Instantiate model với empty weights

```python
with init_empty_weights() if device_map is not None else nullcontext():
    model = cls(config, *model_args, **model_kwargs)
```

`init_empty_weights` (Accelerate context) override `nn.Module.__init__` để tensor không allocate memory (chỉ shape + dtype). Cho phép instantiate model lớn (70B) trên máy có ít RAM.

Sau khi instantiate, model có "skeleton" với meta tensor. Load state_dict sau đó allocate thật.

## Step 5: Load state_dict

```python
state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)
```

`load_state_dict` đọc safetensors / pickle. `weights_only=True` (PyTorch 2.5+) chỉ load tensor, không allow code execution -> an toàn cho file pickle download từ internet.

State dict là `dict[str, torch.Tensor]` với key như `"model.embed_tokens.weight"`, `"model.layers.0.self_attn.q_proj.weight"`, ...

## Step 6: Match state_dict với model

```python
missing_keys, unexpected_keys, mismatched_keys = _load_state_dict_into_model(model, state_dict, ...)
```

Iterate qua mọi parameter của model. Mỗi parameter:

1. Lookup trong `state_dict` theo key.
2. Nếu có: copy tensor vào parameter.
3. Nếu không: nằm trong `missing_keys`.

Tương tự cho state_dict key không có trong model: `unexpected_keys`.

Shape mismatch: param model shape `(4096, 4096)`, state_dict `(2048, 2048)` -> `mismatched_keys`. Raise trừ khi `ignore_mismatched_sizes=True`.

## Step 7: Tie weights

```python
model.tie_weights()
```

Sau load, tie `lm_head.weight = embed_tokens.weight` nếu config bảo. Đã đi chương trước.

## Step 8: Device placement

```python
model = dispatch_model(model, device_map, ...)
```

`device_map` các giá trị:

- `None`: model trên CPU (default).
- `"auto"`: Accelerate split layer giữa các device.
- `"cuda:0"`: full model trên một GPU.
- `{"model.layers.0": 0, "model.layers.1": 1, ...}`: custom map.

`dispatch_model` move tensor đúng device, add hook (`AlignDevicesHook`) tự move input/output mỗi forward.

`torch_dtype` cast model về dtype tương ứng (bf16, fp16, fp32, ...).

## Step 9: TP/FSDP apply

```python
if tp_plan == "auto" or isinstance(tp_plan, dict):
    plan = model._tp_plan if tp_plan == "auto" else tp_plan
    apply_tp_plan(model, plan, distributed_config.mesh)
```

Đã đi chương trước. Apply TP plan qua `torch.distributed.tensor.parallel`.

## Step 10: Quantizer + adapter

```python
if hf_quantizer is not None:
    hf_quantizer.postprocess_model(model)
```

Nếu user pass `quantization_config=BitsAndBytesConfig(...)`, model được quantize sau load. Phổ biến:

- `load_in_8bit=True`: bitsandbytes int8.
- `load_in_4bit=True`: bitsandbytes NF4.
- `quantization_config=GPTQConfig(...)`: GPTQ quant.

Adapter (PEFT, LoRA) similar: load adapter weight sau base model.

## Step 11: Missing/unexpected warning

```python
def _log_missing_unexpected(missing_keys, unexpected_keys):
    # Filter ignored keys
    filtered_missing = [k for k in missing_keys if not any(re.search(p, k) for p in cls._keys_to_ignore_on_load_missing)]
    filtered_unexpected = [k for k in unexpected_keys if not any(re.search(p, k) for p in cls._keys_to_ignore_on_load_unexpected)]
    if filtered_missing:
        logger.warning(f"Some weights of {model.__class__.__name__} were not initialized: {filtered_missing}")
    if filtered_unexpected:
        logger.warning(f"Some weights of pretrained model not used: {filtered_unexpected}")
```

Filter keys bị `_keys_to_ignore_on_load_*` skip. Phần còn lại warning. User cần xem warning để phát hiện:

- Missing key: model có nhưng checkpoint không -> weight random, generate ra junk.
- Unexpected key: checkpoint có nhưng model không -> không load -> waste.

## Step 12: GenerationConfig

```python
if model.can_generate():
    try:
        model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, ...)
    except OSError:
        # No generation_config.json, use default from config
        pass
```

Hub thường có `generation_config.json` riêng (`{"max_length": 4096, "do_sample": true, "temperature": 0.7, "eos_token_id": [128001, 128009]}`). Load và attach vào model.

User sau đó `model.generate(input_ids)` mà không cần pass config riêng. Default đã được set trong `model.generation_config`.

## Pitfall

**1. Hub rate limit**: download fail. Set `HF_HOME` để cache hoặc dùng `local_files_only=True` sau lần đầu.

**2. OOM khi load full precision**: model 70B fp32 = 280 GB RAM. Phải dùng `torch_dtype=torch.bfloat16` (140 GB) hoặc quantize.

**3. Missing tied weight**: `lm_head.weight` thiếu trong state_dict mà không tie -> warning. Check `config.tie_word_embeddings`.

**4. attn_implementation mismatch**: model checkpoint export với SDPA, load với eager -> work nhưng slow. Khuyến nghị: cho HF auto-select.

**5. weights_only=False với pickle**: nguy hiểm. Default từ PyTorch 2.5+: `weights_only=True`. HF inherit. Đừng set False trừ khi tin source.

## Tổng kết

`from_pretrained` là composition của 12 step. Mỗi step có vai trò. Hiểu flow giúp debug nhanh khi load fail.

Phần 5 kết thúc ở đây. Bạn đã đi qua:

- PreTrainedModel/Config foundation.
- Modular vs modeling file.
- Auto class registry.
- Task head pattern và output dataclass.
- Tied weights, attention flags.
- TP/FSDP plan.
- from_pretrained flow.

Toàn bộ chuỗi bài giảng đã hoàn thành. Hãy mở `resources/cheatsheet.md` để có quick lookup, hoặc `resources/glossary.md` để tra thuật ngữ. References cho người muốn đào sâu hơn ngoài transformers.

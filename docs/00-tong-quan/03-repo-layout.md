---
title: Layout repo transformers
---

# Layout của `src/transformers/`

Mở folder `src/transformers/` lần đầu, bạn thấy hơn 70 file Python và một số subfolder. Chương này là bản đồ: file nào quan trọng, file nào hỗ trợ, file nào có thể bỏ qua trong lần đọc đầu.

## Cấu trúc thư mục cấp cao

```text
src/transformers/
├── __init__.py              # Public API, re-export mọi class quan trọng
├── modeling_utils.py        # PreTrainedModel base class, AttentionInterface, save/load
├── configuration_utils.py   # PreTrainedConfig base class
├── cache_utils.py           # Cache, CacheLayerMixin, DynamicCache, StaticCache
├── modeling_attn_mask_utils.py    # create_causal_mask, attention mask helpers
├── modeling_flash_attention_utils.py  # FlashAttention wrapper
├── modeling_layers.py       # GenericForSequenceClassification, GenericForTokenClassification...
├── modeling_outputs.py      # BaseModelOutputWithPast, CausalLMOutputWithPast, ...
├── modeling_rope_utils.py   # ROPE_INIT_FUNCTIONS, scaling strategies
├── masking_utils.py         # create_causal_mask, create_sliding_window_mask
├── generation/              # Generate logic
│   ├── utils.py             # GenerationMixin + generate() pipeline
│   ├── configuration_utils.py  # GenerationConfig
│   ├── logits_process.py    # LogitsProcessor classes
│   ├── stopping_criteria.py # StoppingCriteria classes
│   ├── candidate_generator.py  # For assisted decoding
│   ├── streamers.py         # TextStreamer, TextIteratorStreamer
│   └── continuous_batching/ # Paged attention CB
├── integrations/            # Backend integrations
│   ├── sdpa_attention.py    # SDPA forward
│   ├── flash_attention.py   # FlashAttention forward
│   ├── flex_attention.py    # FlexAttention forward
│   ├── eager_paged.py, flash_paged.py  # Paged variants
│   ├── peft.py              # PEFT (LoRA) integration
│   └── ... (quantization, deepspeed, accelerate)
├── models/                  # Mỗi model gia đình một folder
│   ├── llama/
│   │   ├── modeling_llama.py
│   │   ├── configuration_llama.py
│   │   ├── tokenization_llama.py
│   │   └── convert_llama_weights_to_hf.py
│   ├── t5/, gpt2/, bert/, qwen2/, ...
│   ├── auto/                # AutoConfig, AutoModel, AutoTokenizer registry
│   └── ...
├── data/                    # DataCollator, datasets utility
├── trainer.py, trainer_pt_utils.py  # Trainer class
├── pipelines/               # High-level Pipeline API
└── utils/                   # Logging, hub_utils, generic helpers
```

## Bốn file đáng đọc đầu tiên

Nếu bạn chỉ có thời gian đọc 4 file, đọc bốn file này theo thứ tự:

**1. `models/llama/modeling_llama.py`** (~520 dòng). Đây là kiến trúc tham chiếu. Hầu hết LLM hiện đại (Mistral, Qwen, Gemma) đều theo template Llama. Đọc file này một lần là bạn hiểu được 80% các file `modeling_*.py` khác.

**2. `cache_utils.py`** (~1620 dòng). Đây là module chứa toàn bộ logic KV cache. Hiểu file này là hiểu cách generate hoạt động bên trong.

**3. `generation/utils.py`** (~3940 dòng). Đây là `generate` method và mọi decoding strategy. File dài nhưng có cấu trúc rõ.

**4. `modeling_utils.py`** (~5200 dòng). `PreTrainedModel` class, `AttentionInterface`, save/load. Không cần đọc hết, chỉ đọc các section: `class PreTrainedModel`, `def from_pretrained`, `def _load_pretrained_model`, `class AttentionInterface`.

Tổng cộng: ~11000 dòng. Đọc kỹ trong 1 tuần là khả thi.

## Mỗi model folder có gì

Lấy `models/llama/` làm ví dụ:

```text
models/llama/
├── __init__.py             # Re-export
├── modeling_llama.py       # Kiến trúc PyTorch (LlamaModel, LlamaForCausalLM, ...)
├── configuration_llama.py  # LlamaConfig (hyperparameter)
├── tokenization_llama.py   # LlamaTokenizer (legacy), thường có thêm tokenization_llama_fast.py
├── convert_llama_weights_to_hf.py  # Script convert từ checkpoint Meta
└── modular_llama.py        # (nếu có) diff với một model gốc, được dùng để generate modeling_llama.py
```

Convention chung:

- File có prefix `modeling_*` là PyTorch (cũng có `modeling_tf_*` cho TensorFlow và `modeling_flax_*` cho JAX, hiện đang phase out).
- File có prefix `configuration_*` chỉ chứa dataclass-like config, không có logic.
- File có prefix `tokenization_*` chứa tokenizer custom (rare, hầu hết model dùng tokenizer fast từ `tokenizers` library).
- File có prefix `image_processing_*`, `feature_extraction_*` cho multi-modal.

Khi đọc một model mới, mở `modeling_<model>.py` đầu tiên. Nó self-contained.

## `models/auto/` registry

Folder `auto/` là nơi đăng ký class ánh xạ:

```text
models/auto/
├── auto_factory.py          # _LazyAutoMapping, _BaseAutoModelClass
├── configuration_auto.py    # AutoConfig, CONFIG_MAPPING_NAMES
├── modeling_auto.py         # AutoModelForCausalLM, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, ...
├── tokenization_auto.py     # AutoTokenizer
└── ...
```

Khi bạn gọi `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")`:

1. Đọc `config.json` từ checkpoint, parse thành `LlamaConfig`.
2. Tra `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` với key `LlamaConfig` để tìm class `LlamaForCausalLM`.
3. Gọi `LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8B")`.

Chi tiết flow ở chương 4.

## `integrations/`

Folder này chứa các "side car" tích hợp với backend bên ngoài:

- `sdpa_attention.py`, `flash_attention.py`, `flex_attention.py`: ba backend attention chính. Mỗi file là một function `<name>_attention_forward`.
- `bitsandbytes.py`, `awq.py`, `gptq.py`, `bitnet.py`: quantization. Mỗi backend có module riêng để load và replace `nn.Linear` thành `Linear8bitLt` hoặc tương đương.
- `peft.py`: integration với PEFT library (LoRA, prefix tuning).
- `deepspeed.py`, `accelerate.py`: integration với distributed library.

Bạn không cần đọc folder này trong lần đầu. Khi nào dùng đến quantization hoặc backend cụ thể, mở file tương ứng.

## `generation/`

Folder này là tim của text generation:

- `utils.py`: `GenerationMixin`, `generate`, `_sample`, `_beam_search`, `_assisted_decoding`. File trung tâm.
- `configuration_utils.py`: `GenerationConfig` dataclass.
- `logits_process.py`: `LogitsProcessor` classes (temperature, top_k, top_p, repetition penalty, ...).
- `stopping_criteria.py`: `StoppingCriteria` classes (max_length, eos_token, ...).
- `candidate_generator.py`: cho assisted decoding (speculative).
- `streamers.py`: `TextStreamer` để stream token ra terminal.
- `continuous_batching/`: implementation continuous batching với paged attention.

Phần 4 sẽ đi sâu folder này.

## File hỗ trợ thường gặp

- `modeling_outputs.py`: dataclass cho output của model (`BaseModelOutputWithPast`, `CausalLMOutputWithPast`, `Seq2SeqLMOutput`, ...). Đọc lướt khi cần biết shape của output.
- `modeling_layers.py`: mixin cho task heads (`GenericForSequenceClassification`, ...). Đọc khi cần hiểu vì sao `LlamaForSequenceClassification` chỉ một dòng.
- `masking_utils.py`: hàm `create_causal_mask`, `create_sliding_window_mask`. Đọc khi cần tự định nghĩa attention mask.
- `modeling_rope_utils.py`: ROPE init function cho nhiều scaling strategy (linear, dynamic, yarn). Đọc khi cần extend context length.
- `activations.py`: dict `ACT2FN` map từ string sang activation function. Tham chiếu nhanh.

## File bạn có thể bỏ qua

Trong lần đọc đầu, không cần đọc:

- `pipelines/`: high-level API cho user, nhiều convention nhưng không thay đổi cốt lõi.
- `data/`: data collator và utility, chỉ liên quan training.
- `trainer.py`: huge, chỉ liên quan training với Trainer API.
- `convert_*` scripts: chỉ dùng để convert weight từ format gốc sang HF format. Đọc khi cần port model mới.
- `*_tf.py`, `*_flax.py`: TensorFlow/JAX backend, đang phase out.

## Tóm tắt

| File | Mục đích | Khi nào đọc |
|------|----------|--------------|
| `models/llama/modeling_llama.py` | Kiến trúc tham chiếu LLM | Đọc đầu tiên |
| `cache_utils.py` | KV cache | Phần 3 |
| `generation/utils.py` | Generate logic | Phần 4 |
| `modeling_utils.py` | Base class + AttentionInterface | Khi debug from_pretrained |
| `integrations/sdpa_attention.py` | SDPA backend | Phần 1 |
| `integrations/flash_attention.py` | FlashAttention backend | Phần 1 |
| `models/t5/modeling_t5.py` | Encoder-decoder reference | Phần 2 |
| `models/auto/modeling_auto.py` | Auto class registry | Phần 5 |

Bạn không phải đọc theo thứ tự trên ngay. Cứ đi vào từng phần của chuỗi này, mỗi phần sẽ chỉ chính xác file cần mở.

Chương sau giải thích AutoModel registry.

---
title: AutoClass registry
---

# Model registry và `AutoModel`

Một dòng `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")` thực ra làm rất nhiều việc. Chương này mở hộp đen đó.

## Vấn đề mà AutoClass giải

Thư viện có ~300 model gia đình. Mỗi gia đình có một class cho mỗi task: `LlamaForCausalLM`, `LlamaForSequenceClassification`, `T5ForConditionalGeneration`, `BertForMaskedLM`, ...

Nếu user phải biết tên class chính xác để load:

```python
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
```

Họ sẽ phải nhớ ~300 tên class. Tệ hơn, code không generic: thay model là phải sửa import.

`AutoModel*` giải bằng dispatch dựa trên config:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# Tự dispatch tới LlamaForCausalLM
```

`AutoModelForCausalLM` đọc `config.json` từ checkpoint, biết `model_type = "llama"`, tra registry, gọi `LlamaForCausalLM`.

## Registry cấu trúc

Có hai registry chính trong `models/auto/`:

**1. `CONFIG_MAPPING_NAMES`** trong `configuration_auto.py`. Map từ `model_type` string sang config class name string.

```python
CONFIG_MAPPING_NAMES = OrderedDict([
    ("llama", "LlamaConfig"),
    ("t5", "T5Config"),
    ("qwen2", "Qwen2Config"),
    # ... ~300 entries
])
```

**2. `MODEL_FOR_*_MAPPING_NAMES`** trong `modeling_auto.py`. Một registry riêng cho mỗi task: causal LM, masked LM, sequence classification, ...

```python
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([
    ("llama", "LlamaForCausalLM"),
    ("t5", "T5ForCausalLM"),
    ("qwen2", "Qwen2ForCausalLM"),
    # ...
])

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict([
    ("t5", "T5ForConditionalGeneration"),
    ("bart", "BartForConditionalGeneration"),
    # ...
])
```

Lưu ý: giá trị là **string** tên class, không phải class object. Lý do: lazy loading. Class thực chỉ được import khi cần.

## `_LazyAutoMapping`

```python
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
```

`_LazyAutoMapping` là dict đặc biệt: khi bạn `mapping[LlamaConfig]`, nó:

1. Lấy `model_type = "llama"` từ `LlamaConfig.model_type`.
2. Tra trong `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["llama"]` ra string `"LlamaForCausalLM"`.
3. Import lazy module `transformers.models.llama` rồi `getattr(module, "LlamaForCausalLM")`.
4. Trả về class.

Lazy là quan trọng. Nếu import mọi class lúc khởi động, `import transformers` sẽ rất chậm. Lazy delay import đến khi thực sự cần.

## `AutoModelForCausalLM` flow

Khi gọi `.from_pretrained("meta-llama/Llama-3-8B")`:

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    # Bước 1: load config từ checkpoint
    config, kwargs = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
    )

    # Bước 2: tra registry để tìm model class
    if type(config) in cls._model_mapping:
        model_class = _get_model_class(config, cls._model_mapping)
        # Bước 3: gọi from_pretrained của class thật
        return model_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **kwargs
        )
```

Đây là code mô tả, code thật phức tạp hơn vì có trust_remote_code, dynamic module loading, custom code support.

`AutoConfig.from_pretrained` cũng có flow tương tự: nó đọc `config.json`, parse field `model_type`, tra `CONFIG_MAPPING_NAMES["llama"]` ra `"LlamaConfig"`, import và parse.

## Có bao nhiêu `AutoModel*`

Nhiều, mỗi task có một. Một số phổ biến:

| Class | Task |
|-------|------|
| `AutoModel` | Encoder thuần, không head (cho feature extraction) |
| `AutoModelForCausalLM` | Causal LM (Llama, GPT, Mistral) |
| `AutoModelForMaskedLM` | Masked LM (BERT) |
| `AutoModelForSequenceClassification` | Classification |
| `AutoModelForTokenClassification` | NER, POS tagging |
| `AutoModelForQuestionAnswering` | Extractive QA |
| `AutoModelForSeq2SeqLM` | Encoder-decoder (T5, BART) |
| `AutoModelForSpeechSeq2Seq` | Whisper |
| `AutoModelForImageClassification` | ViT, DeiT |
| `AutoModelForObjectDetection` | DETR, YOLOS |
| `AutoModelForCausalLMWithValueHead` | Cho RLHF |

Mỗi class có registry riêng `MODEL_FOR_*_MAPPING_NAMES`. Một model có thể nằm trong nhiều registry: `LlamaForCausalLM` trong CAUSAL_LM, `LlamaForSequenceClassification` trong SEQUENCE_CLASSIFICATION, ...

## Cách thêm một model mới vào registry

Khi bạn thêm một model `MyModel`:

1. Tạo folder `models/my_model/` với `configuration_my_model.py` và `modeling_my_model.py`.
2. Trong `configuration_my_model.py`, định nghĩa `MyModelConfig` với class attribute `model_type = "my_model"`.
3. Thêm vào `models/auto/configuration_auto.py`: `("my_model", "MyModelConfig")` trong `CONFIG_MAPPING_NAMES`.
4. Thêm vào `models/auto/modeling_auto.py`: `("my_model", "MyModelForCausalLM")` trong `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`.
5. Export trong `__init__.py`.

Sau bốn bước, `AutoModelForCausalLM.from_pretrained("checkpoint-with-my_model")` sẽ tự load `MyModelForCausalLM`.

Đây là một trong những lý do tại sao convention nhất quán quan trọng: thêm model mới chỉ cần follow checklist, không phải nghĩ lại từ đầu.

## Trust remote code và dynamic module

Khi bạn load một model không có trong registry chính thức (ví dụ một version fork):

```python
model = AutoModelForCausalLM.from_pretrained("custom-org/custom-model", trust_remote_code=True)
```

Cờ `trust_remote_code=True` cho phép load class từ repo HuggingFace Hub thay vì từ thư viện local. Cụ thể:

1. Đọc field `auto_map` trong `config.json`. Ví dụ: `"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"`.
2. Download file `modeling_custom.py` từ Hub.
3. Dynamic import class `CustomForCausalLM` từ file đó.
4. Gọi `.from_pretrained` của class đó.

`trust_remote_code=True` là **không an toàn**: code Python từ Hub được execute trên máy của bạn. Chỉ dùng với repo bạn tin tưởng.

## Tóm tắt

Một string checkpoint qua AutoClass đi qua các bước:

```text
"meta-llama/Llama-3-8B"
   |
   v
[AutoConfig] đọc config.json
   |
   v
parse model_type = "llama"
   |
   v
[CONFIG_MAPPING_NAMES] -> "LlamaConfig"
   |
   v
import + instantiate LlamaConfig
   |
   v
[MODEL_FOR_CAUSAL_LM_MAPPING_NAMES] -> "LlamaForCausalLM"
   |
   v
import + LlamaForCausalLM.from_pretrained(...)
   |
   v
load weights, return model
```

Sáu bước, ẩn sau một dòng API. Hiểu được điều này là bạn đã hiểu một nửa cốt lõi của thư viện.

Chương cuối Phần 0: roadmap đọc.

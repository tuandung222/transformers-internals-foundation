---
title: Auto class registry
---

# Auto class registry

`AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")` magic ra `LlamaForCausalLM`. Chương này mở registry để hiểu cơ chế.

File: `src/transformers/models/auto/modeling_auto.py`, `configuration_auto.py`, `tokenization_auto.py`.

## Hai tầng dispatch

`from_pretrained` cần biết hai thứ:

1. **Config class**: để load `config.json` đúng kiểu (`LlamaConfig` vs `T5Config` vs ...).
2. **Model class**: để instantiate đúng class (`LlamaForCausalLM` vs `LlamaModel` vs `LlamaForSequenceClassification`).

Hai tầng được giải bởi hai dict global.

## `CONFIG_MAPPING`

```python
CONFIG_MAPPING_NAMES = OrderedDict([
    ("llama", "LlamaConfig"),
    ("mistral", "MistralConfig"),
    ("qwen2", "Qwen2Config"),
    ("t5", "T5Config"),
    ...
])

CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
```

`CONFIG_MAPPING` map từ `model_type` string sang class config. `_LazyConfigMapping` import class chỉ khi cần (tránh import 200 file ở startup).

Khi `AutoConfig.from_pretrained("path")`:

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    config_dict = json.load(open(f"{path}/config.json"))
    model_type = config_dict["model_type"]
    config_class = CONFIG_MAPPING[model_type]  # lookup
    return config_class(**config_dict)
```

Đơn giản: đọc field `model_type`, lookup class, instantiate.

## `MODEL_MAPPING` và task-specific mapping

```python
MODEL_MAPPING_NAMES = OrderedDict([
    ("llama", "LlamaModel"),
    ("mistral", "MistralModel"),
    ("t5", "T5Model"),
    ...
])

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([
    ("llama", "LlamaForCausalLM"),
    ("mistral", "MistralForCausalLM"),
    ...
])

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    ("llama", "LlamaForSequenceClassification"),
    ("bert", "BertForSequenceClassification"),
    ...
])
```

Mỗi task có một mapping riêng. `AutoModelForCausalLM` dùng `MODEL_FOR_CAUSAL_LM_MAPPING`. `AutoModelForSequenceClassification` dùng `MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING`.

Cùng một `model_type` "llama" có thể map tới nhiều class khác nhau tuỳ task. `AutoModel.from_pretrained` instantiate `LlamaModel` (body), `AutoModelForCausalLM.from_pretrained` instantiate `LlamaForCausalLM` (causal LM head).

## `_BaseAutoModelClass`

Mọi `Auto*` subclass `_BaseAutoModelClass`:

```python
class _BaseAutoModelClass:
    _model_mapping = None

    @classmethod
    def from_config(cls, config):
        model_class = cls._model_mapping[type(config)]
        return model_class._from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if type(config) in cls._model_mapping:
            model_class = cls._model_mapping[type(config)]
            return model_class.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
        raise ValueError(...)
```

```python
class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
```

`AutoModelForCausalLM.from_pretrained(path)`:

1. Gọi `AutoConfig.from_pretrained(path)` -> `LlamaConfig` instance.
2. Lookup `MODEL_FOR_CAUSAL_LM_MAPPING[LlamaConfig]` -> `LlamaForCausalLM`.
3. Gọi `LlamaForCausalLM.from_pretrained(path, config=config)` -> instance.

Một số người prefer hỏi: tại sao không dispatch theo `architectures` field của config (`["LlamaForCausalLM"]`)? Lý do: architectures có thể không có (config cũ), hoặc có nhiều class. Dispatch qua mapping rõ ràng hơn.

## Auto class cho non-modeling

Tương tự pattern cho tokenizer, processor, feature extractor:

```python
TOKENIZER_MAPPING_NAMES = OrderedDict([
    ("llama", ("LlamaTokenizer", "LlamaTokenizerFast")),
    ...
])

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config = AutoConfig.from_pretrained(path)
        tokenizer_class = TOKENIZER_MAPPING[type(config)]
        return tokenizer_class.from_pretrained(path, **kwargs)
```

Tuple `(slow_tokenizer, fast_tokenizer)` cho phép user chọn implementation (slow Python vs fast Rust). `use_fast=True` (default) pick fast.

## Register custom model

User có thể register model custom vào registry:

```python
from transformers import AutoConfig, AutoModel, CONFIG_MAPPING, MODEL_MAPPING

class MyConfig(PreTrainedConfig):
    model_type = "my_model"

class MyModel(PreTrainedModel):
    config_class = MyConfig

CONFIG_MAPPING.register("my_model", MyConfig)
MODEL_MAPPING.register(MyConfig, MyModel)

# Now this works:
config = AutoConfig.from_pretrained("local/my_model")
model = AutoModel.from_pretrained("local/my_model")
```

Đây là extension point cho user khi muốn dùng Auto API với model riêng.

## `_LazyAutoMapping`

```python
class _LazyAutoMapping(OrderedDict):
    def __init__(self, config_mapping, model_mapping_names):
        self._config_mapping = config_mapping
        self._model_mapping_names = model_mapping_names
        self._modules = {}
        self._extra_content = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        # Determine model_type from config class
        model_type = self._reverse_config_mapping[key.__name__]
        model_name = self._model_mapping_names[model_type]
        return self._load_attr_from_module(model_type, model_name)
```

Lazy: chỉ import module model khi user lookup. Tránh import 200 module ở startup, tiết kiệm vài giây và memory.

## `trust_remote_code` và custom model trên Hub

```python
model = AutoModel.from_pretrained("user/custom-model", trust_remote_code=True)
```

Nếu Hub repo có file `modeling_custom.py` và `configuration_custom.py`, HF download và import dynamic. Tương tự với registry nhưng không cần pre-register.

`trust_remote_code` cảnh báo user: code này từ Hub, có thể độc hại. User opt-in.

## Pitfall

**1. `model_type` không khớp**: config có `model_type="llama"`, nhưng user muốn load `LlamaForSequenceClassification`. `AutoModelForSequenceClassification.from_pretrained` lookup `MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[LlamaConfig]`. Nếu Llama không có sequence classification class -> raise.

**2. Custom model không register**: AutoModel raise KeyError. Phải register hoặc dùng class trực tiếp.

**3. `from_pretrained` không tìm thấy file**: hub không có `pytorch_model.bin` hoặc `model.safetensors`. Check `revision` argument hoặc `trust_remote_code`.

**4. Auto class trả về class sai task**: ví dụ user `AutoModel.from_pretrained` để dùng cho generate. Class trả về là `LlamaModel` (body), không phải `LlamaForCausalLM`. Generate fail vì không có lm_head. Phải dùng đúng Auto class cho task.

Chương sau ta đi vào generic mixin cho task head.

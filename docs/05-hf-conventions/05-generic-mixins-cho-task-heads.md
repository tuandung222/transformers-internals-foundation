---
title: Generic mixins cho task heads
---

# Generic mixins cho task heads

Mỗi model có hàng chục task class: `LlamaModel`, `LlamaForCausalLM`, `LlamaForSequenceClassification`, `LlamaForTokenClassification`, `LlamaForQuestionAnswering`. Mỗi task class thêm một head mỏng trên body. Chương này xem các pattern shared qua mixin và output dataclass.

## Pattern task head

```python
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, cache_position=None, logits_to_keep=0, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

Pattern bốn bước:

1. Call body model (`self.model`).
2. Lấy `hidden_states` từ output body.
3. Apply task head (`self.lm_head`, `self.classifier`, ...).
4. Compute loss nếu có labels, return output dataclass.

## `GenerationMixin`

```python
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    ...
```

`GenerationMixin` cung cấp `generate`, `_sample`, `_beam_search`, ... mọi method generation. Phần 4 đã đi chi tiết.

Chỉ class có generation mới inherit. Body model `LlamaModel` không (vì không có lm_head). Classification class cũng không (không generate sequence).

Mixin cho phép class task có generate. Đa thừa kế:

```python
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
```

MRO: `LlamaForCausalLM -> LlamaPreTrainedModel -> PreTrainedModel -> GenerationMixin -> object`. `generate` từ `GenerationMixin` accessible.

## Output dataclass

File: `src/transformers/modeling_outputs.py`. Khoảng 30+ dataclass cho mọi output shape.

```python
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
```

```python
@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
```

```python
@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
```

Mỗi task có dataclass tương ứng. `loss` cho train, `logits` cho inference. `past_key_values` cho generation (chỉ task generative).

## `ModelOutput` base

```python
class ModelOutput(OrderedDict):
    def __post_init__(self):
        class_fields = fields(self)
        ...
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.__dict__[k]
        else:
            # tuple-style indexing for backward compat
            return self.to_tuple()[k]

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())
```

Hai access pattern:

- Attribute style: `output.logits`, `output.hidden_states`.
- Tuple style: `output[0]` = first non-None field. Cho backward compat (legacy code unpack tuple).

Tách `None` field: nếu `output_hidden_states=False`, `hidden_states=None`. `to_tuple()` chỉ trả non-None.

User typical:

```python
outputs = model(**inputs)
logits = outputs.logits  # attribute style
# hoặc
logits = outputs["logits"]  # dict style
# hoặc
logits, past_kv, hidden, attn = outputs  # tuple (legacy)
```

## Vì sao tuple style legacy?

Trước transformers 4.0 (cũ lâu rồi), output là tuple. Code legacy:

```python
hidden, past_kv = model(input_ids)
```

Migration sang dataclass break compat. Solution: dataclass implement `__iter__` và `__getitem__(int)` để giả vờ tuple. Code legacy chạy được.

Hiện tại nên dùng attribute style. `return_dict=False` deprecated.

## Task head custom

Khi bạn viết task head riêng, follow pattern:

```python
class MyModelForCustomTask(MyPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyModel(config)
        self.custom_head = nn.Linear(config.hidden_size, config.num_custom_classes)
        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.model(input_ids, **kwargs)
        hidden = outputs.last_hidden_state
        logits = self.custom_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return MyCustomOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```

Dataclass custom:

```python
@dataclass
class MyCustomOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: tuple | None = None
```

## `loss_function`

```python
if labels is not None:
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
```

`loss_function` là attribute trên `PreTrainedModel`, default là `ForCausalLMLoss` (cross-entropy shift labels). Override cho task khác:

```python
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    loss_function = ForSequenceClassificationLoss
```

Tách function cho phép swap loss strategy (label smoothing, focal loss, ...) qua subclass.

## `logits_to_keep`

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

Optimization: thay vì project toàn bộ `hidden_states` (shape `(B, T, D)`) qua `lm_head` (output `(B, T, vocab_size)`), chỉ project các position cần thiết. Khi inference, ta thường chỉ cần logit của token cuối -> `logits_to_keep=1`. Tiết kiệm `(T-1)` matmul mỗi forward.

`logits_to_keep=0` (default) project toàn bộ. `logits_to_keep=1` project chỉ token cuối. `logits_to_keep=tensor([1, 5, 10])` project các position cụ thể (cho speculative decoding).

Đây là optimization tinh tế nhưng đáng kể với vocab lớn (Llama-3: 128k vocab).

## Tổng kết

Task head pattern: thin layer trên body, output dataclass. Mixin cho behavior shared (generation). Loss function tách function. Output dataclass cho type-safe API. Đây là phần "ngoại tầng" của model HF.

Chương sau ta đi vào tied weights và attention support flags.

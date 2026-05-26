---
title: TP/FSDP plans
---

# TP/FSDP plan trong `PreTrainedModel`

Distributed training/inference với mô hình lớn (>10B) cần shard weight qua nhiều GPU. Hai paradigm chính: TP (tensor parallel, shard tensor) và FSDP (fully sharded data parallel, shard parameter theo data-parallel rank). HF declare các plan này ngay trong model code, không cần wrap ngoài.

## Vì sao declare trong model?

Trước đây, user wrap model với `FSDP(model)` hoặc `torch.distributed.tensor.parallel.parallelize_module(model, plan)`. Vấn đề:

1. **Plan độ chính xác cao**: TP cần biết tensor nào shard column-wise (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`), tensor nào shard row-wise (`o_proj`, `down_proj`). Sai dimension -> wrong math.
2. **Model author hiểu cấu trúc**, user thường không.
3. **Inconsistency**: mỗi user wrap khác nhau, kết quả khác.

Solution: model author declare plan, framework apply. User chỉ cần `from_pretrained(tp_plan="auto")`.

## `_tp_plan` declaration

```python
class LlamaPreTrainedModel(PreTrainedModel):
    _tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    }
```

Dict map từ parameter name pattern (glob style với `*`) đến shard strategy.

**Colwise**: shard output dim. `q_proj` shape `(out=num_heads*head_dim, in=hidden)` -> shard along `out`. Mỗi rank giữ subset head.

**Rowwise**: shard input dim. `o_proj` shape `(out=hidden, in=num_heads*head_dim)` -> shard along `in`. Mỗi rank xử lý subset head.

Pattern Q,K,V colwise (split head) + output projection rowwise (gather lại) là classic TP cho attention layer (Megatron-LM paper).

Tương tự FFN: gate/up colwise, down rowwise.

## `_fsdp_plan`

```python
_fsdp_plan = {
    "model.layers.*": "shard",
    "model.embed_tokens": "shard",
    "lm_head": "shard",
}
```

Đơn giản hơn: chỉ định module nào shard. FSDP shard parameter theo data-parallel rank, không cần biết colwise/rowwise.

`"shard"` báo FSDP apply `FullyShardedDataParallel(module)` cho mọi instance match pattern.

## `_sp_plan` (sequence parallel)

```python
_sp_plan = {
    "model.layers.*.input_layernorm": ("scatter", "all_reduce"),
    "model.layers.*.post_attention_layernorm": ("scatter", "all_reduce"),
}
```

Sequence parallel: shard sequence dimension (T) thay vì hidden dim. Hữu ích cho long context. Plan declare communication pattern (scatter/all_reduce) cho từng module.

Phức tạp hơn, ít model HF declare. Cần TP + SP cùng nhau.

## `_pp_plan` (pipeline parallel)

```python
_pp_plan = {
    "embed_tokens": ("input_ids", "inputs_embeds"),
    "layers.*": ("hidden_states", "hidden_states"),
    "norm": ("hidden_states", "hidden_states"),
    "lm_head": ("hidden_states", "logits"),
}
```

Pipeline parallel: split model theo layer, mỗi rank xử lý vài layer. Plan declare input/output mỗi stage.

## `post_init` aggregate

Trong `PreTrainedModel.post_init`:

```python
cls_tp_plan = getattr(self, "_tp_plan", None) or {}
self._tp_plan = dict(cls_tp_plan)

if self.base_model is self:
    self._tp_plan = self.config.base_model_tp_plan.copy() if ... else {}

for name, module in self.named_children():
    if plan := getattr(module, "_tp_plan", None):
        self._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
```

Plan từ:

1. Class-level `_tp_plan` (declare static trên class).
2. Base model's `config.base_model_tp_plan` (override per-checkpoint).
3. Child modules' `_tp_plan` (composite model: ImageEncoder + TextDecoder).

Aggregate tất cả thành một dict ở top-level model. `from_pretrained` đọc dict này để apply TP.

## Apply TP lúc load

```python
def from_pretrained(cls, ...):
    ...
    if distributed_config.tp_size > 1:
        active_tp_plan = model._tp_plan
        # Apply TP plan to model
        for name, strategy in active_tp_plan.items():
            module = get_module_by_name(model, name)
            parallelize_module(module, mesh, strategy)
    ...
```

Khi user pass `tp_size > 1`, HF resolve plan và apply qua `torch.distributed.tensor.parallel`.

API user:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    tp_plan="auto",
    torch_dtype=torch.bfloat16,
)
# Model is now sharded across 8 GPUs (if launched with 8 ranks)
```

`tp_plan="auto"` báo HF dùng `_tp_plan` declared. User cũng có thể pass custom plan.

## `verify_tp_plan`

Sau load:

```python
if logger.level >= logging.WARNING:
    verify_tp_plan(expected_keys, load_config.tp_plan)
```

Check mọi param trong state_dict có khớp với plan. Warn nếu có param không nằm trong plan (sẽ replicate, không shard) hoặc plan reference param không tồn tại.

## `_no_split_modules` reminder

Đã đề cập chương trước: `_no_split_modules = ["LlamaDecoderLayer"]`. Cho `device_map="auto"` (Accelerate). Khác với TP plan: device_map là model parallel coarse-grained (layer X ở GPU 0, layer Y ở GPU 1). TP là intra-layer shard.

Hai có thể kết hợp: pipeline parallel (device_map) + tensor parallel (tp_plan) cho model siêu lớn.

## Tổng hợp 4 paradigm

| Paradigm | Granularity | Communication | Use case |
|---|---|---|---|
| Device map | Module (layer) | None (forward sequential) | Inference single-host, model > GPU memory |
| FSDP | Parameter | All-gather forward/backward | Training, batch lớn |
| TP | Tensor dim | All-reduce forward + backward | Inference latency, training với batch nhỏ |
| PP | Layer group | Send/recv between stages | Training extreme model với multi-node |
| SP | Sequence dim | Scatter/all-reduce | Long context |

HF declare plan cho TP, FSDP, SP, PP. Device map là Accelerate handle, không trong model.

## Pitfall

**1. Plan không khớp với weight shape**: ví dụ `q_proj` declare colwise nhưng tensor shape không chia hết cho world_size. Raise.

**2. Quên collective sync sau forward**: TP cần all-reduce output của row-wise. Framework handle, không phải user. Nhưng nếu user wrap manual, có thể quên.

**3. Mixed batch shape giữa rank**: TP yêu cầu shape giống nhau mọi rank. Padding mask khác nhau giữa rank gây mismatch.

**4. KV cache với TP**: cache phải shard tương ứng K/V projection. HF cache class handle tự động.

Chương cuối: `from_pretrained` flow.

---
title: StaticCache và torch.compile
---

# `StaticCache` cho `torch.compile`

DynamicCache cat tensor mỗi step, shape thay đổi, không compile được. StaticCache preallocate tensor max size, mutate in-place. Cùng kết quả, compile được.

## `StaticLayer.lazy_initialization`

```python
class StaticLayer(CacheLayerMixin):
    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.cumulative_length = torch.tensor([0], dtype=int)

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.cumulative_length = self.cumulative_length.to(self.device)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)

        self.is_initialized = True
```

Vài điểm:

1. **Allocate full tensor max size** ngay từ đầu bằng `torch.zeros`. Vì preallocate, ta cần biết `max_cache_len` lúc construct.
2. **`torch._dynamo.mark_static_address`**: Báo Dynamo rằng buffer này có địa chỉ memory cố định, dùng được trong CUDA graph. Nếu không mark, compile từ chối cudagraph mode hoặc recompile khi địa chỉ thay đổi.
3. **`cumulative_length` là tensor, không int**: Compile cần value tách rời compile-time. Nếu int, mỗi step có cumulative_length khác, recompile. Nếu tensor, cùng compile graph mỗi step, mutation trên tensor.

## `update`: in-place mutation

```python
def update(self, key_states, value_states, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)

    kv_length = key_states.shape[-2]
    cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
    self.cumulative_length.add_(kv_length)

    try:
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
    except NotImplementedError:
        self.keys[:, :, cache_position] = key_states
        self.values[:, :, cache_position] = value_states

    return self.keys, self.values
```

`index_copy_(dim, index, source)` ghi `source` vào `self.keys` tại các vị trí `cache_position` trên dim 2 (sequence dim). In-place mutation, không cấp tensor mới. Compile-friendly.

Lưu ý: ta **return `self.keys` full** (chứa cả vùng zero chưa fill). Attention không biết vùng nào valid vùng nào zero, nên cần mask. Mask được tạo bởi caller dựa trên `cumulative_length` và `query_length`.

Attention `Q @ K^T` cho cả vùng zero vẫn ra giá trị, nhưng softmax sau mask sẽ zero ra. Hơi lãng phí FLOPs cho vùng zero, nhưng compile-friendly đáng giá hơn.

## `cumulative_length` tensor pattern

Việc dùng `torch.tensor([0])` thay vì Python int là cốt lõi cho compile. Một lỗi thường gặp khi viết cache custom:

```python
# SAI: int sẽ làm recompile
self.cumulative_length = 0
self.cumulative_length += kv_length

# ĐÚNG: tensor + in-place add
self.cumulative_length = torch.tensor([0], dtype=int, device=device)
self.cumulative_length.add_(kv_length)
```

Pattern này lan ra mọi state mà cache cần track: dùng tensor, mutate in-place.

## `StaticSlidingWindowLayer.update`

```python
def update(self, key_states, value_states, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)

    kv_length = key_states.shape[-2]
    current_length = self.cumulative_length_int
    is_full = current_length >= self.max_cache_len
    self.cumulative_length_int += kv_length

    if is_full:
        if key_states.shape[-2] == 1:
            new_keys = self.keys.roll(-1, dims=-2)
            new_values = self.values.roll(-1, dims=-2)
            index = torch.tensor([-1], dtype=int, device=self.device)
            new_keys[:, :, index] = key_states
            new_values[:, :, index] = value_states
            self.keys.copy_(new_keys)
            self.values.copy_(new_values)
        ...
```

Khi sliding window đầy: roll tensor sang trái 1 vị trí (drop token cũ nhất), ghi K/V mới vào slot cuối. `torch.roll` là phép data-independent (không nhánh theo value), compile-friendly. `self.keys.copy_(new_keys)` thay vì `self.keys = new_keys` để giữ địa chỉ tĩnh đã `mark_static_address`.

Khi sliding window chưa đầy: chỉ ghi vào vị trí kế tiếp như StaticLayer thường.

## `StaticCache` parent

```python
class StaticCache(Cache):
    def __init__(self, config, max_cache_len=None, offloading=False, ...):
        ...
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        layers = []
        for layer_type in layer_types:
            if "sliding" in layer_type:
                layers.append(StaticSlidingWindowLayer(max_cache_len, config.sliding_window))
            else:
                layers.append(StaticLayer(max_cache_len))
        super().__init__(layers=layers, offloading=offloading)
```

Constructor đọc `config.layer_types` để phân loại layer (full vs sliding) và allocate đúng class.

## Khi nào dùng StaticCache?

`StaticCache` chỉ cần thiết khi:

1. **`torch.compile(model)` cho decode loop**: muốn compile cả model forward để dùng CUDA graph. Decode 100 token nhanh hơn nhiều so với eager nhờ kernel fusion.
2. **`torch.export`**: muốn save graph cho deployment (ONNX, AOT inductor, ...). Export yêu cầu shape tĩnh.
3. **Inference latency-critical**: mỗi step decode tiết kiệm vài microsecond cộng dồn đáng kể với throughput cao.

User dùng qua `generate`:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="cuda")
model.generation_config.cache_implementation = "static"
out = model.generate(input_ids, max_new_tokens=100)
```

`cache_implementation="static"` báo `generate` khởi tạo `StaticCache` với `max_cache_len = input_len + max_new_tokens`.

## Pitfall: max_cache_len phải đủ lớn

```python
cache = StaticCache(config, max_cache_len=100)
model.generate(input_ids=very_long_prompt, max_new_tokens=10)  # OOB error
```

Nếu `input_len + max_new_tokens > max_cache_len`, `index_copy_` ghi ra ngoài tensor, crash. Khác với DynamicCache (grow tự do), StaticCache cần plan trước.

## Pitfall: compile prefill

Prefill (input_len token) khác shape với decode (1 token). Nếu compile cả prefill, phải compile hai graph khác nhau. HF default **không compile prefill**, chỉ compile decode. Khi user muốn compile prefill (qua `model.compile(...)`), shape động phải được xử lý qua dynamic shape, nhưng trong nhiều case fail.

Convention: compile decode, eager prefill. Tốc độ vẫn cải thiện nhiều vì decode chiếm 99% thời gian generate.

Chương kế tiếp ta đọc Quantized và Offloaded cache.

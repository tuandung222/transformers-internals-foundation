---
title: DynamicCache
---

# `DynamicCache` chi tiết

`DynamicCache` là cache mặc định cho generation. Đặc trưng: tensor K, V grow theo `torch.cat` mỗi step. Linh hoạt nhưng không compile-friendly.

## `DynamicLayer.update`

```python
class DynamicLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, config: PreTrainedConfig | None = None):
        super().__init__()

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values
```

Hết. Hai dòng `torch.cat`, một dòng return. Đơn giản.

Vài quan sát:

1. **Lazy init**: Layer được tạo rỗng. Khi `update` đầu tiên gọi, ta đọc dtype/device từ K mới. Cho phép linh hoạt với dtype mixed (model bf16 nhưng cache fp16) hoặc multi-device.
2. **Reallocate mỗi cat**: `torch.cat` cấp tensor mới, copy data cũ và mới vào. Cost O(T * D) memory write mỗi step. Đây là overhead chính của DynamicCache.
3. **Return luôn full tensor**: cache giữ K/V history, return cho attention. Attention sẽ dùng tensor return để compute, không tính lại từ history.

## Tại sao `torch.cat` mỗi step chấp nhận được?

Cat là O(T * D) write, plus allocate. Tưởng chừng chậm, nhưng so với attention `Q @ K^T` (cũng O(T * D) per step), cost cat không dominating. Trên GPU bf16, allocate + cat tốc độ khoảng vài microsecond cho cache nhỏ. Bottleneck thực sự là attention + linear, không phải cat.

Tuy nhiên, có hai vấn đề:

1. **Fragmentation memory**: mỗi cat cấp tensor mới. Tensor cũ vẫn nằm đó cho tới khi GC. Tổng peak memory có thể 2x cache size trong moment cat.
2. **torch.compile không tối ưu được**: shape tensor thay đổi mỗi step, recompile mỗi step. Compile vô dụng.

`StaticCache` giải hai vấn đề này bằng cách preallocate.

## `DynamicCache` parent class

```python
class DynamicCache(Cache):
    """A cache that grows dynamically as more tokens are generated."""

    def __init__(self, config=None, ddp_cache_data=None, layers=None, offloading=False, ...):
        if layers is None and ddp_cache_data is None:
            if config is not None:
                # Build layers per config.layer_types
                layers = []
                for layer_type in config.layer_types:
                    layer_class = LAYER_TYPE_CACHE_MAPPING[layer_type]
                    layers.append(layer_class(config))
            else:
                # Use layer_class_to_replicate (lazy)
                super().__init__(layer_class_to_replicate=DynamicLayer, offloading=offloading)
                return
        ...
        super().__init__(layers=layers, offloading=offloading)
```

Hai chế độ khởi tạo:

**Chế độ A: với config**, đọc `config.layer_types`. Đây là list string như `["full_attention", "sliding_attention", "full_attention", ...]` cho Gemma-2. Cache build mỗi layer với class đúng từ `LAYER_TYPE_CACHE_MAPPING`. Mỗi layer có thể là `DynamicLayer` hoặc `DynamicSlidingWindowLayer`.

**Chế độ B: không config**, dùng `layer_class_to_replicate=DynamicLayer`. Cache khởi tạo rỗng, layer được append lazy khi `update(..., layer_idx=k)` gọi với `k >= len(layers)`.

## Sliding window layer

```python
class DynamicSlidingWindowLayer(DynamicLayer):
    is_sliding = True

    def __init__(self, config=None, sliding_window=None):
        super().__init__()
        if sliding_window is None:
            sliding_window = getattr(config, "sliding_window", None)
        self.sliding_window = sliding_window
        self.cumulative_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]

        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)

        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        return full_key_states, full_value_states
```

Tinh tế: lưu cache đã crop `[..., -sliding_window+1:]`, nhưng **return full** (cache cũ + K mới). Tại sao? Để attention thấy đủ token mới `+ sliding_window-1` token cũ. Bước decode kế tiếp, K cũ nhất trong full sẽ bị drop khỏi cache nhưng K mới của step này sẽ được giữ.

Sliding window logic giúp model như Mistral (window 4096) không cần lưu cache vô hạn. Cache bị bound ở `sliding_window` token.

## `__iter__` cho compatibility

```python
def __iter__(self):
    for layer in self.layers:
        yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)
```

Cho phép unpack: `for k, v, sw in cache: ...`. Tương thích với DDP wrapping cũ (chuyển cache thành tuple cho serialization). Tránh dùng pattern này trực tiếp; gọi `.layers[idx].keys` rõ ràng hơn.

## Vì sao đây là default?

DynamicCache linh hoạt nhất: không cần biết trước max_length, không cần config phức tạp. User chỉ cần `model.generate(...)` và HF tự khởi tạo. Trade-off: không compile-friendly, allocate mỗi step. Cho hầu hết use case (chat, eval), trade-off chấp nhận được.

Chương kế tiếp ta đọc `StaticCache` cho khi muốn compile.

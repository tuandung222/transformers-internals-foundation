---
title: CacheLayerMixin
---

# `CacheLayerMixin` interface

Một `Cache` trong HF là **list của các layer cache**, mỗi layer là một subclass của `CacheLayerMixin`. Tổ chức này linh hoạt: có thể mix các layer khác loại (ví dụ Gemma-2 có sliding window mỗi 2 layer, full attention mỗi 2 layer khác).

## Class abstract

Trong `src/transformers/cache_utils.py`:

```python
class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False
    layer_type: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        layer_type = cls.__dict__.get("layer_type", None)
        if layer_type is not None:
            LAYER_TYPE_CACHE_MAPPING[layer_type] = cls

    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False

    @abstractmethod
    def lazy_initialization(self, key_states, value_states) -> None: ...

    @abstractmethod
    def update(self, key_states, value_states, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, query_length: int) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...
```

Mọi cache layer cần implement 5 method abstract. Đọc từng method.

## `__init_subclass__` hook và registry

```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    layer_type = cls.__dict__.get("layer_type", None)
    if layer_type is not None:
        LAYER_TYPE_CACHE_MAPPING[layer_type] = cls
```

Pattern Pythonic: mỗi subclass có thể đặt `layer_type = "sliding_attention"` (ví dụ). Khi class được import, `__init_subclass__` chạy và register vào `LAYER_TYPE_CACHE_MAPPING`. Sau đó, `DynamicCache` dùng dict này để chọn loại layer cache cho từng layer dựa trên `config.layer_types`.

Đây là plugin pattern: subclass mới chỉ cần khai báo `layer_type`, không cần sửa code central.

## `is_compileable`

Class attribute `is_compileable = False` (default). `StaticLayer` override thành `True`. Cho phép `Cache` (parent) biết layer này có thể compile được không, để dispatch path đúng (compile vs eager).

## `lazy_initialization`

```python
@abstractmethod
def lazy_initialization(self, key_states, value_states) -> None: ...
```

Mục đích: lấy thông tin (dtype, device, num_heads, head_dim) **runtime** từ K/V mới đến lần đầu, không cần biết trước ở constructor. Lợi ích:

1. Dtype/device theo data thực tế, không cần ép user pass `dtype=torch.bfloat16, device="cuda:0"`.
2. Hỗ trợ TP/FSDP: shape K/V có thể là sharded, không thể biết trước.
3. Đảm bảo tensor được allocate đúng device, tránh implicit copy.

`DynamicLayer` initialize tensor rỗng `torch.tensor([])` rồi cat sau. `StaticLayer` allocate full tensor `torch.zeros((B, H, max_len, d))`.

## `update`: trái tim của cache

```python
@abstractmethod
def update(self, key_states, value_states, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Nhận K/V mới (`(B, H_kv, q_len, d_h)`), trả về `(K, V)` đầy đủ cho attention compute. Implementation khác nhau:

- `DynamicLayer`: `torch.cat([self.keys, key_states], dim=-2)`, trả lại tensor cat.
- `StaticLayer`: `self.keys[:, :, cache_position] = key_states`, trả `self.keys` full (mask vùng chưa fill bằng `attention_mask`).
- `DynamicSlidingWindowLayer`: cat trước, sau đó slice giữ `sliding_window-1` token cuối. Trả về **full** K/V để attention thấy mọi token mới + cache, **rồi** crop cache.

Chú ý: `update` không chỉ "append", mà còn **return** tensor để attention compute. Tensor return có thể khác với tensor lưu trong cache (sliding window case).

## `get_mask_sizes`: tính shape mask

```python
@abstractmethod
def get_mask_sizes(self, query_length: int) -> tuple[int, int]: ...
```

Trả `(kv_length, kv_offset)`. Khi tạo mask cho attention, ta cần biết:

- `kv_length`: tổng số K/V mà mask cần che cho query này.
- `kv_offset`: offset của K/V đầu tiên trong context tuyệt đối (cho sliding window, một số K cũ đã bị drop).

`DynamicLayer`:

```python
def get_mask_sizes(self, query_length):
    return self.get_seq_length() + query_length, 0
```

`DynamicSlidingWindowLayer`:

```python
def get_mask_sizes(self, query_length):
    is_full = self.cumulative_length >= self.sliding_window
    kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
    kv_length = (self.sliding_window - 1 + query_length) if is_full else (self.cumulative_length + query_length)
    return kv_length, kv_offset
```

Logic sliding window phức tạp hơn: khi cache đầy, ta chỉ giữ `sliding_window-1` token, nên mask phải reflect đúng range.

## `get_seq_length` và `get_max_cache_shape`

```python
@abstractmethod
def get_seq_length(self) -> int: ...

@abstractmethod
def get_max_cache_shape(self) -> int: ...
```

`get_seq_length`: số token thật cache đang giữ. Dùng để compute position ids.

`get_max_cache_shape`: capacity. DynamicLayer trả `-1` (không giới hạn). StaticLayer trả `max_cache_len`. SlidingWindowLayer trả `sliding_window`.

## Method extra: offload, prefetch, reset, reorder_cache

Không abstract, có default implementation:

```python
def offload(self):
    if self.is_initialized:
        self.keys = self.keys.to("cpu", non_blocking=True)
        self.values = self.values.to("cpu", non_blocking=True)

def prefetch(self):
    if self.is_initialized and self.keys.device != self.device:
        self.keys = self.keys.to(self.device, non_blocking=True)
        self.values = self.values.to(self.device, non_blocking=True)

def reset(self):
    if self.is_initialized:
        self.keys.zero_()
        self.values.zero_()
    ...

def reorder_cache(self, beam_idx):
    if self.get_seq_length() > 0:
        self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
        self.values = self.values.index_select(0, beam_idx.to(self.values.device))
```

Các method này được parent class `Cache` gọi khi cần. `reorder_cache` cho beam search là quan trọng nhất; sai sẽ làm beam search ra kết quả không hợp lệ.

## Tóm lại

`CacheLayerMixin` định nghĩa contract: cache layer phải biết cách initialize lazy, update, trả length, trả mask sizes, trả max shape. Đây là interface gọn nhưng đầy đủ. Mọi cache class (Dynamic, Static, Quantized, Offloaded) đều subclass với policy update khác nhau. Chương sau ta đọc `DynamicCache` chi tiết.

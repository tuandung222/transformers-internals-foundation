---
title: Quantized và Offloaded Cache
---

# `QuantizedCache` và `OffloadedCache`

Long context (32k, 128k, 1M) làm KV cache vượt VRAM. Hai kỹ thuật để mở rộng: quantize (lưu ít bit hơn) và offload (đẩy ra CPU). Cả hai có cost: quantize chậm hơn vì quant/dequant overhead, offload chậm vì PCIe transfer. HF cài đặt cả hai như cache class.

## QuantizedCache: ý tưởng KIVI

Paper KIVI (2024) đề xuất: lưu K, V cũ ở dạng quantized (2-bit, 4-bit), giữ N token gần nhất ở precision gốc. Khi attend, dequantize chỉ phần đã quantize. Vì dequant nhanh hơn nhiều so với cost của attention, overhead chấp nhận được, trong khi memory giảm 4-8 lần.

```python
class QuantizedLayer(DynamicLayer):
    def __init__(self, nbits=4, axis_key=0, axis_value=0, q_group_size=64, residual_length=128):
        super().__init__()
        self.nbits = nbits
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.cumulative_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        self.cumulative_length += key_states.shape[-2]

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self._quantized_keys = self._quantize(key_states.contiguous(), axis=self.axis_key)
            self._quantized_values = self._quantize(value_states.contiguous(), axis=self.axis_value)
            return key_states, value_states

        dequant_keys = self._dequantize(self._quantized_keys)
        dequant_values = self._dequantize(self._quantized_values)
        keys_to_return = torch.cat([dequant_keys, self.keys, key_states], dim=-2)
        values_to_return = torch.cat([dequant_values, self.values, value_states], dim=-2)

        if self.keys.dim() == 4 and self.keys.shape[-2] + 1 >= self.residual_length:
            # Flush residual buffer vào quantized
            self._quantized_keys = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
            self._quantized_values = self._quantize(values_to_return.contiguous(), axis=self.axis_value)
            self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            self.values = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        return keys_to_return, values_to_return
```

Logic:

1. **Hai buffer**: `_quantized_keys/values` (quantized phần cũ) và `self.keys/values` (residual, precision gốc). Residual buffer giữ tối đa `residual_length` token.
2. **Khi `update`**: dequant phần quantized, cat với residual và K mới, trả về cho attention. Sau đó:
   - Nếu residual đủ đầy (`>= residual_length`): flush, re-quantize toàn bộ history (cũ + residual + mới), reset residual.
   - Nếu chưa: chỉ append K mới vào residual.
3. **Lợi**: phần lớn cache ở dạng quantized, chỉ dequant lúc attention. Trade-off: cost dequant mỗi step nhỏ hơn cost memory bandwidth load full precision cache.

## Backend: quanto và HQQ

`QuantizedLayer` là abstract; hai backend cụ thể:

```python
class QuantoQuantizedLayer(QuantizedLayer):
    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
        return quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
    def _dequantize(self, qtensor):
        return qtensor.dequantize()

class HQQQuantizedLayer(QuantizedLayer):
    def _quantize(self, tensor, axis):
        qtensor, meta = self.quantizer.quantize(tensor, ...)
        ...
        return qtensor, meta
    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        return self.quantizer.dequantize(quant_tensor, meta)
```

- **Quanto**: HuggingFace's optimum-quanto, hỗ trợ INT2/INT4 per-channel.
- **HQQ**: Half-Quadratic Quantization, hỗ trợ INT1/2/3/4/8 với group_size tuỳ chỉnh.

Cả hai là external lib, HF chỉ wrap. User config qua `generation_config.cache_implementation = "quantized"` và `cache_config = QuantizedCacheConfig(backend="quanto", nbits=4)`.

## Số liệu thực

Llama-3 8B, batch=1, context 128k:

| Cache | Size | Decode latency | Quality |
|---|---|---|---|
| Dynamic bf16 | 17.2 GB | baseline | baseline |
| Quantized 4-bit (HQQ) | 4.3 GB | +15% | dưới 1% perplexity loss |
| Quantized 2-bit (KIVI/Quanto) | 2.1 GB | +30% | 1-3% perplexity loss |

4-bit gần như lossless và memory giảm 4x. 2-bit aggressive hơn, có quality drop nhưng cho phép context dài.

## OffloadedCache

Ý tưởng đơn giản hơn: lưu K/V trên CPU RAM, prefetch về GPU khi cần. Class trong code:

```python
class Cache:
    def __init__(self, ..., offloading=False, offload_only_non_sliding=True):
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def update(self, key_states, value_states, layer_idx, ...):
        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, ...)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values
```

Flow mỗi `update`:

1. **Wait** prefetch stream của step trước hoàn tất.
2. **Prefetch** layer `layer_idx + 1` (layer kế tiếp): bắt đầu copy từ CPU lên GPU trên prefetch_stream (không block).
3. **Update + attention** trên layer hiện tại (data đã sẵn trên GPU từ prefetch step trước).
4. **Offload** layer vừa update về CPU sau khi xong attention.

Pattern double-buffering: trong lúc GPU làm attention layer `k`, prefetch_stream tải layer `k+1`. Khi attention layer `k` xong, layer `k+1` đã sẵn sàng. Hiệu năng gần với in-VRAM cache nếu PCIe bandwidth đủ.

## `offload_only_non_sliding`

Optimization: nếu một số layer dùng sliding window (cache nhỏ, vài MB), không cần offload chúng. Chỉ offload các layer full attention (cache vài GB). Logic:

```python
def offload(self, layer_idx, only_non_sliding=True):
    if not (only_non_sliding and self.is_sliding[layer_idx]):
        self.layers[layer_idx].offload()
```

Gemma-2 (full mỗi 2 layer, sliding mỗi 2 layer khác) hưởng lợi nhiều: chỉ offload nửa số layer, vẫn tiết kiệm phần lớn memory.

## Sequence diagram offload step

```
Step k decode:
  GPU stream (default):
    | layer 0 attention (cache đã có trên GPU) |
    |   -> offload layer 0 sang CPU sau khi xong |
    | layer 1 attention (chờ prefetch xong)      |
    ...
  Prefetch stream (parallel):
    | prefetch layer 1 từ CPU lên GPU |
    | prefetch layer 2 |
    ...
```

Vài tinh tế:

- **Non-blocking H2D copy**: dùng `non_blocking=True` để CPU không chờ. GPU schedule copy trên prefetch_stream.
- **Stream sync**: `default_stream.wait_stream(prefetch_stream)` đảm bảo dữ liệu prefetched đã đến trước khi attention dùng.
- **Bandwidth limit**: PCIe Gen4 x16 ~ 32 GB/s. Layer 1 GB cache offload + prefetch tốn ~60 ms. Nếu attention nhanh hơn 60 ms, prefetch là bottleneck.

## Kết hợp: QuantizedOffloadedCache?

Có thể quantize 4-bit (size 1/4) **rồi** offload (CPU bandwidth load 1/4). Combo này được dùng cho context cực dài (1M token Llama-3.1-128k extended). HF không có class đóng gói sẵn; user compose qua subclass.

## Khi nào không dùng?

- Decode latency-critical với short context: Dynamic cache là đủ.
- Compile mode: offload không compile-friendly (stream sync data-dependent).
- Single token decode rate priority: quant overhead nuốt mất tiết kiệm.

Phần cache còn một chương cuối: sequence diagram end-to-end của một decode step.

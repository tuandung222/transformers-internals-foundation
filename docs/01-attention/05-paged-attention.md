---
title: Paged attention
---

# Paged attention và continuous batching

Khi serve LLM cho nhiều request đồng thời, ta gặp vấn đề: mỗi request có context khác nhau, prompt khác nhau, decode tới độ dài khác nhau. Batching tĩnh (pad tới max length) lãng phí compute. Paged attention, đề xuất bởi vLLM, là kỹ thuật KV cache page-based cho phép **continuous batching**: kết hợp tự do nhiều request trong cùng một batch GPU.

## Vì sao cần paged?

Cache KV của một request có thể dài 4k, request khác dài 32k. Nếu allocate contiguous block cho mọi request theo max length, GPU memory cạn nhanh. Nếu allocate đúng kích thước thì khi extend phải realloc và copy, rất chậm.

Paged attention chia KV cache thành các **page** cố định (ví dụ 16 token mỗi page). Mỗi request có một list page indices. Khi request cần thêm slot, allocator cấp page mới từ pool. Tương tự virtual memory của OS.

Lợi ích:

1. Không fragment: page có kích thước cố định, allocator đơn giản.
2. Share prompt: nếu nhiều request share prefix (system prompt), các page đó được share, tiết kiệm memory.
3. Continuous batching: batch chứa request ở các stage khác nhau (vài đang prefill, vài đang decode), kernel tự xử lý qua page table.

## Trong transformers

Trong `AttentionInterface._global_mapping`:

```python
"paged|flash_attention_4": paged_attention_forward,
"paged|flash_attention_3": paged_attention_forward,
"paged|flash_attention_2": paged_attention_forward,
"paged|sdpa": sdpa_attention_paged_forward,
"paged|eager": eager_paged_attention_forward,
```

Pattern key `paged|<backend>` báo cho dispatcher: dùng paged-aware variant của backend. Trong `integrations/`, có các file:

- `paged_attention.py`: paged wrapper cho FlashAttention.
- `sdpa_paged_attention.py`: paged variant cho SDPA.
- `eager_paged_attention.py`: paged variant cho eager (chậm, chỉ để debug).

## Page table structure

Paged cache không lưu K, V dưới dạng `(batch, num_heads, seq_len, head_dim)` mà dưới dạng:

- `k_cache`: tensor `(num_pages, page_size, num_kv_heads, head_dim)`.
- `v_cache`: tensor `(num_pages, page_size, num_kv_heads, head_dim)`.
- `block_table`: tensor `(batch, max_num_pages)` map request_id -> list page indices.
- `seq_lens`: tensor `(batch,)` độ dài thực của từng request.

Khi gọi `paged_attention_forward`, kernel nhận `block_table` và `seq_lens`. Bên trong kernel, vòng lặp qua block, dereference qua `block_table` để lấy K, V đúng page. Đây là indirection thêm một bước so với contiguous, nhưng kernel của vLLM tối ưu rất tốt.

## Khi nào dùng paged trong HF?

HF transformers bản thân không tự build serving stack (đã có vLLM, TGI cho việc đó). Nhưng các function paged được expose để:

1. User test logic continuous batching ở local.
2. Integration test cho các script benchmark.
3. Library downstream (TGI) gọi trực tiếp.

Trong production, hầu hết user vẫn dùng vLLM hoặc TGI thay vì paged backend của HF trực tiếp.

## Code path đơn giản hoá

Pseudo của `paged_attention_forward`:

```python
def paged_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs):
    cache = kwargs["cache"]
    block_table = cache.block_table
    seq_lens = cache.seq_lens
    new_k_pages, new_v_pages = cache.update(key, value, module.layer_idx)

    attn_output = flash_attn_with_kvcache(
        query,
        k_cache=new_k_pages,
        v_cache=new_v_pages,
        block_table=block_table,
        cache_seqlens=seq_lens,
        softmax_scale=scaling,
        causal=module.is_causal,
    )
    return attn_output, None
```

Hai khác biệt chính so với backend thường:

1. K, V mới không materialize thành full tensor mà ghi vào page slot tương ứng.
2. Kernel attention nhận page table thay vì K, V contiguous.

## Pitfall thường gặp

**Page size sai.** Nếu page size không khớp giữa allocator và kernel (ví dụ allocator dùng 16, kernel hardcode 32), kết quả sai im lặng. Luôn config nhất quán.

**Block table dtype.** Kernel mong đợi `int32`. Pass `int64` có thể crash hoặc sai.

**Pad token trong page cuối.** Page cuối của request thường chỉ chứa vài token thật, phần còn lại là rác. Kernel phải đọc `seq_lens` để mask đúng. Nếu quên pass, model attend cả phần rác, output bị nhiễu.

**Cache offset khi append.** Khi decode (q_len=1), token mới phải ghi vào slot `seq_lens[i] % page_size` của page cuối của request `i`. Nếu sai offset, K/V mới ghi đè token cũ, model "quên" context.

Phần 3 (KV Cache) đi sâu vào các Cache class nói chung. Tại đây ta chỉ cần nắm: paged backend là một biến thể của attention, được kích hoạt qua `attn_implementation="paged|sdpa"` hay tương tự, và dùng cho continuous batching.

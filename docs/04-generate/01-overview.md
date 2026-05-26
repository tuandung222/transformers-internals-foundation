---
title: Tổng quan Phần 4
---

# Phần 4: `generate` và sampling

`model.generate(...)` là API user-facing nhất của transformers. Bên trong nó là một orchestration tinh tế: parse config, validate, chuẩn bị logits processor và stopping criteria, prefill, loop decode, sample, broadcast. Phần này mở từng tầng.

## Tại sao đáng học?

User thường thấy `generate` như một black box. Nhưng khi:

- Sample không đúng (output lệch, lặp, ngắt sớm): cần debug logits processor.
- Latency cao: cần biết prefill vs decode, biết bật compile.
- Custom decoding (constrained, speculative): cần biết hook chỗ nào.
- Multimodal generation: cần biết encoder_outputs chảy thế nào.

Hiểu `generate` cho phép can thiệp đúng chỗ, không monkey-patch toàn bộ.

## Mục tiêu

Sau Phần 4, bạn:

1. Vẽ flow từ `model.generate(...)` xuống `_sample` xuống `model.forward`.
2. Đọc `_sample` và giải thích vòng while: prefill, decode, logit process, sample, append, stopping check.
3. Hiểu `LogitsProcessor` (temperature, top-k, top-p, repetition penalty) hoạt động ra sao.
4. Hiểu `StoppingCriteria` (max length, eos, stopping string).
5. Biết các sample strategy: greedy, sample multinomial, beam search, assisted.
6. Hiểu compile path: khi nào `generate` compile model forward, khi nào không.

## Cấu trúc Phần 4

8 chương:

- Chương 2: Top-level `generate`. Parse config, validate, chọn mode.
- Chương 3: `_sample` chi tiết. Sample/greedy chính.
- Chương 4: `LogitsProcessor` và transform logits.
- Chương 5: `StoppingCriteria` và terminate.
- Chương 6: Beam search. Multi-candidate exploration.
- Chương 7: Assisted decoding (speculative). Draft model + verify.
- Chương 8: Compile và CUDA graph. Khi nào bật, pitfall.

## Mô hình tinh thần

`generate` là một state machine với ba phase:

```
┌────────────────────────────────┐
│ Phase 1: Setup                 │
│  - parse config                 │
│  - choose mode (sample/beam/...)│
│  - build processors             │
│  - encoder forward (nếu cần)    │
└────────────┬───────────────────┘
             │
             v
┌────────────────────────────────┐
│ Phase 2: Loop                  │
│  - prefill (1 lần)              │
│  - while not done:              │
│     - decode 1 token            │
│     - process logits            │
│     - sample/pick               │
│     - append to input_ids       │
│     - check stopping            │
└────────────┬───────────────────┘
             │
             v
┌────────────────────────────────┐
│ Phase 3: Finalize              │
│  - format output (tensor or     │
│    GenerateOutput dataclass)    │
│  - cleanup (cache, streamer)    │
└────────────────────────────────┘
```

Phase 2 là nơi đa số thời gian được tiêu. Phase 1 và 3 không thường thấy ở user code nhưng có nhiều subtle.

## Quan sát: thiết kế qua function dispatch

Trong `generation/utils.py`:

```python
GENERATION_MODES_MAPPING = {
    GenerationMode.SAMPLE: "_sample",
    GenerationMode.GREEDY_SEARCH: "_sample",
    GenerationMode.BEAM_SEARCH: "_beam_search",
    GenerationMode.BEAM_SAMPLE: "_beam_search",
    GenerationMode.ASSISTED_GENERATION: "_assisted_decoding",
    ...
}
```

Đúng pattern dispatch ta đã thấy ở attention backend: dict map từ enum sang method name. Mỗi mode là một method `_sample`, `_beam_search`, `_assisted_decoding`. Một số mode (DOLA, contrastive search) đã được deprecate ra repo cộng đồng, chỉ còn ba mode mainstream.

Quan sát thú vị: greedy search và sample dùng cùng `_sample`. Khác biệt là `do_sample=True` vs `False` flag. Hiệu quả gộp hai method giống nhau, chỉ khác chỗ pick token.

## Lưu ý cho phần học

Phần 4 không cố che dấu phức tạp của `generate`. Code thật của HF có khoảng 600 dòng cho `generate()` (validation, deprecation, ...), 200 dòng cho `_sample`, 400 dòng cho `_beam_search`. Chúng ta đọc các phần cốt lõi, không đọc deprecation và edge case. Mục tiêu là hiểu **mạch suy nghĩ**, không nhớ từng dòng.

Sẵn sàng, ta sang chương 2.

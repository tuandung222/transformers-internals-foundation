---
title: Beam search
---

# Beam search

Beam search là sample strategy giữ `num_beams` candidate đồng thời, mở rộng song song, cuối cùng chọn beam có log-prob tổng cao nhất. Hữu ích khi cần output deterministic và high quality, ví dụ translation, summarization.

## Ý tưởng

Greedy chọn 1 token tốt nhất mỗi step. Beam search giữ K candidate (beam) tốt nhất theo cumulative log-prob.

Mỗi step:

1. Mở rộng mỗi beam bằng cách thử mọi token kế tiếp.
2. Có `num_beams * vocab_size` candidate.
3. Pick top `num_beams` theo total log-prob.
4. Cập nhật state cho `num_beams` beam mới.

Sau N step, return beam tốt nhất.

## Code skeleton

File: `src/transformers/generation/utils.py`, method `_beam_search`. Khoảng 350 dòng. Skeleton (đã rút gọn):

```python
def _beam_search(self, input_ids, logits_processor, stopping_criteria, generation_config, ...):
    num_beams = generation_config.num_beams
    batch_size = input_ids.shape[0] // num_beams  # flatten đã được thực hiện ở generate

    # State per beam
    running_beam_scores = torch.zeros((batch_size, num_beams), device=device)
    running_beam_scores[:, 1:] = -1e9  # chỉ beam 0 active ban đầu

    running_sequences = input_ids.view(batch_size, num_beams, -1)
    running_beam_indices = ...

    # Best finished beams
    sequences = torch.full((batch_size, num_beams, max_length), pad_token_id)
    scores = torch.full((batch_size, num_beams), -1e9)
    beam_indices = ...
    is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool)

    # Loop
    while not all_finished:
        # Forward
        model_inputs = prepare_inputs_for_generation(...)
        outputs = model_forward(**model_inputs)

        # Compute log_softmax
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids, next_token_logits)
        log_probs = nn.functional.log_softmax(next_token_scores, dim=-1)

        # Add to cumulative
        log_probs_processed = log_probs + running_beam_scores[:, :, None]
        # Shape: (batch_size, num_beams, vocab_size)

        # Pick top 2*num_beams (need extra for EOS handling)
        log_probs_flat = log_probs_processed.view(batch_size, num_beams * vocab_size)
        topk_log_probs, topk_indices = torch.topk(log_probs_flat, 2 * num_beams, dim=-1)

        # Decode topk_indices to (beam_idx, token_id)
        topk_beam_indices = topk_indices // vocab_size
        topk_token_ids = topk_indices % vocab_size

        # Gather running state along beam dim using topk_beam_indices
        running_sequences = gather_beams(running_sequences, topk_beam_indices)
        ...

        # Append topk_token_ids
        running_sequences = torch.cat([running_sequences, topk_token_ids[:, :, None]], dim=-1)
        running_beam_scores = topk_log_probs

        # Handle EOS: if token is EOS, move beam to "finished", drop from running
        eos_mask = is_eos(topk_token_ids)
        # ... complex bookkeeping

        # Reorder cache to match new beam permutation
        past_key_values.reorder_cache(flatten(topk_beam_indices))

        # Stopping check
        all_finished = check_finished(...)

    return best_finished_beam_per_batch
```

(Rất rút gọn, code thật phức tạp hơn nhiều cho early stopping, length penalty, ...)

## Vài tinh tế đáng chú ý

### `2 * num_beams` topk

Vì sao top `2 * num_beams`, không phải `num_beams`? Lý do: vài beam có thể end (EOS). Ta cần backup. Lấy 2x đảm bảo có đủ beam non-EOS để tiếp tục.

### Cache reorder

```python
past_key_values.reorder_cache(flatten(topk_beam_indices))
```

Đây là method ta đã thấy ở Phần 3. Beam search reshuffle beam mỗi step (best beam của step `t` có thể là continuation của beam `j` ở step `t-1`, không phải beam `j` cũ). Cache phải reorder theo `beam_idx` để K/V khớp với beam mới.

Cost: `index_select` trên tensor cache. Với `DynamicCache`, mỗi layer K/V `(B*num_beams, H, T, d)` -> reorder O(B*num_beams*T*d) per layer. Không nhỏ. Đây là lý do beam search chậm hơn greedy 3-10x.

### Flatten beam vào batch

```python
input_ids_flat = input_ids.view(batch_size * num_beams, seq_len)
```

Trước khi forward, beam dim flatten vào batch dim. Model thấy batch size = `batch_size * num_beams`. Sau forward, unflatten.

Đây là tinh tế: model không biết về beam, chỉ thấy batch lớn. Toàn bộ beam logic ở `_beam_search`, không xuyên xuống model.

### Length penalty

```python
final_scores = beam_scores / (length ** generation_config.length_penalty)
```

Chỉ scaling ở cuối. `length_penalty > 0` favor sequence dài, `< 0` favor sequence ngắn, `= 0` không penalize. Default thường 1.0 (trung tính).

Beam search có xu hướng ra sequence ngắn vì log-prob trừ dần. Length penalty 1.0 chia score cho length nên cân lại.

### Early stopping

```python
generation_config.early_stopping  # True | False | "never"
```

- True: stop khi tất cả beam finish.
- False: stop khi không có cải thiện possible (heuristic).
- "never": chạy tới max_length.

True thường đúng nhất cho production.

## Sample variant: beam sample

`do_sample=True, num_beams>1` -> `BEAM_SAMPLE`. Khác beam search: thay vì `torch.topk`, dùng `torch.multinomial` để sample 2*num_beams candidate từ distribution. Tăng diversity, giảm determinism.

Cả `_beam_search` và `_beam_sample` cùng method, branch theo `do_sample`.

## Khi nào dùng beam search?

| Use case | Recommend |
|---|---|
| Chatbot (creativity) | Sample (`do_sample=True, top_p=0.9`) |
| Translation (quality) | Beam (`num_beams=4-8`) |
| Code generation | Sample low temp + top-p |
| Summarization | Beam (`num_beams=4`) |
| RAG/QA factoid | Greedy hoặc beam=2-4 |

Modern LLM trend là giảm dùng beam. Beam search có cost (3-10x slower, cache reorder), gain chất lượng không nhiều với LLM lớn well-trained. Nhưng vẫn standard cho translation và summarization benchmark.

## Pitfall

**1. Memory blow up**: cache giữ `B * num_beams` copy. Với `num_beams=8`, memory 8x. Nếu OOM, giảm `num_beams` hoặc dùng QuantizedCache.

**2. Trùng output**: beam search có thể ra `num_return_sequences` output gần giống nhau. Diversity penalty (`diversity_penalty`) giúp; nhưng giải pháp cleaner là dùng sample.

**3. EOS quá sớm**: nếu beam EOS sớm với prob cao, các beam khác bị "forget". Tăng `length_penalty` hoặc giảm `early_stopping=False`.

**4. Compile không hỗ trợ**: beam search có data-dependent control flow (drop beam khi EOS), khó compile. HF thường disable compile cho beam search.

Chương sau ta đi vào assisted decoding.

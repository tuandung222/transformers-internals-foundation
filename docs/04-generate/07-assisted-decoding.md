---
title: Assisted decoding
---

# Assisted decoding (speculative)

Assisted decoding (a.k.a. speculative decoding) là kỹ thuật accelerate inference: dùng model nhỏ (draft) để sinh nhanh nhiều token, model lớn (target) verify song song. Nếu draft đúng, ta accept; sai, ta reject từ chỗ sai.

Lợi: model lớn forward một lần thay vì N lần. Nếu draft accuracy cao, tăng tốc 2-3x.

## Ý tưởng cốt lõi

Mỗi step decode chuẩn: 1 forward target -> 1 token. N step -> N forward target.

Assisted: 1 forward draft (sinh K token) + 1 forward target (verify K token) -> chấp nhận tới K token. Nếu mọi token được accept, ta được K token chỉ với 2 forward (1 nhỏ + 1 lớn). Nếu chỉ accept j < K token (token thứ j+1 sai), ta vẫn được j+1 token (token thứ j+1 lấy từ target).

Số token accept trung bình phụ thuộc agreement giữa draft và target. Llama-3 8B draft cho Llama-3 70B: thường accept 3-5 token mỗi vòng.

## Implementation skeleton

File: `src/transformers/generation/utils.py`, method `_assisted_decoding`. Skeleton:

```python
def _assisted_decoding(self, input_ids, candidate_generator, logits_processor, stopping_criteria, generation_config, ...):
    while not done:
        # 1. Draft generate K candidate token
        candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)
        # candidate_input_ids shape: (B, T + K)

        # 2. Target forward toàn bộ candidate (1 forward, q_len = K+1)
        candidate_kwargs = prepare_inputs_for_generation(candidate_input_ids, ...)
        outputs = self(**candidate_kwargs, return_dict=True)

        # 3. Verify từng token
        new_logits = outputs.logits[:, -K-1:, :]
        new_logits = logits_processor(candidate_input_ids[:, :-1], new_logits)

        # 4. Compute accept mask
        selected_tokens, n_matches = _speculative_sampling(
            candidate_input_ids, candidate_logits, new_logits, K,
            last_assistant_token_is_eos, do_sample, ...
        )

        # 5. Append accepted token + 1 token mới từ target
        input_ids = torch.cat([input_ids, selected_tokens[:, :n_matches + 1]], dim=-1)

        # 6. Crop cache to keep only accepted tokens
        past_key_values.crop(input_ids.shape[-1] - 1)

        # 7. Update candidate generator state
        candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

    return input_ids
```

## `CandidateGenerator` interface

```python
class CandidateGenerator:
    def get_candidates(self, input_ids) -> tuple[torch.LongTensor, torch.FloatTensor | None]:
        ...

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        ...
```

Hai method:

- `get_candidates`: input current `input_ids`, return `(candidate_ids, candidate_logits)`. Candidate có K token mới.
- `update_candidate_strategy`: feedback bao nhiêu token đã được accept. Cho phép draft adapt (giảm K nếu accuracy thấp, tăng K nếu cao).

Implementations:

- `AssistedCandidateGenerator`: dùng `assistant_model` (HF model nhỏ hơn).
- `PromptLookupCandidateGenerator`: tìm n-gram trong prompt làm candidate. Cho repetitive task (code edit, summarization).
- `EarlyExitCandidateGenerator`: dùng exit sớm của chính target model.

## `_speculative_sampling`: thuật toán accept/reject

Đây là phần toán đẹp nhất. Với mỗi candidate token position `i`:

- `p_target(x_i | x_{<i})`: prob từ target model.
- `p_draft(x_i | x_{<i})`: prob từ draft model.
- Accept với probability `min(1, p_target / p_draft)`.

Nếu accept: token được giữ, sang vị trí kế.
Nếu reject ở vị trí `i`: tất cả token sau bị bỏ. Sample lại token ở vị trí `i` từ distribution adjusted `(p_target - p_draft)^+ / norm`.

```python
def _speculative_sampling(candidate_input_ids, candidate_logits, new_logits, K, ...):
    # candidate_logits: (B, K, vocab) - draft logits cho K token
    # new_logits: (B, K+1, vocab) - target logits cho K+1 vị trí

    if do_sample:
        # Probability-based accept/reject
        q = nn.functional.softmax(candidate_logits, dim=-1)
        p = nn.functional.softmax(new_logits[:, :-1, :], dim=-1)
        accept_probs = torch.minimum(torch.ones_like(p), p / q)

        # Sample uniform, accept if u < accept_prob[token_id]
        candidate_tokens = candidate_input_ids[:, -K:]
        token_probs = accept_probs.gather(2, candidate_tokens[..., None]).squeeze(-1)
        u = torch.rand_like(token_probs)
        accept_mask = u < token_probs

        # Find first reject
        n_matches = first_false_index(accept_mask)

        # Token at reject position: sample from (p - q)^+ normalized
        ...
    else:
        # Greedy: token được accept nếu match argmax target
        target_tokens = new_logits[:, :-1, :].argmax(dim=-1)
        candidate_tokens = candidate_input_ids[:, -K:]
        accept_mask = (target_tokens == candidate_tokens)
        n_matches = first_false_index(accept_mask)
```

Tinh tế của thuật toán: distribution sample ra **identical** với target standalone. Tức là speculative không bias output; chỉ accelerate.

## Crop cache

```python
past_key_values.crop(input_ids.shape[-1] - 1)
```

Sau verify, cache có K+1 token mới (target forward đã ghi cache). Nếu chỉ accept `n_matches`, các slot K+1, K, ..., n_matches+1 phải bị crop. Phương thức `crop` của `DynamicCache` cắt tensor tới length yêu cầu.

`StaticCache` cũng có `crop` nhưng phức tạp hơn (in-place reset cumulative_length).

## Pitfall

**1. Draft model phải cùng tokenizer**: cùng vocab. Nếu khác (Llama draft cho Mistral target), output không hợp lệ vì token ID khác nghĩa. HF có `UniversalAssistedCandidateGenerator` cho khác tokenizer, decode/re-encode, nhưng cost cao.

**2. K (num_assistant_tokens) tuning**: K quá lớn -> draft chậm + nhiều reject. K quá nhỏ -> ít token saved. Default 5, có thể tune theo task.

**3. Cache crop với StaticCache**: cần handle careful. HF có path riêng. Khi viết custom cache, phải implement `crop`.

**4. Memory**: cache cho target K+1 token, plus draft cache. Tổng memory cao hơn standard decode.

## Khi nào dùng?

- Có draft model nhỏ tốt: 2-3x speedup.
- Task structured (code, JSON, repetitive): PromptLookup hiệu quả.
- Real-time serving với latency critical: pay-off đáng giá.

Khi không:

- Throughput optimization với batch lớn: speculative cost batch overhead, không đáng.
- Compile + StaticCache: speculative khó compile do dynamic shape.

## API user

```python
draft = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
out = target.generate(input_ids, assistant_model=draft, do_sample=False)
```

HF tự build `AssistedCandidateGenerator`, gọi `_assisted_decoding`.

Chương cuối Phần 4: compile và CUDA graph.

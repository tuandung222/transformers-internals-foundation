---
title: Glossary
---

# Glossary

Thuật ngữ thường gặp trong chuỗi bài giảng, kèm tham chiếu chương ngắn.

## A

**`AttentionInterface`**: dict-like singleton chứa mọi backend attention. Truy cập qua `ALL_ATTENTION_FUNCTIONS`. Xem Phần 1 chương 3.

**`apply_rotary_pos_emb`**: helper áp dụng RoPE lên Q và K bằng cos, sin. Xem Phần 1 chương 2.

**Assisted decoding** (speculative): kỹ thuật accelerate generate dùng draft model. Xem Phần 4 chương 7.

**`AutoConfig`, `AutoModel*`**: dispatcher đọc config string -> class tương ứng. Xem Phần 5 chương 4.

## B

**Backend attention**: function tính attention (eager, sdpa, flash, flex, paged). Pass qua `attn_implementation`. Xem Phần 1 chương 4.

**Beam search**: sample strategy giữ K candidate đồng thời. Xem Phần 4 chương 6.

**`BaseModelOutputWithPast`**: dataclass output của body model, chứa `last_hidden_state` + `past_key_values`. Xem Phần 5 chương 5.

## C

**Cache (KV Cache)**: lưu K/V của các token đã thấy để tránh tính lại. Xem Phần 3.

**`CacheLayerMixin`**: interface abstract cho cache của một layer. Subclass: DynamicLayer, StaticLayer, ... Xem Phần 3 chương 3.

**Cross-attention**: attention với Q từ một sequence, K/V từ sequence khác (thường K/V từ encoder). Xem Phần 2.

**Colwise / Rowwise**: TP shard strategy. Xem Phần 5 chương 7.

**CUDA graph**: kernel call captured thành graph, launch một phát thay vì nhiều. Xem Phần 4 chương 8.

## D

**DynamicCache**: cache mặc định, grow theo `torch.cat`. Xem Phần 3 chương 4.

**Decode**: phase generate token sau khi prefill, mỗi step 1 token mới.

**`device_map`**: argument cho `from_pretrained` để Accelerate split model giữa device. Xem Phần 5 chương 8.

## E

**Eager**: backend attention reference implementation, materialize attn_weights. Xem Phần 1 chương 4.

**EncoderDecoderCache**: cache cho encoder-decoder model, đóng gói self_attention_cache và cross_attention_cache. Xem Phần 2 chương 4.

**EOS (end-of-sequence)**: token đặc biệt đánh dấu kết thúc. `EosTokenCriteria` check. Xem Phần 4 chương 5.

## F

**FlashAttention 2/3/4**: kernel CUDA tối ưu cho attention, không materialize attn_weights. Xem Phần 1 chương 4.

**FlexAttention**: PyTorch 2.5+ flexible attention compile từ score modifier function. Xem Phần 1 chương 4.

**FSDP**: Fully Sharded Data Parallel, shard parameter theo data-parallel rank. Xem Phần 5 chương 7.

**`from_pretrained`**: load model từ Hub hoặc path. 12 step. Xem Phần 5 chương 8.

## G

**GQA (Grouped Query Attention)**: Q có nhiều head, K/V có ít head, mỗi K/V phục vụ nhiều Q. `num_key_value_groups` = Q heads / K/V heads. Xem Phần 1 chương 2.

**`generate`**: top-level API sinh sequence. Xem Phần 4.

**`GenerationConfig`**: dataclass chứa hyper-parameter cho generation. Xem Phần 4 chương 2.

**`GenerationMixin`**: mixin cung cấp generate, sample, beam_search method. Xem Phần 5 chương 5.

## H

**HQQ (Half-Quadratic Quantization)**: quantize backend cho cache. INT1-8. Xem Phần 3 chương 6.

**Hidden state**: tensor sequence `(B, T, D)` trong các layer của Transformer.

## I

**`is_causal`**: flag attention báo có cần causal mask không. Self attention decoder: True. Cross attention: False. Xem Phần 1 chương 2.

**`is_updated`**: dict trong EncoderDecoderCache track mỗi layer cross-attention đã được fill chưa. Xem Phần 2 chương 4.

**Index_copy**: in-place write tensor tại index. StaticCache dùng. Xem Phần 3 chương 5.

## K

**KV cache**: viết tắt phổ biến cho KeyValueCache. Xem Phần 3.

## L

**`LogitsProcessor`**: transform `(input_ids, scores) -> scores` áp dụng trước sample. Temperature, top-k, top-p, ... Xem Phần 4 chương 4.

**`logits_to_keep`**: optimization chỉ project lm_head cho các position cần. Xem Phần 5 chương 5.

**`layer_idx`**: index của layer (0-based), dùng để address cache slot. Xem Phần 1 chương 2.

## M

**Mask (attention mask)**: tensor additive (`-inf` cho vị trí cần che) hoặc boolean. Xem Phần 1 chương 2.

**`mark_static_address`**: torch._dynamo helper báo buffer có địa chỉ tĩnh, dùng được với CUDA graph. Xem Phần 3 chương 5.

**Modular file**: source of truth `modular_*.py` mà HF compile thành `modeling_*.py`. Xem Phần 5 chương 3.

**MHA / MQA / GQA**: Multi-Head Attention / Multi-Query (Q nhiều K/V một) / Grouped-Query Attention.

## N

**`num_attention_heads`**: số attention head của Q. Llama-3: 32.

**`num_key_value_heads`**: số head của K/V. GQA: nhỏ hơn `num_attention_heads`. Llama-3: 8.

**Nucleus sampling (top-p)**: giữ token với cumulative prob = p. Xem Phần 4 chương 4.

## O

**OffloadedCache**: cache đẩy K/V sang CPU RAM, prefetch khi cần. Xem Phần 3 chương 6.

**`output_attentions`**: flag forward trả attn_weights. Eager hỗ trợ, SDPA/Flash không.

## P

**Paged attention**: KV cache page-based, allow continuous batching. Xem Phần 1 chương 5.

**`past_key_values`**: argument forward chứa Cache hoặc tuple K/V của các step trước.

**Prefill**: phase forward toàn bộ prompt một lần, fill cache. Khác decode (token-by-token).

**`post_init()`**: method PreTrainedModel chạy sau __init__: init weights, tie, fp32 keep. Xem Phần 5 chương 2.

**`prepare_inputs_for_generation`**: method model-specific build input dict cho mỗi decode step. Xem Phần 4 chương 3.

**`PreTrainedModel`**: base class mọi model. Xem Phần 5 chương 2.

**`PreTrainedConfig`**: base class mọi config. Xem Phần 5 chương 2.

## Q

**QuantizedCache**: cache với K/V quantize 2-4 bit. KIVI, HQQ, Quanto backend. Xem Phần 3 chương 6.

**`q_proj, k_proj, v_proj, o_proj`**: linear projection của attention. Xem Phần 1 chương 2.

## R

**RoPE (Rotary Position Embedding)**: rotate Q và K theo position. Llama, Qwen dùng. Xem Phần 1 chương 2.

**`repeat_kv`**: helper expand K/V cho GQA (`num_kv_heads` -> `num_heads`). Xem Phần 1 chương 4.

**`return_dict_in_generate`**: flag generate trả `GenerateOutput` dataclass thay vì tensor.

## S

**Self-attention**: attention với Q, K, V cùng nguồn (hidden_states).

**SDPA**: scaled_dot_product_attention, native PyTorch backend. Xem Phần 1 chương 4.

**`_sample`**: method cho greedy (`do_sample=False`) và multinomial sample (`do_sample=True`). Xem Phần 4 chương 3.

**`StoppingCriteria`**: check khi nào stop generate. MaxLength, EOS, StopString. Xem Phần 4 chương 5.

**StaticCache**: cache preallocate, compile-friendly. Xem Phần 3 chương 5.

**Sliding window**: attention pattern chỉ attend `window_size` token gần nhất. Mistral, Gemma-2. Xem Phần 3 chương 4.

**Speculative decoding**: xem Assisted decoding.

**`_supports_sdpa` / `_supports_flash_attn`**: flag class báo backend nào model hỗ trợ. Xem Phần 5 chương 6.

## T

**TP (Tensor Parallel)**: shard tensor giữa GPU theo colwise/rowwise. Xem Phần 5 chương 7.

**`_tp_plan` / `_fsdp_plan`**: dict plan declare strategy parallel. Xem Phần 5 chương 7.

**`tie_weights()`**: share weight giữa hai module (`lm_head = embed_tokens`). Xem Phần 5 chương 6.

**Temperature**: scale logit trước softmax. T < 1 sharpen, T > 1 flatten.

**Top-k / Top-p**: filter token theo rank/cumsum prob. Xem Phần 4 chương 4.

**`torch.compile`**: PyTorch JIT compile module thành kernel optimized. Xem Phần 4 chương 8.

## V

**Vision Encoder-Decoder**: pattern ViT + text decoder cho image-to-text (TrOCR, Pix2Struct). Xem Phần 2 chương 5.

**vLLM**: serving engine với paged attention. Tham chiếu trong Phần 1 chương 5.

## W

**Whisper**: speech-to-text encoder-decoder model. Xem Phần 2 chương 5.

**`weights_only`**: argument `from_pretrained` báo không load code, chỉ tensor. Default True từ PyTorch 2.5+. Xem Phần 5 chương 8.

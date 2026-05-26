---
title: References
---

# References

Tài liệu để đào sâu hơn ngoài source code transformers.

## Source code chính

- HuggingFace transformers: [github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- File `modeling_utils.py`, `cache_utils.py`, `generation/utils.py` là ba file đọc nhiều nhất.
- Module `integrations/` chứa backend attention.
- Module `models/X/modeling_X.py` cho từng model cụ thể.

## Papers nền tảng

### Attention

- **Attention Is All You Need** (Vaswani et al., 2017). Bài gốc Transformer.
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022).
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023).
- **FlashAttention-3** (Shah et al., 2024). Hopper-optimized.

### Position embedding

- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021). RoPE original.
- **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023). Mở rộng context với RoPE.

### Architecture variants

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023).
- **Mistral 7B** (Jiang et al., 2023). Sliding window attention.
- **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023).
- **Llama 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023).

### Inference optimization

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023). vLLM paper.
- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** (Liu et al., 2024).
- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2023).

### Encoder-decoder

- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)** (Raffel et al., 2020).
- **BART: Denoising Sequence-to-Sequence Pre-training** (Lewis et al., 2019).
- **Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)** (Radford et al., 2022).

## Blog posts và lectures

- HuggingFace blog: [huggingface.co/blog](https://huggingface.co/blog). Đặc biệt các bài về cache, quantization, FlashAttention.
- Andrej Karpathy, "Let's build GPT": YouTube series xây Transformer từ đầu.
- Lilian Weng, "The Transformer Family Version 2.0": [lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/).
- Chip Huyen, "Building LLM applications for production": blog series về production.

## PyTorch tài liệu

- **`torch.nn.functional.scaled_dot_product_attention`**: [pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- **FlexAttention**: [pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention/)
- **`torch.compile`**: [pytorch.org/docs/stable/torch.compiler.html](https://pytorch.org/docs/stable/torch.compiler.html)
- **CUDA graph**: [pytorch.org/blog/accelerating-pytorch-with-cuda-graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- **Tensor Parallel API**: [pytorch.org/docs/stable/distributed.tensor.parallel.html](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- **FSDP**: [pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html)

## Other libraries cùng hệ sinh thái

- **Accelerate**: [github.com/huggingface/accelerate](https://github.com/huggingface/accelerate). Device map, dispatch, multi-GPU.
- **PEFT**: [github.com/huggingface/peft](https://github.com/huggingface/peft). LoRA, adapter.
- **vLLM**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm). Production serving với paged attention.
- **TGI** (Text Generation Inference): [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference). HuggingFace's serving.
- **FlashAttention library**: [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).
- **optimum-quanto**: [github.com/huggingface/optimum-quanto](https://github.com/huggingface/optimum-quanto). Quantize.
- **HQQ**: [github.com/mobiusml/hqq](https://github.com/mobiusml/hqq). Half-Quadratic Quantization.

## Practical guides

- **HuggingFace LLM Course**: [huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course).
- **Generation Strategies docs**: [huggingface.co/docs/transformers/generation_strategies](https://huggingface.co/docs/transformers/generation_strategies).
- **Cache class docs**: [huggingface.co/docs/transformers/kv_cache](https://huggingface.co/docs/transformers/kv_cache).
- **Modular transformers**: [huggingface.co/docs/transformers/modular_transformers](https://huggingface.co/docs/transformers/modular_transformers).

## Đọc tiếp khi cần

- **Building custom model**: docs HF có guide tổng quát: [huggingface.co/docs/transformers/custom_models](https://huggingface.co/docs/transformers/custom_models).
- **Contributing model**: nếu muốn đóng góp model mới lên transformers, đọc `docs/source/en/add_new_model.md`.
- **Performance optimization**: docs HF perf benchmark: [huggingface.co/docs/transformers/perf_infer_gpu_one](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

## Đề xuất lộ trình tiếp theo

Sau chuỗi này, đề xuất:

1. **Fine-tune một model**: pick một dataset, fine-tune Llama-3.2-1B với PEFT/LoRA, deploy.
2. **Implement custom attention**: viết một backend custom (ví dụ logn-sparse attention), register vào AttentionInterface.
3. **Speculative decoding**: pair draft model nhỏ + target model lớn, đo throughput.
4. **Long context**: thử YaRN + StaticCache + FlashAttention cho 128k context.
5. **Multimodal**: học BLIP-2, LLaVA. Cross-attention từ vision encoder vào language decoder.
6. **Serving production**: deploy với vLLM hoặc TGI, đo latency p50/p99.

Mỗi tasks trên là một dự án 1-2 tuần. Hoàn thành 2-3 cái là đã ở mức intermediate-advanced ML engineer.

## Lời cảm ơn cuối

Chuỗi bài giảng này dựa trên codebase transformers ở thời điểm viết. Codebase tiến hoá liên tục: file path, dòng code có thể đã thay đổi. Tinh thần cốt lõi (attention dispatch, cache API, generate orchestration, conventions) sẽ ổn định lâu hơn.

Chúc bạn đọc source code với tự tin hơn.

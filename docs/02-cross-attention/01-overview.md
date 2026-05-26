---
title: Tổng quan Phần 2
---

# Phần 2: Cross-attention và encoder-decoder

Self-attention chiếm phần lớn các bài viết về Transformer. Nhưng các model như T5, BART, mBART, Whisper, Vision Encoder-Decoder, Pix2Struct, BLIP, ... đều dựa trên cross-attention. Phần này mở rộng kiến thức Phần 1 sang cấu trúc encoder-decoder.

## Câu hỏi mở đầu

Cross-attention khác self-attention ở chỗ nào về mặt code? Phép toán có lẽ đã quen: Q từ một sequence, K/V từ sequence khác. Nhưng implementation phải xử lý ba việc mà self-attention không cần:

1. **Source K/V đến từ encoder**, không từ hidden state hiện tại. Encoder chạy một lần, decoder gọi lại K/V mỗi layer mỗi step.
2. **Cache đặc biệt**: cross-attention K/V không đổi qua từng decoding step, nên cache đầy ngay sau prefill và không update. Cache self-attention vẫn append từng step. Hai loại cache phải sống song song trong cùng object.
3. **Mask khác**: cross-attention không causal. Mọi Q decoder được phép nhìn mọi K encoder. Mask chỉ dùng để che padding của source.

`EncoderDecoderCache` của HF được thiết kế để giải bài toán 2 và 3. Ta sẽ đọc kỹ.

## Mục tiêu

Sau Phần 2, bạn:

1. Hiểu vì sao mỗi decoder layer có hai sub-layer attention: self và cross.
2. Đọc `T5Attention.forward` và giải thích đường code cho cross-attention.
3. Hiểu `EncoderDecoderCache.is_updated` cho phép reuse K/V cross attention qua các step decoder.
4. Biết cấu trúc tương tự ở Whisper (speech-to-text) và Vision Encoder-Decoder (TrOCR, ...).

## Cấu trúc Phần 2

5 chương:

- Chương 2: So sánh self-attention và cross-attention về mặt formula, shape, mask.
- Chương 3: Walkthrough `T5Attention.forward` chi tiết. T5 là model encoder-decoder mainstream nhất trong HF.
- Chương 4: `EncoderDecoderCache` và flag `is_updated`. Đây là một invariant tinh tế: tính K/V cross attention một lần, dùng N step.
- Chương 5: Whisper và Vision Encoder-Decoder. Hai variant phổ biến của pattern encoder-decoder, mở rộng sang multimodal.

## Tại sao encoder-decoder vẫn quan trọng?

Một câu hỏi thực tế: trong thời đại decoder-only (Llama, GPT, Qwen), tại sao học encoder-decoder? Vài lý do:

1. Speech: Whisper vẫn là baseline mạnh nhất cho ASR đa ngôn ngữ. Hiểu cross-attention là điều kiện để fine-tune Whisper.
2. Vision-Language: BLIP-2, Pix2Struct, Donut, TrOCR đều dùng encoder-decoder. Decoder phải attend trên token visual của encoder.
3. T5 là backbone của nhiều benchmark text-to-text. Nhiều variant (Flan-T5, T0, UL2) vẫn được nghiên cứu.
4. Cross-attention pattern xuất hiện trong các speculation/draft model setup, trong tool-calling, trong retrieval-augmented generation.

Hiểu cross-attention không chỉ là chuyện T5. Nó là một primitive xuất hiện ở mọi nơi cần "Q từ A, K/V từ B".

## Pattern encoder-decoder cấp cao

```
input_ids -> Encoder (self-attention thuần) -> encoder_hidden_states [shape: (B, T_src, D)]
                                                       |
                                                       v
decoder_input_ids -> Decoder:
    for layer in decoder_layers:
        h = SelfAttention(h, h, h)     # causal, dùng past_key_values.self_attention_cache
        h = CrossAttention(h, K_enc, V_enc)  # không causal, dùng past_key_values.cross_attention_cache
        h = FFN(h)
    -> lm_head -> logits
```

Đây là pattern chuẩn từ "Attention Is All You Need". HF cài đặt nhất quán cho mọi encoder-decoder model.

Sẵn sàng, ta sang chương 2 để xem hai loại attention khác nhau cụ thể ở chỗ nào.

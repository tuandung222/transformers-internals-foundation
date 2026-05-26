---
title: Whisper và Vision Encoder-Decoder
---

# Whisper và Vision Encoder-Decoder

Pattern encoder-decoder không giới hạn ở text-to-text. Chương này nhìn hai variant: Whisper (audio-to-text) và Vision Encoder-Decoder (image-to-text). Cả hai dùng cùng `EncoderDecoderCache`, cùng cách dispatch cross-attention. Khác biệt nằm ở encoder.

## Whisper: audio-to-text

File chính: `src/transformers/models/whisper/modeling_whisper.py`.

**Encoder Whisper** không phải Transformer text. Input là mel spectrogram shape `(B, n_mels, T_audio)` (thường `T_audio = 3000` cho 30 giây audio, `n_mels = 80`). Encoder bắt đầu bằng hai Conv1D để downsample audio xuống `(B, 1500, D)`, sau đó là stack các Transformer encoder layer self-attention. Output: `encoder_hidden_states` shape `(B, 1500, D)`.

**Decoder Whisper** là Transformer decoder điển hình. Mỗi decoder layer:

```python
class WhisperDecoderLayer:
    def __init__(self, config):
        self.self_attn = WhisperAttention(config, is_causal=True, ...)
        self.encoder_attn = WhisperAttention(config, is_causal=False, ...)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        ...
```

Quan sát: cross-attention được đặt tên là `encoder_attn`, một class `WhisperAttention` được dùng cho cả hai loại (giống pattern T5), chỉ khác `is_causal` flag.

**Cache flow**:

```python
past_key_values = EncoderDecoderCache(
    DynamicCache(config=config),  # self attention
    DynamicCache(config=config),  # cross attention
)
encoder_out = model.encoder(input_features)  # (B, 1500, D)
generated = model.decoder.generate(
    decoder_input_ids=...,
    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
    past_key_values=past_key_values,
)
```

Bước decode đầu tiên: cross-attention compute K/V từ `encoder_out` cho mọi layer, set `is_updated[i] = True`. Các step sau reuse.

Một subtlety: Whisper có **special tokens** (`<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, ...) định nghĩa task. Token đầu tiên decoder thấy là `<|startoftranscript|>`. Trong code, không có gì đặc biệt: chỉ là token bình thường được embed và đi qua decoder. Special token chỉ định ngữ nghĩa qua training.

## Vision Encoder-Decoder

File chính: `src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py`.

`VisionEncoderDecoderModel` là **wrapper composition**: chứa một encoder vision (ViT, BeiT, DeiT, ...) và một decoder text (GPT-2, RoBERTa, ...). Class này không định nghĩa mới attention; nó pass `encoder_hidden_states` vào decoder.cross_attention.

Constructor:

```python
class VisionEncoderDecoderModel(PreTrainedModel):
    def __init__(self, config=None, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder or AutoModel.from_config(config.encoder)
        self.decoder = decoder or AutoModelForCausalLM.from_config(config.decoder)
        self.enc_to_dec_proj = nn.Linear(...) if needed
```

Pattern composition này cho phép user mix-and-match: ViT + GPT-2 (TrOCR), DeiT + RoBERTa, BeiT + BERT, ... Mỗi combination chia sẻ cùng forward logic:

```python
def forward(self, pixel_values, decoder_input_ids, ...):
    encoder_outputs = self.encoder(pixel_values=pixel_values, ...)
    encoder_hidden_states = encoder_outputs[0]
    if self.enc_to_dec_proj is not None:
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        ...
    )
```

`enc_to_dec_proj` là linear layer để đồng bộ chiều D. ViT có `D=768`, GPT-2 có `D=768` (khớp luôn) hoặc `D=1024` (lớn hơn, cần project xuống).

## Yêu cầu cho decoder

Decoder phải **support cross-attention**. Không phải mọi decoder-only model hỗ trợ. Cụ thể, decoder phải có:

1. Constructor nhận `add_cross_attention=True`.
2. Mỗi decoder layer có một sub-layer cross-attention.
3. `forward` chấp nhận `encoder_hidden_states` và `encoder_attention_mask`.

GPT-2 implementation HF có flag `add_cross_attention` trong config. Khi `True`, mỗi `GPT2Block` thêm `crossattention = GPT2Attention(is_cross_attention=True)`. Llama, Qwen không hỗ trợ cross-attention. Pix2Struct, TrOCR dùng GPT-2 fork hoặc decoder-specific.

## TrOCR walkthrough nhanh

TrOCR (Microsoft) là OCR encoder-decoder điển hình:

- Encoder: BeiT hoặc DeiT (ViT-style), input `(B, 3, 384, 384)`, output `(B, 577, 1024)` (577 = 24*24 patch + 1 CLS).
- Decoder: RoBERTa-style decoder với cross-attention. Vocab ~50k. Generate text từ image.

Khi inference:

```python
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
```

Bên trong `generate`, encoder chạy một lần, decoder generate từng token với cross-attention reuse K/V. Đúng pattern đã học.

## So sánh ba pattern

| Model | Encoder input | Cross K/V length | Decoder dùng RoPE? | Cache class |
|---|---|---|---|---|
| T5 | text token | `T_src` | không (relative bias) | EncoderDecoderCache(Dynamic, Dynamic) |
| Whisper | mel `(80, 3000)` | 1500 (audio frame) | không (sinusoidal) | EncoderDecoderCache(Dynamic, Dynamic) |
| TrOCR | image `(3, 384, 384)` | 577 (patch token) | tuỳ decoder | EncoderDecoderCache(Dynamic, Dynamic) |

Mọi model chia sẻ pattern cache. Khác biệt ở dạng encoder và length cross K/V.

## Kết luận Phần 2

Cross-attention không phải khái niệm hoàn toàn mới so với self-attention. Code khác biệt nằm ở ba điểm cụ thể: nguồn K/V (từ encoder), cache reuse (`is_updated` flag), và mask (không causal). Hiểu T5 đủ để mở rộng sang Whisper, BART, TrOCR. Phần 3 đào sâu hơn vào KV Cache, bao gồm các variant Static, Quantized, Offloaded.

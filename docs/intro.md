---
title: Giới thiệu chuỗi bài giảng
slug: /intro
---

# Giới thiệu

Có hai cách để dùng một thư viện như HuggingFace transformers. Cách thứ nhất, gọi `AutoModelForCausalLM.from_pretrained(...).generate(...)`, có kết quả, và tin rằng nó đúng. Cách thứ hai, mở `modeling_llama.py`, đọc từng dòng, hiểu vì sao `LlamaAttention.forward` được viết theo cách đó, và sau đó dùng API với sự tự tin của một người đã thấy bộ máy bên trong.

Chuỗi bài giảng này nhằm vào cách thứ hai. Nó không thay thế tài liệu chính thức của HuggingFace, nó bổ sung cho tài liệu đó bằng cách kéo người đọc đi xuyên qua code, dừng lại ở những chỗ then chốt, và giải thích vì sao thiết kế lại như vậy.

## Đối tượng

Bạn nên đã biết:

- PyTorch cơ bản, viết được một `nn.Module` với forward đơn giản.
- Khái niệm cốt lõi của Transformer: attention, multi-head, feed-forward, layer norm.
- Đã dùng HuggingFace ở mức API: load model, tokenize, gọi `generate`, fine-tune đơn giản.

Bạn sẽ học được:

- Cách thư viện dispatch giữa các backend attention (eager, SDPA, FlashAttention, FlexAttention) thông qua `AttentionInterface`.
- Cấu trúc KV cache với `CacheLayerMixin`, `DynamicCache`, `StaticCache`, `EncoderDecoderCache`.
- Toàn bộ pipeline của `generate()`: chuẩn bị inputs, cache, logits processors, stopping criteria, rồi dispatch vào `_sample`, `_beam_search`, `_assisted_decoding`.
- Convention thiết kế: tách `PreTrainedConfig` khỏi `PreTrainedModel`, dùng `modular_*.py` để tạo `modeling_*.py`, đăng ký vào `AutoModel` registry, dùng `Generic*ForX` mixin cho task heads.

## Triết lý đọc source

Có một câu nói trong cộng đồng PyTorch: "Code is read more often than written". Với một thư viện rộng như transformers (gần 300 model, hơn 200K dòng code), tác giả đã chọn một triết lý cụ thể là **một-model-một-file**: mỗi kiến trúc có một file `modeling_*.py` chứa toàn bộ logic, không chia nhỏ thành nhiều file con. Điều này tăng độ dài file, nhưng giảm số file bạn phải mở để hiểu một model.

Triết lý này có ưu và nhược điểm. Ưu: bạn đọc một file là đủ để debug, không phải nhảy qua nhiều module. Nhược: nhiều code lặp lại giữa các model, vì mỗi model phải tự định nghĩa lại RMSNorm, RotaryEmbedding, MLP.

HuggingFace giải quyết nhược điểm bằng `modular_*.py`: file ngắn chứa diff so với một model gốc, sau đó được expand tự động thành `modeling_*.py` đầy đủ. Chương sau sẽ giải thích chi tiết.

## Phương pháp

Mỗi phần đều tuân theo cấu trúc:

1. **Trực giác trước**: vấn đề là gì, vì sao cần giải, ý tưởng cốt lõi của HuggingFace.
2. **Toán hoặc đặc tả**: derive hoặc trình bày chính xác cái gì đang được implement.
3. **Walkthrough source**: mở file thật, đọc từng đoạn, gắn trở lại với toán hoặc đặc tả.
4. **Pitfall và edge case**: những chỗ dễ sai khi tự viết model hoặc khi debug.

Mọi đoạn code trích dẫn đều từ thư viện thật, không phải pseudocode. Nếu bạn clone repo `huggingface/transformers` cùng version, bạn sẽ tìm thấy các dòng đó nguyên văn.

## Cấu trúc các phần

Phần 0 giới thiệu repo transformers, cấu trúc thư mục, philosophy `one-model one-file`, AutoClass registry, và roadmap đọc.

Phần 1 đi vào attention internals: walkthrough `LlamaAttention.forward`, hiểu `AttentionInterface` dispatch, so sánh các backend eager, SDPA, FlashAttention, FlexAttention, và bàn về paged attention cho continuous batching.

Phần 2 giải thích cross-attention trong encoder-decoder model như T5 và Whisper, vì sao K/V chỉ tính một lần từ encoder, và `EncoderDecoderCache` với cờ `is_updated`.

Phần 3 đi sâu KV cache: lớp abstract `CacheLayerMixin`, các implementation `DynamicLayer`, `StaticLayer`, `QuantizedLayer`, và cách `DynamicCache` quản lý chúng. Cuối phần là luồng update past_key_values khi sinh từng token.

Phần 4 mổ xẻ `generate()`: pipeline 9 bước, các generation mode, vòng lặp `_sample` với prefill và decode, beam search, và speculative (assisted) decoding.

Phần 5 tổng kết convention thiết kế model của HF: `PreTrainedModel` + `PreTrainedConfig`, modular vs modeling, Auto registry, Generic mixins, tied weights, `_tp_plan`/`_sp_plan`/`_fsdp_plan`, và toàn bộ flow của `from_pretrained`.

## Source reference

Mọi trích dẫn trong chuỗi này dựa trên branch `main` của repo `huggingface/transformers` tại thời điểm biên soạn. API có thể thay đổi giữa các version. Khi đọc, bạn nên đối chiếu với version bạn đang dùng nếu khác.

Sẵn sàng, chúng ta vào Phần 0.

---
title: Tổng quan Phần 0
---

# Phần 0: Tổng quan thư viện transformers

Trước khi mở bất kỳ file `modeling_*.py` nào, ta cần một bản đồ. Phần 0 cho bạn bản đồ đó: thư viện được tổ chức như thế nào, vì sao tác giả chọn tổ chức như vậy, và bạn nên đọc các file theo thứ tự nào để không bị ngợp.

## Mục tiêu

Sau Phần 0, bạn có thể:

1. Giải thích triết lý **one-model-one-file** và biết khi nào nó là điểm mạnh, khi nào là điểm yếu.
2. Mở `src/transformers/` và biết ngay folder nào chứa gì, file nào quan trọng nhất.
3. Hiểu `AutoModel*` registry hoạt động ra sao và vì sao một string như `"meta-llama/Llama-3-8B"` đủ để load đúng class.
4. Có lộ trình rõ ràng để đọc các phần tiếp theo, biết phần nào dựa trên phần nào.

## Cấu trúc Phần 0

Phần 0 có 5 chương:

- Chương 2 (`02-vi-sao-doc-source`) trả lời câu hỏi triết học: tại sao đáng bỏ công đọc source code của một thư viện thay vì chỉ dùng API. Đây là chương ngắn, đặt tone cho cả chuỗi.
- Chương 3 (`03-repo-layout`) đi qua từng folder của `src/transformers/`: `models/`, `generation/`, `cache_utils.py`, `modeling_utils.py`, `integrations/`. Mỗi folder có vai trò gì, file nào đáng đọc trước.
- Chương 4 (`04-model-registry-va-auto-class`) giải thích cơ chế `AutoModel`, `AutoConfig`, `AutoTokenizer`. Vì sao một dòng `AutoModelForCausalLM.from_pretrained("...")` đủ để load model. `MODEL_MAPPING_NAMES`, `_LazyAutoMapping`, `auto_class_update`.
- Chương 5 (`05-roadmap-bai-giang`) là bản đồ đọc các phần còn lại. Phần nào dựa trên phần nào, đâu là entry point cho từng câu hỏi.

## Ngữ cảnh

Repo `huggingface/transformers` được sinh ra năm 2018 với BERT, ban đầu chỉ vài chục file. Năm 2026 nó chứa khoảng 300 model gia đình khác nhau (Llama, Qwen, Mistral, T5, Whisper, ViT, CLIP, BLIP, Stable Diffusion, ...), tổng hơn 200K dòng code Python. Để quản lý độ phức tạp đó, HuggingFace dùng vài convention then chốt:

**One-model-one-file**: mỗi kiến trúc có file riêng, không chia sẻ code qua import. Trade-off: nhiều code lặp lại, nhưng dễ đọc một mình.

**Modular**: file `modular_*.py` chỉ chứa diff so với một model gốc, hệ thống tự expand thành `modeling_*.py`. Trade-off: tác giả viết ít hơn, nhưng người đọc cuối vẫn thấy file đầy đủ.

**AutoClass registry**: model class được đăng ký vào `MODEL_MAPPING_NAMES`. Một string config đủ để dispatch tới class đúng.

**Mixin cho task heads**: `LlamaForSequenceClassification` không cần viết lại logic classification, kế thừa từ `GenericForSequenceClassification`. Trade-off: hơi khó tìm code thật, nhưng giảm trùng lặp ở task head.

**AttentionInterface dispatch**: thay vì if-else trong forward, có một dict global mapping `attn_implementation` string sang function. Cho phép user override mà không sửa modeling file.

**Cache abstract**: `Cache` và `CacheLayerMixin` tách logic KV cache khỏi modeling. Bạn có thể swap `DynamicCache` thành `StaticCache` không sửa model.

Sáu convention trên là DNA của thư viện. Mọi thứ trong các phần sau là instance của các convention này. Khi đọc, bạn nên thường xuyên tự hỏi: "Đây là convention nào?".

## Khi nào quay lại Phần 0

Phần 0 là tham chiếu. Khi bạn quên `AutoModel` hoạt động ra sao, hoặc cần map từ file `modular_*.py` sang `modeling_*.py`, quay lại đây.

Sẵn sàng, ta vào chương đầu.

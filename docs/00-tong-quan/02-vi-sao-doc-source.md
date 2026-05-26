---
title: Vì sao đọc source
---

# Vì sao đọc source code, không chỉ dùng API

Có một lập luận hợp lý chống lại việc đọc source: thư viện được thiết kế để bạn không cần đọc source. Nếu API tốt, bạn chỉ cần tài liệu. Trong nhiều trường hợp lập luận này đúng. Bạn không cần đọc source của `numpy` để dùng `np.dot`. Vậy vì sao transformers lại khác.

## Ba lý do thực dụng

**Lý do thứ nhất: debug**. Khi `model.generate(...)` ra kết quả không như mong đợi, không tutorial nào trên StackOverflow giải thích được tại sao. Bạn phải mở `generation/utils.py`, đặt breakpoint, và lần ngược. Người đã đọc qua một lần sẽ tìm ra trong vài phút. Người chưa đọc sẽ mất nhiều giờ.

**Lý do thứ hai: tối ưu**. Mỗi phiên bản transformers có hàng chục flag và option có thể tăng tốc. `attn_implementation="sdpa"`, `cache_implementation="static"`, `_supports_logits_to_keep`, `torch.compile` với `mode="reduce-overhead"`. Tài liệu liệt kê chúng, nhưng không nói khi nào dùng cái nào. Đọc source cho bạn câu trả lời.

**Lý do thứ ba: customize**. Một ngày bạn cần thêm hỗ trợ cho một backend attention mới, hoặc thay đổi behavior của `generate` cho một use case đặc thù. Tài liệu sẽ không hướng dẫn. Source thì có. Hơn nữa, vì transformers theo convention rõ ràng (Phần 5), customize đúng cách rất ngắn.

## Lý do triết học

Đó là về thái độ. Một kỹ sư xem framework như hộp đen sẽ bị giới hạn bởi bề mặt API. Một kỹ sư xem framework như code có thể đọc được sẽ tự nhiên đặt câu hỏi sâu hơn: vì sao K được transpose ở chiều này, vì sao cache có cờ `is_updated`, vì sao position_ids được tính lại mỗi step thay vì lưu.

Mỗi câu hỏi có một câu trả lời nằm trong source. Khi bạn tìm ra, bạn không chỉ học một sự thật, bạn học một cách suy nghĩ. Lâu dần, bạn có thể tự thiết kế thư viện tương tự.

## Vì sao transformers đáng đọc

Không phải mọi codebase đều đáng đọc. Có thư viện code rất tốt nhưng kiến trúc khó theo. Có thư viện code dễ theo nhưng thiết kế lỗi thời. Transformers nằm ở giao điểm: code Python sạch, convention rõ ràng, và quan trọng nhất, chứa đựng các quyết định thiết kế đang là **state-of-the-art** trong ngành.

Cụ thể, các quyết định sau là điển hình cho thư viện ML production:

- Tách config khỏi model: `LlamaConfig` định nghĩa hyperparameter, `LlamaModel` định nghĩa logic. Hai class song song.
- Tách cache khỏi attention: `LlamaAttention` chỉ biết gọi `past_key_values.update(k, v, layer_idx)`. Cache có thể là bất kỳ implementation nào tuân theo interface.
- Dispatch qua dict: `AttentionInterface` là dict global, không phải if-else trong forward. Cho phép user thêm backend mà không sửa source thư viện.
- Mixin cho task heads: `LlamaForSequenceClassification` là một dòng `class X(GenericForSequenceClassification, LlamaPreTrainedModel): ...`. Logic classification dùng chung qua mixin.

Bốn pattern này áp dụng được cho mọi thư viện ML, không chỉ NLP. Đọc transformers là cách học các pattern đó qua một implementation thực tế thay vì paper abstract.

## Cảnh báo

Đừng cố đọc toàn bộ thư viện. Bạn sẽ kiệt sức và không nhớ gì. Chiến lược tốt hơn:

1. Đọc một model gia đình bạn quan tâm nhất (ví dụ Llama).
2. Đọc một file cốt lõi (ví dụ `generation/utils.py` cho generate).
3. Đọc một abstract (ví dụ `cache_utils.py` cho cache).

Sau ba phần đó, bạn có đủ nền để đọc các phần khác khi cần. Phần 0 chương 3 sẽ liệt kê các file đáng ưu tiên.

## Cách đọc hiệu quả

Đọc một file lớn (3000-5000 dòng như `generation/utils.py`) không phải đọc từ trên xuống. Cách hiệu quả:

1. **Đọc tên class và method trước**: dùng search "class " hoặc "def " để có outline. Hiểu các block chính trước khi sa vào chi tiết.
2. **Đọc docstring**: docstring của tác giả thường giải thích vì sao, code chỉ nói cái gì.
3. **Đặt breakpoint trong code thật**: chạy một script tiny, dừng trong `LlamaAttention.forward`, in shape, in placement. Trực giác hình thành rất nhanh.
4. **Đọc comment liên quan đến edge case**: comment thường ở chỗ phức tạp, đó là chỗ thiết kế đang trade-off.
5. **Đối chiếu với paper hoặc blog gốc**: ví dụ FlashAttention paper, RoPE paper. Code là implementation của ý tưởng paper.

Năm bước trên áp dụng cho mọi file lớn, không chỉ transformers.

Sẵn sàng, chương sau ta đi vào layout của repo.

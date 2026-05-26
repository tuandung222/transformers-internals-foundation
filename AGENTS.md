# AGENTS.md

Hướng dẫn cho AI agent (Cascade, Cursor, Aider, Codex, ...) làm việc trên repo này. Đọc trước khi sửa bất kỳ file nào.

## 1. Project context

Repo là một **chuỗi bài giảng tiếng Việt** (Docusaurus 3) về internals của HuggingFace `transformers`. Audience: ML engineer đã quen Transformer ở mức conceptual, muốn đọc được source code thật. Triết lý: bottom-up, neo mọi khái niệm vào file/class/dòng code cụ thể trong codebase transformers thật.

**Codebase tham chiếu (source of truth)**: `/Users/admin/TuanDung/repos/transformers` (HF transformers checkout, tag tại thời điểm viết). Mọi citation phải khớp class/file thực tồn tại ở đây.

## 2. Hard rules (phải tuân thủ, có CI/QA enforce)

- **`README.md` phải rỗng** (0 bytes). Lý do: privacy. Check bởi `scripts/qa_docs.py`. Nếu cần giữ nội dung README cũ, chuyển sang `README_archived.md`; tuyệt đối không để nội dung public-facing trong `README.md`.
- **Không em-dash (ký tự U+2014)** trong bất kỳ file nào ngoài `README.md`. Dùng dấu phẩy, dấu hai chấm, hoặc parenthesis. Check bởi `qa_docs.py`.
- **Không leak identity cá nhân**: pattern `Small-?Qwen|CLIP-?HAR|LLMs?-with-Semantic-Search|Open-?vocabulary-Action-Recognition|tuandung222` không được xuất hiện trong `docs/`. Check bởi `qa_docs.py`.
- **Privacy infra**: `static/robots.txt` phải `User-agent: *\nDisallow: /`. `docusaurus.config.ts` phải có meta `noindex,nofollow,noarchive,nosnippet` và `sitemap: false`. Đừng đụng vào.
- **Sidebar ID phải khớp file slug**. Mỗi `id` trong `sidebars.ts` tương ứng với `docs/<id>.md` hoặc `docs/<id>/index.md`. Check bởi `qa_docs.py`.
- **Link nội bộ tuyệt đối `](/docs/...)` phải trỏ tới file tồn tại**. Check bởi `qa_docs.py`.
- **Chỉ commit khi `npm run verify` pass** (gộp qa_docs + typecheck + build).

## 3. Writing conventions

**Ngôn ngữ**: Tiếng Việt cho prose. Giữ thuật ngữ tiếng Anh khi: tên class/method (`PreTrainedModel`, `from_pretrained`), tên kỹ thuật phổ biến (attention, cache, embedding, tokenizer, decoder, prefill, decode), tên paper.

**Tone**: Trực tiếp, không filler. Không mở đầu bằng "Trong chương này chúng ta sẽ...". Vào thẳng nội dung. Không xưng "chúng tôi" hoa mỹ; dùng "ta" hoặc giọng impersonal khi cần.

**Bottom-up**: Luôn dẫn code/snippet/số liệu trước, giải thích trừu tượng sau. Không tổng quát hoá khi chưa có một ví dụ cụ thể.

**Không emoji** trong docs (trừ khi user yêu cầu).

**Structure mặc định mỗi chapter**:

1. Frontmatter `title:`.
2. `# Heading` trùng title.
3. 1 đoạn mở đầu nêu vì sao chapter tồn tại (1-3 câu).
4. Section chính với `## H2`.
5. Snippet code (Python hoặc pseudocode) đi kèm giải thích.
6. Số liệu cụ thể nếu có (memory, FLOPs, latency).
7. Section `## Pitfall` hoặc `## Pitfalls` ở cuối khi có nhiều case fail thường gặp.
8. Câu cuối trỏ tới chapter kế ("Chương sau ta...").

**Citation source code**: Khi tham chiếu, ghi tên file + class + (tuỳ chọn) dòng. Ví dụ: `src/transformers/modeling_utils.py`, `class PreTrainedModel`. Không bịa dòng số nếu chưa verify.

**Số liệu**: Khi nêu memory/latency, ghi rõ model + batch + context. Ví dụ: "Llama-3 8B, batch=1, context 128k: 17.18 GB". Tránh hand-wave kiểu "memory rất lớn".

## 4. Diagram and chart teaching guidance

Repo này sẽ có các chương dùng nhiều diagram để giải thích flow, tensor shape, cache lifecycle, dispatch path và software design. Mục tiêu không phải sưu tầm càng nhiều sơ đồ càng tốt. Mục tiêu là dạy người đọc nhìn diagram có phương pháp, biết sơ đồ nào trả lời câu hỏi nào, và không bị choáng khi gặp diagram phức tạp trên mạng.

**Nguyên tắc chính**: mỗi diagram phải trả lời một câu hỏi cụ thể trước khi vẽ. Nếu không nói được câu hỏi, đừng vẽ. Ví dụ:

- **Flowchart**: control flow rẽ nhánh thế nào?
- **Sequence diagram**: object/function nào gọi object/function nào theo thời gian?
- **Dataflow diagram**: tensor/data đi qua các bước nào và shape đổi ra sao?
- **Dependency graph**: phần nào phụ thuộc phần nào?
- **State machine**: object có những trạng thái nào và transition nào hợp lệ?
- **Component diagram**: boundary giữa module/service/class lớn nằm ở đâu?
- **Architecture diagram**: system-level boxes giao tiếp bằng protocol nào?
- **Timeline chart**: prefill/decode/training/inference diễn ra theo pha nào?
- **Matrix/heatmap style diagram**: attention pattern hoặc masking pattern có cấu trúc gì?
- **Table**: dùng khi cần so sánh thuộc tính, không ép thành hình nếu bảng rõ hơn.

**Cách dạy người đọc không bị ngợp**:

1. Đọc title trước: diagram đang trả lời câu hỏi gì?
2. Xác định node type: node là class, tensor, process, request, cache layer hay hardware resource?
3. Xác định edge meaning: edge là call, dependency, data movement, ownership, hoặc time order?
4. Tìm entry point và exit point: bắt đầu từ đâu, kết thúc ở đâu?
5. Ignore styling trước: màu sắc, icon, grouping chỉ đọc sau khi hiểu semantics.
6. Trace một path cụ thể: chọn một request, một token, một tensor hoặc một function call rồi đi hết đường.
7. Quay lại invariant: sau khi đọc diagram, người đọc phải phát biểu được 2-5 invariant kiểm chứng được trong code.

**Progressive disclosure**: với diagram phức tạp, không dump full diagram ngay. Dạy theo 3 lớp:

1. Skeleton: chỉ node chính và hướng đi tổng quát.
2. Critical path: thêm path quan trọng nhất, ví dụ `generate -> model.forward -> attention -> cache.update`.
3. Edge cases: thêm branch, fallback, backend khác, error path hoặc optimization.

**Diagram trong repo này phải neo vào source code**. Sau diagram nên có ít nhất một trong các thứ sau:

- File/class/function thật trong `/Users/admin/TuanDung/repos/transformers`.
- Snippet code ngắn chứng minh một edge quan trọng.
- Tensor shape cụ thể trước và sau một node.
- Invariant có thể verify bằng đọc code.

**Không biến chương thành gallery diagram**. Một chương tốt thường có 1-3 diagram chính. Nếu cần hơn, mỗi diagram phải có vai trò khác nhau: overview, critical path, pitfall. Tránh 5 diagram cùng nói một ý bằng hình thức khác nhau.

**Khi viết về diagram phổ biến trong software design**, phải nhấn mạnh lựa chọn theo câu hỏi, không theo trend. Sequence diagram không thay dataflow diagram. Component diagram không thay dependency graph. Architecture diagram cao tầng không chứng minh được tensor shape. Nếu reader chọn sai loại diagram, họ sẽ tự tạo confusion.

**Mermaid preference**: ưu tiên Mermaid khi đủ biểu đạt vì diff được, review được, build cùng docs. Nếu Mermaid quá rối, dùng ASCII diagram trong code fence. Không dùng hình ảnh raster trừ khi user yêu cầu hoặc Mermaid không diễn đạt được.

## 5. MDX gotchas (lessons từ session đầu tiên, painful)

Docusaurus 3 dùng MDX 3. MDX parse `{...}` thành JSX expression **ngay cả trong khối toán `$$...$$`**. Hậu quả:

- `$$H_{kv}$$` → MDX coi `{kv}` là JSX, raise `ReferenceError: kv is not defined` lúc build SSR. **Fix**: dùng `\text{kv}` (`$$H_{\text{kv}}$$`), hoặc bỏ math, dùng code fence ASCII: `H_kv` trong ` ``` ` block.
- `\text{cache}` thì OK vì MDX không parse khi có backslash macro phía trước. An toàn nhất: mọi subscript identifier dùng `\text{...}`.
- `<1%` ngoài code block → MDX coi `<1` là JSX tag mở, raise "Unexpected character `1` before name". **Fix**: viết "dưới 1%" hoặc bọc backtick `` `<1%` ``.
- `<->` ngoài code block → cùng vấn đề. **Fix**: backtick `` `<->` ``.
- Curly literal trong prose `{foo}` cũng có rủi ro. Bọc backtick nếu là code, hoặc escape `\{foo\}`.

**Quy tắc**: chạy `npm run build` mỗi khi thêm math hoặc dấu `<`. `qa_docs.py` và `tsc` không bắt được lỗi MDX.

## 6. Workflow chuẩn

Trước khi commit bất kỳ thay đổi nào trong `docs/`:

```bash
npm run verify
# = python3 scripts/qa_docs.py && tsc && docusaurus build
```

Nếu chỉ sửa file nhỏ, có thể chạy từng bước:

```bash
python3 scripts/qa_docs.py   # nhanh, bắt em-dash, sidebar, README
npm run typecheck            # bắt lỗi TS trong sidebars.ts, docusaurus.config.ts
npm run build                # CRITICAL: bắt lỗi MDX (xem mục 5)
```

`npm run start` chỉ cho dev preview, **không phát hiện hết** lỗi build-time. Phải `build`.

## 7. Cấu trúc repo

```
docs/
  intro.md                    # Trang giới thiệu
  00-tong-quan/               # Part 0: orientation
  01-attention/               # Part 1: attention internals (6 chương)
  02-cross-attention/         # Part 2: cross-attention + encoder-decoder (5)
  03-kv-cache/                # Part 3: KV cache (7)
  04-generate/                # Part 4: generate method (8)
  05-hf-conventions/          # Part 5: PreTrainedModel conventions (8)
  resources/
    glossary.md
    cheatsheet.md
    references.md
sidebars.ts                   # Phải đồng bộ với docs/ tree
docusaurus.config.ts          # baseUrl, organizationName, math plugin
scripts/qa_docs.py            # QA gate
.github/workflows/deploy.yml  # GitHub Pages auto-deploy on push main
```

**Naming**: file chapter dạng `NN-slug-tieng-viet.md` (NN = 01-99, slug snake-case không dấu). Khi đổi tên file, **phải** update `sidebars.ts` đồng thời.

**Thêm Part mới**: tạo thư mục `06-...`, viết `01-overview.md`, thêm category block vào `sidebars.ts`, cập nhật `docs/00-tong-quan/05-roadmap-bai-giang.md`.

## 8. Patterns đã thiết lập

- **Overview chapter** mỗi part: nêu mục tiêu, list các chương, đặt bối cảnh trong toàn series.
- **Walkthrough chapter**: đi qua một class/function thực, copy snippet, annotate dòng.
- **Comparison chapter**: bảng so sánh backend/strategy/variant.
- **Pitfall section** cuối chapter dài: liệt kê 3-5 case fail thường gặp với fix.
- **Glossary entry**: 1-2 câu định nghĩa + chỉ chapter chi tiết.
- **Cheatsheet**: snippet copy-paste-ready, không giải thích dài.

## 9. Source-of-truth references

Khi cần verify code transformers:

```bash
# Codebase HF transformers local
/Users/admin/TuanDung/repos/transformers/src/transformers/
  modeling_utils.py           # PreTrainedModel, from_pretrained
  cache_utils.py              # Cache, DynamicCache, StaticCache, QuantizedCache
  generation/utils.py         # GenerationMixin, _sample, _beam_search
  models/llama/modeling_llama.py    # Reference model
  models/t5/modeling_t5.py          # Encoder-decoder reference
  integrations/                # Attention backend (sdpa, flash, flex, paged)
```

Tool agent dùng: `grep_search` cho targeted search, `code_search` cho exploratory với câu hỏi cụ thể, `read_file` để verify dòng số.

## 10. Deployment

- Repo: `tuandung222/transformers-internals-foundation` trên GitHub.
- Pages URL: https://tuandung222.github.io/transformers-internals-foundation/
- Auto-deploy: `.github/workflows/deploy.yml` chạy on push `main`. Verify build local trước khi push (workflow sẽ fail nếu build fail).
- Config `docusaurus.config.ts` hard-code `organizationName: tuandung222`, `baseUrl: /transformers-internals-foundation/`. Đừng đổi trừ khi migrate account.

## 11. Anti-patterns đã gặp (avoid)

- **Tạo file mới khi chỉ cần edit**. Repo này đã có cấu trúc, đừng thêm scratch file (`notes.md`, `tmp.md`).
- **Liệt kê inline bullet** thay vì list block. Markdown list phải có endline trước.
- **Em-dash trong văn bản** (style preference + qa enforce).
- **Mock số liệu không reasonable**. Ví dụ memory 10x sai. Nếu không chắc, ghi "tham khảo" hoặc bỏ.
- **Hand-wave reference**: "transformers có class abc xử lý việc này". Phải ghi chính xác tên class và file.
- **Generic intro section**: "Cross-attention là một kỹ thuật quan trọng...". Bỏ. Vào thẳng code.
- **Math block với `{identifier}` không bọc `\text`**: gây MDX fail (xem mục 5).
- **Đụng vào `README.md`**: phải giữ rỗng.
- **Sửa modeling file của HF transformers**: repo này chỉ documentation, không fork transformers.

## 12. Quick checklist khi thêm 1 chương

- [ ] File đặt đúng folder, naming `NN-slug.md`.
- [ ] Frontmatter `title:` set.
- [ ] Heading 1 trùng title.
- [ ] Không em-dash.
- [ ] Math block với identifier dùng `\text{}` hoặc bỏ math.
- [ ] Mọi `<digit` và `<-` bọc backtick.
- [ ] Citation code có file path + class name thực.
- [ ] Số liệu có context (model + batch + seq_len).
- [ ] Pitfall section nếu chapter dài (>200 dòng).
- [ ] Câu cuối trỏ chapter kế.
- [ ] `sidebars.ts` có entry cho file mới.
- [ ] `npm run verify` pass.
- [ ] Commit message ngắn gọn, mô tả thay đổi.

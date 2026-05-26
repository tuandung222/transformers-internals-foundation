# AGENTS.md

Hướng dẫn cho AI agent (Cascade, Cursor, Aider, Codex, ...) làm việc trên repo này. Đọc trước khi sửa bất kỳ file nào.

## 1. Project context

Repo là một **chuỗi bài giảng tiếng Việt** (Docusaurus 3) về internals của HuggingFace `transformers`. Audience: ML engineer đã quen Transformer ở mức conceptual, muốn đọc được source code thật. Triết lý: bottom-up, neo mọi khái niệm vào file/class/dòng code cụ thể trong codebase transformers thật.

**Codebase tham chiếu (source of truth)**: `/Users/admin/TuanDung/repos/transformers` (HF transformers checkout, tag tại thời điểm viết). Mọi citation phải khớp class/file thực tồn tại ở đây.

## 2. Hard rules (phải tuân thủ, có CI/QA enforce)

- **`README.md` phải rỗng** (0 bytes). Lý do: privacy. Check bởi `scripts/qa_docs.py`.
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

## 4. MDX gotchas (lessons từ session đầu tiên, painful)

Docusaurus 3 dùng MDX 3. MDX parse `{...}` thành JSX expression **ngay cả trong khối toán `$$...$$`**. Hậu quả:

- `$$H_{kv}$$` → MDX coi `{kv}` là JSX, raise `ReferenceError: kv is not defined` lúc build SSR. **Fix**: dùng `\text{kv}` (`$$H_{\text{kv}}$$`), hoặc bỏ math, dùng code fence ASCII: `H_kv` trong ` ``` ` block.
- `\text{cache}` thì OK vì MDX không parse khi có backslash macro phía trước. An toàn nhất: mọi subscript identifier dùng `\text{...}`.
- `<1%` ngoài code block → MDX coi `<1` là JSX tag mở, raise "Unexpected character `1` before name". **Fix**: viết "dưới 1%" hoặc bọc backtick `` `<1%` ``.
- `<->` ngoài code block → cùng vấn đề. **Fix**: backtick `` `<->` ``.
- Curly literal trong prose `{foo}` cũng có rủi ro. Bọc backtick nếu là code, hoặc escape `\{foo\}`.

**Quy tắc**: chạy `npm run build` mỗi khi thêm math hoặc dấu `<`. `qa_docs.py` và `tsc` không bắt được lỗi MDX.

## 5. Workflow chuẩn

Trước khi commit bất kỳ thay đổi nào trong `docs/`:

```bash
npm run verify
# = python3 scripts/qa_docs.py && tsc && docusaurus build
```

Nếu chỉ sửa file nhỏ, có thể chạy từng bước:

```bash
python3 scripts/qa_docs.py   # nhanh, bắt em-dash, sidebar, README
npm run typecheck            # bắt lỗi TS trong sidebars.ts, docusaurus.config.ts
npm run build                # CRITICAL: bắt lỗi MDX (xem mục 4)
```

`npm run start` chỉ cho dev preview, **không phát hiện hết** lỗi build-time. Phải `build`.

## 6. Cấu trúc repo

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

## 7. Patterns đã thiết lập

- **Overview chapter** mỗi part: nêu mục tiêu, list các chương, đặt bối cảnh trong toàn series.
- **Walkthrough chapter**: đi qua một class/function thực, copy snippet, annotate dòng.
- **Comparison chapter**: bảng so sánh backend/strategy/variant.
- **Pitfall section** cuối chapter dài: liệt kê 3-5 case fail thường gặp với fix.
- **Glossary entry**: 1-2 câu định nghĩa + chỉ chapter chi tiết.
- **Cheatsheet**: snippet copy-paste-ready, không giải thích dài.

## 8. Source-of-truth references

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

## 9. Deployment

- Repo: `tuandung222/transformers-internals-foundation` trên GitHub.
- Pages URL: https://tuandung222.github.io/transformers-internals-foundation/
- Auto-deploy: `.github/workflows/deploy.yml` chạy on push `main`. Verify build local trước khi push (workflow sẽ fail nếu build fail).
- Config `docusaurus.config.ts` hard-code `organizationName: tuandung222`, `baseUrl: /transformers-internals-foundation/`. Đừng đổi trừ khi migrate account.

## 10. Anti-patterns đã gặp (avoid)

- **Tạo file mới khi chỉ cần edit**. Repo này đã có cấu trúc, đừng thêm scratch file (`notes.md`, `tmp.md`).
- **Liệt kê inline bullet** thay vì list block. Markdown list phải có endline trước.
- **Em-dash trong văn bản** (style preference + qa enforce).
- **Mock số liệu không reasonable**. Ví dụ memory 10x sai. Nếu không chắc, ghi "tham khảo" hoặc bỏ.
- **Hand-wave reference**: "transformers có class abc xử lý việc này". Phải ghi chính xác tên class và file.
- **Generic intro section**: "Cross-attention là một kỹ thuật quan trọng...". Bỏ. Vào thẳng code.
- **Math block với `{identifier}` không bọc `\text`**: gây MDX fail (xem mục 4).
- **Đụng vào `README.md`**: phải giữ rỗng.
- **Sửa modeling file của HF transformers**: repo này chỉ documentation, không fork transformers.

## 11. Quick checklist khi thêm 1 chương

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

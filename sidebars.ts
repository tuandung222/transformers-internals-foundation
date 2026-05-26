import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  lectureSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Giới thiệu chuỗi bài giảng',
    },
    {
      type: 'category',
      label: 'Phần 0: Tổng quan thư viện transformers',
      link: {type: 'doc', id: '00-tong-quan/01-overview'},
      collapsed: false,
      items: [
        '00-tong-quan/01-overview',
        '00-tong-quan/02-vi-sao-doc-source',
        '00-tong-quan/03-repo-layout',
        '00-tong-quan/04-model-registry-va-auto-class',
        '00-tong-quan/05-roadmap-bai-giang',
      ],
    },
    {
      type: 'category',
      label: 'Phần 1: Attention internals',
      link: {type: 'doc', id: '01-attention/01-overview'},
      collapsed: false,
      items: [
        '01-attention/01-overview',
        '01-attention/02-llama-attention-walkthrough',
        '01-attention/03-attention-interface-va-dispatch',
        '01-attention/04-eager-vs-sdpa-vs-flash',
        '01-attention/05-paged-attention',
        '01-attention/06-tong-ket-attention',
      ],
    },
    {
      type: 'category',
      label: 'Phần 2: Cross-attention và encoder-decoder',
      link: {type: 'doc', id: '02-cross-attention/01-overview'},
      collapsed: false,
      items: [
        '02-cross-attention/01-overview',
        '02-cross-attention/02-self-vs-cross-attention',
        '02-cross-attention/03-t5-cross-attention',
        '02-cross-attention/04-encoder-decoder-cache',
        '02-cross-attention/05-whisper-va-vision-encoder-decoder',
      ],
    },
    {
      type: 'category',
      label: 'Phần 3: KV Cache',
      link: {type: 'doc', id: '03-kv-cache/01-overview'},
      collapsed: false,
      items: [
        '03-kv-cache/01-overview',
        '03-kv-cache/02-vi-sao-can-kv-cache',
        '03-kv-cache/03-cache-layer-mixin',
        '03-kv-cache/04-dynamic-cache',
        '03-kv-cache/05-static-cache-va-compile',
        '03-kv-cache/06-quantized-va-offloaded-cache',
        '03-kv-cache/07-sequence-diagram-update',
      ],
    },
    {
      type: 'category',
      label: 'Phần 4: Generate method',
      link: {type: 'doc', id: '04-generate/01-overview'},
      collapsed: false,
      items: [
        '04-generate/01-overview',
        '04-generate/02-top-level-generate',
        '04-generate/03-sample-greedy',
        '04-generate/04-logits-processors',
        '04-generate/05-stopping-criteria',
        '04-generate/06-beam-search',
        '04-generate/07-assisted-decoding',
        '04-generate/08-compile-va-cuda-graph',
      ],
    },
    {
      type: 'category',
      label: 'Phần 5: Convention thiết kế model của HF',
      link: {type: 'doc', id: '05-hf-conventions/01-overview'},
      collapsed: false,
      items: [
        '05-hf-conventions/01-overview',
        '05-hf-conventions/02-pretrained-model-va-config',
        '05-hf-conventions/03-modular-va-modeling-files',
        '05-hf-conventions/04-auto-class-registry',
        '05-hf-conventions/05-generic-mixins-cho-task-heads',
        '05-hf-conventions/06-tied-weights-va-attn-flags',
        '05-hf-conventions/07-tp-fsdp-plans-trong-pretrained-model',
        '05-hf-conventions/08-from-pretrained-flow',
      ],
    },
    {
      type: 'category',
      label: 'Tài nguyên',
      collapsed: true,
      items: [
        'resources/glossary',
        'resources/cheatsheet',
        'resources/references',
      ],
    },
  ],
};

export default sidebars;

import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const parts = [
  ['00', 'Tổng quan thư viện transformers', 'Cấu trúc repo, AutoClass registry, philosophy "one-model one-file", tại sao đọc source giúp bạn làm chủ thư viện thay vì dùng như hộp đen.', '/docs/00-tong-quan/01-overview'],
  ['01', 'Attention internals', 'Walkthrough LlamaAttention, AttentionInterface dispatch, các backend eager/SDPA/FlashAttention/FlexAttention, paged variants cho continuous batching.', '/docs/01-attention/01-overview'],
  ['02', 'Cross-attention và encoder-decoder', 'Vì sao cross-attention chỉ tính K/V một lần, T5Attention forward bao trùm cả self và cross, EncoderDecoderCache với cờ is_updated.', '/docs/02-cross-attention/01-overview'],
  ['03', 'KV Cache', 'CacheLayerMixin abstract, DynamicCache vs StaticCache, quantized và offloaded cache, luồng update của past_key_values khi sinh từng token.', '/docs/03-kv-cache/01-overview'],
  ['04', 'Generate method', 'GenerationConfig, pipeline 9 bước trong generate(), 5 chiến lược decoding, vòng lặp _sample với prefill + decode, beam search và assisted decoding.', '/docs/04-generate/01-overview'],
  ['05', 'Convention thiết kế model của HF', 'PreTrainedModel + PreTrainedConfig, modular vs modeling, Auto registry, mixin GenericForX, tied weights, _tp_plan và from_pretrained flow.', '/docs/05-hf-conventions/01-overview'],
];

function HomepageHeader(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <Heading as="h1" className={styles.heroTitle}>{siteConfig.title}</Heading>
        <p className={styles.heroTagline}>{siteConfig.tagline}</p>
        <div className={styles.heroButtons}>
          <Link className={`button button--primary button--lg ${styles.heroButton}`} to="/docs/intro">Bắt đầu đọc</Link>
          <Link className={`button button--secondary button--lg ${styles.heroButton}`} to="/docs/00-tong-quan/01-overview">Vào Phần 0</Link>
          <Link className={`button button--secondary button--lg ${styles.heroButton}`} to="/docs/resources/how-to-read-diagrams">Cách đọc diagram</Link>
        </div>
      </div>
    </header>
  );
}

function PartGrid(): ReactNode {
  return (
    <section className={styles.gridSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>Chuỗi bài giảng đi sâu</Heading>
        <p className={styles.sectionSubtitle}>
          Mục tiêu: bạn đọc xong sẽ tự tin mở `modeling_llama.py`, `cache_utils.py`, hoặc `generation/utils.py` và hiểu từng quyết định thiết kế. Mỗi phần đều có walkthrough source code thật, kèm derivation toán học, sơ đồ luồng dữ liệu và một tài nguyên riêng về cách đọc diagram không bị quá tải.
        </p>
        <div className={styles.grid}>
          {parts.map(([number, title, description, to]) => (
            <Link key={number} to={to} className={styles.card}>
              <div className={styles.cardNumber}>PHẦN {number}</div>
              <Heading as="h3" className={styles.cardTitle}>{title}</Heading>
              <p className={styles.cardDescription}>{description}</p>
              <span className={styles.badgeReady}>Đọc phần này</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function PhilosophySection(): ReactNode {
  return (
    <section className={styles.philosophy}>
      <div className="container">
        <blockquote className={styles.quote}>
          <p><em>Một thư viện tốt là một thư viện mà bạn có thể đọc được khi cần. HuggingFace transformers cố tình tối thiểu hóa abstraction giữa code và mô hình: mỗi model có một file `modeling_*.py` riêng, mọi attention pattern lộ ra rõ ràng, KV cache được tách thành module độc lập, và generate là một pipeline có thể mở ra từng bước.</em></p>
        </blockquote>
        <p className={styles.philosophyText}>
          Chuỗi này dạy bạn cách đọc thư viện đó. Không phải để memorize API, mà để hiểu vì sao API được thiết kế như vậy. Khi đó bạn có thể tự viết một model mới đúng convention, debug được production issue, và tận dụng tối đa các backend attention hoặc cache strategy mà không cần đợi tutorial.
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline as string}>
      <HomepageHeader />
      <main>
        <PartGrid />
        <PhilosophySection />
      </main>
    </Layout>
  );
}

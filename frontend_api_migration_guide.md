# フロントエンド向け API 移行ガイド

## 📋 概要

レコメンド文生成フローを最適化するため、3つの統合APIを新規作成しました。これにより、**最大14回のAPIコールを最小4回に削減**できます。

---

## 🔄 変更前後のフロー比較

```
【変更前】最大14回のAPIコール
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → (選択) → 10 → 11 → 12 → 13 → 14 → 15

【変更後】最小4回のAPIコール
[cascade] → [similarity/multi] → (選択) → [cascade] → [related-graphs/batch] → 14 → 15
    ↓              ↓                              ↓                 ↓
  1-6統合        7-8統合                        10             11-13統合
```

---

## 📖 API 1: グラフ検索の統合

### エンドポイント
```
GET /api/v1/manga-anime-neo4j/graph/cascade
```

### 変更前（6回のAPIコール）
```typescript
// ステップ 1-6: 順次実行
const strategies = [
  { lang: 'japanese', mode: 'simple' },
  { lang: 'japanese', mode: 'fulltext' },
  { lang: 'japanese', mode: 'ranked' },
  { lang: 'english', mode: 'simple' },
  { lang: 'english', mode: 'fulltext' },
  { lang: 'english', mode: 'ranked' },
];

let result = null;
for (const { lang, mode } of strategies) {
  const res = await fetch(
    `/api/v1/manga-anime-neo4j/graph?q=${query}&limit=3&lang=${lang}&mode=${mode}`
  );
  const data = await res.json();
  if (data.nodes?.length > 0) {
    result = data;
    break;
  }
}
```

### 変更後（1回のAPIコール）
```typescript
// ステップ 1-6: 統合API
const response = await fetch(
  `/api/v1/manga-anime-neo4j/graph/cascade?q=${encodeURIComponent(query)}&limit=3&languages=japanese,english`
);
const result = await response.json();

// 結果が空かどうかチェック
if (result.nodes?.length > 0) {
  // ステップ 11 へ進む
  proceedToRelatedGraphs(result);
} else {
  // ステップ 7 (類似検索) へ進む
  proceedToSimilaritySearch(query);
}
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|------------|------|
| `q` | string | - | 検索キーワード |
| `limit` | number | `3` | 取得するWork数の上限 |
| `languages` | string | `"japanese,english"` | 検索言語の優先順（カンマ区切り） |
| `include_hentai` | boolean | `false` | Hentaiコンテンツを含めるか |

### レスポンス例
```json
{
  "nodes": [
    {
      "id": "4:xxx:123",
      "label": "Work",
      "properties": {
        "title_name": "Jujutsu Kaisen",
        "japanese_name": "呪術廻戦",
        "mal_id": 113415
      }
    }
  ],
  "edges": [...],
  "total_nodes": 5,
  "total_edges": 4
}
```

---

## 📖 API 2: 類似検索の統合

### エンドポイント
```
POST /api/v1/manga-anime-neo4j/vector/similarity/multi
```

### 変更前（2回のAPIコール + マージ処理）
```typescript
// ステップ 7: title_en で検索
const enResponse = await fetch('/api/v1/manga-anime-neo4j/vector/similarity', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: userInput,
    embedding_type: 'title_en',
    embedding_dims: 128,
    limit: 10,
    threshold: 0.3,
    include_hentai: false
  })
});

// ステップ 8: title_ja で検索
const jaResponse = await fetch('/api/v1/manga-anime-neo4j/vector/similarity', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: userInput,
    embedding_type: 'title_ja',
    embedding_dims: 128,
    limit: 10,
    threshold: 0.3,
    include_hentai: false
  })
});

// フロントエンドでマージ・重複排除
const enResults = await enResponse.json();
const jaResults = await jaResponse.json();
const merged = mergeAndDeduplicate(enResults.results, jaResults.results);
```

### 変更後（1回のAPIコール）
```typescript
// ステップ 7-8: 統合API
const response = await fetch('/api/v1/manga-anime-neo4j/vector/similarity/multi', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: userInput,
    embedding_types: ['title_en', 'title_ja'],
    embedding_dims: 128,
    limit: 10,
    threshold: 0.3,
    include_hentai: false
  })
});

const data = await response.json();
// data.results は既にマージ・重複排除・ソート済み

// ステップ 9: ユーザーに選択肢を表示
showMangaSelectionPopup(data.results);
```

### リクエストボディ

| フィールド | 型 | デフォルト | 説明 |
|------------|------|------------|------|
| `query` | string | (必須) | 検索クエリテキスト |
| `embedding_types` | string[] | `["title_en", "title_ja"]` | 検索対象の埋め込みタイプ |
| `embedding_dims` | number | `128` | 埋め込み次元数 |
| `limit` | number | `10` | 返却件数 |
| `threshold` | number | `0.3` | 類似度閾値 |
| `include_hentai` | boolean | `false` | Hentaiを含めるか |

### レスポンス例
```json
{
  "results": [
    {
      "work_id": "4:xxx:123",
      "title_en": "Jujutsu Kaisen",
      "title_ja": "呪術廻戦",
      "description": "...",
      "similarity_score": 0.92,
      "media_type": "manga",
      "genres": ["Action", "Supernatural"]
    },
    {
      "work_id": "4:xxx:456",
      "title_en": "Sorcery Fight",
      "title_ja": null,
      "description": "...",
      "similarity_score": 0.78,
      "media_type": "manga",
      "genres": ["Action"]
    }
  ],
  "total": 2,
  "query": "呪術廻戦",
  "embedding_types": ["title_en", "title_ja"],
  "embedding_dims": 128,
  "threshold": 0.3
}
```

---

## 📖 API 3: 関連グラフの一括取得

### エンドポイント
```
POST /api/v1/manga-anime-neo4j/related-graphs/batch
```

### 変更前（3回のAPIコール）
```typescript
// ステップ 11: Author関連作品
const authorResponse = await fetch(
  `/api/v1/manga-anime-neo4j/author/${authorNodeId}/works?limit=5`
);

// ステップ 12: Magazine関連作品
const magazineResponse = await fetch(
  `/api/v1/manga-anime-neo4j/magazine/${magazineNodeId}/works?limit=5`
);

// ステップ 13: Publisher関連雑誌
const publisherResponse = await fetch(
  `/api/v1/manga-anime-neo4j/publisher/${publisherNodeId}/magazines?limit=3`
);

const authorGraph = await authorResponse.json();
const magazineGraph = await magazineResponse.json();
const publisherGraph = await publisherResponse.json();
```

### 変更後（1回のAPIコール）
```typescript
// ステップ 11-13: 統合API
const response = await fetch('/api/v1/manga-anime-neo4j/related-graphs/batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    author_node_id: authorNodeId,       // グラフ検索結果から取得
    magazine_node_id: magazineNodeId,   // グラフ検索結果から取得
    publisher_node_id: publisherNodeId, // グラフ検索結果から取得
    author_limit: 5,
    magazine_limit: 5,
    publisher_limit: 3,
    include_hentai: false
  })
});

const data = await response.json();
// data.author_graph, data.magazine_graph, data.publisher_graph
```

### リクエストボディ

| フィールド | 型 | デフォルト | 説明 |
|------------|------|------------|------|
| `author_node_id` | string \| null | `null` | Author ノードの elementId |
| `magazine_node_id` | string \| null | `null` | Magazine ノードの elementId |
| `publisher_node_id` | string \| null | `null` | Publisher ノードの elementId |
| `author_limit` | number | `5` | Author関連作品の上限 |
| `magazine_limit` | number | `5` | Magazine関連作品の上限 |
| `publisher_limit` | number | `3` | Publisher関連雑誌の上限 |
| `reference_work_id` | string \| null | `null` | Magazine検索時のソート基準Work |
| `exclude_magazine_id` | string \| null | `null` | Publisher検索での除外雑誌 |
| `include_hentai` | boolean | `false` | Hentaiを含めるか |

### レスポンス例
```json
{
  "author_graph": {
    "nodes": [...],
    "edges": [...],
    "total_nodes": 6,
    "total_edges": 5
  },
  "magazine_graph": {
    "nodes": [...],
    "edges": [...],
    "total_nodes": 8,
    "total_edges": 7
  },
  "publisher_graph": {
    "nodes": [...],
    "edges": [...],
    "total_nodes": 4,
    "total_edges": 3
  }
}
```

### 注意事項
- 各 `*_node_id` は省略可能。`null` の場合、対応するグラフは `null` で返されます
- エラーが発生した場合も、他のグラフの取得は続行されます

---

## 🔧 完全な実装例

```typescript
// types.ts
interface GraphNode {
  id: string;
  label: string;
  properties: Record<string, any>;
}

interface GraphResponse {
  nodes: GraphNode[];
  edges: any[];
  total_nodes: number;
  total_edges: number;
}

interface SimilarityResult {
  work_id: string;
  title_en: string | null;
  title_ja: string | null;
  description: string | null;
  similarity_score: number;
  media_type: string | null;
  genres: string[] | null;
}

interface RelatedGraphsResponse {
  author_graph: GraphResponse | null;
  magazine_graph: GraphResponse | null;
  publisher_graph: GraphResponse | null;
}

// recommendationService.ts
async function generateRecommendation(userInput: string): Promise<void> {
  // ============================================
  // STEP 1-6: グラフ検索（統合API）
  // ============================================
  const cascadeResponse = await fetch(
    `/api/v1/manga-anime-neo4j/graph/cascade?q=${encodeURIComponent(userInput)}&limit=3&languages=japanese,english`
  );
  let graphResult: GraphResponse = await cascadeResponse.json();

  // ============================================
  // STEP 7-10: 類似検索（グラフが見つからない場合）
  // ============================================
  if (!graphResult.nodes?.length) {
    // STEP 7-8: 類似検索（統合API）
    const similarityResponse = await fetch(
      '/api/v1/manga-anime-neo4j/vector/similarity/multi',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userInput,
          embedding_types: ['title_en', 'title_ja'],
          embedding_dims: 128,
          limit: 10,
          threshold: 0.3,
          include_hentai: false
        })
      }
    );
    const similarityData = await similarityResponse.json();

    if (!similarityData.results?.length) {
      throw new Error('作品が見つかりませんでした');
    }

    // STEP 9: ユーザーに選択させる
    const selectedWork = await showMangaSelectionPopup(similarityData.results);

    // STEP 10: 選択された作品でグラフ再検索
    const reSearchResponse = await fetch(
      `/api/v1/manga-anime-neo4j/graph/cascade?q=${encodeURIComponent(selectedWork.title_en || selectedWork.title_ja || '')}&limit=3&languages=japanese,english`
    );
    graphResult = await reSearchResponse.json();
  }

  // ============================================
  // グラフからノードIDを抽出
  // ============================================
  const authorNode = graphResult.nodes.find(n => n.label === 'Author');
  const magazineNode = graphResult.nodes.find(n => n.label === 'Magazine');
  const publisherNode = graphResult.nodes.find(n => n.label === 'Publisher');

  // ============================================
  // STEP 11-13: 関連グラフ取得（統合API）
  // ============================================
  const relatedResponse = await fetch(
    '/api/v1/manga-anime-neo4j/related-graphs/batch',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        author_node_id: authorNode?.id ?? null,
        magazine_node_id: magazineNode?.id ?? null,
        publisher_node_id: publisherNode?.id ?? null,
        author_limit: 5,
        magazine_limit: 5,
        publisher_limit: 3,
        include_hentai: false
      })
    }
  );
  const relatedGraphs: RelatedGraphsResponse = await relatedResponse.json();

  // ============================================
  // STEP 14: Publisher + Magazine + Work グラフ取得
  // ============================================
  const magazineIds = relatedGraphs.publisher_graph?.nodes
    ?.filter(n => n.label === 'Magazine')
    ?.map(n => n.id) ?? [];

  let magazineWorkGraph: GraphResponse | null = null;
  if (magazineIds.length > 0) {
    const workGraphResponse = await fetch(
      '/api/v1/manga-anime-neo4j/magazines/work-graph',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          magazine_element_ids: magazineIds,
          work_limit: 3,
          include_hentai: false
        })
      }
    );
    magazineWorkGraph = await workGraphResponse.json();
  }

  // ============================================
  // STEP 15: JSON化
  // ============================================
  const graphJson = {
    main_graph: graphResult,
    author_works: relatedGraphs.author_graph,
    magazine_works: relatedGraphs.magazine_graph,
    publisher_magazines: relatedGraphs.publisher_graph,
    magazine_work_graph: magazineWorkGraph
  };

  // ============================================
  // STEP 16: レコメンド文生成
  // ============================================
  await generateRecommendationText(graphJson);
}

// UI Helper
async function showMangaSelectionPopup(
  results: SimilarityResult[]
): Promise<SimilarityResult> {
  // ポップアップUIを表示してユーザーに選択させる
  // 実装はUIフレームワークに依存
  return new Promise((resolve) => {
    // ... popup implementation
  });
}
```

---

## 📊 パフォーマンス比較

| シナリオ | 変更前 | 変更後 | 削減率 |
|----------|--------|--------|--------|
| グラフ検索で即ヒット | 1回 | 1回 | 0% |
| グラフ検索6回目でヒット | 6回 | 1回 | **83%** |
| 類似検索経由 | 8回 + 選択 + 1回 | 2回 + 選択 + 1回 | **67%** |
| 関連グラフ取得 | 3回 | 1回 | **67%** |
| **最悪ケース合計** | 14回 | 5回 | **64%** |

---

## ⚠️ 移行時の注意点

1. **エラーハンドリング**
   - 統合APIでもHTTPステータスコードは従来通り（200, 400, 500など）
   - `related-graphs/batch` は部分的な失敗を許容（1つがエラーでも他は返却）

2. **後方互換性**
   - 既存の個別APIは引き続き利用可能
   - 段階的な移行が可能

3. **キャッシュ戦略**
   - 統合APIのレスポンスは従来より大きいため、キャッシュ戦略の見直しを推奨

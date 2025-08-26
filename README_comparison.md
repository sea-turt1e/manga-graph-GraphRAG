# GraphRAG vs Standard LLM Comparison

GraphRAGを使った漫画推薦システムと標準的なLLMを使った推薦システムの比較を行うツールです。

## ファイル構成

```
manga-graph-GraphRAG/
├── comparison_service.py      # メインの比較サービス
├── main_comparison.py         # エントリーポイント
├── demo_scenarios.py          # デモシナリオの定義
├── prompts/
│   └── manga_prompts.py       # プロンプトテンプレート
├── sample_with_manga_graph_api.py  # GraphRAG実装
└── README_comparison.md       # このファイル
```

## 機能

### 1. StandardMangaRecommender
- 標準的なLLMのみを使用した漫画推薦
- 一般的な知識ベースから推薦を生成

### 2. ComparisonService
- GraphRAGと標準LLMの両方を実行して比較
- パフォーマンスメトリクスの計算と保存
- 成功率や文字数などの統計情報を提供

### 3. DemoRunner
- 定義済みシナリオの実行
- カスタムシナリオの実行
- エッジケースのテスト

### 4. ComparisonFormatter
- 比較結果の見やすい表示
- サマリー情報のフォーマット

## 使用方法

### 基本的な使用方法

```bash
# 定義済みデモシナリオを実行
python main_comparison.py

# または明示的に
python main_comparison.py --demo
```

### インタラクティブモード

```bash
python main_comparison.py --interactive
```

インタラクティブモードでは以下のコマンドが使用できます：
- `compare-rec <query>` - 推薦の比較
- `compare-ana <title>` - 分析の比較
- `summary` - パフォーマンスサマリーの表示
- `help` - ヘルプの表示
- `quit` - 終了

### カスタムデモモード

```bash
python main_comparison.py --custom
```

独自の推薦クエリや分析タイトルを入力して比較を実行できます。

### クイックテスト

```bash
python main_comparison.py --quick
```

開発時の動作確認用の簡単なテストを実行します。

## デモシナリオ

### 推薦テストシナリオ
- 冒険・少年漫画（ONE PIECE類似）
- 忍者・アクション（NARUTO類似）
- 心理戦・サスペンス
- 学園・恋愛
- ダークファンタジー
- スポーツ漫画

### 分析テストタイトル
- ONE PIECE
- NARUTO
- 進撃の巨人
- 鬼滅の刃
- ドラゴンボール

### エッジケース
- 非常に具体的な条件
- 矛盾する条件
- 曖昧な質問
- 存在しない可能性の高い条件

## カスタマイズ

### 新しいプロンプトの追加

`prompts/manga_prompts.py`でプロンプトテンプレートを追加・修正できます：

```python
@staticmethod
def get_custom_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["variable"],
        template="カスタムプロンプト: {variable}"
    )
```

### 新しいデモシナリオの追加

`demo_scenarios.py`の`DemoScenarios`クラスでシナリオを追加できます：

```python
@staticmethod
def get_custom_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "id": "custom_test",
            "query": "カスタムテストクエリ",
            "category": "カスタム",
            "expected_themes": ["テーマ1", "テーマ2"]
        }
    ]
```

## API作成への活用

このコードは以下のように細分化されており、API開発の参考になります：

### エンドポイント設計の参考
- `/recommend/compare` - 推薦比較
- `/analyze/compare` - 分析比較
- `/metrics/summary` - パフォーマンスサマリー

### サービス層の分離
- `ComparisonService` - ビジネスロジック
- `StandardMangaRecommender` - 標準LLM処理
- `ComparisonFormatter` - レスポンス整形

### データモデルの例
```python
{
    "query": "ユーザークエリ",
    "graphrag_result": {
        "recommendation": "推薦内容",
        "method": "graphrag",
        "success": True,
        "graph_data_size": 100
    },
    "standard_result": {
        "recommendation": "推薦内容", 
        "method": "standard_llm",
        "success": True
    },
    "comparison_metrics": {
        "both_successful": True,
        "length_ratio": 1.5,
        "entities_linked": 5
    }
}
```

## 必要な環境設定

### 環境変数
```bash
OPENAI_API_KEY=your_openai_api_key
```

### 前提条件
1. Manga Graph APIが`http://localhost:8000`で起動していること
2. 必要なPythonパッケージがインストールされていること：
   - langchain
   - langchain-openai
   - python-dotenv
   - requests（sample_with_manga_graph_api.pyで使用）

## トラブルシューティング

### よくあるエラー

1. **GraphRAG APIに接続できない**
   ```
   GraphRAG error: Connection refused
   ```
   → Manga Graph APIが起動していることを確認

2. **OpenAI APIキーエラー**
   ```
   OpenAI API error: Invalid API key
   ```
   → `.env`ファイルに正しいAPIキーが設定されていることを確認

3. **プロンプトモジュールが見つからない**
   ```
   Warning: prompts.manga_prompts not found
   ```
   → フォールバックプロンプトが使用されますが、完全な機能を使用するには`prompts/manga_prompts.py`が必要

## 拡張可能性

このツールは以下のような拡張が可能です：

1. **新しい評価指標の追加**
2. **A/Bテスト機能の実装**
3. **ユーザーフィードバックの統合**
4. **パフォーマンスログの永続化**
5. **Web UIの追加**
6. **REST API化**

各機能が独立したモジュールとして設計されているため、段階的な拡張が容易です。

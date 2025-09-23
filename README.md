# manga-graph-GraphRAG

漫画向けのGraphRAG（Graph Retrieval-Augmented Generation）デモと比較ツールの実装です。ローカル/外部の「Manga Graph API」から取得したグラフ（作品・作者・雑誌・出版社などの関係）を文脈として、LLM生成を強化します。StreamlitによるUI、GraphRAGと素のLLMの比較スクリプト、プロンプト群、Railway向け設定が含まれます。

## 主要機能

- GraphRAG推薦生成（Streamlit UI）: タイトル抽出→グラフ検索→文脈生成→ストリーミング生成
- 漫画レコメンド/分析の「GraphRAG vs 素のLLM」比較デモ（CLI）
- プロンプトテンプレート集（標準/GraphRAG/比較/メタ評価）
- Railwayでのホスティング用設定

## アーキテクチャ概要

- データ取得: Manga Graph API（ENV: `API_BASE`。既定は `http://localhost:8000`）
	- 厳格検索: `/api/v1/neo4j/search`（発行巻数によるフィルタや並び替えに対応）
	- ベクトル類似: `/api/v1/neo4j/vector/title-similarity`
	- ヘルスチェック: `/health`
- 生成器（LLM）: バックエンドの `/text-generation/generate` にプロンプトを送り、SSEでストリーミング受信（フロントからはOpenAIキー不要）
- 推薦フロー（`graphrag_service.py`）:
	1) ユーザー入力から作品名を抽出（LLM）
	2) 厳格検索（該当なしならベクトル類似→最適候補で再検索）
	3) 作者/同誌/同出版社の関係から文脈を構築
	4) GraphRAG用プロンプトで生成（ストリーミング）

## リポジトリ構成（主なファイル）

- `streamlit_app.py` — UI。素のLLMとGraphRAGの結果を左右で比較し、曖昧一致時は候補選択UIを提供
- `graphrag_service.py` — GraphRAG実装（検索・文脈組み立て・生成呼び出しの中核）
- `prompts/manga_prompts.py` — 標準/GraphRAG/比較/メタ評価の各プロンプト
- `main_comparison.py` / `comparison_service.py` / `demo_scenarios.py` — GraphRAGと標準LLMの比較ツール
- `README_comparison.md` — 比較ツールの詳細
- `README_manga_graph_api.md` — API連携のサンプル解説
- `sample_with_manga_graph_api.py` — APIクライアント＋GraphRAGサンプル実装
- `requirements.txt` — 依存関係（最小構成）
- `Procfile` / `railway.json` — Railwayデプロイ用

## 必要条件

- Python 3.12 以降（推奨）
- 稼働中の Manga Graph API
	- 既定: `http://localhost:8000`
	- 別ホストを使う場合は `API_BASE` を設定
- （任意）バックエンドAPIの認証キー: `BACKEND_API_KEY`
- フロント（本リポジトリ）からOpenAIキーは不要（生成はバックエンド経由）

## セットアップ（macOS / zsh）

```zsh
# 仮想環境の作成（任意）
python3 -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数（必要に応じて）
# API_BASE: Manga Graph APIのベースURL
# BACKEND_API_KEY: バックエンド側で要求される場合のみ
export API_BASE="http://localhost:8000"
# export BACKEND_API_KEY="your-backend-key"
```

## 使い方

### 1) Streamlitデモ（GraphRAG推薦の比較表示）
```zsh
streamlit run streamlit_app.py
```
- 左: 素のLLM（GraphRAGなし）
- 右: GraphRAG（タイトル抽出→グラフ検索→推薦生成）。候補が複数ヒットした際はUIで選択
- 入力横の「n巻以上発行」で最低巻数フィルタを適用

### 2) 比較ツール（CLI）
```zsh
# 定義済みデモ一式（推薦・分析・エッジケース）
python main_comparison.py --demo

# インタラクティブ
python main_comparison.py --interactive

# カスタム入力
python main_comparison.py --custom
```
- 詳細: `README_comparison.md`

### 3) API連携サンプル
```zsh
python sample_with_manga_graph_api.py
```
- APIクライアント（`MangaGraphClient`）とGraphRAGパイプライン例
- 詳細: `README_manga_graph_api.md`

## GraphRAGの文脈構築（例）
`graphrag_service.build_graph_context()` は、抽出作品に対して以下の情報をまとめ、プロンプトに渡します。
- 作者の別作品
- 同じ雑誌の他作品（作者が不明なら抑制）
- 同出版社の別誌に掲載の作品

## 設定・環境変数

- `API_BASE`（既定: `http://localhost:8000`）: Manga Graph APIのベースURL
- `BACKEND_API_KEY`（任意）: バックエンドの認証トークン（`Authorization: Bearer` として送信）
- バックエンドがOpenAIを利用する場合は、そちら側で `OPENAI_API_KEY` を設定してください（本フロントでは不要）

## エンドポイント（使用例）

- 生成API: `POST /text-generation/generate`（SSEストリーミング対応）
- 厳格検索: `GET /api/v1/neo4j/search`（`sort_total_volumes=desc`、`min_total_volumes` 等を付与）
- 類似検索: `GET /api/v1/neo4j/vector/title-similarity`
- ヘルスチェック: `GET /health`

## デプロイ（Railway）

- `railway.json` と `Procfile` を用意済み
- 起動コマンド: `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`
- 環境変数 `API_BASE`/`BACKEND_API_KEY` をRailwayのダッシュボードから設定

## トラブルシューティング

- API接続不可: APIサーバーが起動しているか、`API_BASE` が正しいか確認。Streamlitの「サーバー接続確認」ボタンでもチェック可能
- 類似候補が多数: 右ペインの候補選択UIで対象作品を選んでから生成
- 依存のずれ: 一部スクリプトで `langchain_openai` をimportしていますが、実行経路では直接使用していません。必要に応じて追加インストールするか、該当箇所を無効化してください
- `test_recommend.py`: モックの例示用で、未整合のimportが含まれます（必要なら調整してください）

## ライセンス

Apache License 2.0（`LICENSE` を参照）

## 出典
このアプリケーションは、以下のデータセットを使用しています：
- [メディア芸術データベース](https://mediaarts-db.artmuseums.go.jp/)
  - 出典：独立行政法人国立美術館国立アートリサーチセンター「メディア芸術データベース」 （https://mediaarts-db.artmuseums.go.jp/）
  - 独立行政法人国立美術館国立アートリサーチセンター「メディア芸術データベース」（https://mediaarts-db.artmuseums.go.jp/）を加工してデータを作成
- [OpenBD](https://openbd.jp/)
  - 「OpenBD」 （https://openbd.jp/） を利用しています。
- [MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist)
  - 本プロジェクトはMyAnimeList Dataset（MyAnimeList.net） のデータを利用しています。データベースは Open Database License (ODbL) v1.0、個々のコンテンツは Database Contents License (DbCL) v1.0 に基づきます。ライセンス条件に従い帰属表示と通知保持を行っています。」

# Claude Code Configuration

## Repository 設定
- **リポジトリ名**: manga-graph-GraphRAG


## issue作成時の注意事項
issue作成時の注意事項を@ai-rules/ISSUE_GUIDELINES.mdにまとめています
issue作成時は必ず確認して必ずこの内容に従ってissue作成を行ってください。

## アプリケーションポート設定
以下のポートで各サービスが動作します：

### 開発環境（Docker Compose）
- `http://localhost:8000` (ポート: 8000)

## 関数・エンドポイント作成時の注意事項
- 命名規則などを@ai-rules/API_FUNCTION_NAMING.mdにまとめています
- 関数やエンドポイントの作成時には必ず確認し、内容に従って実装を行ってください。

## 開発時の注意点
- バックエンドAPIはFastAPIで定義し、リクエスト/レスポンススキーマにはPydanticモデルを使うこと
- .envファイル内のキーはUPPER_SNAKE_CASEで記述し、値にクォートは付けないこと
- @ai-rules/COMMIT_AND_PR_GUIDELINES.mdにガイドラインを記述しています。git commitやPR作成時は必ず確認し、内容に従ってください。

## ファイル作成時の注意点（ファイル作成時必ず確認）
- ファイル作成時に、そのファイルがGithubに挙げられるべきではないと判断した場合には、必ず.gitignoreに指定してください。


## 動作確認・テスト時の必須確認事項（コミット前に必ず実施されるべきです）
- テスト・動作確認は修正を行って際は必ず行ってください。
- E2Eテストとしてユーザ目線での動作が問題ないかしっかりと確認してください。

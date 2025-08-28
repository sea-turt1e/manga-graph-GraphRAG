#!/usr/bin/env python3
"""
Prompt templates for manga recommendation and analysis

This module contains all prompt templates used for:
1. Standard LLM-based manga recommendations
2. Standard LLM-based manga analysis
3. Comparison prompts
"""

from langchain.prompts import PromptTemplate


class StandardMangaPrompts:
    """Prompt templates for standard manga recommendations without GraphRAG"""

    @staticmethod
    def get_recommendation_prompt() -> PromptTemplate:
        """Get prompt template for manga recommendations"""
        return PromptTemplate(
            input_variables=["user_query"],
            template="""
あなたは漫画に詳しい専門家です。ユーザーの好みに基づいて、適切な漫画を推薦してください。

ユーザーの質問: {user_query}

推薦する際は以下の点を考慮してください：
1. 一般的に知られている漫画の知識
2. ジャンルや作風の類似性
3. 人気や評価の高い作品
4. ユーザーの好みとの関連性
5. なぜその作品を推薦するのか具体的な理由

推薦作品を3〜5作品選んで、それぞれについて以下の形式で回答してください：

【作品名】作品タイトル
【作者】作者名
【ジャンル】ジャンル
【推薦理由】なぜこの作品を推薦するのか詳細に説明

推薦:""",
        )

    @staticmethod
    def get_analysis_prompt() -> PromptTemplate:
        """Get prompt template for manga analysis"""
        return PromptTemplate(
            input_variables=["manga_title"],
            template="""
「{manga_title}」について詳細な分析を行ってください。

分析に含めるべき要素:
1. 作品の基本情報（作者、ジャンル、連載期間など）
2. ストーリーの特徴と魅力
3. 主要キャラクターの魅力と成長
4. 作画やアートスタイルの特徴
5. 作品のテーマやメッセージ
6. 文化的影響や社会的意義
7. 類似作品との比較
8. 読者層や評価

以下の形式で分析してください：

【基本情報】
- 作者：
- ジャンル：
- 連載期間：
- 巻数：

【ストーリー分析】
（ストーリーの特徴、構成、魅力について詳細に分析）

【キャラクター分析】
（主要キャラクターの魅力、成長、関係性について分析）

【作品の特徴】
（作画、テーマ、独自性について分析）

【文化的意義】
（作品の影響、評価、位置づけについて分析）

【類似作品】
（似たような作品との比較）

分析結果:""",
        )

    @staticmethod
    def get_genre_analysis_prompt() -> PromptTemplate:
        """Get prompt template for genre-specific analysis"""
        return PromptTemplate(
            input_variables=["genre", "user_preferences"],
            template="""
{genre}ジャンルの漫画について、以下のユーザーの好みを考慮して分析と推薦を行ってください。

ユーザーの好み: {user_preferences}

分析内容:
1. {genre}ジャンルの特徴と魅力
2. このジャンルの代表的な作品
3. ユーザーの好みに合致する要素
4. 推薦する理由

推薦:""",
        )


class ComparisonPrompts:
    """Prompt templates for comparing different approaches"""

    @staticmethod
    def get_quality_evaluation_prompt() -> PromptTemplate:
        """Get prompt template for evaluating recommendation quality"""
        return PromptTemplate(
            input_variables=["user_query", "graphrag_result", "standard_result"],
            template="""
以下のユーザーの質問に対する2つの推薦結果を評価してください。

ユーザーの質問: {user_query}

【GraphRAG推薦結果】
{graphrag_result}

【標準LLM推薦結果】
{standard_result}

以下の観点から評価してください：
1. 推薦の適切性（ユーザーの好みとの一致度）
2. 推薦理由の具体性と説得力
3. 推薦作品の多様性
4. 情報の正確性
5. 回答の有用性

各観点を1-5点で評価し、総合的な判断を提供してください。

評価:""",
        )

    @staticmethod
    def get_difference_analysis_prompt() -> PromptTemplate:
        """Get prompt template for analyzing differences between approaches"""
        return PromptTemplate(
            input_variables=["graphrag_result", "standard_result"],
            template="""
GraphRAGと標準LLMの推薦結果の違いを分析してください。

【GraphRAG結果】
{graphrag_result}

【標準LLM結果】
{standard_result}

分析観点:
1. 推薦作品の重複と相違
2. 推薦理由の違い
3. 情報の詳細度の違い
4. アプローチの特徴の違い
5. それぞれの強みと弱み

分析結果:""",
        )


class GraphRAGPrompts:
    """Prompt Template for GraphRAG"""

    @staticmethod
    def get_recommendation_prompt() -> PromptTemplate:
        """Get prompt template for manga recommendations"""
        return PromptTemplate(
            input_variables=["user_query", "graph_data", "context"],
            template="""
以下のユーザーの好みと、グラフデータベースから取得した情報を基に、
最適な漫画の推薦を行ってください。

ユーザーの質問: {user_query}

グラフデータベースの検索結果:
{graph_data}

追加コンテキスト:
{context}

推薦する際は以下の点を考慮してください：
1. 作品間の関係性（同じ作者、同じ雑誌、類似ジャンル）
2. 時代背景や影響関係
3. ユーザーの好みとの関連性
4. なぜその作品を推薦するのか具体的な理由

推薦:""",
        )

    @staticmethod
    def get_analysis_prompt() -> PromptTemplate:
        """Get prompt template for manga analysis"""
        return PromptTemplate(
            input_variables=["manga_title", "graph_data"],
            template="""
以下のグラフデータを基に、「{manga_title}」について詳細な分析を行ってください。

グラフデータ:
-{graph_data}

分析に含めるべき要素:
1. 作品の基本情報
2. 作者とその他の作品
3. 同じ雑誌に掲載された関連作品
4. ジャンルや時代の文脈
5. 作品の影響や系譜

分析結果:""",
        )

    @staticmethod
    def get_multi_hop_prompt() -> PromptTemplate:
        """Get prompt template for multi-hop reasoning"""
        return PromptTemplate(
            input_variables=["start_entity", "end_entity", "path_data"],
            template="""
「{start_entity}」から「{end_entity}」への関係性について、
以下のグラフパスデータを基に説明してください。

パスデータ:
-{path_data}

説明には以下を含めてください：
1. 直接的な関係
2. 間接的な関係（共通の要素を通じた関係）
3. 歴史的・文化的な文脈
4. その関係性が持つ意味

関係性の説明:""",
        )

    @staticmethod
    def get_entity_extraction_prompt() -> PromptTemplate:
        """Get prompt template for entity extraction"""
        return PromptTemplate(
            input_variables=["user_text"],
            template="""
あなたは漫画に関するエンティティ抽出器です。ユーザーの自由記述から、以下のJSON形式だけを出力してください。説明文は不要です。

必ずこのスキーマ:
{{"titles": [], "authors": [], "magazines": [], "keywords": []}}

-- titles: 作品名の候補（不確実でも候補を入れる）
-- authors: 作者名の候補
-- magazines: 掲載誌の候補
-- keywords: ジャンル/特徴/キーワード

ユーザー入力:
-{user_text}
""",
        )

    @staticmethod
    def get_title_extraction_prompt() -> PromptTemplate:
        """Extract a single (most likely) formal manga title from user input."""
        return PromptTemplate(
            input_variables=["user_input"],
            template="""
以下のユーザー入力から漫画の正式な作品名を1つだけ抽出して出力してください。

制約:
1. 出力は作品名のみ（余計な記号、句読点、引用符、説明、番号は禁止）
2. 複数候補がある場合は最も一般的/代表的なものを選択
3. 入力が曖昧な場合は推測して最も関連性が高い既存の有名漫画作品名1つを出力
4. 作品名が存在しない/不明な場合は「不明」と出力
5. カタカナ/英語/漢字など公式で一般的に用いられる表記を使用

ユーザー入力:
{user_input}
""",
        )


class MetaPrompts:
    """Meta prompts for system evaluation and improvement"""

    @staticmethod
    def get_system_performance_prompt() -> PromptTemplate:
        """Get prompt template for evaluating system performance"""
        return PromptTemplate(
            input_variables=["comparison_results", "metrics"],
            template="""
以下の比較結果とメトリクスに基づいて、システムの性能を評価してください。

比較結果: {comparison_results}
メトリクス: {metrics}

評価観点:
1. システムの安定性
2. 推薦の質
3. レスポンス時間
4. ユーザー満足度
5. 改善点

評価結果:""",
        )

    @staticmethod
    def get_improvement_suggestion_prompt() -> PromptTemplate:
        """Get prompt template for improvement suggestions"""
        return PromptTemplate(
            input_variables=["current_performance", "user_feedback"],
            template="""
現在のシステム性能とユーザーフィードバックに基づいて、改善提案を作成してください。

現在の性能: {current_performance}
ユーザーフィードバック: {user_feedback}

改善提案:
1. 短期的改善（1-2週間で実装可能）
2. 中期的改善（1-3ヶ月で実装可能）
3. 長期的改善（3ヶ月以上）
4. 優先度の評価

提案:""",
        )

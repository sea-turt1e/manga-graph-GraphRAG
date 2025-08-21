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

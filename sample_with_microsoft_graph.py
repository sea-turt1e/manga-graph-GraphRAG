import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()  # .envファイルから環境変数を読み込む


class MangaGraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """GraphRAGシステムの初期化"""
        # Neo4jグラフの接続（APOCプラグインなしでも動作するように設定）
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False,  # APOCプラグインを使用しないようにする
        )

        # 手動でスキーマを設定（APOCなしで動作させるため）
        self._setup_manual_schema()

        # LLMの初期化
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))

        # GraphRAG用のプロンプトテンプレート
        self.cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
あなたは漫画データベースのエキスパートです。
以下のグラフスキーマを使用して、質問に答えるためのCypherクエリを生成してください。

スキーマ:
{schema}

質問: {question}

重要な注意事項:
1. リレーション名は英語のみを使用し、日本語は使わない
2. 有効なリレーション名: CREATED, PUBLISHED_BY, SIMILAR_TO, INFLUENCED_BY
3. WHERE句では正しい構文を使用する
4. 日本語の固有名詞（漫画タイトル、作者名）は部分一致（CONTAINS）を使用
5. 必ずLIMIT句を含める（最大20件）
6. 構文エラーを避けるため、シンプルなクエリを生成する

正しいCypherクエリの例:
MATCH (m:Manga) WHERE m.title CONTAINS 'NARUTO' RETURN m LIMIT 20
MATCH (a:Author)-[:CREATED]->(m:Manga) WHERE a.name CONTAINS '作者名' RETURN m LIMIT 20

Cypherクエリ:
""",
        )

        self.answer_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
質問: {question}

グラフデータベースから取得した情報:
{context}

上記の情報を基に、以下の点に注意して回答してください：
1. なぜその漫画がおすすめなのか、具体的な理由を説明
2. 作者の関係性（師弟関係、アシスタント経験など）があれば言及
3. ジャンルや出版社の共通点があれば指摘
4. データに基づいた客観的な説明を心がける

回答:
""",
        )

        # GraphCypherQAChainの初期化
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=self.cypher_prompt,
            return_intermediate_steps=True,
            verbose=True,
            allow_dangerous_requests=True,  # セキュリティ警告を承認（デモ用）
        )

    def _setup_manual_schema(self):
        """APOCプラグインなしで手動でスキーマを設定"""
        # サンプルスキーマを手動設定（英語リレーション名のみ使用）
        sample_schema = """
        Node properties:
        Manga {title: STRING, genre: STRING, year: INTEGER, status: STRING}
        Author {name: STRING, birth_year: INTEGER, nationality: STRING}
        Publisher {name: STRING, country: STRING}
        Genre {name: STRING}
        
        Relationship properties:
        CREATED {year: INTEGER}
        PUBLISHED_BY {year: INTEGER}
        SIMILAR_TO {similarity_score: FLOAT}
        INFLUENCED_BY {influence_type: STRING}
        HAS_GENRE {}
        
        The relationships are:
        (:Author)-[:CREATED]->(:Manga)
        (:Manga)-[:PUBLISHED_BY]->(:Publisher)
        (:Manga)-[:SIMILAR_TO]->(:Manga)
        (:Author)-[:INFLUENCED_BY]->(:Author)
        (:Manga)-[:HAS_GENRE]->(:Genre)
        
        Available relationship types (English only):
        - CREATED
        - PUBLISHED_BY
        - SIMILAR_TO
        - INFLUENCED_BY
        - HAS_GENRE
        """
        # スキーマを直接設定
        self.graph.schema = sample_schema

    def enhance_query_with_graph_context(self, question: str) -> str:
        """質問にグラフコンテキストを追加"""
        # 質問からエンティティを抽出
        entities = self._extract_entities(question)

        # エンティティの周辺情報を取得
        context = []
        for entity in entities:
            neighbors = self._get_entity_neighbors(entity)
            if neighbors:
                context.append(f"{entity}の関連情報: {neighbors}")

        enhanced_question = question
        if context:
            enhanced_question += f"\n\n参考情報:\n" + "\n".join(context)

        return enhanced_question

    def _extract_entities(self, text: str) -> List[str]:
        """テキストから漫画・作者名を抽出"""
        # 簡易実装：実際はNERやLLMを使用
        query = """
        MATCH (m:Manga)
        WHERE $text CONTAINS m.title
        RETURN m.title as entity
        UNION
        MATCH (a:Author)
        WHERE $text CONTAINS a.name
        RETURN a.name as entity
        LIMIT 5
        """
        result = self.graph.query(query, params={"text": text})
        return [r["entity"] for r in result]

    def _get_entity_neighbors(self, entity: str) -> Dict[str, Any]:
        """エンティティの1ホップ隣接情報を取得"""
        query = """
        MATCH (n)
        WHERE n.name = $entity OR n.title = $entity
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, type(r) as relation, collect(m) as neighbors
        LIMIT 1
        """
        result = self.graph.query(query, params={"entity": entity})
        if result:
            return {"relations": result[0].get("relation", []), "neighbor_count": len(result[0].get("neighbors", []))}
        return {}

    def recommend_manga(self, user_preference: str) -> Dict[str, Any]:
        """ユーザーの好みに基づいて漫画を推薦"""
        # クエリの拡張
        enhanced_query = self.enhance_query_with_graph_context(user_preference)

        # GraphRAGで回答生成
        result = self.qa_chain.invoke({"query": enhanced_query})

        # 中間ステップから推薦理由を抽出
        cypher_query = "クエリ情報なし"
        database_results = []

        # resultが辞書形式で中間ステップを含む場合
        if isinstance(result, dict):
            # intermediate_stepsの取得を試みる
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                # Cypherクエリの取得
                for step in intermediate_steps:
                    if isinstance(step, dict) and "query" in step:
                        cypher_query = step["query"]
                        break
                    elif isinstance(step, tuple) and len(step) > 0:
                        cypher_query = str(step[0])
                        break

                # データベース結果の取得
                for step in intermediate_steps:
                    if isinstance(step, dict) and "context" in step:
                        database_results = step["context"]
                        break
                    elif isinstance(step, tuple) and len(step) > 1:
                        database_results = step[1]
                        break

            # もしintermediate_stepsがない場合、他のキーを探す
            if not intermediate_steps:
                # 可能性のあるキー名
                for key in ["cypher", "generated_cypher", "query"]:
                    if key in result:
                        cypher_query = result[key]
                        break

                for key in ["context", "graph_data", "results"]:
                    if key in result:
                        database_results = result[key]
                        break

        # 推薦理由の生成
        reasoning = self._generate_recommendation_reasoning(user_preference, database_results)

        return {
            "recommendation": result.get("result", result) if isinstance(result, dict) else str(result),
            "reasoning": reasoning,
            "cypher_query": cypher_query,
            "graph_data": database_results,
        }

    def _generate_recommendation_reasoning(self, preference: str, graph_data: List[Dict]) -> str:
        """グラフデータから推薦理由を生成"""
        prompt = PromptTemplate(
            input_variables=["preference", "data"],
            template="""
ユーザーの好み: {preference}

データベースの検索結果から、以下の関連性が見つかりました:
{data}

これらの情報を基に、推薦の根拠を箇条書きで説明してください：
- 作者の関係性（同じ師匠、アシスタント経験など）
- ジャンルやテーマの共通点
- 出版社や連載誌の関連
- 時代背景や影響関係

推薦理由:
""",
        )

        reasoning_chain = prompt | self.llm
        return reasoning_chain.invoke({"preference": preference, "data": str(graph_data)}).content


# プロンプトエンジニアリングの例
class MangaPromptTemplates:
    """漫画推薦用の最適化されたプロンプト集"""

    @staticmethod
    def get_similarity_search_prompt() -> PromptTemplate:
        """類似作品検索用プロンプト"""
        return PromptTemplate(
            input_variables=["manga_title", "aspect"],
            template="""
{manga_title}と{aspect}が似ている漫画を探しています。
以下の観点で類似性を評価してください：
1. ストーリーの構成
2. キャラクターの設定
3. 世界観
4. 画風（同じ作者やアシスタント経験者）
""",
        )

    @staticmethod
    def get_author_lineage_prompt() -> PromptTemplate:
        """作者の系譜探索用プロンプト"""
        return PromptTemplate(
            input_variables=["author_name"],
            template="""
{author_name}の系譜を調べてください。
含めるべき情報：
- 師匠（アシスタント先）
- 弟子（元アシスタント）
- 影響を受けた/与えた作家
- 作風の変遷
""",
        )


# 使用例
def demo_graphrag():
    try:
        print("Neo4jデータベースに接続しています...")
        # GraphRAGシステムの初期化
        graphrag = MangaGraphRAG(neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password")
        print("✅ Neo4jデータベースへの接続に成功しました")

        # 推薦の実行
        print("\n漫画推薦システムをテストしています...")
        result = graphrag.recommend_manga("NARUTOが好きです。似たような作品を教えてください。")

        print("\n=== 推薦結果 ===")
        print("推薦作品:", result["recommendation"])
        print("\n推薦理由:", result["reasoning"])
        print("\n使用したCypherクエリ:", result["cypher_query"])
        print(
            "\nグラフデータ数:", len(result["graph_data"]) if isinstance(result["graph_data"], list) else "データなし"
        )

        # 作者の系譜を調査
        lineage_result = graphrag.qa_chain.invoke({"query": "岸本斉史の作品と、同じ系譜の作者の作品を教えてください"})
        print("\n系譜調査結果:", lineage_result)

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("\n対処方法:")
        print("1. Neo4jデータベースが起動していることを確認してください")
        print("   - Docker（APOCプラグイン付き）:")
        print("     docker run --publish=7474:7474 --publish=7687:7687")
        print("     --env NEO4J_PLUGINS='[\"apoc\"]'")
        print("     --volume=$HOME/neo4j/data:/data neo4j")
        print("   - Neo4j Desktop: APOCプラグインをインストールしてください")
        print("2. 接続設定（URI、ユーザー名、パスワード）を確認してください")
        print("3. OpenAI APIキーが設定されていることを確認してください")
        print("   - export OPENAI_API_KEY='your-api-key'")
        print("4. APOCプラグインに関するエラーの場合:")
        print("   - Neo4j設定でapoc.*の手続きを有効にしてください")
        print("   - または、修正版のコードではAPOCなしでも動作するはずです")

        # デモ用にサンプル出力を表示
        print("\n=== デモ出力（実際の処理なし）===")
        print("推薦作品: GraphRAGシステムは正常に動作します。Neo4jデータベース接続後に実際の推薦が実行されます。")
        print("推薦理由: 作者の関係性やジャンルの類似性を分析して推薦理由を生成します。")
        print(
            "使用したクエリ: MATCH (m:Manga)-[:SIMILAR_TO]-(similar:Manga) WHERE m.title CONTAINS 'NARUTO' RETURN similar LIMIT 20"
        )

        # デバッグ情報
        print("\n=== デバッグ情報 ===")
        print(f"エラータイプ: {type(e).__name__}")
        print(f"エラー詳細: {str(e)}")


if __name__ == "__main__":
    demo_graphrag()

#!/usr/bin/env python3
"""
Manga GraphRAG Implementation using Manga Graph API

This script demonstrates how to implement GraphRAG (Graph Retrieval-Augmented Generation)
using the Manga Graph API to:
- Search and retrieve manga information from the graph database
- Use graph relationships to enhance recommendations
- Generate contextual responses based on graph data
- Implement multi-hop reasoning across the knowledge graph
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from retry_utils import request_with_retry

# from langchain_openai import ChatOpenAI  # 未使用のためコメントアウト

load_dotenv()  # Load environment variables from .env file


class MangaGraphClient:
    """Client for interacting with the Manga Graph API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        backend_api_key = os.getenv("BACKEND_API_KEY", "").strip()
        if backend_api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {backend_api_key}",
                    "X-API-Key": backend_api_key,
                }
            )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = request_with_retry(
                method,
                url,
                session=self.session,
                params=params,
                json=json_data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return {}

    # Search endpoints
    def search_manga(self, query: str, depth: int = 2) -> Dict[str, Any]:
        """Search for manga and related data"""
        data = {"query": query, "depth": depth}
        return self._make_request("POST", "/api/v1/search", json_data=data)

    # Basic data endpoints
    def get_authors(self) -> List[Dict[str, Any]]:
        """Get all authors"""
        return self._make_request("GET", "/api/v1/authors")

    def get_works(self) -> List[Dict[str, Any]]:
        """Get all works"""
        return self._make_request("GET", "/api/v1/works")

    def get_magazines(self) -> List[Dict[str, Any]]:
        """Get all magazines"""
        return self._make_request("GET", "/api/v1/magazines")

    # Media Arts Database endpoints
    def search_media_arts(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search media arts database"""
        params = {"q": query, "limit": limit}
        return self._make_request("GET", "/api/v1/media-arts/search", params=params)

    def get_creator_works(self, creator_name: str, limit: int = 50) -> Dict[str, Any]:
        """Get works by a specific creator"""
        params = {"limit": limit}
        return self._make_request("GET", f"/api/v1/media-arts/creator/{creator_name}", params=params)

    def get_manga_magazines(self, limit: int = 100) -> Dict[str, Any]:
        """Get manga magazines from media arts database"""
        params = {"limit": limit}
        return self._make_request("GET", "/api/v1/media-arts/magazines", params=params)

    def fulltext_search(self, query: str, search_type: str = "simple_query_string", limit: int = 50) -> Dict[str, Any]:
        """Perform full-text search"""
        params = {"q": query, "search_type": search_type, "limit": limit}
        return self._make_request("GET", "/api/v1/media-arts/fulltext-search", params=params)

    def search_with_related(self, query: str, limit: int = 50, include_related: bool = True) -> Dict[str, Any]:
        """Search with related works"""
        params = {"q": query, "limit": limit, "include_related": include_related}
        return self._make_request("GET", "/api/v1/media-arts/search-with-related", params=params)

    def get_magazine_relationships(
        self, magazine_name: Optional[str] = None, year: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        """Get magazine relationships"""
        params = {"limit": limit}
        if magazine_name:
            params["magazine_name"] = magazine_name
        if year:
            params["year"] = year
        return self._make_request("GET", "/api/v1/media-arts/magazine-relationships", params=params)

    # Neo4j endpoints
    def search_neo4j(self, query: str, limit: int = 50, include_related: bool = True) -> Dict[str, Any]:
        """Search using Neo4j"""
        # Added sort_total_volumes & min_total_volumes per requirement
        params = {
            "q": query,
            "limit": limit,
            "include_related": include_related,
            "sort_total_volumes": "desc",
            "min_total_volumes": 5,
        }
        return self._make_request("GET", "/api/v1/neo4j/search", params=params)

    def get_creator_works_neo4j(self, creator_name: str, limit: int = 50) -> Dict[str, Any]:
        """Get creator works using Neo4j"""
        params = {"limit": limit}
        return self._make_request("GET", f"/api/v1/neo4j/creator/{creator_name}", params=params)

    def get_neo4j_stats(self) -> Dict[str, Any]:
        """Get Neo4j database statistics"""
        return self._make_request("GET", "/api/v1/neo4j/stats")

    # Cover image endpoints
    def get_work_cover(self, work_id: str) -> Dict[str, Any]:
        """Get cover image URL for a work"""
        return self._make_request("GET", f"/api/v1/covers/work/{work_id}")

    def update_work_cover(self, work_id: str) -> Dict[str, Any]:
        """Update cover image URL for a work"""
        return self._make_request("POST", f"/api/v1/covers/work/{work_id}/update")

    def bulk_update_covers(self, limit: int = 100) -> Dict[str, Any]:
        """Bulk update cover images"""
        params = {"limit": limit}
        return self._make_request("POST", "/api/v1/covers/bulk-update", params=params)

    # Utility functions
    def format_graph_response(self, response: Dict[str, Any]) -> str:
        """Format graph response for display"""
        if not response:
            return "No data returned"

        output = []
        # Handle the response from /api/v1/neo4j/search endpoint
        if "nodes" in response:
            output.append(f"Nodes: {response.get('node_count', len(response['nodes']))}")
            for node in response.get("nodes", [])[:5]:  # Show first 5 nodes
                output.append(f"  - {node.get('labels', ['Unknown'])[0]}: {node.get('name', node.get('title', 'N/A'))}")
                if node.get("properties"):
                    for key, value in list(node["properties"].items())[:3]:
                        if key not in ["name", "title"]:  # Avoid duplicating name/title
                            output.append(f"    {key}: {value}")

        if "edges" in response:
            output.append(f"\nRelationships: {response.get('relationship_count', len(response['edges']))}")
            for rel in response.get("edges", [])[:5]:  # Show first 5 relationships
                output.append(
                    f"  - {rel.get('type', 'UNKNOWN')}: {rel.get('source', 'N/A')} -> {rel.get('target', 'N/A')}"
                )

        return "\n".join(output)


class MangaGraphRAG:
    """GraphRAG implementation for manga recommendations and analysis"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """Initialize the GraphRAG system"""
        self.client = MangaGraphClient(api_base_url)
        # Initialize prompt templates
        self._init_prompts()

    def _init_prompts(self):
        """Initialize prompt templates for various GraphRAG tasks"""
        self.recommendation_prompt = PromptTemplate(
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

        self.analysis_prompt = PromptTemplate(
            input_variables=["manga_title", "graph_data"],
            template="""
以下のグラフデータを基に、「{manga_title}」について詳細な分析を行ってください。

グラフデータ:
{graph_data}

分析に含めるべき要素:
1. 作品の基本情報
2. 作者とその他の作品
3. 同じ雑誌に掲載された関連作品
4. ジャンルや時代の文脈
5. 作品の影響や系譜

分析結果:""",
        )

        self.multi_hop_prompt = PromptTemplate(
            input_variables=["start_entity", "end_entity", "path_data"],
            template="""
「{start_entity}」から「{end_entity}」への関係性について、
以下のグラフパスデータを基に説明してください。

パスデータ:
{path_data}

説明には以下を含めてください：
1. 直接的な関係
2. 間接的な関係（共通の要素を通じた関係）
3. 歴史的・文化的な文脈
4. その関係性が持つ意味

関係性の説明:""",
        )

        # エンティティ抽出（自然言語→候補）
        self.entity_extraction_prompt = PromptTemplate(
            input_variables=["user_text"],
            template="""
あなたは漫画に関するエンティティ抽出器です。ユーザーの自由記述から、以下のJSON形式だけを出力してください。説明文は不要です。

必ずこのスキーマ:
{{"titles": [], "authors": [], "magazines": [], "keywords": []}}

- titles: 作品名の候補（不確実でも候補を入れる）
- authors: 作者名の候補
- magazines: 掲載誌の候補
- keywords: ジャンル/特徴/キーワード

ユーザー入力:
{user_text}
""",
        )

    # --- helper methods for entity linking ---
    def _node_label(self, node: Dict[str, Any]) -> str:
        return (
            node.get("label")
            or node.get("name")
            or node.get("title")
            or (node.get("properties", {}) or {}).get("name")
            or (node.get("properties", {}) or {}).get("title")
            or "N/A"
        )

    def _node_type(self, node: Dict[str, Any]) -> str:
        t = node.get("type")
        if isinstance(t, str) and t:
            return t
        labels = node.get("labels")
        if isinstance(labels, list) and labels:
            return labels[0]
        return "Unknown"

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Use LLM to extract entity candidates from natural language text"""
        try:
            chain = self.entity_extraction_prompt
            out = chain.invoke({"user_text": text})
            content = (out.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", content)
            json_str = m.group(0) if m else content
            data = json.loads(json_str)
        except Exception:
            data = {"titles": [], "authors": [], "magazines": [], "keywords": []}
        # Normalize and uniquify
        result: Dict[str, List[str]] = {k: [] for k in ["titles", "authors", "magazines", "keywords"]}
        for k in result.keys():
            vals = data.get(k, []) or []
            if isinstance(vals, str):
                vals = [vals]
            # keep order while uniquifying
            seen = set()
            cleaned: List[str] = []
            for v in vals:
                if not v:
                    continue
                s = str(v).strip()
                if s and s not in seen:
                    seen.add(s)
                    cleaned.append(s)
            result[k] = cleaned
        return result

    def link_entities(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Link extracted entities to graph nodes via Neo4j search and fuzzy scoring"""
        candidates: List[Dict[str, Any]] = []
        search_targets: List[Tuple[str, str]] = []
        for mention in entities.get("titles", []):
            search_targets.append(("Title", mention))
        for mention in entities.get("authors", []):
            search_targets.append(("Author", mention))
        for mention in entities.get("magazines", []):
            search_targets.append(("Magazine", mention))

        for expected_type, mention in search_targets:
            search = self.client.search_neo4j(mention, limit=10, include_related=True)
            for node in search.get("nodes", []) or []:
                label = str(self._node_label(node))

                candidates.append(
                    {
                        "mention": mention,
                        "expected_type": expected_type,
                        "entity_type": self._node_type(node),
                        "label": label,
                        "node": node,
                        # "score": score,
                    }
                )

        # sort and dedupe by (label, entity_type)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top: List[Dict[str, Any]] = []
        seen = set()
        for c in candidates:
            key = (c["label"], c["entity_type"])
            if key in seen:
                continue
            seen.add(key)
            top.append(c)
            if len(top) >= 10:
                break

        return {"candidates": top}

    def recommend_from_text(self, user_text: str) -> Dict[str, Any]:
        """End-to-end: natural language -> entity candidates -> graph -> recommendation"""
        # Step 0: Extract and link entities
        entities = self.extract_entities_from_text(user_text)
        linked = self.link_entities(entities)
        # Step 1: Aggregate graph around top candidates
        agg_nodes: List[Dict[str, Any]] = []
        agg_edges: List[Dict[str, Any]] = []
        for c in linked.get("candidates", [])[:5]:
            res = self.client.search_neo4j(c["label"], limit=50, include_related=True)
            agg_nodes.extend(res.get("nodes", []) or [])
            agg_edges.extend(res.get("edges", []) or [])

        # Fallback if nothing resolved
        if not agg_nodes:
            res = self.client.search_neo4j(user_text, limit=30, include_related=True)
            agg_nodes = res.get("nodes", []) or []
            agg_edges = res.get("edges", []) or []

        graph_data = {
            "nodes": agg_nodes,
            "edges": agg_edges,
            "node_count": len(agg_nodes),
            "relationship_count": len(agg_edges),
        }

        # ノードIDのマッピングを事前に作成（高速化のため）
        node_id_to_label = {}
        for node in agg_nodes:
            node_id = node.get("id")
            if node_id:
                label = self._node_label(node)
                node_id_to_label[node_id] = label

        # Step 2: Analyze nodes/edges by type
        nodes_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for n in agg_nodes:
            nt = self._node_type(n)
            nodes_by_type.setdefault(nt, []).append(n)
        relationships_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in agg_edges:
            rt = e.get("type", "Unknown")
            relationships_by_type.setdefault(rt, []).append(e)

        # Step 3: Build context and formatted data
        context = self._build_recommendation_context(nodes_by_type, relationships_by_type)
        context += "\n\nエンティティリンク結果:\n"
        if linked.get("candidates"):
            for c in linked["candidates"][:5]:
                context += f"- {c['mention']} -> {c['label']} ({c['entity_type']}, 信頼度: {c['score']:.2f})\n"
        else:
            context += "- 候補が見つかりませんでした（全文検索にフォールバック）\n"

        formatted_data = "取得したグラフデータ:\n\n"
        formatted_data += f"ノード数: {len(agg_nodes)}\n"
        formatted_data += f"関係数: {len(agg_edges)}\n\n"

        formatted_data += "ノードタイプ別情報:\n"
        for node_type, nodes in nodes_by_type.items():
            formatted_data += f"- {node_type}: {len(nodes)}件\n"
            for node in nodes:
                node_name = self._node_label(node)
                formatted_data += f"  • {node_name}\n"

        formatted_data += "\n関係タイプ別情報:\n"
        for rel_type, rels in relationships_by_type.items():
            formatted_data += f"- {rel_type}: {len(rels)}件\n"
            for rel in rels:
                # 高速化：事前に作成した辞書を使用（O(1)）
                source = node_id_to_label.get(rel.get("source"), "N/A")
                target = node_id_to_label.get(rel.get("target"), "N/A")
                formatted_data += f"  • {source} → {target}\n"

        # Step 4: Generate recommendation using LLM
        recommendation = self.recommendation_prompt
        result = recommendation.invoke({"user_query": user_text, "graph_data": formatted_data, "context": context})

        return {
            "recommendation": result.content,
            "graph_data": graph_data,
            "nodes_analysis": nodes_by_type,
            "relationships_analysis": relationships_by_type,
            "context": context,
            "entities": entities,
            "linked_candidates": linked.get("candidates", []),
        }

    def enhance_query_with_graph_context(self, query: str) -> Tuple[Dict[str, Any], str]:
        """Enhance user query with graph context"""
        # First, search for relevant entities
        search_result = self.client.search_neo4j(query, limit=10)

        # Extract entities and relationships
        entities = []
        relationships = []

        if "nodes" in search_result:
            for node in search_result["nodes"]:
                entities.append(
                    {
                        "type": node.get("labels", ["Unknown"])[0],
                        "label": node.get("name", node.get("title", "N/A")),
                        "properties": node.get("properties", {}),
                    }
                )

        if "edges" in search_result:
            for rel in search_result["edges"]:
                relationships.append(
                    {
                        "type": rel.get("type", "UNKNOWN"),
                        "source": rel.get("source", "N/A"),
                        "target": rel.get("target", "N/A"),
                    }
                )

        # Build context
        context = f"Found {len(entities)} entities and {len(relationships)} relationships."

        return search_result, context

    def recommend_manga(self, user_preference: str) -> Dict[str, Any]:
        """Recommend manga based on user preferences using GraphRAG"""
        print(f"Processing recommendation request: {user_preference}")
        # Step 1: Get graph data using neo4j search
        graph_data = self.client.search_neo4j(user_preference, limit=30, include_related=True)

        # ノードIDのマッピングを事前に作成（高速化のため）
        node_id_to_label = {}
        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                node_id = node.get("id")
                if node_id:
                    label = node.get("label", "N/A")
                    node_id_to_label[node_id] = label

        # Step 2: Extract and analyze nodes and relationships
        nodes_by_type = {}
        relationships_by_type = {}

        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                # Get node type from labels array
                node_type = node.get("type", ["Unknown"])
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node)

        if "edges" in graph_data:
            for rel in graph_data["edges"]:
                rel_type = rel.get("type", "Unknown")
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                # 高速化：辞書ルックアップを使用（O(1)）
                source = node_id_to_label.get(rel["source"], "N/A")
                target = node_id_to_label.get(rel["target"], "N/A")
                relationships_by_type[rel_type].append({"source": source, "target": target})

        # Step 3: Build recommendation context from graph structure
        context = self._build_recommendation_context(nodes_by_type, relationships_by_type)

        # Step 4: Format detailed graph data for LLM
        formatted_data = "取得したグラフデータ:\n\n"
        formatted_data += f"ノード数: {graph_data.get('node_count', len(graph_data.get('nodes', [])))}\n"
        formatted_data += f"関係数: {graph_data.get('relationship_count', len(graph_data.get('edges', [])))}\n\n"

        formatted_data += "ノードタイプ別情報:\n"
        for node_type, nodes in nodes_by_type.items():
            formatted_data += f"- {node_type}: {len(nodes)}件\n"
            # for node in nodes[:5]:  # 最初の5件を表示
            for node in nodes:
                node_name = node.get("label", "N/A")
                formatted_data += f"  • {node_name}"
                # if node.get("properties"):
                #     props = []
                #     for k, v in list(node["properties"].items())[:3]:
                #         if v and k not in ["name", "title"]:
                #             props.append(f"{k}: {v}")
                #     if props:
                #         formatted_data += f" ({', '.join(props)})"
                formatted_data += "\n"

        formatted_data += "\n関係タイプ別情報:\n"
        for rel_type, rels in relationships_by_type.items():
            formatted_data += f"- {rel_type}: {len(rels)}件\n"
            # for rel in rels[:3]:  # 最初の3件を表示
            for rel in rels:
                formatted_data += f"  • {rel.get('source', 'N/A')} → {rel.get('target', 'N/A')}\n"

        # Step 5: Generate recommendation using LLM with graph data
        recommendation = self.recommendation_prompt
        result = recommendation.invoke(
            {"user_query": user_preference, "graph_data": formatted_data, "context": context}
        )

        return {
            "recommendation": result.content,
            "graph_data": graph_data,
            "nodes_analysis": nodes_by_type,
            "relationships_analysis": relationships_by_type,
            "context": context,
        }

    def _build_recommendation_context(self, nodes_by_type: Dict, relationships_by_type: Dict) -> str:
        """Build context from graph structure for recommendations"""
        context = "グラフ構造から抽出した情報:\n"

        # 作品情報
        if "Work" in nodes_by_type or "Manga" in nodes_by_type:
            works = nodes_by_type.get("Work", []) + nodes_by_type.get("Manga", [])
            context += f"- 関連作品数: {len(works)}\n"

        # 作者情報
        if "Author" in nodes_by_type or "Creator" in nodes_by_type:
            authors = nodes_by_type.get("Author", []) + nodes_by_type.get("Creator", [])
            context += f"- 関連作者数: {len(authors)}\n"

        # 出版社・雑誌情報
        if "Publisher" in nodes_by_type or "Magazine" in nodes_by_type:
            publishers = nodes_by_type.get("Publisher", []) + nodes_by_type.get("Magazine", [])
            context += f"- 関連出版社/雑誌数: {len(publishers)}\n"

        # 関係性の分析
        if relationships_by_type:
            context += "\n主要な関係性:\n"
            for rel_type, rels in relationships_by_type.items():
                context += f"- {rel_type}: {len(rels)}件の関係\n"

        return context

    def analyze_manga(self, manga_title: str) -> Dict[str, Any]:
        """Analyze a specific manga using graph data"""
        print(f"Analyzing manga: {manga_title}")

        # Get comprehensive data about the manga using neo4j search
        search_data = self.client.search_neo4j(manga_title, limit=50, include_related=True)
        # For additional related data, we can search with a higher limit
        related_data = self.client.search_neo4j(manga_title, limit=30, include_related=True)

        # Format data
        formatted_data = self.client.format_graph_response(search_data)
        if related_data:
            formatted_data += "\n\n関連情報:\n" + self.client.format_graph_response(related_data)

        # Generate analysis
        analysis = self.analysis_prompt
        result = analysis.invoke({"manga_title": manga_title, "graph_data": formatted_data})

        return {"analysis": result.content, "graph_data": search_data, "related_data": related_data}

    def find_multi_hop_relationships(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """Find multi-hop relationships between two entities"""
        print(f"Finding relationships between '{entity1}' and '{entity2}'")

        # Search for both entities
        data1 = self.client.search_neo4j(entity1, limit=15)
        data2 = self.client.search_neo4j(entity2, limit=15)

        # Find common nodes or paths
        common_nodes = []
        if "nodes" in data1 and "nodes" in data2:
            # Use node name/title for comparison since id might not be available
            nodes1_names = {node.get("name", node.get("title", "")) for node in data1["nodes"]}
            nodes2_names = {node.get("name", node.get("title", "")) for node in data2["nodes"]}
            common_names = nodes1_names.intersection(nodes2_names)

            for node in data1["nodes"]:
                node_name = node.get("name", node.get("title", ""))
                if node_name in common_names:
                    common_nodes.append(node)

        # Format path data
        path_data = f"Entity 1 ({entity1}) connections:\n"
        path_data += self.client.format_graph_response(data1)
        path_data += f"\n\nEntity 2 ({entity2}) connections:\n"
        path_data += self.client.format_graph_response(data2)

        if common_nodes:
            path_data += f"\n\nCommon connections: {len(common_nodes)} nodes"
            for node in common_nodes[:5]:
                path_data += f"\n  - {node.get('labels', ['Unknown'])[0]}: {node.get('name', node.get('title', 'N/A'))}"

        # Generate explanation
        explanation = self.multi_hop_prompt
        result = explanation.invoke({"start_entity": entity1, "end_entity": entity2, "path_data": path_data})

        return {
            "explanation": result.content,
            "entity1_data": data1,
            "entity2_data": data2,
            "common_nodes": common_nodes,
        }

    def explore_author_lineage(self, author_name: str) -> Dict[str, Any]:
        """Explore the lineage and influence network of an author"""
        print(f"Exploring lineage of: {author_name}")

        # Get author's works using neo4j search with author name
        author_works = self.client.search_neo4j(author_name, limit=50, include_related=True)

        # Search for connections
        connections = self.client.search_neo4j(author_name, limit=30)

        # Build lineage context
        lineage_data = "作者情報:\n"
        lineage_data += self.client.format_graph_response(author_works)
        lineage_data += "\n\n関連情報:\n"
        lineage_data += self.client.format_graph_response(connections)

        # Create lineage analysis prompt
        lineage_prompt = PromptTemplate(
            input_variables=["author_name", "lineage_data"],
            template="""
「{author_name}」の系譜と影響関係について、以下のデータを基に分析してください。

{lineage_data}

分析に含めるべき要素:
1. 主要作品とその特徴
2. 同時代の作家との関係
3. 影響を与えた/受けた作家
4. 作風の変遷
5. 漫画史における位置づけ

系譜分析:""",
        )

        # Generate lineage analysis
        analysis = lineage_prompt
        result = analysis.invoke({"author_name": author_name, "lineage_data": lineage_data})

        return {"lineage_analysis": result.content, "author_works": author_works, "connections": connections}


def demo_graphrag():
    """Demonstrate GraphRAG capabilities"""
    try:
        print("=== Manga GraphRAG Demo ===")
        print("Initializing GraphRAG system...")

        graphrag = MangaGraphRAG()
        print("✅ GraphRAG system initialized\n")

        # Demo examples with ONE PIECE, NARUTO, and るろうに剣心
        demos = [
            {
                "title": "レベルE",
                "func": lambda: graphrag.recommend_manga("レベルE"),
                "query": "レベルE",
            },
            # {
            #     "title": "2. NARUTOベースの推薦",
            #     "func": lambda: graphrag.recommend_manga(
            #         "NARUTOが好きです。忍者や友情をテーマにした作品を探しています。"
            #     ),
            #     "query": "NARUTOが好きです。忍者や友情をテーマにした作品を探しています。",
            # },
            # {
            #     "title": "3. るろうに剣心ベースの推薦",
            #     "func": lambda: graphrag.recommend_manga(
            #         "るろうに剣心が好きです。歴史物や剣術をテーマにした作品を教えてください。"
            #     ),
            #     "query": "るろうに剣心が好きです。歴史物や剣術をテーマにした作品を教えてください。",
            # },
        ]

        for i, demo in enumerate(demos):
            print(f"\n{demo['title']}")
            print(f"User query: '{demo['query']}'")
            result = demo["func"]()
            print("\n推薦結果:")
            print(result["recommendation"])

            # グラフ情報の要約を表示
            if "nodes_analysis" in result:
                print("\n取得したグラフ情報の要約:")
                for node_type, nodes in result["nodes_analysis"].items():
                    print(f"  - {node_type}: {len(nodes)}件")

            print("\n" + "=" * 70 + "\n")

        # Demo 4: ONE PIECEの詳細分析
        print("4. ONE PIECEの詳細分析")
        print("Analyzing: 'ONE PIECE'")
        analysis = graphrag.analyze_manga("ONE PIECE")
        print("\n分析結果:")
        print(analysis["analysis"])
        print("\n" + "=" * 70 + "\n")

        # Demo 5: 作者間の関係性分析
        print("5. 作者間の関係性分析")
        print("Finding relationships between '尾田栄一郎' and 'ONE PIECE'")
        relationships = graphrag.find_multi_hop_relationships("尾田栄一郎", "ONE PIECE")
        print("\n関係性の説明:")
        print(relationships["explanation"])

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\n対処方法:")
        print("1. Manga Graph APIが起動していることを確認してください")
        print("   - APIはlocalhost:8000で動作している必要があります")
        print("2. OpenAI APIキーが設定されていることを確認してください")
        print("   - .envファイルにOPENAI_API_KEY=your-api-keyを設定")
        print("3. 必要なパッケージがインストールされていることを確認してください")
        print("   - pip install requests langchain langchain-openai python-dotenv")


def interactive_graphrag_demo():
    """Interactive GraphRAG demo"""
    graphrag = MangaGraphRAG()

    print("=== Manga GraphRAG Interactive Demo ===")
    print("Available commands:")
    print("  recommend <自然言語>   - 自由記述からエンティティリンクしてレコメンド")
    print("  analyze <title>      - Analyze a specific manga")
    print("  relate <e1> | <e2>   - Find relationships between entities")
    print("  lineage <author>     - Explore author lineage")
    print("  help                 - Show this help")
    print("  quit                 - Exit demo")
    print()

    while True:
        try:
            command = input("\nEnter command: ").strip()

            if command == "quit":
                break
            elif command == "help":
                print("Commands: recommend, analyze, relate, lineage, help, quit")
            elif command.startswith("recommend "):
                query = command[10:]
                print(f"\nProcessing recommendation for (NL): {query}")
                result = graphrag.recommend_from_text(query)
                print("\n推薦結果:")
                print(result["recommendation"])
            elif command.startswith("analyze "):
                title = command[8:]
                print(f"\nAnalyzing: {title}")
                result = graphrag.analyze_manga(title)
                print("\n分析結果:")
                print(result["analysis"])
            elif command.startswith("relate "):
                parts = command[7:].split(" | ")
                if len(parts) == 2:
                    print(f"\nFinding relationships between '{parts[0]}' and '{parts[1]}'")
                    result = graphrag.find_multi_hop_relationships(parts[0], parts[1])
                    print("\n関係性:")
                    print(result["explanation"])
                else:
                    print("Format: relate <entity1> | <entity2>")
            elif command.startswith("lineage "):
                author = command[8:]
                print(f"\nExploring lineage of: {author}")
                result = graphrag.explore_author_lineage(author)
                print("\n系譜分析:")
                print(result["lineage_analysis"])
            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nThank you for using Manga GraphRAG!")


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_graphrag_demo()
    # else:
    #     demo_graphrag()

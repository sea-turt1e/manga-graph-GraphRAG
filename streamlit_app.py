import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

from prompts.manga_prompts import GraphRAGPrompts, StandardMangaPrompts
from retry_utils import request_with_retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GraphRAGを使用した生成デモ", page_icon="📚", layout="wide", initial_sidebar_state="collapsed"
)
load_dotenv()

# Optional API key for backend
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "").strip()
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# API Endpoints
# 統合API (1-6, 7-8, 11-13)
GRAPH_CASCADE_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/graph/cascade"
VECTOR_SIMILARITY_MULTI_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/vector/similarity/multi"
RELATED_GRAPHS_BATCH_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/related-graphs/batch"
MAGAZINES_WORK_GRAPH_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/magazines/work-graph"
TEXT_GEN_ENDPOINT = f"{API_BASE}/text-generation/generate"


def _auth_headers(extra: dict | None = None) -> dict:
    headers: dict = {}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
    headers["X-API-Key"] = BACKEND_API_KEY
    if extra:
        headers.update(extra)
    return headers


# =============================================================================
# Backend API Functions for GraphRAG (統合API使用)
# =============================================================================


def search_graph_cascade(query: str, limit: int = 3, languages: str = "japanese,english") -> Dict[str, Any]:
    """
    グラフ検索統合API呼び出し (1-6 統合)
    japanese/simple -> japanese/fulltext -> japanese/ranked -> english/simple -> english/fulltext -> english/ranked
    を1回のAPI呼び出しで実行
    """
    params = {
        "q": query,
        "limit": limit,
        "languages": languages,
        "include_hentai": False,
    }
    try:
        r = request_with_retry(
            "GET",
            GRAPH_CASCADE_ENDPOINT,
            params=params,
            headers=_auth_headers(),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Graph cascade search error: %s", e)
        return {}


def get_related_graphs_batch(
    author_node_id: str | None = None,
    magazine_node_id: str | None = None,
    publisher_node_id: str | None = None,
    author_limit: int = 5,
    magazine_limit: int = 5,
    publisher_limit: int = 3,
    reference_work_id: str | None = None,
    exclude_magazine_id: str | None = None,
) -> Dict[str, Any]:
    """
    関連グラフ一括取得API呼び出し (11-13 統合)
    著者の他作品、雑誌の他作品、出版社の他雑誌を1回のAPI呼び出しで取得
    """
    body = {
        "author_node_id": author_node_id,
        "magazine_node_id": magazine_node_id,
        "publisher_node_id": publisher_node_id,
        "author_limit": author_limit,
        "magazine_limit": magazine_limit,
        "publisher_limit": publisher_limit,
        "include_hentai": False,
    }
    if reference_work_id:
        body["reference_work_id"] = reference_work_id
    if exclude_magazine_id:
        body["exclude_magazine_id"] = exclude_magazine_id
    
    try:
        r = request_with_retry(
            "POST",
            RELATED_GRAPHS_BATCH_ENDPOINT,
            json=body,
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Related graphs batch error: %s", e)
        return {}


def search_vector_similarity_multi(
    query: str,
    embedding_types: List[str] | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    ベクトル類似検索統合API呼び出し (7-8 統合)
    title_en と title_ja での検索を1回のAPI呼び出しで実行
    結果は既にマージ・重複排除・ソート済み
    """
    if embedding_types is None:
        embedding_types = ["title_en", "title_ja"]
    
    body = {
        "query": query,
        "embedding_types": embedding_types,
        "embedding_dims": 256,
        "limit": limit,
        "threshold": threshold,
        "include_hentai": False,
    }
    try:
        r = request_with_retry(
            "POST",
            VECTOR_SIMILARITY_MULTI_ENDPOINT,
            json=body,
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()
        logger.info(f"Vector similarity multi search: {len(result.get('results', []))} results")
        return result
    except Exception as e:
        logger.warning("Vector similarity multi search error: %s", e)
        return {}


def get_magazines_work_graph(magazine_ids: List[str], work_limit: int = 3, reference_work_id: str | None = None) -> Dict[str, Any]:
    """複数雑誌の作品グラフを取得"""
    # 空の場合は早期リターン
    if not magazine_ids:
        logger.warning("get_magazines_work_graph: magazine_ids is empty")
        return {}
    
    body = {
        "magazine_element_ids": magazine_ids,  # APIスキーマに合わせたフィールド名
        "work_limit": work_limit,
        "include_hentai": False,
    }
    if reference_work_id:
        body["reference_work_id"] = reference_work_id
    
    logger.info(f"Magazines work graph request body: {body}")
    
    try:
        r = request_with_retry(
            "POST",
            MAGAZINES_WORK_GRAPH_ENDPOINT,
            json=body,
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        # エラーレスポンスの詳細をログ出力
        try:
            error_detail = e.response.json()
            logger.warning("Magazines work graph error: %s - Detail: %s", e, error_detail)
        except Exception:
            logger.warning("Magazines work graph error: %s - Response: %s", e, e.response.text)
        return {}
    except Exception as e:
        logger.warning("Magazines work graph error: %s", e)
        return {}


def extract_ids_from_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """グラフレスポンスからノードIDを抽出"""
    result = {
        "work_id": None,
        "work_title": None,
        "author_ids": [],
        "magazine_ids": [],
        "publisher_ids": [],
    }
    
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or graph.get("relationships", []) or []
    
    # ノードをタイプ別に分類
    for node in nodes:
        node_type = node.get("type", "").lower()
        node_id = node.get("id") or node.get("elementId")
        
        if node_type == "work":
            if result["work_id"] is None:
                result["work_id"] = node_id
                # japanese_nameを優先的に使用
                props = node.get("properties", {})
                result["work_title"] = (
                    props.get("japanese_name") 
                    or props.get("title") 
                    or node.get("title") 
                    or node.get("label")
                )
        elif node_type == "author":
            if node_id and node_id not in result["author_ids"]:
                result["author_ids"].append(node_id)
        elif node_type == "magazine":
            if node_id and node_id not in result["magazine_ids"]:
                result["magazine_ids"].append(node_id)
        elif node_type == "publisher":
            if node_id and node_id not in result["publisher_ids"]:
                result["publisher_ids"].append(node_id)
    
    return result


def get_work_title(node: Dict[str, Any]) -> str:
    """Workノードから漫画名を取得（japanese_nameを優先）"""
    props = node.get("properties", {})
    return (
        props.get("japanese_name")
        or props.get("title")
        or node.get("title")
        or node.get("label")
        or ""
    )


def perform_graph_search(query: str) -> tuple[Dict[str, Any], str]:
    """
    グラフ検索を実行（統合API使用）
    1-6の検索を1回のAPI呼び出しで実行
    
    Returns: (graph_response, search_mode_used)
    """
    result = search_graph_cascade(query, limit=3, languages="japanese,english")
    nodes = result.get("nodes", []) or []
    if nodes:
        logger.info(f"Graph cascade search found {len(nodes)} nodes")
        return result, "cascade"
    
    return {}, ""


def fetch_extended_graph_info(base_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    11-13: 追加のグラフ情報を取得（統合API使用）
    11. 著者の他作品
    12. 雑誌の他作品
    13. 出版社の他雑誌
    14. 他雑誌の作品グラフ
    """
    ids = extract_ids_from_graph(base_graph)
    
    extended_info = {
        "base_graph": base_graph,
        "author_works": [],
        "magazine_works": [],
        "publisher_magazines": [],
        "other_magazines_works": [],
        "extracted_ids": ids,
    }
    
    work_id = ids.get("work_id")
    author_ids = ids.get("author_ids", [])
    magazine_ids = ids.get("magazine_ids", [])
    publisher_ids = ids.get("publisher_ids", [])
    
    # 11-13: 関連グラフを一括取得（統合API）
    author_node_id = author_ids[0] if author_ids else None
    magazine_node_id = magazine_ids[0] if magazine_ids else None
    publisher_node_id = publisher_ids[0] if publisher_ids else None
    exclude_mag = magazine_ids[0] if magazine_ids else None
    
    related_graphs = get_related_graphs_batch(
        author_node_id=author_node_id,
        magazine_node_id=magazine_node_id,
        publisher_node_id=publisher_node_id,
        author_limit=5,
        magazine_limit=5,
        publisher_limit=3,
        reference_work_id=work_id,
        exclude_magazine_id=exclude_mag,
    )
    
    # 統合APIの結果を従来の形式に変換
    if related_graphs.get("author_graph"):
        extended_info["author_works"].append(related_graphs["author_graph"])
    
    if related_graphs.get("magazine_graph"):
        extended_info["magazine_works"].append(related_graphs["magazine_graph"])
    
    if related_graphs.get("publisher_graph"):
        extended_info["publisher_magazines"].append(related_graphs["publisher_graph"])
        # 他雑誌IDを収集
        other_magazine_ids = []
        for node in related_graphs["publisher_graph"].get("nodes", []):
            if node.get("type", "").lower() == "magazine":
                mag_id = node.get("id") or node.get("elementId")
                if mag_id and mag_id not in other_magazine_ids:
                    other_magazine_ids.append(mag_id)
        
        # 14. 他雑誌の作品グラフ
        if other_magazine_ids:
            other_works = get_magazines_work_graph(other_magazine_ids, work_limit=3, reference_work_id=work_id)
            if other_works:
                extended_info["other_magazines_works"].append(other_works)
    
    return extended_info


def build_graph_context_from_extended(extended_info: Dict[str, Any], query_title: str) -> str:
    """
    15: 拡張グラフ情報からコンテキスト文字列を構築
    """
    lines: List[str] = []
    
    base_graph = extended_info.get("base_graph", {})
    ids = extended_info.get("extracted_ids", {})
    
    # 基本情報
    work_title = ids.get("work_title") or query_title
    lines.append(f"クエリ: {work_title}")
    
    # 著者情報
    base_nodes = base_graph.get("nodes", []) or []
    authors = [n for n in base_nodes if n.get("type", "").lower() == "author"]
    magazines = [n for n in base_nodes if n.get("type", "").lower() == "magazine"]
    publishers = [n for n in base_nodes if n.get("type", "").lower() == "publisher"]
    
    if authors:
        author_name = authors[0].get("properties", {}).get("name") or authors[0].get("name") or authors[0].get("label") or "不明"
        lines.append(f"クエリの作品の作者: {author_name}")
    
    if magazines:
        mag_name = magazines[0].get("properties", {}).get("name") or magazines[0].get("name") or magazines[0].get("label") or "不明"
        lines.append(f"クエリが掲載された雑誌: {mag_name}")
    
    if publishers:
        pub_name = publishers[0].get("properties", {}).get("name") or publishers[0].get("name") or publishers[0].get("label") or "不明"
        lines.append(f"クエリが掲載された雑誌の出版社: {pub_name}")
    
    # 著者の別作品
    author_works_list = extended_info.get("author_works", [])
    if author_works_list:
        lines.append("")
        author_name = authors[0].get("properties", {}).get("name") or authors[0].get("name") if authors else "作者"
        lines.append(f"### {author_name}の別作品")
        work_titles_added = set()
        for aw in author_works_list:
            for node in aw.get("nodes", []):
                if node.get("type", "").lower() == "work":
                    title = get_work_title(node)
                    if title and title.lower() != work_title.lower() and title not in work_titles_added:
                        lines.append(f"- {title}")
                        work_titles_added.add(title)
        if not work_titles_added:
            lines.append("- なし")
    
    # 同雑誌の別作品
    magazine_works_list = extended_info.get("magazine_works", [])
    if magazine_works_list:
        lines.append("")
        lines.append("### 同雑誌の別作品")
        work_titles_added = set()
        for mw in magazine_works_list:
            nodes_dict = {n.get("id") or n.get("elementId"): n for n in mw.get("nodes", [])}
            for node in mw.get("nodes", []):
                if node.get("type", "").lower() == "work":
                    title = get_work_title(node)
                    if title and title.lower() != work_title.lower() and title not in work_titles_added:
                        # 作者を探す
                        work_author = "不明"
                        for edge in mw.get("edges", []) or mw.get("relationships", []) or []:
                            if edge.get("type") == "created" and edge.get("target") == (node.get("id") or node.get("elementId")):
                                author_node = nodes_dict.get(edge.get("source"))
                                if author_node:
                                    work_author = author_node.get("properties", {}).get("name") or author_node.get("name") or "不明"
                        mag_name = magazines[0].get("properties", {}).get("name") if magazines else "不明"
                        pub_name = publishers[0].get("properties", {}).get("name") if publishers else "不明"
                        lines.append(f"- {title}（作者: {work_author}、雑誌: {mag_name}、出版社: {pub_name}）")
                        work_titles_added.add(title)
        if not work_titles_added:
            lines.append("- なし")
    
    # 同出版社の他誌の作品
    other_mag_works = extended_info.get("other_magazines_works", [])
    if other_mag_works:
        lines.append("")
        lines.append("### 同出版社の他誌に掲載された作品")
        work_titles_added = set()
        for omw in other_mag_works:
            nodes_dict = {n.get("id") or n.get("elementId"): n for n in omw.get("nodes", [])}
            for node in omw.get("nodes", []):
                if node.get("type", "").lower() == "work":
                    title = get_work_title(node)
                    if title and title.lower() != work_title.lower() and title not in work_titles_added:
                        # 作者と雑誌を探す
                        work_author = "不明"
                        work_mag = "不明"
                        for edge in omw.get("edges", []) or omw.get("relationships", []) or []:
                            node_id = node.get("id") or node.get("elementId")
                            if edge.get("type") == "created" and edge.get("target") == node_id:
                                author_node = nodes_dict.get(edge.get("source"))
                                if author_node:
                                    work_author = author_node.get("properties", {}).get("name") or author_node.get("name") or "不明"
                            if edge.get("type") == "published" and edge.get("target") == node_id:
                                mag_node = nodes_dict.get(edge.get("source"))
                                if mag_node:
                                    work_mag = mag_node.get("properties", {}).get("name") or mag_node.get("name") or "不明"
                        pub_name = publishers[0].get("properties", {}).get("name") if publishers else "不明"
                        lines.append(f"- {title}（作者: {work_author}、雑誌: {work_mag}、出版社: {pub_name}）")
                        work_titles_added.add(title)
                        if len(work_titles_added) >= 5:
                            break
            if len(work_titles_added) >= 5:
                break
        if not work_titles_added:
            lines.append("- なし")
    
    return "\n".join(lines)


def generate_graphrag_recommendation(
    user_input: str,
    context: str,
    token_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """GraphRAGレコメンド文を生成"""
    rec_prompt = GraphRAGPrompts.get_recommendation_prompt()
    prompt_text = rec_prompt.format(user_query=user_input, context=context)
    
    body = {
        "text": prompt_text,
        "model": "gpt-4.1-nano",
        "temperature": 0.7,
        "max_tokens": 1000,
        "streaming": True,
    }
    
    full_text = ""
    try:
        r = request_with_retry(
            "POST",
            TEXT_GEN_ENDPOINT,
            json=body,
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=180,
            stream=True,
        )
        with r:
            r.raise_for_status()
            buffer = ""
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                buffer += chunk
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    if message.startswith("data: "):
                        line = message[6:].strip()
                        if not line:
                            continue
                        appended = ""
                        try:
                            if line.startswith("{") and line.endswith("}"):
                                data = json.loads(line)
                                if isinstance(data, dict) and "text" in data:
                                    appended = str(data["text"])
                            else:
                                appended = line
                        except Exception:
                            appended = line
                        if appended:
                            full_text += appended
                            if token_callback:
                                token_callback(appended)
    except Exception as e:
        logger.error("GraphRAG generation failed: %s", e)
        return full_text + f"\n[GraphRAG生成エラー] {e}"
    
    return full_text or "(生成結果なし)"


def run_graphrag_pipeline_new(
    user_input: str,
    token_callback: Optional[Callable[[str], None]] = None,
    selected_title: str | None = None,
) -> Dict[str, Any]:
    """
    新しいGraphRAGパイプライン（統合API使用）
    1-6: グラフ検索（cascade統合API）
    7-8: 類似検索（similarity/multi統合API）- グラフ検索で見つからない場合
    11-14: 拡張グラフ情報取得（batch統合API + work-graph）
    15-16: コンテキスト生成とレコメンド生成
    
    類似検索で候補が見つかった場合は候補を返し、ユーザーに選択させる
    """
    query = selected_title or user_input
    
    # 1-6: グラフ検索（統合API）
    base_graph, search_mode = perform_graph_search(query)
    
    fuzzy_used = False
    similarity_candidates = []
    
    # グラフ検索で見つからなかった場合は類似検索（ただし、ユーザーが既に選択済みの場合はスキップ）
    if not base_graph.get("nodes") and selected_title is None:
        fuzzy_used = True
        # 7-8: ベクトル類似検索（統合API）
        similarity_result = search_vector_similarity_multi(query, limit=10, threshold=0.3)
        results = similarity_result.get("results", []) or []
        
        # 候補リストを構築
        for r in results:
            title = r.get("title_ja") or r.get("title_en") or ""
            score = r.get("similarity_score") or 0
            if title and title not in [c["title"] for c in similarity_candidates]:
                similarity_candidates.append({
                    "title": title,
                    "score": score,
                    "work_id": r.get("work_id"),
                })
        
        # 候補がある場合、選択を待つ（候補を返す）
        if similarity_candidates:
            return {
                "extracted_title": query,
                "fuzzy_used": True,
                "fuzzy_best_title": similarity_candidates[0]["title"],
                "user_selected_candidate": False,
                "search_mode": "",
                "graph_summary": "",
                "graph_debug": "",
                "recommendation": "",
                "raw_graph": {},
                "similarity_candidates": similarity_candidates,
                "not_found": False,
                "awaiting_selection": True,  # ユーザー選択待ちフラグ
            }
    
    # グラフ検索で見つからなかった場合（類似検索後も含む）
    if not base_graph.get("nodes"):
        return {
            "extracted_title": query,
            "fuzzy_used": fuzzy_used,
            "fuzzy_best_title": similarity_candidates[0]["title"] if similarity_candidates else None,
            "user_selected_candidate": selected_title is not None,
            "search_mode": "",
            "graph_summary": "",
            "graph_debug": "",
            "recommendation": "",
            "raw_graph": {},
            "similarity_candidates": similarity_candidates,
            "not_found": True,  # 検索結果なしフラグ
            "awaiting_selection": False,
        }
    
    # 11-14: 拡張グラフ情報取得（統合API）
    extended_info = fetch_extended_graph_info(base_graph)
    
    # 15: コンテキスト構築
    context = build_graph_context_from_extended(extended_info, query)
    
    # 16: レコメンド生成
    recommendation = generate_graphrag_recommendation(user_input, context, token_callback)
    
    return {
        "extracted_title": query,
        "fuzzy_used": fuzzy_used,
        "fuzzy_best_title": similarity_candidates[0]["title"] if similarity_candidates else None,
        "user_selected_candidate": selected_title is not None,
        "search_mode": search_mode,
        "graph_summary": context,
        "graph_debug": json.dumps(extended_info, ensure_ascii=False, indent=2)[:2000],
        "recommendation": recommendation,
        "raw_graph": base_graph,
        "similarity_candidates": similarity_candidates,
        "not_found": False,
        "awaiting_selection": False,
    }


def stream_generate(text, container, title):
    """APIからストリーミングレスポンスを取得して表示"""
    try:
        api_base = os.getenv("API_BASE", "http://localhost:8000")
        url = f"{api_base}/text-generation/generate"
        headers = _auth_headers({"Content-Type": "application/json"})
        data = {"text": text, "streaming": "true"}

        # ストリーミングレスポンスを処理
        # 502等が出ることがあるため、接続確立までリトライ
        # on_retryでUIに起動待ちメッセージを表示
        def on_retry(ctx: dict):
            wait = ctx.get("wait")
            status = ctx.get("status")
            if status in (502, 503, 504) or status is None:
                container.info(
                    f"バックエンド起動待ち中... リトライ{ctx.get('attempt')}回目。"
                    + (f" 次の試行まで約{wait:.1f}秒" if wait else "")
                )

        response = request_with_retry(
            "POST",
            url,
            json=data,
            headers=headers,
            stream=True,
            timeout=180,
            on_retry=on_retry,
        )
        response.raise_for_status()  # エラーチェック

        full_text = ""
        buffer = ""
        with container.container():
            st.subheader(title)
            text_placeholder = st.empty()

            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                buffer += chunk
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    if message.startswith("data: "):
                        line = message[len("data: ") :].strip()
                        if not line:
                            continue

                        # lineがJSON形式（"{...}"）であるかチェック
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                json_data = json.loads(line)
                                if isinstance(json_data, dict):
                                    if "text" in json_data:
                                        full_text += str(json_data["text"])
                                    elif "content" in json_data:
                                        full_text += str(json_data["content"])
                                    else:
                                        # 他のキーも考慮
                                        full_text += " ".join(
                                            [str(v) for v in json_data.values() if isinstance(v, (str, int, float))]
                                        )
                                else:
                                    full_text += str(json_data)
                            except json.JSONDecodeError:
                                # JSONデコードに失敗した場合は、文字列としてそのまま追加
                                full_text += line
                        else:
                            # JSON形式でない場合は、そのままテキストとして追加
                            full_text += line

                        # リアルタイムで表示を更新
                        text_placeholder.markdown(full_text)
                        # セッションに保持して再描画時も表示できるようにする
                        st.session_state["raw_llm_output"] = full_text
                        time.sleep(0.01)  # 少し遅延を入れて表示を見やすくする
        # 完了フラグ
        st.session_state["raw_llm_done"] = True
    except requests.exceptions.HTTPError as e:
        with container.container():
            st.subheader(title)
            st.error(f"API呼び出しに失敗しました。ステータスコード: {e.response.status_code}")
            st.text(f"レスポンス: {e.response.text}")
        st.session_state["raw_llm_output"] = f"APIエラー: {e.response.status_code}\n{e.response.text}"
        st.session_state["raw_llm_done"] = True

    except requests.exceptions.ConnectionError:
        with container.container():
            st.subheader(title)
            st.error("APIサーバーに接続できません。API_Serverが起動していることを確認してください。")
        st.session_state["raw_llm_output"] = "APIサーバーに接続できませんでした。"
        st.session_state["raw_llm_done"] = True
    except Exception as e:
        with container.container():
            st.subheader(title)
            st.error(f"エラーが発生しました: {str(e)}")
        st.session_state["raw_llm_output"] = f"エラー: {str(e)}"
        st.session_state["raw_llm_done"] = True


def main():
    st.title("📚 GraphRAGを使用した生成デモ")
    st.markdown("同じテキストに対して素のLLM（GraphRAGなし）とGraphRAGを使用した生成の結果を比較表示します。")
    # 右下に小さな「出典」リンク（フローティング）
    st.markdown(
        """
        <style>
        .floating-citation-link {
            position: fixed;
            right: 16px;
            bottom: 12px;
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(6px);
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            padding: 4px 8px;
            font-size: 12px;
            z-index: 9999;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .floating-citation-link a {
            color: #4f46e5;
            text-decoration: none;
        }
        .floating-citation-link a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="floating-citation-link">
            🔗 <a href="/source_link" target="_self">出典</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 入力欄 + 巻数フィルタ（PCでは横並び 4:1 / モバイルでは自動縦積み）
    st.subheader("🔤 漫画入力とフィルタ")
    col_title, col_vol = st.columns([4, 1], gap="small")
    with col_title:
        input_text = st.text_area(
            "おすすめ文を生成したい漫画名を入力してください。:",
            height=100,
            placeholder="例: NARUTO",
        )
    with col_vol:
        min_vol = st.number_input(
            "n巻以上発行 (≤10)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="指定した巻数以上の単行本が発行されている作品に限定します",
        )

    # 比較用に素のLLMを実行するかの切り替え
    show_raw_llm = st.checkbox(
        "素のLLM（GraphRAGなし）も実行して比較する",
        value=True,
        help="オフにすると素のLLMをスキップしてGraphRAGのみ実行します",
    )

    # 右カラムにGraphRAGの結果を書き込むヘルパー
    def run_graphrag_into(
        right_container,
        status_text,
        progress_bar,
        user_text: str,
        min_volumes: int,
        selected_title: str | None = None,
    ):
        status_text.text("🔄 GraphRAGパイプラインを実行中...")
        progress_bar.progress(60)
        with right_container:
            st.subheader("🕸️ GraphRAGを使用した生成")
            with st.spinner("Graph / 推薦生成中..."):
                try:
                    reco_placeholder = st.empty()
                    buffer = []

                    def on_token(t: str):
                        buffer.append(t)
                        if "\n" in t or len(buffer) % 5 == 0 or t.endswith(("。", "!", "?")):
                            reco_placeholder.markdown("".join(buffer))

                    result = run_graphrag_pipeline_new(
                        user_text,
                        token_callback=on_token,
                        selected_title=selected_title,
                    )
                    
                    # 類似検索で候補が見つかり、ユーザー選択待ちの場合
                    if result.get("awaiting_selection"):
                        reco_placeholder.empty()
                        st.session_state["fuzzy_candidates"] = result.get("similarity_candidates", [])
                        st.session_state["awaiting_candidate_selection"] = True
                        st.session_state["pending_user_input"] = user_text
                        st.session_state["pending_min_vol"] = min_volumes
                        st.rerun()
                        return
                    
                    # 検索結果が見つからなかった場合
                    if result.get("not_found"):
                        reco_placeholder.warning("検索結果が見つかりませんでした。別のキーワードをお試しください。")
                    else:
                        reco_placeholder.markdown(result["recommendation"])
                        with st.expander("抽出・検索メタ情報"):
                            meta_info = {
                                "extracted_title": result.get("extracted_title"),
                                "search_mode": result.get("search_mode"),
                                "fuzzy_used": result.get("fuzzy_used"),
                                "fuzzy_best_title": result.get("fuzzy_best_title"),
                                "user_selected_candidate": result.get("user_selected_candidate"),
                                "node_count": len(result.get("raw_graph", {}).get("nodes", []) or []),
                                "relationship_count": len(result.get("raw_graph", {}).get("edges", []) or []),
                            }
                            st.write(meta_info)
                            st.caption("グラフコンテキスト")
                            st.text(result.get("graph_summary"))
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:  # noqa: BLE001
                    st.error(f"GraphRAG実行中にエラー: {e}")
        progress_bar.progress(90)
        progress_bar.progress(100)
        status_text.text("✅ 生成が完了しました！")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        st.success("✅ 生成が完了しました！")

    # 候補選択パネルを表示するヘルパー
    def render_candidate_selector_panel(right_container):
        cands = st.session_state.get("fuzzy_candidates", [])
        base_query = st.session_state.get("pending_user_input", "")
        with right_container:
            st.subheader("🔎 類似する候補が見つかりました")
            st.write("正しい作品を選んでください。選択後に生成を開始します。")
            st.caption(f"検索語: {base_query}")
            st.caption(f"候補件数: {len(cands)} 件")

            if not cands:
                st.info("候補が見つかりませんでした。検索条件を変えてお試しください。")
                return

            # 候補をラジオボタンで表示
            options = []
            for c in cands:
                score_percent = c.get("score", 0) * 100
                options.append(f"{c['title']} (類似度: {score_percent:.1f}%)")
            
            idx = st.radio(
                "候補",
                options=range(len(options)),
                format_func=lambda i: options[i],
                index=0,
                key="cand_idx",
            )
            
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("この作品で生成する", type="primary"):
                    chosen = cands[idx]
                    st.session_state["chosen_title"] = chosen["title"]
                    st.session_state["awaiting_candidate_selection"] = False
                    st.session_state["start_generation"] = True
                    st.rerun()
            with cols[1]:
                if st.button("キャンセル"):
                    # セッション状態をクリア
                    for k in ["fuzzy_candidates", "awaiting_candidate_selection", "pending_user_input", "pending_min_vol"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

    # 選択待ちなら、生LLM結果を左に保持表示しつつ、候補選択パネルを出す（GraphRAGは未実行）
    if st.session_state.get("awaiting_candidate_selection"):
        st.markdown("---")
        st.subheader("📊 生成結果の比較")
        col1, col2 = st.columns(2)
        with col1.container():
            st.subheader("💬 素のLLM（GraphRAGなし）")
            raw_out = st.session_state.get("raw_llm_output")
            if raw_out:
                st.markdown(raw_out)
            else:
                st.info("素のLLMの結果はここに表示されます。")
        with col2.container():
            st.subheader("🕸️ GraphRAGを使用した生成")
            st.info("候補を選択するとGraphRAGの生成を開始します。")
        st.markdown("---")
        render_candidate_selector_panel(col2.container())
        st.stop()

    # 選択後に自動実行
    if st.session_state.get("start_generation"):
        st.markdown("---")
        st.subheader("📊 生成結果の比較")
        col1, col2 = st.columns(2)
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 左に保存済みの素のLLM結果を表示（再リクエストはしない）
        with col1.container():
            st.subheader("💬 素のLLM（GraphRAGなし）")
            raw_out = st.session_state.get("raw_llm_output")
            if raw_out:
                st.markdown(raw_out)
            else:
                st.info("素のLLMの結果はここに表示されます。")

        run_graphrag_into(
            col2.container(),
            status_text,
            progress_bar,
            st.session_state.get("pending_user_input", input_text),
            st.session_state.get("pending_min_vol", int(min_vol)),
            selected_title=st.session_state.get("chosen_title"),
        )
        # 後片付け
        for k in [
            "fuzzy_candidates",
            "awaiting_candidate_selection",
            "pending_user_input",
            "pending_min_vol",
            "chosen_title",
            "start_generation",
        ]:
            if k in st.session_state:
                del st.session_state[k]

    # 実行ボタン押下時の処理（素のLLM→GraphRAG）
    if st.button("🚀 生成開始", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ テキストを入力してください。")
        else:
            try:
                # レイアウト（比較表示）と生成
                st.markdown("---")
                st.subheader("📊 生成結果の比較")
                col1, col2 = st.columns(2)
                progress_bar = st.progress(0)
                status_text = st.empty()

                if show_raw_llm:
                    with col1.container():
                        prompt = get_standard_recommend_prompt(input_text)
                        stream_generate(prompt, col1, "💬 素のLLM（GraphRAGなし）")

                # GraphRAGパイプラインを実行
                run_graphrag_into(
                    col2.container(),
                    status_text,
                    progress_bar,
                    input_text,
                    int(min_vol),
                    selected_title=None,
                )
            except Exception as e:
                st.error(f"前処理中にエラーが発生しました: {e}")

    # APIサーバーの状態チェック
    st.markdown("---")
    st.subheader("🔧 サーバー状態")

    if st.button("サーバー接続確認"):
        check_server_connection(os.getenv("API_BASE", "http://localhost:8000"))


def check_server_connection(api_base: str):
    try:
        response = request_with_retry("GET", f"{api_base}/health", headers=_auth_headers(), timeout=5)
        if response.status_code == 200:
            st.success("✅ APIサーバーに正常に接続できます")
        else:
            st.warning(f"⚠️ サーバーからの応答が異常です (ステータス: {response.status_code})")
    except requests.exceptions.ConnectionError:
        st.error("❌ APIサーバーに接続できません。API_Serverが起動していることを確認してください。")
    except Exception as e:
        st.error(f"❌ 接続確認中にエラーが発生しました: {str(e)}")


def get_standard_recommend_prompt(user_query: str) -> str:
    prompt_template = StandardMangaPrompts.get_recommendation_prompt()
    return prompt_template.format(user_query=user_query)


if __name__ == "__main__":
    main()

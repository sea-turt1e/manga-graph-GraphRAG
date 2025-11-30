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
GRAPH_SEARCH_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/graph"
VECTOR_SIMILARITY_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/vector/similarity"
AUTHOR_WORKS_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/author"
MAGAZINE_WORKS_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/magazine"
PUBLISHER_MAGAZINES_ENDPOINT = f"{API_BASE}/api/v1/manga-anime-neo4j/publisher"
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
# Backend API Functions for GraphRAG
# =============================================================================


def search_graph(query: str, lang: str = "japanese", mode: str = "simple", limit: int = 3) -> Dict[str, Any]:
    """グラフ検索API呼び出し"""
    params = {
        "q": query,
        "lang": lang,
        "mode": mode,
        "limit": limit,
        "include_hentai": False,
    }
    try:
        r = request_with_retry(
            "GET",
            GRAPH_SEARCH_ENDPOINT,
            params=params,
            headers=_auth_headers(),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Graph search error: %s", e)
        return {}


def search_vector_similarity(query: str, embedding_type: str = "title_en") -> Dict[str, Any]:
    """ベクトル類似検索API呼び出し"""
    body = {
        "query": query,
        "embedding_type": embedding_type,
        "embedding_dims": 256,
        "limit": 10,
        "threshold": 0.3,
        "include_hentai": False,
    }
    try:
        r = request_with_retry(
            "POST",
            VECTOR_SIMILARITY_ENDPOINT,
            json=body,
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Vector similarity search error: %s", e)
        return {}


def get_author_works(author_node_id: str, limit: int = 5) -> Dict[str, Any]:
    """著者の他作品を取得"""
    url = f"{AUTHOR_WORKS_ENDPOINT}/{author_node_id}/works"
    params = {"limit": limit, "include_hentai": False}
    try:
        r = request_with_retry("GET", url, params=params, headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Author works error: %s", e)
        return {}


def get_magazine_works(magazine_node_id: str, limit: int = 5, reference_work_id: str | None = None) -> Dict[str, Any]:
    """雑誌の他作品を取得"""
    url = f"{MAGAZINE_WORKS_ENDPOINT}/{magazine_node_id}/works"
    params = {"limit": limit, "include_hentai": False}
    if reference_work_id:
        params["reference_work_id"] = reference_work_id
    try:
        r = request_with_retry("GET", url, params=params, headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Magazine works error: %s", e)
        return {}


def get_publisher_magazines(publisher_node_id: str, limit: int = 3, exclude_magazine_id: str | None = None) -> Dict[str, Any]:
    """出版社の他雑誌を取得"""
    url = f"{PUBLISHER_MAGAZINES_ENDPOINT}/{publisher_node_id}/magazines"
    params = {"limit": limit, "include_hentai": False}
    if exclude_magazine_id:
        params["exclude_magazine_id"] = exclude_magazine_id
    try:
        r = request_with_retry("GET", url, params=params, headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Publisher magazines error: %s", e)
        return {}


def get_magazines_work_graph(magazine_ids: List[str], work_limit: int = 3, reference_work_id: str | None = None) -> Dict[str, Any]:
    """複数雑誌の作品グラフを取得"""
    body = {
        "magazine_ids": magazine_ids,
        "work_limit": work_limit,
        "include_hentai": False,
    }
    if reference_work_id:
        body["reference_work_id"] = reference_work_id
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
                result["work_title"] = node.get("properties", {}).get("title") or node.get("title") or node.get("label")
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


def perform_graph_search(query: str) -> tuple[Dict[str, Any], str]:
    """
    グラフ検索を段階的に実行
    1. japanese/simple -> 2. japanese/fulltext -> 3. japanese/ranked
    4. english/simple -> 5. english/fulltext -> 6. english/ranked
    
    Returns: (graph_response, search_mode_used)
    """
    search_modes = [
        ("japanese", "simple"),
        ("japanese", "fulltext"),
        ("japanese", "ranked"),
        ("english", "simple"),
        ("english", "fulltext"),
        ("english", "ranked"),
    ]
    
    for lang, mode in search_modes:
        result = search_graph(query, lang=lang, mode=mode, limit=3)
        nodes = result.get("nodes", []) or []
        if nodes:
            logger.info(f"Graph search found results with lang={lang}, mode={mode}")
            return result, f"{lang}/{mode}"
    
    return {}, ""


def perform_vector_similarity_search(query: str) -> List[Dict[str, Any]]:
    """
    ベクトル類似検索を実行
    7. title_en -> 8. title_ja
    
    Returns: 候補リスト
    """
    candidates = []
    
    # 7. title_en で検索
    result_en = search_vector_similarity(query, embedding_type="title_en")
    results_en = result_en.get("results", []) or result_en.get("nodes", []) or []
    for r in results_en:
        title = r.get("title") or r.get("properties", {}).get("title") or ""
        score = r.get("similarity_score") or r.get("score") or 0
        if title and title not in [c["title"] for c in candidates]:
            candidates.append({"title": title, "score": score, "source": "title_en"})
    
    # 8. title_ja で検索
    result_ja = search_vector_similarity(query, embedding_type="title_ja")
    results_ja = result_ja.get("results", []) or result_ja.get("nodes", []) or []
    for r in results_ja:
        title = r.get("title") or r.get("properties", {}).get("title") or ""
        score = r.get("similarity_score") or r.get("score") or 0
        if title and title not in [c["title"] for c in candidates]:
            candidates.append({"title": title, "score": score, "source": "title_ja"})
    
    # スコアでソート
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    return candidates


def fetch_extended_graph_info(base_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    11-14: 追加のグラフ情報を取得
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
    
    # 11. 著者の他作品
    for author_id in ids.get("author_ids", [])[:2]:  # 最大2著者
        author_works = get_author_works(author_id, limit=5)
        if author_works:
            extended_info["author_works"].append(author_works)
    
    # 12. 雑誌の他作品
    for magazine_id in ids.get("magazine_ids", [])[:2]:  # 最大2雑誌
        magazine_works = get_magazine_works(magazine_id, limit=5, reference_work_id=work_id)
        if magazine_works:
            extended_info["magazine_works"].append(magazine_works)
    
    # 13. 出版社の他雑誌
    other_magazine_ids = []
    for publisher_id in ids.get("publisher_ids", [])[:1]:  # 最大1出版社
        exclude_mag = ids.get("magazine_ids", [None])[0] if ids.get("magazine_ids") else None
        publisher_mags = get_publisher_magazines(publisher_id, limit=3, exclude_magazine_id=exclude_mag)
        if publisher_mags:
            extended_info["publisher_magazines"].append(publisher_mags)
            # 他雑誌IDを収集
            for node in publisher_mags.get("nodes", []):
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
                    title = node.get("properties", {}).get("title") or node.get("title") or node.get("label")
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
                    title = node.get("properties", {}).get("title") or node.get("title") or node.get("label")
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
                    title = node.get("properties", {}).get("title") or node.get("title") or node.get("label")
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
    新しいGraphRAGパイプライン
    1-6: グラフ検索
    7-10: 類似検索（必要な場合）
    11-14: 拡張グラフ情報取得
    15-16: コンテキスト生成とレコメンド生成
    """
    query = selected_title or user_input
    
    # 1-6: グラフ検索
    base_graph, search_mode = perform_graph_search(query)
    
    fuzzy_used = False
    similarity_candidates = []
    
    # グラフ検索で見つからなかった場合は類似検索
    if not base_graph.get("nodes"):
        fuzzy_used = True
        # 7-8: ベクトル類似検索
        similarity_candidates = perform_vector_similarity_search(query)
        
        # 候補がある場合、最上位の候補で再検索
        if similarity_candidates:
            best_candidate = similarity_candidates[0]["title"]
            base_graph, search_mode = perform_graph_search(best_candidate)
            query = best_candidate
    
    # 11-14: 拡張グラフ情報取得
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
    }


def get_similarity_candidates_for_selection(query: str) -> List[Dict[str, Any]]:
    """UI用: 類似検索の候補を取得"""
    # まずグラフ検索を試す
    base_graph, _ = perform_graph_search(query)
    if base_graph.get("nodes"):
        return []  # グラフで見つかった場合は候補選択不要
    
    # 類似検索
    candidates = perform_vector_similarity_search(query)
    processed = []
    for c in candidates[:10]:
        score_percent = c.get("score", 0) * 100
        processed.append({
            "title": c["title"],
            "score": c.get("score", 0),
            "display": f"{c['title']} (類似度: {score_percent:.1f}%)",
        })
    return processed


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
                    reco_placeholder.markdown(result["recommendation"])
                    with st.expander("抽出・検索メタ情報"):
                        st.write(
                            {
                                "extracted_title": result.get("extracted_title"),
                                "search_mode": result.get("search_mode"),
                                "fuzzy_used": result.get("fuzzy_used"),
                                "fuzzy_best_title": result.get("fuzzy_best_title"),
                                "user_selected_candidate": result.get("user_selected_candidate"),
                                "node_count": len(result.get("raw_graph", {}).get("nodes", []) or []),
                                "relationship_count": len(result.get("raw_graph", {}).get("edges", []) or []),
                            }
                        )
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

    # インライン（ページ内）候補選択パネル
    def render_candidate_selector_panel(right_container):  # uses session_state
        cands = st.session_state.get("fuzzy_candidates", [])
        base_query = st.session_state.get("dialog_extracted_title") or st.session_state.get("pending_user_input")
        with right_container:
            st.subheader("🔎 候補が複数見つかりました")
            st.write("正しい作品を選んでください。選択後に生成を開始します。")
            st.caption(f"検索語: {base_query}")
            st.caption(f"候補件数: {len(cands)} 件")

            if not cands:
                st.info("候補が見つかりませんでした。検索条件を変えてお試しください。")
                return

            options = [c["display"] for c in cands]
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
                if st.button("上位候補で生成"):
                    # 上位候補または抽出タイトルで続行
                    fallback = cands[0]["title"] if cands else (st.session_state.get("dialog_extracted_title") or "")
                    st.session_state["chosen_title"] = fallback
                    st.session_state["awaiting_candidate_selection"] = False
                    st.session_state["start_generation"] = True
                    st.rerun()

    # 旧フラグ（モーダル用）が残っていれば新フラグに移行
    if st.session_state.get("open_candidate_dialog"):
        st.session_state["awaiting_candidate_selection"] = True
        del st.session_state["open_candidate_dialog"]

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

    # 実行ボタン押下時の処理（まず素のLLM→その後にグラフ検索/類似検索→必要なら候補選択→GraphRAG）
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

                # 曖昧性解消（候補選択）を完了させる。解決後に生成を開始する。
                # スピナーと結果UIは右カラムに表示
                with col2.container():
                    with st.spinner("グラフから漫画名を検索中..."):
                        # 1-6) グラフ検索を段階的に実行
                        graph_result, search_mode = perform_graph_search(input_text)

                        selected_title_for_run: str | None = None
                        processed = []
                        
                        if graph_result.get("nodes"):
                            # グラフ検索で見つかった
                            selected_title_for_run = None  # 入力テキストでそのまま実行
                        else:
                            # 7-8) ベクトル類似検索
                            st.markdown(
                                "🔍 **:red[一致する漫画作品が見つからなかったため、近そうな漫画作品名をリストアップします。]**"
                            )
                            candidates = perform_vector_similarity_search(input_text)
                            
                            for c in candidates[:10]:
                                score_percent = c.get("score", 0) * 100
                                processed.append({
                                    "title": c["title"],
                                    "score": c.get("score", 0),
                                    "display": f"{c['title']} (類似度: {score_percent:.1f}%)",
                                })

                    # 曖昧性の結果に応じて分岐
                    if len(processed) > 1:
                        # 9) 2件以上 → ページ内パネルで選択、選択後に生成開始
                        st.session_state["fuzzy_candidates"] = processed
                        st.session_state["dialog_extracted_title"] = input_text
                        st.session_state["awaiting_candidate_selection"] = True
                        st.session_state["pending_user_input"] = input_text
                        st.session_state["pending_min_vol"] = int(min_vol)
                        st.session_state["pending_show_raw_llm"] = bool(show_raw_llm)
                        # 現在のランで右カラムにパネル表示へ移行
                        st.markdown("---")
                        render_candidate_selector_panel(col2.container())
                        st.stop()
                    else:
                        # 候補0/1件 → そのまま生成開始
                        if processed:
                            selected_title_for_run = processed[0]["title"]
                        # 10) 選択された候補でグラフ検索して以降の処理を実行

                    run_graphrag_into(
                        col2.container(),
                        status_text,
                        progress_bar,
                        input_text,
                        int(min_vol),
                        selected_title=selected_title_for_run,
                    )
            except Exception as e:
                st.error(f"前処理中にエラーが発生しました: {e}")

    # 選択後に自動実行（左に素のLLM結果を再掲）
    if st.session_state.get("start_generation"):
        # 選択後は比較表示を再構築して生成
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
        # 後片付け（モーダルを閉じたままに）
        for k in [
            "fuzzy_candidates",
            "dialog_extracted_title",
            "awaiting_candidate_selection",
            "pending_user_input",
            "pending_min_vol",
            "pending_show_raw_llm",
            "chosen_title",
            "start_generation",
        ]:
            if k in st.session_state:
                del st.session_state[k]

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

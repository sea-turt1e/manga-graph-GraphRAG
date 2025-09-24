#!/usr/bin/env python3
"""GraphRAG service module for Streamlit app.

Steps implemented:
1. Extract single formal manga title via /text-generation/generate (LLM prompt)
2. Try strict search /api/v1/neo4j/search
3. If no nodes, fallback to /api/v1/neo4j/vector/title-similarity
4. If fuzzy used, re-query strict search with best candidate's properties.title
5. Build context + call GraphRAG recommendation prompt
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from prompts.manga_prompts import GraphRAGPrompts

logger = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env file

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TEXT_GEN_ENDPOINT = f"{API_BASE}/text-generation/generate"
STRICT_SEARCH_ENDPOINT = f"{API_BASE}/api/v1/neo4j/search"
# Use vector-based title similarity API instead of legacy search-fuzzy
TITLE_SIMILARITY_ENDPOINT = f"{API_BASE}/api/v1/neo4j/vector/title-similarity"
DEFAULT_GEN_BODY = {"max_tokens": 1000, "temperature": 0.7, "model": "gpt-4.1-nano"}

# Optional API key for backend auth
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "").strip()


def _auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Compose request headers with optional Authorization.

    Assumes backend expects Authorization: Bearer <BACKEND_API_KEY>.
    """
    headers: Dict[str, str] = {}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
        headers["X-API-Key"] = BACKEND_API_KEY
    if extra:
        headers.update(extra)
    return headers


def _post_text_generation(prompt: str) -> str:
    body = {**DEFAULT_GEN_BODY, "text": prompt}
    try:
        r = requests.post(
            TEXT_GEN_ENDPOINT,
            json=body,
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("text-generation error: %s", e)
        return ""

    # non-streaming assumed JSON; sometimes SSE lines -> handle both
    try:
        data = r.json()
        if isinstance(data, dict):
            # API might return {'text': '...'}
            return data.get("generated_text", "")
    except ValueError:
        # fallback: raw text
        return r.text.strip()[:100]
    return ""


def extract_formal_title(user_input: str) -> str:
    """Use LLM endpoint to extract a single formal title.

    We build a dedicated minimal instruction so model outputs only the title.
    """
    prompt_template: PromptTemplate = GraphRAGPrompts.get_title_extraction_prompt()
    # Render prompt locally (no direct OpenAI call; we'll send to local endpoint)
    prompt_text = prompt_template.format(user_input=user_input)
    title = _post_text_generation(prompt_text)
    # sanitize: keep first line only; remove quotes
    # remove surrounding quotes (single or double)
    first_line = title.splitlines()[0].strip()
    if (first_line.startswith('"') and first_line.endswith('"')) or (
        first_line.startswith("'") and first_line.endswith("'")
    ):
        first_line = first_line[1:-1].strip()
    title = first_line
    if not title:
        return ""
    # heuristic: avoid model echo of instruction
    if len(title) > 40 and " " in title and "ユーザー" in title:
        return ""
    return title


def strict_search(
    title: str,
    limit: int = 30,
    include_related: bool = True,
    include_same_publisher_other_magazines: bool = True,
    same_publisher_other_magazines_limit: int = 5,
    min_total_volumes: int = 5,
) -> Dict[str, Any]:
    # Added sort_total_volumes & min_total_volumes per requirement
    params = {
        "q": title,
        "limit": limit,
        "include_related": str(include_related).lower(),
        "include_same_publisher_other_magazines": str(include_same_publisher_other_magazines).lower(),
        "same_publisher_other_magazines_limit": same_publisher_other_magazines_limit,
        "sort_total_volumes": "desc",
        "min_total_volumes": min_total_volumes,
    }
    try:
        r = requests.get(STRICT_SEARCH_ENDPOINT, params=params, headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("strict search error: %s", e)
        return {}


def fuzzy_search(
    query: str, limit: int = 5, similarity_threshold: float = 0.8, embedding_method: str = "huggingface"
) -> Dict[str, Any]:
    """Vector-based title similarity search.

    Kept function name for compatibility with callers, but internally this
    now calls /api/v1/neo4j/vector/title-similarity.
    """
    params = {
        "q": query,
        "limit": limit,
        "similarity_threshold": similarity_threshold,
        "embedding_method": embedding_method,
    }
    try:
        r = requests.get(TITLE_SIMILARITY_ENDPOINT, params=params, headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("title similarity search error: %s", e)
        return {}


def _node_label(node: Dict[str, Any]) -> str:
    props = node.get("properties", {}) or {}
    return (
        node.get("label") or node.get("name") or node.get("title") or props.get("title") or props.get("name") or "N/A"
    )


def build_graph_context(graph: Dict[str, Any]) -> str:
    """Build a structured context string from graph data for prompt consumption.

    Output format example:

    クエリ: Hunter×hunter
    クエリの作品の作者: 富樫義博
    クエリが掲載された雑誌: 週刊少年ジャンプ
    クエリが掲載された雑誌の出版社: 集英社

    ### {クエリの作品の作者}の別作品
    - てんで性悪キューピッド
    - 幽★遊★白書

    ### 同雑誌の別作品
    - ダンダダン = DAN DA DAN（作者: 龍幸伸、雑誌: 週刊少年ジャンプ、 出版社: 集英社）
    - 神のまにまに（作者: 猗笠怜司、雑誌: 週刊少年ジャンプ、 出版社: 集英社）

    ### 同出版社の他誌に掲載された作品
    - 末永くよろしくお願いします（作者: 池ジュン子、雑誌: 花とゆめ、 出版社: 集英社）
    """
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []

    # Cache: id -> display name; also keep raw node for later
    id_to_name: Dict[Any, str] = {n.get("id"): _node_label(n) for n in nodes if n.get("id") is not None}
    id_to_node: Dict[Any, Dict[str, Any]] = {n.get("id"): n for n in nodes if n.get("id") is not None}

    def norm(s: Any) -> str:
        """Normalize any value to a trimmed string.

        Ensures downstream .lower() calls are safe even if input is None or non-str.
        """
        return str(s or "").strip()

    query_title = norm(graph.get("_extracted_title") or "")

    # Identify the query work node id by name match if possible (exact match on display label)
    query_work_id = None
    if query_title:
        for nid, name in id_to_name.items():
            # guard in case name is not a string
            if str(name).lower() == query_title.lower():
                query_work_id = nid
                break
        # fallback: try properties.title exact match if display name differs
        if query_work_id is None:
            for nid, node in id_to_node.items():
                props = node.get("properties", {}) or {}
                # props.get("title") can be None; normalize before .lower()
                if norm(props.get("title")).lower() == query_title.lower():
                    query_work_id = nid
                    break

    # Index edges by type for quick lookup
    created_edges = [e for e in edges if e.get("type") == "created"]
    published_edges = [e for e in edges if e.get("type") == "published"]
    published_by_edges = [e for e in edges if e.get("type") == "published_by"]

    # Helper: collect authors for a work id
    def authors_of(work_id: Any) -> list[str]:
        authors = []
        for e in created_edges:
            if e.get("target") == work_id:
                a = id_to_name.get(e.get("source"))
                if a and a not in authors:
                    authors.append(a)
        return authors

    # Helper: collect magazines for a work id
    def magazines_of(work_id: Any) -> list[Any]:
        mags = []
        for e in published_edges:
            if e.get("target") == work_id:
                mid = e.get("source")
                if mid is not None and mid not in mags:
                    mags.append(mid)
        return mags

    # Helper: publisher of a magazine id
    def publisher_of(mag_id: Any) -> Optional[tuple[Any, str]]:
        for e in published_by_edges:
            if e.get("source") == mag_id:
                pid = e.get("target")
                return pid, id_to_name.get(pid, "")
        return None

    # Derive query's author, magazine, publisher
    query_author_name = ""
    query_mag_id = None
    query_mag_name = ""
    query_publisher_id = None
    query_publisher_name = ""

    if query_work_id is not None:
        a_list = authors_of(query_work_id)
        if a_list:
            query_author_name = a_list[0]
        mags = magazines_of(query_work_id)
        if mags:
            query_mag_id = mags[0]
            query_mag_name = id_to_name.get(query_mag_id, "")
            pub = publisher_of(query_mag_id)
            if pub:
                query_publisher_id, query_publisher_name = pub[0], pub[1]

    # Build sections
    lines: list[str] = []
    if query_title:
        lines.append(f"クエリ: {query_title}")
    if query_author_name:
        lines.append(f"クエリの作品の作者: {query_author_name}")
    if query_mag_name:
        lines.append(f"クエリが掲載された雑誌: {query_mag_name}")
    if query_publisher_name:
        lines.append(f"クエリが掲載された雑誌の出版社: {query_publisher_name}")

    # Section A: author's other works
    other_works: list[str] = []
    if query_author_name:
        # get author's node id via created_edges where source name == author
        author_ids = {e.get("source") for e in created_edges if id_to_name.get(e.get("source")) == query_author_name}
        for e in created_edges:
            if e.get("source") in author_ids and e.get("target") != query_work_id:
                wname = id_to_name.get(e.get("target"))
                if wname and wname not in other_works:
                    other_works.append(wname)
        lines.append("")
        lines.append(f"### {query_author_name}の別作品")
        if other_works:
            for t in other_works:
                lines.append(f"- {t}")
        else:
            lines.append("- なし")

    # query_mag_nameが入っていて、かつSection Aでは抽出できなかった作品
    if query_title:
        wnames = [
            g["label"]
            for g in graph["nodes"]
            if query_title.lower() in g.get("label", "").lower()
            and g.get("type") == "work"
            and g.get("label") not in other_works
            and g.get("label") != query_title
        ]
        if wnames:
            lines.append("")
            lines.append(
                '### 別の作者の作品。ただしスピンオフなどで関係が深い"かもしれない"作品（全くの別作品の可能性あり）'
            )
            for t in wnames:
                lines.append(f"- {t}")

    # Section B: same magazine other works
    if query_mag_id is not None:
        same_mag_works: list[Any] = []
        for e in published_edges:
            if e.get("source") == query_mag_id and e.get("target") != query_work_id:
                wid = e.get("target")
                if wid not in same_mag_works:
                    same_mag_works.append(wid)
        lines.append("")
        lines.append("### 同雑誌の別作品")
        if same_mag_works:
            for wid in same_mag_works:
                title = id_to_name.get(wid, "")
                auths = authors_of(wid)
                author_txt = "、".join(auths) if auths else "不明"
                pub_txt = query_publisher_name or ""
                mag_txt = query_mag_name or ""
                # show: タイトル（作者: X、雑誌: Y、 出版社: Z）
                meta = []
                meta.append(f"作者: {author_txt}")
                if mag_txt:
                    meta.append(f"雑誌: {mag_txt}")
                if pub_txt:
                    meta.append(f"出版社: {pub_txt}")
                meta_str = "、".join(meta)
                lines.append(f"- {title}（{meta_str}）")
        else:
            lines.append("- なし")

    # Section C: same publisher other magazines
    if query_publisher_id is not None:
        # magazines under same publisher except the query magazine
        other_mag_ids: list[Any] = []
        for e in published_by_edges:
            if e.get("target") == query_publisher_id and e.get("source") != query_mag_id:
                mid = e.get("source")
                if mid not in other_mag_ids:
                    other_mag_ids.append(mid)
        # collect works from those magazines
        lines.append("")
        lines.append("### 同出版社の他誌に掲載された作品")
        collected = 0
        for mid in other_mag_ids:
            # works published by this magazine
            for e in published_edges:
                if e.get("source") == mid:
                    wid = e.get("target")
                    # avoid duplicates and the query work
                    if wid == query_work_id:
                        continue
                    title = id_to_name.get(wid, "")
                    mag_name = id_to_name.get(mid, "")
                    auths = authors_of(wid)
                    author_txt = "、".join(auths) if auths else "不明"
                    pub_txt = query_publisher_name or id_to_name.get(query_publisher_id, "")
                    meta = [f"作者: {author_txt}"]
                    if mag_name:
                        meta.append(f"雑誌: {mag_name}")
                    if pub_txt:
                        meta.append(f"出版社: {pub_txt}")
                    lines.append(f"- {title}（{'、'.join(meta)}）")
                    collected += 1
                    if collected >= 10:
                        break
            # if collected >= 10:
            #     break
        if collected == 0:
            lines.append("- なし")

    return "\n".join(lines)


def format_graph_data(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    out = ["取得したグラフデータ:", ""]
    # out.append("ノード一覧(最大50):")
    # for n in nodes[:30]:
    #     out.append(f"- {_node_label(n)}")
    out.append("\n関係(最大30件):")
    name_cache = {n.get("id"): _node_label(n) for n in nodes if n.get("id")}
    for e in edges[:30]:
        s = name_cache.get(e.get("source"), e.get("source"))
        t = name_cache.get(e.get("target"), e.get("target"))
        edge_type = e.get("type", "REL")
        out.append(f"- {s} -> {t} ({edge_type})")
    return "\n".join(out)


def build_debug_graph_edges(graph: Dict[str, Any]) -> str:
    """Build a full, non-truncated edge list for debugging in the form:
    ・source -[TYPE]-> target

    Edges are sorted by relation type (created, published, published_by, others),
    then by source and target display names for readability.
    """
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    name_cache = {n.get("id"): _node_label(n) for n in nodes if n.get("id") is not None}

    type_order = {"created": 0, "published": 1, "published_by": 2}

    def key(e: Dict[str, Any]):
        t = e.get("type", "")
        s = name_cache.get(e.get("source"), str(e.get("source")))
        d = name_cache.get(e.get("target"), str(e.get("target")))
        return (type_order.get(t, 3), s, d)

    lines: list[str] = []
    for e in sorted(edges, key=key):
        s = name_cache.get(e.get("source"), str(e.get("source")))
        d = name_cache.get(e.get("target"), str(e.get("target")))
        rel = e.get("type", "REL")
        lines.append(f"・{s} -[{rel}]-> {d}")
    return "\n".join(lines)


def fetch_graph_for_user_input(
    user_input: str, min_total_volumes: int = 5, selected_title: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch graph based on user input.

    If selected_title is provided (from UI fuzzy selection), we skip extraction/fuzzy
    and directly run strict search with the selected title.
    """
    # Fast path: respect user-selected title from UI
    if selected_title:
        graph = strict_search(selected_title, min_total_volumes=min_total_volumes)
        graph["_extracted_title"] = selected_title
        graph.setdefault("node_count", len(graph.get("nodes", []) or []))
        graph.setdefault("relationship_count", len(graph.get("edges", []) or []))
        # mark as fuzzy-used via user selection to surface in UI
        graph["_fuzzy_used"] = True
        graph["_fuzzy_best_title"] = selected_title
        graph["_user_selected_candidate"] = True
        return graph
    # 0. strict search
    graph = strict_search(user_input, min_total_volumes=min_total_volumes)
    if graph.get("nodes"):
        graph["_extracted_title"] = user_input
        graph.setdefault("node_count", len(graph.get("nodes", []) or []))
        graph.setdefault("relationship_count", len(graph.get("edges", []) or []))
        return graph

    # 1. extract title
    extracted_title = extract_formal_title(user_input)
    logger.info("Extracted title: %s", extracted_title)

    # 2. strict search
    graph = strict_search(extracted_title, min_total_volumes=min_total_volumes)
    used_fuzzy = False

    nodes = graph.get("nodes", []) or []
    if not nodes:
        # 3. fuzzy search
        fuzzy_res = fuzzy_search(extracted_title)
        print(f"Title similarity results: {json.dumps(fuzzy_res)[:200]}")
        # Tolerant extraction of candidate list across possible response shapes
        if isinstance(fuzzy_res, list):
            candidates = fuzzy_res
        else:
            candidates = (
                fuzzy_res.get("results")
                or fuzzy_res.get("nodes")
                or fuzzy_res.get("matches")
                or fuzzy_res.get("items")
                or []
            )
        if candidates:
            used_fuzzy = True
            # 4. pick highest similarity
            # assume similarity_score field path maybe candidate['similarity_score'] or inside properties

            def sim(c):  # noqa: D401
                """Return similarity score for candidate."""
                return c.get("similarity_score") or c.get("properties", {}).get("similarity_score") or 0

            best = max(candidates, key=sim)
            title_prop = (
                (best.get("properties", {}) or {}).get("title")
                or best.get("title")
                or best.get("name")
                or extracted_title
            )
            graph = strict_search(title_prop, min_total_volumes=min_total_volumes)
            graph["_fuzzy_best_title"] = title_prop
            graph["_fuzzy_used"] = True
        else:
            graph["_fuzzy_used"] = True
            graph["_fuzzy_best_title"] = extracted_title

    graph["_extracted_title"] = extracted_title
    graph.setdefault("node_count", len(graph.get("nodes", []) or []))
    graph.setdefault("relationship_count", len(graph.get("edges", []) or []))
    graph["_fuzzy_used"] = graph.get("_fuzzy_used", used_fuzzy)
    return graph


class GraphRAGRecommender:
    """Recommendation generator that delegates LLM inference to backend API.

    This avoids exposing OPENAI_API_KEY in the Streamlit frontend by sending the
    fully constructed prompt to the server's /text-generation/generate endpoint.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEN_BODY["model"],
        temperature: float = DEFAULT_GEN_BODY["temperature"],
        max_tokens: int = DEFAULT_GEN_BODY["max_tokens"],
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rec_prompt = GraphRAGPrompts.get_recommendation_prompt()

    def recommend(
        self, user_input: str, graph: Dict[str, Any], token_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Generate recommendation via streaming API.

        token_callback: optional callable receiving incremental text chunks.
        Returns the full generated text.
        """
        context = build_graph_context(graph)
        prompt_text = self.rec_prompt.format(
            user_query=user_input,
            context=context,
        )
        print(f"GraphRAG Prompt text:\n{prompt_text}\n")
        body = {
            "text": prompt_text,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": True,
        }
        full_text = ""
        try:
            with requests.post(
                TEXT_GEN_ENDPOINT,
                json=body,
                headers=_auth_headers({"Content-Type": "application/json"}),
                timeout=180,
                stream=True,
            ) as r:
                r.raise_for_status()
                buffer = ""
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    buffer += chunk
                    # SSE format: "data: {...}\n\n" で区切られる
                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)
                        if message.startswith("data: "):
                            line = message[6:].strip()  # "data: " を除去
                            if not line:
                                continue

                            # JSONパースして text フィールドを取得
                            appended = ""
                            try:
                                if line.startswith("{") and line.endswith("}"):
                                    data = json.loads(line)
                                    if isinstance(data, dict) and "text" in data:
                                        appended = str(data["text"])
                                else:
                                    # JSON形式でない場合はそのまま使用
                                    appended = line
                            except Exception:
                                # JSONパースに失敗した場合はそのまま使用
                                appended = line

                            if appended:
                                full_text += appended
                                if token_callback:
                                    token_callback(appended)
        except Exception as e:  # noqa: BLE001
            logger.error("GraphRAG streaming generation failed: %s", e)
            return full_text + f"\n[GraphRAG生成エラー] {e}"
        return full_text or "(生成結果なし)"


def run_graphrag_pipeline(
    user_input: str,
    token_callback: Optional[Callable[[str], None]] = None,
    min_total_volumes: int = 5,
    selected_title: Optional[str] = None,
) -> Dict[str, Any]:
    graph = fetch_graph_for_user_input(user_input, min_total_volumes=min_total_volumes, selected_title=selected_title)
    recommender = GraphRAGRecommender()
    rec_text = recommender.recommend(user_input, graph, token_callback=token_callback)
    return {
        "extracted_title": graph.get("_extracted_title"),
        "fuzzy_used": graph.get("_fuzzy_used", False),
        "fuzzy_best_title": graph.get("_fuzzy_best_title"),
        "user_selected_candidate": graph.get("_user_selected_candidate", False),
        "graph_summary": build_graph_context(graph),
        "graph_debug": build_debug_graph_edges(graph),
        "recommendation": rec_text,
        "raw_graph": graph,
    }

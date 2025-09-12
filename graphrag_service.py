#!/usr/bin/env python3
"""GraphRAG service module for Streamlit app.

Steps implemented:
1. Extract single formal manga title via /text-generation/generate (LLM prompt)
2. Try strict search /api/v1/neo4j/search
3. If no nodes, fallback to /api/v1/neo4j/search-fuzzy
4. If fuzzy used, re-query strict search with best candidate's properties.title
5. Build context + call GraphRAG recommendation prompt
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

import requests
from langchain.prompts import PromptTemplate

from prompts.manga_prompts import GraphRAGPrompts

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TEXT_GEN_ENDPOINT = f"{API_BASE}/text-generation/generate"
STRICT_SEARCH_ENDPOINT = f"{API_BASE}/api/v1/neo4j/search"
FUZZY_SEARCH_ENDPOINT = f"{API_BASE}/api/v1/neo4j/search-fuzzy"

DEFAULT_GEN_BODY = {
    "max_tokens": os.getenv("MAX_TOKENS", 1000),
    "temperature": os.getenv("OPENAI_TEMPERATURE", 0.5),
    "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
}


def _post_text_generation(prompt: str) -> str:
    body = {**DEFAULT_GEN_BODY, "text": prompt}
    try:
        r = requests.post(TEXT_GEN_ENDPOINT, json=body, timeout=60)
        r.raise_for_status()
    except Exception as e:
        logger.warning("text-generation error: %s", e)
        return "不明"

    # non-streaming assumed JSON; sometimes SSE lines -> handle both
    try:
        data = r.json()
        if isinstance(data, dict):
            # API might return {'text': '...'}
            return data.get("generated_text", "")
    except ValueError:
        # fallback: raw text
        return r.text.strip()[:100]
    return "不明"


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
        return "不明"
    # heuristic: avoid model echo of instruction
    if len(title) > 40 and " " in title and "ユーザー" in title:
        return "不明"
    return title


def strict_search(
    title: str, limit: int = 20, include_related: bool = True, min_total_volumes: int = 5
) -> Dict[str, Any]:
    # Added sort_total_volumes & min_total_volumes per requirement
    params = {
        "q": title,
        "limit": limit,
        "include_related": str(include_related).lower(),
        "sort_total_volumes": "desc",
        "min_total_volumes": min_total_volumes,
    }
    try:
        r = requests.get(STRICT_SEARCH_ENDPOINT, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("strict search error: %s", e)
        return {}


def fuzzy_search(
    query: str, limit: int = 5, similarity_threshold: float = 0.8, embedding_method: str = "huggingface"
) -> Dict[str, Any]:
    params = {
        "q": query,
        "limit": limit,
        "similarity_threshold": similarity_threshold,
        "embedding_method": embedding_method,
    }
    try:
        r = requests.get(FUZZY_SEARCH_ENDPOINT, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("fuzzy search error: %s", e)
        return {}


def _node_label(node: Dict[str, Any]) -> str:
    props = node.get("properties", {}) or {}
    return (
        node.get("label") or node.get("name") or node.get("title") or props.get("title") or props.get("name") or "N/A"
    )


def build_graph_context(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    node_count = graph.get("node_count", len(nodes))
    edge_count = graph.get("relationship_count", len(edges))
    ctx = [f"ノード数: {node_count}", f"関係数: {edge_count}"]

    # Group nodes by label/type
    by_type: Dict[str, int] = {}
    sample_by_type: Dict[str, list] = {}
    SAMPLE_LIMIT = 5  # 各タイプのサンプル最大件数
    for n in nodes:
        t = n.get("type") or (n.get("labels") or ["Unknown"])[0]
        by_type[t] = by_type.get(t, 0) + 1
        # サンプル収集（重複は避ける）
        label = _node_label(n)
        if t not in sample_by_type:
            sample_by_type[t] = []
        if label and label not in sample_by_type[t] and len(sample_by_type[t]) < SAMPLE_LIMIT:
            sample_by_type[t].append(label)
    if by_type:
        ctx.append("\nノードタイプ内訳:")
        for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            samples = sample_by_type.get(t) or []
            if samples:
                ctx.append(f"- {t}: {cnt}件 (例: {', '.join(samples)})")
            else:
                ctx.append(f"- {t}: {cnt}件")

    # Sample edges
    if edges:
        ctx.append("\n関係サンプル:")
        node_name_cache = {n.get("id"): _node_label(n) for n in nodes if n.get("id")}
        for e in edges[:10]:
            s = node_name_cache.get(e.get("source"), str(e.get("source")))
            t = node_name_cache.get(e.get("target"), str(e.get("target")))
            rtype = e.get("type", "REL")
            ctx.append(f"  • {s} -[{rtype}]-> {t}")

    return "\n".join(ctx)


def format_graph_data(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    out = ["取得したグラフデータ:", f"ノード数: {len(nodes)}", f"関係数: {len(edges)}", ""]
    out.append("ノード一覧(最大20):")
    for n in nodes[:20]:
        out.append(f"- {_node_label(n)}")
    out.append("\n関係(最大20):")
    name_cache = {n.get("id"): _node_label(n) for n in nodes if n.get("id")}
    for e in edges[:20]:
        s = name_cache.get(e.get("source"), e.get("source"))
        t = name_cache.get(e.get("target"), e.get("target"))
        edge_type = e.get("type", "REL")
        out.append(f"- {s} -> {t} ({edge_type})")
    return "\n".join(out)


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
        print(f"Fuzzy search results: {json.dumps(fuzzy_res)[:200]}")
        candidates = fuzzy_res.get("results") or fuzzy_res.get("nodes") or []
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
        graph_text = format_graph_data(graph)
        context = build_graph_context(graph)
        prompt_text = self.rec_prompt.format(
            user_query=user_input,
            graph_data=graph_text,
            context=context,
        )
        body = {
            "text": prompt_text,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": True,
        }
        full_text = ""
        try:
            with requests.post(TEXT_GEN_ENDPOINT, json=body, timeout=180, stream=True) as r:
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
        "recommendation": rec_text,
        "raw_graph": graph,
    }

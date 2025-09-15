import json
import logging
import os
import time
from copy import deepcopy

import requests
import streamlit as st
from dotenv import load_dotenv

from graphrag_service import extract_formal_title, fuzzy_search, run_graphrag_pipeline, strict_search
from prompts.manga_prompts import StandardMangaPrompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="GraphRAGを使用した生成デモ", page_icon="📚", layout="wide")
load_dotenv()


def stream_generate(text, container, title):
    """APIからストリーミングレスポンスを取得して表示"""
    try:
        api_base = os.getenv("API_BASE", "http://localhost:8000")
        url = f"{api_base}/text-generation/generate"
        headers = {"Content-Type": "application/json"}
        data = {"text": text, "streaming": "true"}

        # ストリーミングレスポンスを処理
        response = requests.post(url, json=data, headers=headers, stream=True)
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

                    result = run_graphrag_pipeline(
                        user_text,
                        token_callback=on_token,
                        min_total_volumes=int(min_volumes),
                        selected_title=selected_title,
                    )
                    reco_placeholder.markdown(result["recommendation"])
                    with st.expander("抽出・検索メタ情報"):
                        st.write(
                            {
                                "extracted_title": result.get("extracted_title"),
                                "fuzzy_used": result.get("fuzzy_used"),
                                "fuzzy_best_title": result.get("fuzzy_best_title"),
                                "user_selected_candidate": result.get("user_selected_candidate"),
                                "node_count": result.get("raw_graph", {}).get("node_count"),
                                "relationship_count": result.get("raw_graph", {}).get("relationship_count"),
                            }
                        )
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

    # 実行ボタン押下時の処理（まず素のLLM→その後に厳格/抽出/あいまい→必要なら候補選択→GraphRAG）
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
                        # 1) 厳格検索（入力テキスト）
                        strict_res = strict_search(input_text, min_total_volumes=int(min_vol))

                        selected_title_for_run: str | None = None
                        if strict_res.get("nodes"):
                            selected_title_for_run = None  # 入力テキストでそのまま実行
                        else:
                            # 2) タイトル抽出 → 厳格
                            extracted = extract_formal_title(input_text)
                            if not extracted:
                                extracted = deepcopy(input_text)
                            strict2 = strict_search(extracted, min_total_volumes=int(min_vol))
                            if strict2.get("nodes"):
                                selected_title_for_run = extracted
                            else:
                                # 3) あいまい検索
                                fz = fuzzy_search(extracted)
                                # さまざまなレスポンス形状に対応
                                raw_candidates = fz.get("results") or fz.get("nodes") or []
                                # nodes配列の場合はworkだけに絞る
                                if (
                                    raw_candidates
                                    and isinstance(raw_candidates[0], dict)
                                    and "type" in raw_candidates[0]
                                ):
                                    raw_candidates = [n for n in raw_candidates if n.get("type") == "work"]

                                processed = []
                                for c in raw_candidates:
                                    title = None
                                    score = None
                                    if isinstance(c, dict):
                                        props = c.get("properties") or {}
                                        # フラット形式 or properties形式 双方対応
                                        title = (
                                            props.get("title")
                                            or c.get("title")
                                            or props.get("name")
                                            or c.get("name")
                                            or props.get("work_title")
                                        )
                                        score = (
                                            props.get("similarity_score") or c.get("similarity_score") or c.get("score")
                                        )
                                    elif isinstance(c, str):
                                        title = c
                                    if title:
                                        disp = f"{title}" if score is None else f"{title} (score: {score:.3f})"
                                        processed.append({"title": title, "score": score, "display": disp})

                    # 曖昧性の結果に応じて分岐
                    if "processed" in locals() and len(processed) > 1:
                        # 2件以上 → ページ内パネルで選択、選択後に生成開始
                        st.session_state["fuzzy_candidates"] = processed
                        st.session_state["dialog_extracted_title"] = extracted
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
                        auto_title = None
                        if "processed" in locals():
                            auto_title = processed[0]["title"] if processed else extracted
                        final_selected_title = (
                            selected_title_for_run if selected_title_for_run is not None else auto_title
                        )

                    run_graphrag_into(
                        col2.container(),
                        status_text,
                        progress_bar,
                        input_text,
                        int(min_vol),
                        selected_title=final_selected_title,
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
        response = requests.get(f"{api_base}/health", timeout=5)
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

import argparse
import json
import logging
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

from graphrag_service import run_graphrag_pipeline
from prompts.manga_prompts import StandardMangaPrompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="GraphRAGを使用した生成デモ", page_icon="📚", layout="wide")
load_dotenv()

# argsでdebugモードを指定可能に
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


def _convert_newlines(text: str) -> str:
    """Convert raw newlines to HTML <br> for reliable rendering in Streamlit markdown."""
    if text is None:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


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
                        time.sleep(0.01)  # 少し遅延を入れて表示を見やすくする
    except requests.exceptions.HTTPError as e:
        with container.container():
            st.subheader(title)
            st.error(f"API呼び出しに失敗しました。ステータスコード: {e.response.status_code}")
            st.text(f"レスポンス: {e.response.text}")

    except requests.exceptions.ConnectionError:
        with container.container():
            st.subheader(title)
            st.error("APIサーバーに接続できません。API_Serverが起動していることを確認してください。")
    except Exception as e:
        with container.container():
            st.subheader(title)
            st.error(f"エラーが発生しました: {str(e)}")


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

    # 実行ボタン
    if st.button("🚀 生成開始", type="primary", use_container_width=True):
        if input_text.strip():
            st.markdown("---")
            st.subheader("📊 生成結果の比較")

            # 2つのカラムを作成
            col1, col2 = st.columns(2)

            # プログレスバーを表示
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 最初のリクエストを実行
            if not args.debug:
                status_text.text("🔄 1つ目のリクエストを実行中...")
                progress_bar.progress(25)
                prompt = get_standard_recommend_prompt(input_text)
                stream_generate(prompt, col1, "🎯 素のLLM（GraphRAGなし）")

            # 2つ目 GraphRAG パイプライン
            status_text.text("🔄 GraphRAGパイプラインを実行中...")
            progress_bar.progress(60)
            with col2.container():
                st.subheader("🎯 GraphRAGを使用した生成")
                with st.spinner("Graph / 推薦生成中..."):
                    try:
                        reco_placeholder = st.empty()
                        buffer = []

                        def on_token(t: str):  # streaming callback
                            buffer.append(t)
                            # 更新タイミング: 5チャンク毎 / 句点 / 改行
                            if "\n" in t or len(buffer) % 5 == 0 or t.endswith(("。", "!", "?")):
                                # GraphRAG出力はMarkdownフォーマットなので、変換せずにそのまま表示
                                reco_placeholder.markdown("".join(buffer))

                        result = run_graphrag_pipeline(
                            input_text, token_callback=on_token, min_total_volumes=int(min_vol)
                        )
                        # 最終更新 - GraphRAG出力はMarkdownフォーマットなので、変換せずにそのまま表示
                        reco_placeholder.markdown(result["recommendation"])
                        with st.expander("抽出・検索メタ情報"):
                            st.write(
                                {
                                    "extracted_title": result.get("extracted_title"),
                                    "fuzzy_used": result.get("fuzzy_used"),
                                    "fuzzy_best_title": result.get("fuzzy_best_title"),
                                    "node_count": result.get("raw_graph", {}).get("node_count"),
                                    "relationship_count": result.get("raw_graph", {}).get("relationship_count"),
                                }
                            )
                            st.text(result.get("graph_summary"))
                    except ValueError as e:
                        # Shouldn't normally occur now, but keep fallback
                        st.error(str(e))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"GraphRAG実行中にエラー: {e}")
            progress_bar.progress(90)

            # 完了
            progress_bar.progress(100)
            status_text.text("✅ 両方の生成が完了しました！")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            st.success("✅ 両方の生成が完了しました！")
        else:
            st.warning("⚠️ テキストを入力してください。")

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

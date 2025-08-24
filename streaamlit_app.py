import json
import time

import requests
import streamlit as st

st.set_page_config(page_title="GraphRAGを使用した生成デモ", page_icon="📚", layout="wide")


def stream_generate_api(text):
    """APIからストリーミングレスポンスを取得"""
    try:
        url = "http://localhost:8000/text-generation/generate"
        headers = {"Content-Type": "application/json"}
        data = {"text": text}

        # ストリーミングレスポンスを処理
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code == 200:
            full_text = ""
            for line in response.iter_lines():
                if line:
                    # バイト文字列をデコード
                    decoded_line = line.decode("utf-8")

                    # SSE形式の場合、"data: "プレフィックスを削除
                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line[6:]

                    # JSON形式のレスポンスを処理
                    try:
                        json_data = json.loads(decoded_line)
                        if "text" in json_data:
                            full_text += json_data["text"]
                        elif "content" in json_data:
                            full_text += json_data["content"]
                        else:
                            full_text += decoded_line
                    except json.JSONDecodeError:
                        # JSONでない場合は直接追加
                        full_text += decoded_line

            return {"success": True, "text": full_text}
        else:
            return {
                "success": False,
                "error": f"API呼び出しに失敗しました。ステータスコード: {response.status_code}",
                "details": response.text,
            }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "APIサーバーに接続できません。http://localhost:8000 が起動していることを確認してください。",
        }
    except Exception as e:
        return {"success": False, "error": f"エラーが発生しました: {str(e)}"}


def stream_generate(text, container, title):
    """APIからストリーミングレスポンスを取得して表示"""
    try:
        url = "http://localhost:8000/text-generation/generate"
        headers = {"Content-Type": "application/json"}
        data = {"text": text}

        # ストリーミングレスポンスを処理
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code == 200:
            full_text = ""
            with container.container():
                st.subheader(title)
                text_placeholder = st.empty()

                for line in response.iter_lines():
                    if line:
                        # バイト文字列をデコード
                        decoded_line = line.decode("utf-8")

                        # SSE形式の場合、"data: "プレフィックスを削除
                        if decoded_line.startswith("data: "):
                            decoded_line = decoded_line[6:]

                        # JSON形式のレスポンスを処理
                        try:
                            json_data = json.loads(decoded_line)
                            if "text" in json_data:
                                full_text += json_data["text"]
                            elif "content" in json_data:
                                full_text += json_data["content"]
                            else:
                                full_text += decoded_line
                        except json.JSONDecodeError:
                            # JSONでない場合は直接追加
                            full_text += decoded_line

                        # リアルタイムで表示を更新
                        text_placeholder.markdown(full_text)
                        time.sleep(0.01)  # 少し遅延を入れて表示を見やすくする
        else:
            with container.container():
                st.subheader(title)
                st.error(f"API呼び出しに失敗しました。ステータスコード: {response.status_code}")
                st.text(f"レスポンス: {response.text}")

    except requests.exceptions.ConnectionError:
        with container.container():
            st.subheader(title)
            st.error("APIサーバーに接続できません。http://localhost:8000 が起動していることを確認してください。")
    except Exception as e:
        with container.container():
            st.subheader(title)
            st.error(f"エラーが発生しました: {str(e)}")


def main():
    st.title("📚 GraphRAGを使用した生成デモ")
    st.markdown("同じテキストに対して素のLLM（GraphRAGなし）とGraphRAGを使用した生成の結果を比較表示します。")

    # 入力欄
    st.subheader("🔤 テキスト入力")
    input_text = st.text_area(
        "生成したいテキストを入力してください:", height=100, placeholder="例: 主人公が冒険の旅に出る物語"
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
            status_text.text("🔄 1つ目のリクエストを実行中...")
            progress_bar.progress(25)
            stream_generate(input_text, col1, "🎯 素のLLM（GraphRAGなし）")

            # 2つ目のリクエストを実行
            status_text.text("🔄 2つ目のリクエストを実行中...")
            progress_bar.progress(75)
            stream_generate(input_text, col2, "🎯 GraphRAGを使用した生成")

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
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ APIサーバーに正常に接続できます")
            else:
                st.warning(f"⚠️ サーバーからの応答が異常です (ステータス: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ APIサーバーに接続できません。http://localhost:8000 が起動していることを確認してください。")
        except Exception as e:
            st.error(f"❌ 接続確認中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()

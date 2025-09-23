import streamlit as st

st.set_page_config(page_title="出典", page_icon="ℹ️")

st.title("出典")
st.markdown(
    """
このアプリケーションは、以下のデータセットを使用しています：

- [メディア芸術データベース](https://mediaarts-db.artmuseums.go.jp/)
  - 出典：独立行政法人国立美術館国立アートリサーチセンター「メディア芸術データベース」 （https://mediaarts-db.artmuseums.go.jp/）
  - 独立行政法人国立美術館国立アートリサーチセンター「メディア芸術データベース」（https://mediaarts-db.artmuseums.go.jp/）を加工してデータを作成
- [OpenBD](https://openbd.jp/)
  - 「OpenBD」 （https://openbd.jp/） を利用しています。
"""
)

st.divider()
st.page_link("streamlit_app.py", label="← トップへ戻る", icon="🏠")

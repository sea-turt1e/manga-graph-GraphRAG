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
- [MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist)
  - 本プロジェクトはMyAnimeList Dataset（MyAnimeList.net） のデータを利用しています。データベースは Open Database License (ODbL) v1.0、個々のコンテンツは Database Contents License (DbCL) v1.0 に基づきます。ライセンス条件に従い帰属表示と通知保持を行っています。」
"""
)

st.divider()
st.page_link("streamlit_app.py", label="← トップへ戻る", icon="🏠")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas.io.formats.excel
from plotly.offline import iplot

#import requests
#from bs4 import BeautifulSoup
#import time
#from llama_cpp import Llama
#import urllib.parse
#import trafilatura
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.schema import Document
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
#import re

#from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
#from reportlab.lib.pagesizes import letter
#from reportlab.lib import colors
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

st.set_page_config(layout="wide")

# --- ここでユーザーIDとパスワードを設定 ---
USER_CREDENTIALS = {
    "admin": "password123",
    "user1": "pass1"
}

# --- セッション状態初期化 ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- ログイン処理 ---
def login():
    username = st.session_state["login_username"]
    password = st.session_state["login_password"]
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success(f"{username}さん、ログイン成功！")
    else:
        st.session_state.logged_in = False
        st.error("ユーザー名またはパスワードが間違っています")

# --- ログイン画面 ---
if not st.session_state.logged_in:
    st.title("ログイン")
    st.text_input("ユーザー名", key="login_username")
    st.text_input("パスワード", type="password", key="login_password")
    st.button("ログイン", on_click=login)
    st.stop()  # ログイン前はここで処理を止める

# --- ログイン成功後に表示する画面 ---
st.title("Price Dush")
st.write(f"ようこそ、{st.session_state.username}さん！")


# --- キャッシュ付きでParquet読み込み ---
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
    

try:
    # ローカルの固定ファイルを読み込み
    df = load_csv("moto_.csv")
    df = df.sort_values(['brandna','syameisei', 'katashiki', 'gradesei', 'nenss', 'hyoka'])
    today = pd.to_datetime("today").normalize()
    df["aaymd"] = pd.to_datetime(df["aaymd"])
    two_years_ago = today - pd.DateOffset(years=2)
    df = df[df["aaymd"] >= two_years_ago]
    # ソート（必要に応じて調整)
    #df = df.sort_values(['brandna','syameisei', 'katashiki', 'gradesei', 'nenss', 'hyoka'])
    #df['aaymd'] = pd.to_datetime(df['aaymd'])
    #df['aaym'] = pd.to_datetime(df['aaym'])

except:
    pass

# --- JSON階層読み込み ---
with open("nestdict_latest.json", "r", encoding="utf-8") as f:
    car_data = json.load(f)
    
def normalize_car_data_years_and_scores(car_data: dict):
    """
    JSON データを整形：
    - 年式を int に変換
    - 評価点を float に変換
    """
    for brand, brand_dict in car_data.items():
        for syameisei, katashiki_dict in brand_dict.items():
            for katashiki, grade_dict in katashiki_dict.items():
                for grade, val in grade_dict.items():
                    if isinstance(val, list):
                        # 評価点リストを float に変換
                        car_data[brand][syameisei][katashiki][grade] = [float(x) for x in val]
                    elif isinstance(val, dict):
                        # 年式ごとの dict のキーを int に、値はリストなら float に変換
                        new_dict = {}
                        for year_str, scores in val.items():
                            year_int = int(year_str)
                            if isinstance(scores, list):
                                new_dict[year_int] = [float(x) for x in scores]
                            else:
                                new_dict[year_int] = scores
                        car_data[brand][syameisei][katashiki][grade] = new_dict
normalize_car_data_years_and_scores(car_data)

st.sidebar.markdown("## Settings")
    #category = st.sidebar.selectbox("カテゴリー", test, index = None, placeholder="選択してください")  # ① 表示名に使うカラムを選択
try:
    # ブランド選択
    brandna = st.sidebar.selectbox("ブランド", list(car_data.keys()), index = None, placeholder="選択してください")

    # 車種選択
    syameisei = st.sidebar.selectbox("車種", list(car_data[brandna].keys()), index = None, placeholder="選択してください")

    # 型式選択
    katashiki = st.sidebar.selectbox("型式", list(car_data[brandna][syameisei].keys()), index = None, placeholder="選択してください")

    # グレード選択
    gradesei = st.sidebar.selectbox("グレード", list(car_data[brandna][syameisei][katashiki].keys()), index = None, placeholder="選択してください")

    # 年式選択
    nenss = st.sidebar.selectbox("年式", list(car_data[brandna][syameisei][katashiki][gradesei].keys()), index = None, placeholder="選択してください")

    # ボディカラー選択
    clrmona = st.sidebar.selectbox("ボディカラー", list(car_data[brandna][syameisei][katashiki][gradesei][nenss].keys()), index = None, placeholder="選択してください")
    
    # 評価点選択
    hyoka_list = car_data[brandna][syameisei][katashiki][gradesei][nenss][clrmona]
    hyoka = st.sidebar.selectbox("外装評価点", hyoka_list, index = None, placeholder="選択してください")

except:
    st.warning("アルファードを選択してください")
    
# --- CSVフィルタリング ---
#if brandna and syameisei and katashiki:
try:
    df3 = df[
        (df['brandna'] == brandna) &
        (df['syameisei'] == syameisei) &
        (df['katashiki'] == katashiki)
    ]
except:
    pass
    
    
# --- CSVフィルタリング ---
#if brandna and syameisei and katashiki and gradesei and nenss:
try:
    df45 = df[
        (df['brandna'] == brandna) &
        (df['syameisei'] == syameisei) &
        (df['katashiki'] == katashiki) &
        (df['gradesei'] == gradesei) &
        (df['nenss'] == int(nenss))
    ]
except:
    pass
    
# --- CSVフィルタリング ---
#if brandna and syameisei and katashiki and gradesei and nenss and clrmona and hyoka:
try:
    df6 = df[
        (df['brandna'] == brandna) &
        (df['syameisei'] == syameisei) &
        (df['katashiki'] == katashiki) &
        (df['gradesei'] == gradesei) &
        (df['nenss'] == int(nenss)) &
        (df['clrmona'] == clrmona) &
        (df['hyoka'] == hyoka)
    ]
except:
    pass
    
# --- 走行距離スライダー ---
try:
    min_s = df6['distance'].min()
    max_s = df6['distance'].max()
except:
    min_s, max_s = None,None
if pd.isna(min_s) or pd.isna(max_s):
    st.warning("評価点が未選択の為、ダミースライダーになっています。")
    #start_s, end_s = None, None
    
    start_s, end_s = st.sidebar.slider(
        '走行距離',
        min_value=0,
        max_value=999999,
        value=(0, 999999)
    )   
else:
    start_s, end_s = st.sidebar.slider(
        '走行距離',
        min_value=float(min_s),
        max_value=float(max_s),
        value=(float(min_s), float(max_s))
    )   
try:
    if start_s is not None and end_s is not None:
        if start_s == end_s:
            df7 = df6[df6['distance'] == start_s]
        else:
            df7 = df6[df6['distance'].between(start_s, end_s)]
    else:
        df7 = df6.copy()  # スライダーが無効なら全件
except:
    pass

span = [5,7,10,15]
col1, col2 = st.columns([2,1])
with col1:
    with st.expander('テクニカル指標'):
        a = st.checkbox(label='MA:移動平均線',value=True)
        b = st.checkbox(label='EMA:指数平滑移動平均線',value=True)
        c = st.checkbox(label='ボリンジャーバンド',value=False)
with col2:
    with st.expander('移動平均日数'):
        window_size = st.selectbox('日数',span)
        
try:
    df8 = df7.groupby(['aaym'],as_index=False,dropna=False).agg(price_ave=('price', 'mean'))
    df8['MA'] = df8['price_ave'].rolling(window=window_size).mean()
    df8['EMA'] = df8['price_ave'].ewm(span=window_size, adjust=False, ignore_na=True).mean()
    df8['std'] = df8['price_ave'].rolling(window=window_size).std()
    df8['up_b'] = df8['MA'] + (2 * df8['std'])
    df8['lo_b'] = df8['MA'] - (2 * df8['std'])
    fig_scatter = go.Scatter(x=df45['aaymd'], y=df45['price'], mode='markers', name='Near', marker=dict(color='white'),customdata=df45[['hyoka','clrmona','distance']],hovertemplate='<b>開催年月日:</b> %{x}<br><b>評価点:</b> %{customdata[0]} 点<br><b>ボディカラー:</b> %{customdata[1]}<br><b>走行距離:</b> %{customdata[2]} km<br><b>落札価格:</b> %{y} 千円<extra></extra>')
    fig_scatter2 = go.Scatter(x=df7['aaymd'], y=df7['price'], mode='markers', name='Equal', marker=dict(color='orange'),customdata=df7[['hyoka','clrmona','distance']],hovertemplate='<b>開催年月日:</b> %{x}<br><b>評価点:</b> %{customdata[0]} 点<br><b>ボディカラー:</b> %{customdata[1]}<br><b>走行距離:</b> %{customdata[2]} km<br><b>落札価格:</b> %{y} 千円<extra></extra>')
    fig_scatter3 = go.Scatter(x=df8['aaym'], y=df8['MA'], mode='lines', name=f'{window_size}日移動平均落札', line=dict(color='red', shape='spline'))
    fig_scatter4 = go.Scatter(x=df8['aaym'], y=df8['EMA'], mode='lines', name=f'{window_size}日指数平滑移動平均落札', line=dict(color='green', shape='spline'))
    fig_scatter5 = go.Scatter(x=df8['aaym'], y=df8['up_b'], mode='lines', name='upper_band', line=dict(color='blue', shape='spline'))
    fig_scatter6 = go.Scatter(x=df8['aaym'], y=df8['lo_b'], mode='lines', name='lower_band',fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='blue', shape='spline'))
    layout = go.Layout(title='価格チャート：ver0.8')
#graph = [fig_scatter,fig_scatter2,fig_scatter3,fig_scatter4,fig_scatter5,fig_scatter6]
    graph = [fig_scatter,fig_scatter2]
    if a:
        graph.append(fig_scatter3)
    if b:
        graph.append(fig_scatter4)
    if c:
        graph.append(fig_scatter5)
        graph.append(fig_scatter6)
    fig = go.Figure(data=graph, layout=layout)
    fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"),
                      legend=dict(x=0.01,y=0.99,xanchor='left',yanchor='top',orientation='h',)) # 例: 西暦4桁、月2桁、日2桁
    st.plotly_chart(fig)
except:
    pass


st.sidebar.markdown("## Settings:残価表")

def weighted_mean_by_date(x, df, value_col, date_col="aaymd", ratio=1.0, decay=0.98):
    """
    x: group内の Series (値)
    df: 元のDataFrame
    value_col: 'price' or 'RV'
    date_col: 基準日付カラム
    ratio: 補正倍率
    decay: 直近重視度（0.95〜0.99くらいが実用的）
    """
    sub_df = df.loc[x.index, [value_col, date_col]].copy()
    sub_df[date_col] = pd.to_datetime(sub_df[date_col])

    max_date = sub_df[date_col].max()
    sub_df["days_diff"] = (max_date - sub_df[date_col]).dt.days
    sub_df["weight"] = decay ** sub_df["days_diff"]

    if sub_df["weight"].sum() == 0:
        return None

    w_avg = np.average(sub_df[value_col], weights=sub_df["weight"]) * ratio
    return int(round(w_avg, 0))

col3, col4 = st.columns(2)
with col3:
    with st.sidebar.expander("レート設定"):
        ratio = st.slider('レート',0.5,1.0,1.0,0.05)
#with col4:
    #with st.sidebar.expander("色系統設定"):
        #clr_set = st.selectbox('色系統',clr_list)

with st.expander("残価表"):
    try:
        df3['新車価格'] = (df3['max_price'] / 1000).fillna(0).round(0).astype(int)
        df3 = df3[df3['新車価格'] != 0]
        df3['clr_adjusted_price'] = df3['clr_adjusted_price'].fillna(0).round(0).astype(int)
        
        # --- 切替ボタン ---
        mode = st.radio("price：（千円）、salvage_rate：（%）", ["clr_adjusted_price", "SV"], horizontal=True)

        # --- ピボットテーブル作成 ---
        #pivot_table = pd.pivot_table(df3,values='price' if mode == "price" else 'RV',index=['grade','新車価格'],columns='nenss',aggfunc=lambda x:str(int(round(x.mean() * ratio ,0))))
        # --- ピボットテーブル作成 ---
        pivot_table = pd.pivot_table(
            df3,
            values='clr_adjusted_price' if mode == "clr_adjusted_price" else 'SV',
            index=['gradesei','新車価格'],
            columns='nenss',
            aggfunc=lambda x: str(int(weighted_mean_by_date(
                x, df3,
                value_col='clr_adjusted_price' if mode == "clr_adjusted_price" else 'SV',
                date_col="aaymd",
                ratio=ratio,
                decay=0.98  # ← ここを調整すると「直近の効き具合」が変わる
                )))
            )
        pivot_table = pivot_table.fillna("")
        pivot_table_sorted = pivot_table.sort_values(by='nenss', axis=1, ascending=False)
        pivot_table_sorted = pivot_table_sorted.sort_values(by='gradesei',ascending=False)
        st.text('残価表：ver0.8')
        st.write(f"{syameisei}：{katashiki}")
        #st.table(pivot_table_sorted)
        st.dataframe(
            pivot_table_sorted,
            column_config={
                "gradesei": st.column_config.TextColumn("グレード", width="medium"),
                "新車価格": st.column_config.NumberColumn("新車価格", width="small"),
                },
            use_container_width=True
            )
    except Exception as e:
        st.error(f"残価表エラー：{e}")
    
    try:
        # 色ごとに補正値の平均を計算
        params_means = df3.groupby('clrgrp')['clr_params'].mean().round(1).to_dict()
        # 文字列として作成
        color_texts = []
        for color_name, avg_value in params_means.items():
            color_texts.append(f"{color_name}：{avg_value}")

        # スペース区切りで一行にまとめて表示
        st.write("色補正値（平均）： " + " ".join(color_texts))
    except:
        pass
    
    try:
        df3_1 = df3[['gradesei','nenss','clr_adjusted_price']].groupby(['gradesei','nenss'],as_index=False).mean() #*ratio 
        df3_1['clr_adjusted_price'] = df3_1['clr_adjusted_price'].astype(int)
        fig_scatter7 = px.line(df3_1,
                               x='nenss',
                               y='clr_adjusted_price',
                               #text = 'price',
                               markers=True ,
                               color='grade',
                               title='残価チャート：ver0.8',
                               line_shape='spline'
                               )
        fig_scatter7.update_xaxes(autorange='reversed')
        fig_scatter7.update_layout(
            legend=dict(
                x=-0.1,  # 右端
                y=1,  # 上端
                xanchor='right', # 右揃え
                yanchor='top', # 上揃え
                )
            )
        st.plotly_chart(fig_scatter7)
    except:
        pass
    
    
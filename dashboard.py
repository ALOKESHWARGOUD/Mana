# dashboard.py
import json, os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config("Movie Intelligence", "🎬", layout="wide")

if not os.path.exists("latest_intelligence.json"):
    st.error("No intelligence data found yet.")
    st.stop()

with open("latest_intelligence.json", "r", encoding="utf-8") as f:
    data = json.load(f)

st.title(f"🎬 {data['movie']} — Intelligence Dashboard")
st.caption(f"Hero: {data['hero']} | Director: {data['director']}")

# 🚨 ALERT
if data.get("real_time_alert"):
    a = data["real_time_alert"]
    st.error(f"🚨 NEGATIVITY SPIKE | {a['date']} | {a['count']} vs avg {a['average']}")

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Total Mentions", data["instances"]["total_mentions"])
k2.metric("Negative Mentions", data["instances"]["negative_mentions"])
k3.metric("Coordinated Attackers", len(data["attack_coordination"]))

# Sentiment by Stage
st.subheader("🎥 Sentiment by Movie Stage")
stage_df = pd.DataFrame.from_dict(data["sentiment_by_stage"], orient="index")
stage_df["negative_pct"] = (stage_df["negative"] / stage_df["total"] * 100).round(2)
st.bar_chart(stage_df[["total", "negative_pct"]])

# Song Analytics
st.subheader("🎵 Song-wise Sentiment")
song_df = pd.DataFrame.from_dict(data["song_analysis"], orient="index").fillna(0)
song_df["negative_pct"] = (song_df["negative"] / song_df["total"] * 100).round(2)
st.dataframe(song_df.sort_values("negative_pct", ascending=False))

# Language
st.subheader("🌐 Language Distribution")
lang_df = pd.DataFrame.from_dict(data["language_distribution"], orient="index")
lang_df["negative_pct"] = (lang_df["negative"] / lang_df["total"] * 100).round(2)
st.dataframe(lang_df)

# Attack Coordination
st.subheader("🚨 Coordinated Attack Users")
if data["attack_coordination"]:
    st.dataframe(pd.DataFrame(data["attack_coordination"]))
else:
    st.success("No coordinated attack patterns detected")

# Evidence
st.subheader("🧾 Comment Evidence")
comments_df = pd.DataFrame(data["comments"])
st.dataframe(
    comments_df[
        ["author", "sentiment", "language", "video_type", "video_title", "comment", "video_url"]
    ],
    use_container_width=True
)
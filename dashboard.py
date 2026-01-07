# frontend/dashboard.py
import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config("Movie Intelligence", "🎬", layout="wide")

# ================= LOAD DATA =================
if not os.path.exists("latest_intelligence.json"):
    st.error("❌ No intelligence data found yet. Run backend first.")
    st.stop()

with open("latest_intelligence.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ================= HEADER =================
st.title(f"🎬 {data.get('movie')} — Intelligence Dashboard")
st.caption(
    f"Hero: {data.get('hero')} | Director: {data.get('director')} | "
    f"Updated: {data.get('generated_at')}"
)

# ================= KPI =================
k1, k2, k3 = st.columns(3)
inst = data.get("instances", {})
k1.metric("Total Mentions", inst.get("total_mentions", 0))
k2.metric("Negative Mentions", inst.get("negative_mentions", 0))
k3.metric("Attackers", len(data.get("attack_coordination", [])))

st.divider()

# ================= STAGE SENTIMENT =================
st.subheader("🎥 Sentiment by Movie Stage")

stage = data.get("sentiment_by_stage", {})
if stage:
    df = pd.DataFrame.from_dict(stage, orient="index").reset_index()
    df.columns = ["Stage", "Total", "Negative"]
    fig = px.bar(df, x="Stage", y=["Total", "Negative"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# ================= SONG BAR GRAPH =================
st.subheader("🎵 Song-wise Negative Sentiment")

songs = data.get("song_analysis", {})
if songs:
    df = pd.DataFrame.from_dict(songs, orient="index").reset_index()
    df.columns = ["Song", "Total", "Negative"]
    df = df.sort_values("Negative", ascending=False)

    fig = px.bar(
        df,
        x="Song",
        y="Negative",
        title="Negative Comments per Song"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================= NEGATIVE SPIKE LINE GRAPH =================
st.subheader("📉 Negative Comment Spike Timeline")

spikes = data.get("negative_spikes", {})
if spikes:
    spike_df = pd.DataFrame(
        sorted(spikes.items()),
        columns=["Time", "Negative"]
    )
    spike_df["Time"] = pd.to_datetime(spike_df["Time"])

    fig = px.line(
        spike_df,
        x="Time",
        y="Negative",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================= LANGUAGE =================
st.subheader("🌐 Language Distribution")

lang = data.get("language_distribution", {})
if lang:
    df = pd.DataFrame.from_dict(lang, orient="index").reset_index()
    df.columns = ["Language", "Total", "Negative"]
    fig = px.bar(df, x="Language", y="Negative")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================= ATTACK USERS =================
st.subheader("🚨 Coordinated Attack Users")

attackers = data.get("attack_coordination", [])
if attackers:
    st.dataframe(pd.DataFrame(attackers), use_container_width=True)
else:
    st.success("✅ No coordinated attacks detected")

st.divider()

# ================= COMMENTS =================
st.subheader("🧾 Comment Evidence")

comments = data.get("comments", [])
if comments:
    df = pd.DataFrame(comments)
    st.dataframe(
        df[
            ["author", "sentiment", "language", "video_type", "video_title", "comment"]
        ],
        use_container_width=True
    )

# dashboard.py
import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Movie Intelligence", layout="wide")

with open("latest_intelligence.json", "r", encoding="utf-8") as f:
    data = json.load(f)

comments = pd.DataFrame(data["comments"])

st.title(f"🎬 {data['movie']} – Intelligence Dashboard")
st.caption(f"Hero: {data['hero']} | Director: {data['director']} | Updated: {data['generated_at']}")

# ================= KPIs =================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Mentions", data["instances"]["total_mentions"])
k2.metric("Negative Mentions", data["instances"]["negative_mentions"])
k3.metric("Flagged Videos", len(data["flagged_negative_videos"]))
k4.metric("Repeat Users", len(data["repeat_users"]))

st.divider()

# ================= FLAGGED VIDEOS =================
st.subheader("🚨 Flagged Negative Uploads")
st.dataframe(pd.DataFrame(data["flagged_negative_videos"]), use_container_width=True)

# ================= SPIKES =================
st.subheader("📈 Negative Comment Spikes")
spike_df = pd.DataFrame(data["negative_spikes"].items(), columns=["Hour","Count"])
st.plotly_chart(px.line(spike_df, x="Hour", y="Count"), use_container_width=True)

# ================= LANGUAGE =================
st.subheader("🌐 Language Distribution")
lang_rows = []
for l,v in data["language_distribution"].items():
    total = v["Positive"] + v["Negative"]
    lang_rows.append({"Language":l,"Negative %":round(v["Negative"]/max(1,total)*100,2)})
st.plotly_chart(px.bar(pd.DataFrame(lang_rows), x="Language", y="Negative %"), use_container_width=True)

# ================= STAGE =================
st.subheader("🎞 Sentiment by Movie Stage")
stage_rows = []
for s,v in data["sentiment_by_stage"].items():
    stage_rows.append({"Stage":s,"Negative %":round(v["negative"]/max(1,v["total"])*100,2)})
st.plotly_chart(px.bar(pd.DataFrame(stage_rows), x="Stage", y="Negative %"), use_container_width=True)

# ================= CATEGORIES =================
st.subheader("🧨 Negative Comment Categories")
st.plotly_chart(px.pie(pd.DataFrame(data["negative_categories"].items(), columns=["Category","Count"]),
                       names="Category", values="Count"), use_container_width=True)

# ================= OFFENDERS =================
st.subheader("🕵️ Top Repeat Offenders")
st.dataframe(pd.DataFrame(data["repeat_users"]), use_container_width=True)

# ================= TAKEAWAYS =================
st.subheader("📌 Key Takeaways")
for t in data["key_takeaways"]:
    st.write("•", t)

# ================= COMMENTS =================
st.subheader("🧾 Comment Evidence")
st.dataframe(comments[["author","sentiment","language","video_type","video_title","comment","video_url"]],
             use_container_width=True)
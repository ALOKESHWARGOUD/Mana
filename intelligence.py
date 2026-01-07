# intelligence.py
# FULL MOVIE-LEVEL YouTube Intelligence Engine
# Hourly automation | .env | Complete intelligence layers

from dotenv import load_dotenv
load_dotenv()

import os
import time
import json
from collections import defaultdict, Counter
from datetime import datetime

from googleapiclient.discovery import build
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not found in .env")

SENTIMENT_MODEL = "tabularisai/multilingual-sentiment-analysis"
MAX_VIDEOS = 300
MAX_COMMENTS = 100
NEGATIVE_VIDEO_THRESHOLD = 60
REPEAT_USER_THRESHOLD = 3

MOVIE_NAME = "Shiva Shankara Vara Prasad"
HERO = "Chiranjeevi"
DIRECTOR = "Anil Ravipudi"

VIDEO_CLASSIFIERS = {
    "Title Glimpse": ["title", "glimpse", "announcement", "first look"],
    "Song": ["song", "lyrical", "audio", "music"],
    "Teaser": ["teaser"],
    "Trailer": ["trailer"],
    "Interview": ["interview", "press meet", "interaction"],
    "Review / Public Talk": ["review", "public talk", "response"]
}

NEGATIVE_CATEGORIES = {
    "Routine Content": ["routine", "boring", "same"],
    "Age / Relevance": ["old", "age", "outdated"],
    "Director Criticism": ["ravipudi", "director"],
    "Story Doubt": ["story", "script", "content"],
    "PR Attack": ["pr", "paid", "fake hype"],
    "Personal Attack": ["acting", "voice", "looks"]
}

# =====================================================
# HELPERS
# =====================================================

def normalize_sentiment(label):
    return "Positive" if "pos" in label.lower() else "Negative"

def classify_video(title):
    t = title.lower()
    for stage, keys in VIDEO_CLASSIFIERS.items():
        if any(k in t for k in keys):
            return stage
    return "Other"

def categorize_negative(text):
    t = text.lower()
    for cat, keys in NEGATIVE_CATEGORIES.items():
        if any(k in t for k in keys):
            return cat
    return "General Negativity"

def get_youtube():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def load_sentiment():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

# =====================================================
# VIDEO SEARCH
# =====================================================

def search_movie_videos(youtube):
    query = f"{MOVIE_NAME} {HERO} {DIRECTOR}"
    videos = []

    req = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=50,
        relevanceLanguage="te",
        regionCode="IN"
    )

    while req and len(videos) < MAX_VIDEOS:
        res = req.execute()
        for it in res.get("items", []):
            vid = it["id"]["videoId"]
            title = it["snippet"]["title"]
            videos.append({
                "video_id": vid,
                "video_title": title,
                "video_type": classify_video(title),
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "published_at": it["snippet"]["publishedAt"]
            })
        req = youtube.search().list_next(req, res)
        time.sleep(0.2)

    return list({v["video_id"]: v for v in videos}.values())

# =====================================================
# COMMENTS
# =====================================================

def fetch_comments(youtube, video):
    out = []
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video["video_id"],
            maxResults=MAX_COMMENTS,
            textFormat="plainText"
        )
        while req:
            res = req.execute()
            for it in res.get("items", []):
                s = it["snippet"]["topLevelComment"]["snippet"]
                out.append({
                    "author": s.get("authorDisplayName"),
                    "comment": s.get("textDisplay"),
                    "published_at": s.get("publishedAt"),
                    "video_title": video["video_title"],
                    "video_type": video["video_type"],
                    "video_url": video["video_url"]
                })
            req = youtube.commentThreads().list_next(req, res)
    except:
        pass
    return out

# =====================================================
# MAIN ENGINE
# =====================================================

def run_intelligence():
    yt = get_youtube()
    sentiment_model = load_sentiment()

    videos = search_movie_videos(yt)

    all_comments = []
    for v in videos:
        all_comments.extend(fetch_comments(yt, v))

    results = []
    video_map = defaultdict(list)
    user_map = defaultdict(list)
    language_map = defaultdict(lambda: {"Positive": 0, "Negative": 0})
    stage_map = defaultdict(lambda: {"total": 0, "negative": 0})
    category_map = Counter()
    spikes = Counter()
    engagement = Counter()

    for c in tqdm(all_comments, desc="Analyzing comments"):
        try:
            out = sentiment_model(c["comment"][:512])[0]
            sentiment = normalize_sentiment(out["label"])
            language = detect(c["comment"])

            ts = datetime.fromisoformat(c["published_at"].replace("Z", "+00:00"))
            hour_bucket = ts.strftime("%Y-%m-%d %H:00")

            neg_cat = categorize_negative(c["comment"]) if sentiment == "Negative" else None

            row = {
                **c,
                "sentiment": sentiment,
                "language": language,
                "negative_category": neg_cat,
                "confidence": out["score"],
                "hour_bucket": hour_bucket
            }

            results.append(row)
            video_map[c["video_title"]].append(row)
            user_map[c["author"]].append(row)
            engagement[c["video_title"]] += 1
            language_map[language][sentiment] += 1
            stage_map[c["video_type"]]["total"] += 1

            if sentiment == "Negative":
                stage_map[c["video_type"]]["negative"] += 1
                category_map[neg_cat] += 1
                spikes[hour_bucket] += 1

        except:
            pass

    # ================= FLAGGED VIDEOS =================

    flagged_videos = []
    for title, rows in video_map.items():
        neg = sum(1 for r in rows if r["sentiment"] == "Negative")
        pct = round((neg / max(1, len(rows))) * 100, 2)
        if pct >= NEGATIVE_VIDEO_THRESHOLD:
            flagged_videos.append({
                "video_title": title,
                "video_type": rows[0]["video_type"],
                "negative_percentage": pct,
                "video_url": rows[0]["video_url"]
            })

    # ================= REPEAT USERS =================

    repeat_users = {
        u: v for u, v in user_map.items()
        if sum(1 for x in v if x["sentiment"] == "Negative") >= REPEAT_USER_THRESHOLD
    }

    top_offenders = [
        {
            "author": u,
            "negative_comments": sum(1 for x in v if x["sentiment"] == "Negative"),
            "videos_targeted": len(set(x["video_title"] for x in v))
        }
        for u, v in repeat_users.items()
    ]

    # ================= KEY TAKEAWAYS =================

    key_takeaways = [
        f"{round((sum(1 for r in results if r['sentiment']=='Negative') / max(1,len(results))) * 100,2)}% of all mentions are negative",
        f"{len(flagged_videos)} videos crossed negative threshold",
        f"{len(repeat_users)} users show repeated negative behavior",
        f"Highest negativity stage: {max(stage_map, key=lambda x: stage_map[x]['negative']/max(1,stage_map[x]['total']))}"
    ]

    output = {
        "movie": MOVIE_NAME,
        "hero": HERO,
        "director": DIRECTOR,
        "generated_at": datetime.utcnow().isoformat(),

        "instances": {
            "total_mentions": len(results),
            "negative_mentions": sum(1 for r in results if r["sentiment"] == "Negative")
        },

        "negative_spikes": dict(spikes),
        "language_distribution": language_map,
        "sentiment_by_stage": stage_map,
        "negative_categories": dict(category_map),
        "top_engaged_videos": engagement.most_common(10),

        "flagged_negative_videos": flagged_videos,
        "repeat_users": top_offenders,
        "repeat_user_details": repeat_users,
        "key_takeaways": key_takeaways,

        "videos": videos,
        "comments": results
    }

    with open("latest_intelligence.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output

if __name__ == "__main__":
    run_intelligence()
    print("✅ Hourly intelligence updated")
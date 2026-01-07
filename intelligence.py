# backend/intelligence.py
# FULL MOVIE-LEVEL YOUTUBE INTELLIGENCE ENGINE

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

# ================= CONFIG =================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY missing in .env")

SENTIMENT_MODEL = "tabularisai/multilingual-sentiment-analysis"

MAX_VIDEOS = 300
MAX_COMMENTS = 100
REPEAT_USER_THRESHOLD = 3

MOVIE_NAME = "Shiva Shankara Vara Prasad"
HERO = "Chiranjeevi"
DIRECTOR = "Anil Ravipudi"

VIDEO_CLASSIFIERS = {
    "Title / Announcement": ["title", "glimpse", "announcement", "first look"],
    "Song": ["song", "lyrical", "audio", "music"],
    "Teaser": ["teaser"],
    "Trailer": ["trailer"],
    "Interview / Press": ["interview", "press", "meet"],
    "Review / Public Talk": ["review", "public"]
}

# ================= HELPERS =================

def normalize_sentiment(label):
    return "Positive" if "pos" in label.lower() else "Negative"

def classify_video(title):
    t = title.lower()
    for k, v in VIDEO_CLASSIFIERS.items():
        if any(x in t for x in v):
            return k
    return "Other"

def normalize_language(text):
    try:
        lang = detect(text)
        return {
            "te": "Telugu",
            "en": "English",
            "hi": "Hindi"
        }.get(lang, "Mixed / Roman")
    except:
        return "Unknown"

def get_youtube():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ================= VIDEO SEARCH =================

def search_movie_videos(youtube):
    query = f"{MOVIE_NAME} {HERO} {DIRECTOR}"
    videos = []

    request = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=50,
        relevanceLanguage="te",
        regionCode="IN"
    )

    while request and len(videos) < MAX_VIDEOS:
        response = request.execute()

        for item in response.get("items", []):
            vid = item["id"]["videoId"]
            title = item["snippet"]["title"]

            videos.append({
                "video_id": vid,
                "video_title": title,
                "video_type": classify_video(title),
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "published_at": item["snippet"]["publishedAt"]
            })

        request = youtube.search().list_next(request, response)
        time.sleep(0.2)

    return list({v["video_id"]: v for v in videos}.values())

# ================= COMMENTS =================

def fetch_comments(youtube, video):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video["video_id"],
            maxResults=MAX_COMMENTS,
            textFormat="plainText"
        )

        while request:
            response = request.execute()
            for item in response.get("items", []):
                s = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": s.get("authorDisplayName"),
                    "comment": s.get("textDisplay"),
                    "published_at": s.get("publishedAt"),
                    "video_title": video["video_title"],
                    "video_type": video["video_type"],
                    "video_url": video["video_url"]
                })

            request = youtube.commentThreads().list_next(request, response)

    except:
        # Comments disabled / restricted
        pass

    return comments

# ================= MAIN ENGINE =================

def run_intelligence():
    youtube = get_youtube()
    sentiment_model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    videos = search_movie_videos(youtube)

    all_comments = []
    for v in videos:
        all_comments.extend(fetch_comments(youtube, v))

    results = []
    user_map = defaultdict(list)
    stage_stats = defaultdict(lambda: {"total": 0, "negative": 0})
    language_stats = defaultdict(lambda: {"total": 0, "negative": 0})
    song_stats = defaultdict(lambda: {"total": 0, "negative": 0})
    spikes = Counter()

    for c in tqdm(all_comments, desc="Analyzing"):
        try:
            out = sentiment_model(c["comment"][:512])[0]
            sentiment = normalize_sentiment(out["label"])
            language = normalize_language(c["comment"])

            hour = datetime.fromisoformat(
                c["published_at"].replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:00")

            row = {**c, "sentiment": sentiment, "language": language}
            results.append(row)
            user_map[c["author"]].append(row)

            stage_stats[c["video_type"]]["total"] += 1
            language_stats[language]["total"] += 1

            if sentiment == "Negative":
                stage_stats[c["video_type"]]["negative"] += 1
                language_stats[language]["negative"] += 1
                spikes[hour] += 1

                if c["video_type"] == "Song":
                    song_stats[c["video_title"]]["negative"] += 1

            if c["video_type"] == "Song":
                song_stats[c["video_title"]]["total"] += 1

        except:
            pass

    attack_users = [
        {
            "author": u,
            "negative_comments": sum(1 for r in rows if r["sentiment"] == "Negative"),
            "stages_targeted": list(set(r["video_type"] for r in rows))
        }
        for u, rows in user_map.items()
        if sum(1 for r in rows if r["sentiment"] == "Negative") >= REPEAT_USER_THRESHOLD
    ]

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "movie": MOVIE_NAME,
        "hero": HERO,
        "director": DIRECTOR,
        "instances": {
            "total_mentions": len(results),
            "negative_mentions": sum(1 for r in results if r["sentiment"] == "Negative")
        },
        "sentiment_by_stage": stage_stats,
        "language_distribution": language_stats,
        "song_analysis": song_stats,
        "negative_spikes": dict(spikes),
        "attack_coordination": attack_users,
        "comments": results
    }

    with open("latest_intelligence.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output

if __name__ == "__main__":
    run_intelligence()
    print("✅ Intelligence generated")

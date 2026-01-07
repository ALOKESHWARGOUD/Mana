# intelligence.py
# FULL MOVIE-LEVEL YouTube Intelligence Engine
# Hourly automation | .env | Attack detection | Spike alerts

from dotenv import load_dotenv
load_dotenv()

import os, time, json
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
    "Interview": ["interview", "press", "meet"],
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
    except:
        return "Unknown"

    if lang in ["te"]:
        return "Telugu"
    if lang in ["en"]:
        return "English"
    if lang in ["hi"]:
        return "Hindi"
    return "Roman / Mixed"

def get_youtube():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def load_sentiment():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

# ================= VIDEO SEARCH =================

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

# ================= COMMENTS =================

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

# ================= ATTACK DETECTION =================

def detect_attack_coordination(user_map):
    attackers = []
    for user, rows in user_map.items():
        neg = [r for r in rows if r["sentiment"] == "Negative"]
        if len(neg) >= REPEAT_USER_THRESHOLD:
            stages = set(r["video_type"] for r in neg)
            if len(stages) >= 2:
                attackers.append({
                    "author": user,
                    "negative_comments": len(neg),
                    "stages_targeted": list(stages),
                    "videos_targeted": len(set(r["video_title"] for r in neg))
                })
    return attackers

def detect_spike(spikes):
    if len(spikes) < 3:
        return None
    values = list(spikes.values())
    avg = sum(values[:-1]) / max(1, len(values[:-1]))
    last_key = list(spikes.keys())[-1]
    last_val = spikes[last_key]
    if last_val > avg * 2:
        return {
            "date": last_key,
            "count": last_val,
            "average": round(avg, 2),
            "severity": "HIGH"
        }
    return None

# ================= MAIN =================

def run_intelligence():
    yt = get_youtube()
    sentiment_model = load_sentiment()

    videos = search_movie_videos(yt)

    all_comments = []
    for v in videos:
        all_comments.extend(fetch_comments(yt, v))

    results, user_map = [], defaultdict(list)
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

    attack_users = detect_attack_coordination(user_map)
    spike_alert = detect_spike(spikes)

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "movie": MOVIE_NAME,
        "hero": HERO,
        "director": DIRECTOR,

        "instances": {
            "total_mentions": len(results),
            "negative_mentions": sum(1 for r in results if r["sentiment"] == "Negative")
        },

        "negative_spikes": dict(spikes),
        "real_time_alert": spike_alert,
        "sentiment_by_stage": stage_stats,
        "language_distribution": language_stats,
        "song_analysis": song_stats,
        "attack_coordination": attack_users,
        "comments": results,
        "videos": videos
    }

    with open("latest_intelligence.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output

if __name__ == "__main__":
    run_intelligence()
    print("✅ Intelligence updated")
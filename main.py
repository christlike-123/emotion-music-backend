import os
import time
import random
import traceback
import requests
import io

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras import layers
from keras.utils import register_keras_serializable
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# === Custom Attention Layer ===
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(scale, mode='fan_avg', distribution='uniform')

@register_keras_serializable()
class Attention(layers.Layer):
    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        super().__init__(**kwargs)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        scale = tf.cast(self.units, tf.float32) ** -0.5
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        attn_score = tf.matmul(q, k, transpose_b=True) * scale
        attn_score = tf.nn.softmax(attn_score, axis=-1)
        context_vector = tf.matmul(attn_score, v)
        context_vector = self.proj(context_vector)
        return inputs + context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# === Face Detection ===
def detect_and_crop_face(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return pil_image
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)

# === FastAPI App ===
app = FastAPI()

model = load_model("facemodel.keras", custom_objects={"Attention": Attention})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# === Spotify API Setup ===
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "c37a556373604e48a727e92549d859fc")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "bef55abea06246c5b8c9ece20aed32ec")

emotion_cache = {}

def get_spotify_token():
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10
        )
        token_data = resp.json()
        if not token_data or "access_token" not in token_data:
            print("[SPOTIFY] Invalid token response:", token_data)
            return None
        return token_data["access_token"]
    except Exception as e:
        print(f"[SPOTIFY TOKEN ERROR] {e}")
        return None

def get_tracks_by_emotion(emotion):
    if emotion in emotion_cache:
        return emotion_cache[emotion]

    token = get_spotify_token()
    if token is None:
        print("[ERROR] Failed to get Spotify token.")
        return []

    headers = {"Authorization": f"Bearer {token}"}

    try:
        search = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params={"q": emotion, "type": "playlist", "limit": 5},
            timeout=10
        )
        data = search.json()
        playlists = data.get("playlists", {}).get("items", [])
        if not playlists:
            print(f"[SPOTIFY] No playlists found for emotion: {emotion}")
            return []
    except Exception as e:
        print(f"[SPOTIFY SEARCH ERROR] {e}")
        return []

    all_tracks = set()

    for playlist in playlists:
        playlist_id = playlist.get("id")
        if not playlist_id:
            print("[SPOTIFY] Invalid playlist object.")
            continue

        try:
            r = requests.get(
                f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
                headers=headers,
                params={"limit": 50},
                timeout=10
            )
            items = r.json().get("items", [])
            for item in items:
                track = item.get("track")
                if track and track.get("external_urls", {}).get("spotify"):
                    all_tracks.add(track["external_urls"]["spotify"])
            time.sleep(0.3)
        except Exception as e:
            print(f"[SPOTIFY TRACK FETCH ERROR] {e}")
            continue

    track_list = list(all_tracks)
    if not track_list:
        print(f"[SPOTIFY] No tracks found for emotion: {emotion}")
        return []

    emotion_cache[emotion] = random.sample(track_list, min(20, len(track_list)))
    return emotion_cache[emotion]

# === Emotion Detection Endpoint ===
@app.post("/emotion-music")
async def detect_and_recommend(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = detect_and_crop_face(img).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        pred = model.predict(img_array)
        if np.sum(pred) == 0:
            raise ValueError("Model returned all zero probabilities.")

        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        detected_emotion = emotions[np.argmax(pred)]

        tracks = get_tracks_by_emotion(detected_emotion)
        if not tracks:
            return {"emotion": detected_emotion, "tracks": [], "error": "No music tracks found."}

        return {"emotion": detected_emotion, "tracks": tracks}
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return {"emotion": "undefined", "tracks": [], "error": str(e)}

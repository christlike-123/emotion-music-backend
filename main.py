import os
import time
import random
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
from tensorflow.keras.applications.mobilenet import preprocess_input
import httpx  # NEW: used instead of requests

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return pil_image  # fallback if no face is detected

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl, a, b))
    face_clahe = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)

    return Image.fromarray(face_clahe)

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

# === Spotify API Setup ===

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "c37a556373604e48a727e92549d859fc")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "bef55abea06246c5b8c9ece20aed32ec")

emotion_cache = {}

# NEW: Async Spotify Functions

async def get_spotify_token():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10.0
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

async def get_tracks_by_emotion(emotion):
    if emotion in emotion_cache:
        return emotion_cache[emotion]

    token = await get_spotify_token()
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        search = await client.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params={"q": emotion, "type": "playlist", "limit": 3},
        )
        search.raise_for_status()
        playlists = search.json().get("playlists", {}).get("items", [])

        all_tracks = set()

        for playlist in playlists:
            playlist_id = playlist["id"]
            try:
                r = await client.get(
                    f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
                    headers=headers,
                    params={"limit": 30},
                )
                r.raise_for_status()
                for item in r.json().get("items", []):
                    track = item.get("track")
                    if track and "external_urls" in track:
                        all_tracks.add(track["external_urls"]["spotify"])
            except Exception as e:
                print(f"Error fetching playlist {playlist_id}: {e}")
                continue

    track_list = list(all_tracks)
    emotion_cache[emotion] = random.sample(track_list, min(10, len(track_list)))
    return emotion_cache[emotion]

# === Emotion Detection Endpoint ===

@app.post("/emotion-music")
async def detect_and_recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = detect_and_crop_face(img).resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0]
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    detected_emotion = emotions[np.argmax(pred)]
    confidence = float(np.max(pred))

    tracks = await get_tracks_by_emotion(detected_emotion)

    return {
        "emotion": detected_emotion,
        "confidence": confidence,
        "tracks": tracks
    }

# === Ping Route ===

@app.get("/ping")
async def ping():
    return {"status": "ok"}

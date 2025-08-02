import os
import time
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras import layers
from keras.utils import register_keras_serializable
from PIL import Image
import numpy as np
import io
import requests
import tensorflow as tf
import cv2

# === Custom Attention Layer ===

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(
        scale, mode='fan_avg', distribution='uniform'
    )

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
        return pil_image  # fallback if no face is detected

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)

# === FastAPI App ===

app = FastAPI()

# Load your face emotion recognition model
model = load_model("facemodel.keras", custom_objects={"Attention": Attention})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Root Route ===

@app.get("/")
def read_root():
    return {"message": "Emotion Music Backend is running."}

# === Spotify API Setup ===

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "c37a556373604e48a727e92549d859fc")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "bef55abea06246c5b8c9ece20aed32ec")

emotion_cache = {}

def get_spotify_token():
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    token = resp.json().get("access_token")
    if not token:
        print("[SPOTIFY TOKEN ERROR]", resp.json())
    return token

def get_tracks_by_emotion(emotion):
    if emotion in emotion_cache:
        return emotion_cache[emotion]

    token = get_spotify_token()
    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}"}
    search = requests.get(
        "https://api.spotify.com/v1/search",
        headers=headers,
        params={"q": emotion, "type": "playlist", "limit": 5},
    )
    playlists = search.json().get("playlists", {}).get("items", [])

    all_tracks = set()

    for playlist in playlists:
        playlist_id = playlist.get("id")
        if not playlist_id:
            continue

        try:
            r = requests.get(
                f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
                headers=headers,
                params={"limit": 50},
            )
            items = r.json().get("items", [])
            for item in items:
                track = item.get("track")
                if track and "external_urls" in track:
                    url = track["external_urls"]["spotify"]
                    all_tracks.add(url)
            time.sleep(0.3)  # avoid rate limiting
        except Exception as e:
            print(f"[ERROR FETCHING PLAYLIST {playlist_id}]:", e)

    track_list = list(all_tracks)
    if track_list:
        sampled = random.sample(track_list, min(20, len(track_list)))
    else:
        sampled = []

    emotion_cache[emotion] = sampled
    return sampled

# === Emotion Detection Endpoint ===

@app.post("/emotion-music")
async def detect_and_recommend(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = detect_and_crop_face(img).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        pred = model.predict(img_array)
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        detected_emotion = emotions[np.argmax(pred)]

        tracks = get_tracks_by_emotion(detected_emotion)
        return {"emotion": detected_emotion, "tracks": tracks}
    except Exception as e:
        print("[ERROR]", e)
        return {"emotion": "undefined", "tracks": []}

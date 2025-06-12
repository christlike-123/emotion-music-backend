import os
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

# === FastAPI App ===

app = FastAPI()

# Load your face emotion recognition model
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

def get_spotify_token():
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    return resp.json()["access_token"]

def get_tracks_by_emotion(emotion):
    token = get_spotify_token()
    headers = {"Authorization": f"Bearer {token}"}
    search = requests.get(
        "https://api.spotify.com/v1/search",
        headers=headers,
        params={"q": emotion, "type": "playlist", "limit": 1},
    )
    playlists = search.json().get("playlists", {}).get("items", [])
    if not playlists:
        return []
    playlist_id = playlists[0]["id"]
    tracks = requests.get(
        f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
        headers=headers,
    ).json()
    return [
        item["track"]["external_urls"]["spotify"]
        for item in tracks["items"] if item.get("track")
    ]

# === API Endpoint ===

@app.post("/emotion-music")
async def detect_and_recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    detected_emotion = emotions[np.argmax(pred)]
    tracks = get_tracks_by_emotion(detected_emotion)
    return {"emotion": detected_emotion, "tracks": tracks}

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.utils import register_keras_serializable
from PIL import Image
import numpy as np
import io
import requests
import tensorflow as tf

# Register custom Attention layer
@register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, units=1024, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units, activation="tanh")

    def call(self, inputs):
        scores = tf.nn.softmax(self.dense(inputs), axis=1)
        context = scores * inputs
        return tf.reduce_sum(context, axis=1)

app = FastAPI()

# Load your face emotion recognition model
model = load_model("facemodel.keras", custom_objects={"Attention": Attention})

# Allow all CORS (adjust if needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Spotify credentials (secure in production)
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
    return [item["track"]["external_urls"]["spotify"] for item in tracks["items"] if item.get("track")]

@app.post("/emotion-music")
async def detect_and_recommend(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((48, 48))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))
    pred = model.predict(img_array)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    detected_emotion = emotions[np.argmax(pred)]
    tracks = get_tracks_by_emotion(detected_emotion)
    return {"emotion": detected_emotion, "tracks": tracks}

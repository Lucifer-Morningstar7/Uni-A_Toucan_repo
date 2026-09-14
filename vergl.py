# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:28:33 2026

@author: gahnluca
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition



# 1. Laden des vorbereiteten ECAPA-TDNN Modells von Hugging Face
print("Lade ECAPA-TDNN Modell...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def get_embedding(audio_path):
    """Lädt eine Audiodatei und extrahiert das hochdimensionale Vektor-Embedding."""
    # SpeechBrain übernimmt automatisch das Resampling auf 16kHz, falls nötig
    signal, fs = verification_model.load_audio(audio_path)
    
    # Embedding extrahieren (Form: [1, 1, 192] -> wir reduzieren es auf [192])
    with torch.no_grad():
        embedding = verification_model.encode_batch(signal).squeeze()
    return embedding

# 2. Pfade zu deinen Audiodateien (Beispiel)
path_reference = r"C:\Users\gahnluca\Downloads\Statements von Olaf Scholz, Sanna Marin und Magdalena Andersson (mp3cut.net).wav"
path_test_user = r"C:\Users\gahnluca\Downloads\Qwen3_schwedin_v4_gut.wav"

# 3. Embeddings extrahieren
emb_reference = get_embedding(path_reference)
emb_test_user = get_embedding(path_test_user)

# 4. Kosinus-Ähnlichkeit (Cosine Similarity) berechnen
# F.cosine_similarity erwartet Tensoren mit einer Batch-Dimension, daher unsqueeze(0)
similarity = F.cosine_similarity(emb_reference.unsqueeze(0), emb_test_user.unsqueeze(0))

# Der Wert liegt zwischen -1 und 1. Wir holen uns die reine Zahl.
ass_score = similarity.item()

print("\n--- Ergebnis ---")
print(f"Accent Similarity Score (ASS): {ass_score:.4f}")

# Optionale Skalierung auf 0 bis 100% für den Nutzer
ass_percentage = max(0.0, ass_score) * 100
print(f"Ähnlichkeit in Prozent: {ass_percentage:.2f}%")
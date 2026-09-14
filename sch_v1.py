import torch
# Wir importieren die Klasse aus der Datei, die du mir gerade geschickt hast
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# 1. Hier wird die Klasse initialisiert.
# Durch dein "schwedin_v1" sucht sie automatisch in:
# Models/ToucanTTS_schwedin_v1/best.pt
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Schwedin_V1",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/minist_schnipsel/schnipsel_001.wav")

# 2. Definiere deine Test-Sätze
meine_texte = [
    "Das ist ein Test mit meinem eigenen, frisch trainierten Modell.",
    "Ich bin gespannt, wie gut die schwedische Stimme klingt!"
]

ausgabe_pfad = "mein_ergebnis.wav"
print("Generiere Audio... Bitte warten.")

# 3. Rufe die Methode auf, die du mir oben geschickt hast
# Diese generiert das Audio und speichert es ab.
tts.read_to_file(text_list=meine_texte, file_location=ausgabe_pfad)

print("Erfolgreich! Die Datei mein_ergebnis.wav wurde erstellt.")
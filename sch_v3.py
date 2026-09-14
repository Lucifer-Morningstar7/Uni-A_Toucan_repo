import torch
# Wir importieren die Klasse aus der Datei, die du mir gerade geschickt hast
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# 1. Hier wird die Klasse initialisiert.
# Durch dein "schwedin_v1" sucht sie automatisch in:
# Models/ToucanTTS_schwedin_v1/best.pt
deutsch_frontend = ArticulatoryCombinedTextFrontend(language="de")

tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Schwedin_V1",
    language="swe"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/minist_schnipsel/schnipsel_001.wav")

# 2. Definiere deine Test-Sätze
deutscher_text = "Hallo, ich spreche Deutsch mit einem sehr schönen schwedischen Akzent."
print("Wandle Text in schwedische IPA-Lautschrift um...")
phoneme_liste = deutsch_frontend.string_to_phoneme_list(deutscher_text)
deutsche_phoneme = " ".join(phoneme_liste)

ausgabe_pfad = "schwedischer_aktzent_v3.wav"
print("Generiere Audio... Bitte warten.")

# 3. Rufe die Methode auf, die du mir oben geschickt hast
# Diese generiert das Audio und speichert es ab.
tts.read_to_file(text_list=[deutsche_phoneme], file_location=ausgabe_pfad, input_is_phones=True, pitch_variance_scale=1.5, duration_scaling_factor=1.15)

print("Erfolgreich! Die Datei mein_ergebnis.wav wurde erstellt.")
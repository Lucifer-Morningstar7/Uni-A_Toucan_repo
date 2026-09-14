import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# 1. Alles rein auf Deutsch aufbauen (für flüssige, menschliche Übergänge)
deutsch_frontend = ArticulatoryCombinedTextFrontend(language="de")
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/minist_schnipsel/schnipsel_001.wav")

# KEIN tts.set_accent_language("swe") -> Das zerstört die Satzmelodie!

deutscher_text = "Sonne, Suppe und ein großes Auto."
print(f"Generiere Text mit natürlichem Akzent-Filter...")

with torch.no_grad():
    # Saubere deutsche Phoneme holen
    phone_string = deutsch_frontend.get_phone_string(deutscher_text)
    print(f"Original: {phone_string}")
    
    # SYSTEMATISCHER SCHWEDISCHER FILTER (Vokale verschieben, Struktur erhalten)
    # Wir ersetzen o durch u, g durch k, etc., lassen aber Pausen (~) intakt
    akzent_string = (phone_string
                     .replace("ɔ", "u")  # offenes o zu u
                     .replace("o", "u")  # geschlossenes o zu u
                     .replace("ɡ", "k")  # g zu k
                     .replace("x", "s")) # ch-Laut zu s
    
    # Das nervige End-Geräusch abschneiden, falls vorhanden
    if akzent_string.endswith("~#"):
        akzent_string = akzent_string[:-2]
        
    print(f"Modifiziert: {akzent_string}")
    
    # Tensor erstellen
    phones_vector = deutsch_frontend.string_to_tensor(akzent_string).to(tts.device)
    
    # Generierung (Reines deutsches Modell -> steuert flüssige Übergänge)
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        return_duration_pitch_energy=True
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

sf.write("schwedischer_akzent_menschlich.wav", wav_tensor, 24000)
print("Fertig! Audiodatei generiert.")
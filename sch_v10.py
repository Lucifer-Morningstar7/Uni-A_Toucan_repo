import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# 1. Interface im DEUTSCHEN Modus laden
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/wahlstrom_2-enhanced-v2 (mp3cut.net).wav")

# Phonemizer bleibt fest auf Deutsch
tts.set_phonemizer_language("deu")

de_lang_id = tts.lang_id.clone()
tts.set_accent_language("swe")
swe_lang_id = tts.lang_id.clone()

# Hier steuerst du die Stärke: 
# 0.0 = reines Deutsch (kein Akzent)
# 1.0 = reines Schwedisch (maximaler Akzent)
accent_strength = 0.35 

with torch.no_grad():
    # 2. Wir holen die echten Repräsentations-Vektoren aus dem Modell
    de_emb = tts.tts.encoder.language_embedding(de_lang_id.to(tts.device))
    swe_emb = tts.tts.encoder.language_embedding(swe_lang_id.to(tts.device))
    
    # 3. Wir mischen die Vektoren mathematisch (jetzt völlig legal, da es Floats sind)
    blended_emb = (accent_strength * swe_emb) + ((1.0 - accent_strength) * de_emb)
    
    # 4. Wir überschreiben den Tabelleneintrag von Deutsch fliegend mit unserem Mix
    de_idx = de_lang_id.item()
    tts.tts.encoder.language_embedding.weight.data[de_idx] = blended_emb.squeeze()

# 5. Dem Interface sagen, dass es die deutsche ID aufrufen soll 
# (hinter der jetzt unser perfekt dosierter Akzent-Mix liegt)
tts.lang_id = de_lang_id.to(tts.device)
# --------------------------------------------

print("Phonemizer-Sprache:", tts.text2phone.language)
print(f"Gemischter Akzent aktiv (Stärke: {accent_strength})")

deutscher_text = "Ich habe heute Morgen einen Spaziergang im Park gemacht und die frische Luft genossen."
ausgabe_pfad = "schwedischer_akzent_new_wahlstrom_v1.wav"

print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    phone_string = tts.text2phone.get_phone_string(deutscher_text)
    phones_vector = tts.text2phone.string_to_tensor(deutscher_text).to(tts.device)
    
    # HIER AUCH DIE PROSODIE ETWAS DEZENTER SETZEN:
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        duration_scaling_factor=1.03,  # Nur noch minimal langsamer (vorher 1.1)
        pitch_variance_scale=1.15,     # Weniger extremen Singsang (vorher 1.3)
        energy_variance_scale=1.0,
        return_duration_pitch_energy=True
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

# Audio zuschneiden und speichern
trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Datei unter '{ausgabe_pfad}' gespeichert.")
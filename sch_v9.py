import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# 1. Wir laden das Interface komplett im DEUTSCHEN Modus ("deu")
# Dadurch greift das Modell auf die flüssige, menschliche deutsche Satzmelodie zu.
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/wahlstrom_2-enhanced-v2 (mp3cut.net).wav")

# 2. DER SYSTEM-TRICK: Wir nutzen das eingebaute Akzent-Feature von Toucan.
# Das sorgt dafür, dass die Artikulationsmerkmale (z.B. weicheres S, Vokalverschiebung)
# automatisch schwedisch eingefärbt werden, OHNE dass der Vektor zerbricht.
tts.set_phonemizer_language("deu")
tts.set_accent_language("swe")

print("Phonemizer-Sprache:", tts.text2phone.language)
print("Aktive Sprach-ID (Akzent):", tts.lang_id)

# Wir nutzen normalen Text, damit das Frontend die Wortgrenzen mathematisch perfekt berechnet.
deutscher_text = "Ich habe heute Morgen einen Spaziergang im Park gemacht und die frische Luft genossen."
ausgabe_pfad = "schwedischer_akzent_new_wahlstrom_v1.wav"


print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    # text2phone holt nun die Phoneme, das Modell kombiniert sie mit dem "swe"-Akzentvektor.
    # Da das Interface oben nicht direkt aufrufbar war, nutzen wir die korrekte Methode:
    phone_string = tts.text2phone.get_phone_string(deutscher_text)
    print("IPA Phoneme:",phone_string)
    phones_vector = tts.text2phone.string_to_tensor(deutscher_text).to(tts.device)

    print("lang_id:", tts.lang_id)
    print("phonemizer language:", tts.text2phone.language)
    
    # WICHTIG: lang_id bleibt auf tts.lang_id (wird durch set_accent_language gesteuert)
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        duration_scaling_factor=0.9,   # Normales Tempo für flüssigen Sprachfluss
        pitch_variance_scale=1.1,      # Etwas mehr Melodie gegen Monotonie (Sweet Spot)
        energy_variance_scale=1.15,    # Mehr Lautstärke-Dynamik (Menschen sprechen nicht immer gleich laut)
        return_duration_pitch_energy=True
    )

    print("Aktuell aktive lang_id im Interface (Rohwert):", tts.lang_id)
    
    # Der Vocoder berechnet das flüssige, menschliche Audio
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()


trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Die Datei wurde unter '{ausgabe_pfad}' gespeichert.")
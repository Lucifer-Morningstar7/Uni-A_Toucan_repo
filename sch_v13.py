import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# 1. ToucanTTS Interface laden
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/opfergang2-enhanced-v2 (mp3cut.net).wav")
tts.set_phonemizer_language("deu")

# 2. PHONETISCH OPTIMIERTER TEXT (Ellen Ammann / Reichsschwedisch):
# Wir nutzen Lautmalerei, damit Toucan flüssige Vektoren ohne Roboter-Artefakte baut.
deutscher_text_schwedisch = (
    "Asch -- nennen sie misch ruhig Ell-se, wie die anderen das Tün. "
    "Der Naame ist für misch hoüte eine liebe Errr-innerung. "
    "Den haabe isch von meiner Mutterrr."
)

ausgabe_pfad = "schwedischer_akzent_new_opfergang_v3.wav"

print(f"Generiere Text: '{deutscher_text_schwedisch}'...")

with torch.no_grad():
    # Toucan baut aus dem Text saubere Phoneme
    phones_vector = tts.text2phone.string_to_tensor(deutscher_text_schwedisch).to(tts.device)
    
    # 3. PROSODIE FÜR OBERSCHICHTEN-SINGSANG:
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        duration_scaling_factor=1.32,  # Leicht getragen / vornehm artikuliert
        pitch_variance_scale=1.35,     # DER SCHWEDISCHE SINGSANG! (Wichtig gegen "osteuropäischen" Sound)
        energy_variance_scale=1.0,
        return_duration_pitch_energy=True
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

# Trimmen und Speichern
trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Audio wurde unter '{ausgabe_pfad}' gespeichert.")
import torch
import soundfile as sf
import re
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# 1. Interface laden
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)

tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/opfergang2-enhanced-v2 (mp3cut.net).wav")
tts.set_phonemizer_language("deu")

deutscher_text = "Ich will mir nicht bei allem überlegen, ob ich es darf, oder ob ich es nicht darf."
ausgabe_pfad = "schwedischer_akzent_änderung_v2.wav"

# --- SCHWEDISCHE SPRACH-ID ERMITTELN ---
swe_lang_key = None
if hasattr(tts, "lang2id"):
    for code in ["swe", "sv", "swedish"]:
        if code in tts.lang2id:
            swe_lang_key = code
            break

if swe_lang_key:
    swe_lang_id = torch.tensor([tts.lang2id[swe_lang_key]], device=tts.device)
    print(f"Schwedische Prosodie geladen: '{swe_lang_key}' (ID: {tts.lang2id[swe_lang_key]})")
else:
    swe_lang_id = tts.lang_id
    print("Sicherheits-Fallback: Schwedisch nicht in lang2id gefunden, verwende Standard-ID.")

# --- REGEL 2: Fehlen der Auslautverhärtung ---
text_fuer_phonemizer = re.sub(r'([bdgBDG])(?=\s|[.,!?~]|$)', r'\1e', deutscher_text)

print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    raw_ipa = tts.text2phone.get_phone_string(text_fuer_phonemizer)
    modified_ipa = raw_ipa
    
    # --- REGELN ANWENDEN ---
    modified_ipa = re.sub(r'([bdg])ə', r'\1', modified_ipa)
    modified_ipa = modified_ipa.replace('ʔ', '') 
    modified_ipa = modified_ipa.replace('ɐ', 'ɛr').replace('ɜ', 'ɛ')
    modified_ipa = modified_ipa.replace('uː', 'yː').replace('ʊ', 'ʏ') 
    modified_ipa = modified_ipa.replace('oː', 'uː').replace('ɔ', 'ʊ')
    modified_ipa = modified_ipa.replace('aː', 'ɑː')
    modified_ipa = modified_ipa.replace('z', 's')
    modified_ipa = modified_ipa.replace('v', 'w')
    modified_ipa = modified_ipa.replace('ʁ', 'r').replace('ʀ', 'r').replace('ɾ', 'r')
    modified_ipa = modified_ipa.replace('iː', 'iːj')

    modified_ipa = re.sub(r'(^|\s|~)ʃt', r'\1st', modified_ipa)
    modified_ipa = re.sub(r'(^|\s|~)ʃp', r'\1sp', modified_ipa)
    
    print("Original IPA (Deutsch):", raw_ipa)
    print("Schwedisch modifiziert: ", modified_ipa)
    
    phones_vector = tts.text2phone.string_to_tensor(modified_ipa, input_phonemes=True).to(tts.device)

    # PASS 1: Vorhersage der Standard-Durations
    _, predicted_durations, _, _ = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id,
        duration_scaling_factor=1.0, 
        pitch_variance_scale=1.45,
        return_duration_pitch_energy=True
    )
        
    my_durations = predicted_durations.clone()

    # --- SCHWEDISCHER RHYTHMUS-ALGORITHMUS ---
    if len(modified_ipa) == my_durations.shape[0]:
        vokale = ['a', 'e', 'i', 'o', 'u', 'y', 'ɑ', 'ɛ', 'ɪ', 'ʊ', 'ɔ', 'ʏ', 'ʉ', 'ɒ', 'œ', 'ø', 'æ']
        konsonanten = ['p','b','t','d','k','g','m','n','ŋ','l','r','f','v','w','s','z','ʃ','ʒ','j','h','ç','x', 'ɕ']

        for i, phonem in enumerate(modified_ipa):
            # Rollendes R
            if phonem == 'r':
                my_durations[i] = my_durations[i] * 1.8
                
            # Dehnung betonter Silben
            if phonem in vokale and i+1 < len(modified_ipa) and modified_ipa[i+1] == 'ː':
                my_durations[i] = my_durations[i] * 2.2
                my_durations[i+1] = my_durations[i+1] * 2.0
                
            # Dehnung von Folgekonsonanten nach kurzen Vokalen
            if phonem in konsonanten and i > 0 and modified_ipa[i-1] in vokale:
                my_durations[i] = my_durations[i] * 1.7
                    
            # J-Gleitlaut
            if phonem == 'j' and i > 0 and modified_ipa[i-1] == 'ː':
                my_durations[i] = my_durations[i] * 1.4
                
    my_durations = torch.round(my_durations).long()

    # PASS 2: Synthese mit modifizierten Durations.
    # WICHTIG: pitch & energy hier NICHT explizit übergeben, 
    # damit das Modell Pitch & Energy dynamisch an die neuen Längen anpasst!
    mel = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id,
        durations=my_durations,
        pitch_variance_scale=1.45, # Erhöhter Singsang bleibt erhalten
        return_duration_pitch_energy=False
    )
    
    # Vocoder Aufruf
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Die Datei wurde unter '{ausgabe_pfad}' gespeichert.")
import torch
import soundfile as sf
import re
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

print("Phonemizer-Sprache:", tts.text2phone.language)
print("Aktive Sprach-ID (Akzent):", tts.lang_id)

# Wir nutzen normalen Text, damit das Frontend die Wortgrenzen mathematisch perfekt berechnet.
deutscher_text = "Ich habe heute Morgen einen Spaziergang im Park gemacht und die frische Luft genossen."
ausgabe_pfad = "schwedischer_akzent_new_wahlstrom_v3.wav"


print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    # 1. Den deutschen Text in die reine Lautschrift (IPA) übersetzen lassen
    raw_ipa = tts.text2phone.get_phone_string(deutscher_text)
    
    # 2. LINGUISTISCHE MANIPULATION FÜR REICHSSCHWEDISCH
    modified_ipa = raw_ipa
    
    # Bereits vorhanden: Das rollende R, weiches S und klares -er
    modified_ipa = modified_ipa.replace('ʁ', 'r').replace('ʀ', 'r')
    modified_ipa = modified_ipa.replace('z', 's')
    modified_ipa = modified_ipa.replace('ɐ', 'ɛr')

    # NEU Regel 4: "st" und "sp" schwedisch aussprechen (S-tadt statt Schtadt)
    modified_ipa = modified_ipa.replace('ʃp', 'sp').replace('ʃt', 'st')

    # NEU Regel 5: Das deutsche "Z" wird oft als "S" gesprochen
    modified_ipa = modified_ipa.replace('ts', 's')

    # NEU Regel 6: Der "ich"-Laut wird zu "sch"
    modified_ipa = modified_ipa.replace('ç', 'ʃ')

    # NEU Regel 7: Überartikulation von Wortenden (hab-e statt hab-uh)
    modified_ipa = modified_ipa.replace('ə', 'ɛ')

    print("Original IPA:  ", raw_ipa)
    print("Schwedisch IPA:", modified_ipa)
    
    # 3. Den modifizierten IPA-String in einen Tensor umwandeln
    # Toucan versteht IPA-Zeichen direkt, wenn wir sie als String übergeben!
    phones_vector = tts.text2phone.string_to_tensor(modified_ipa, input_phonemes=True).to(tts.device)
    
    
    # 4. Prosodie an die historische Sprechweise (Oberschicht) anpassen
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        duration_scaling_factor=1,  # Etwas LANGSAMER. Oberschicht sprach damals sehr gewählt und artikuliert.
        pitch_variance_scale=1.25,     # HÖHER. Schwedisch hat den berühmten "Singsang" (Pitch Accent).
        energy_variance_scale=1.0,
        return_duration_pitch_energy=True
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()


trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Die Datei wurde unter '{ausgabe_pfad}' gespeichert.")
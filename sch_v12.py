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
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/opfergang2-enhanced-v2 (mp3cut.net).wav")

# 2. DER SYSTEM-TRICK: Wir nutzen das eingebaute Akzent-Feature von Toucan.
# Das sorgt dafür, dass die Artikulationsmerkmale (z.B. weicheres S, Vokalverschiebung)
# automatisch schwedisch eingefärbt werden, OHNE dass der Vektor zerbricht.
tts.set_phonemizer_language("deu")

print("Phonemizer-Sprache:", tts.text2phone.language)
print("Aktive Sprach-ID (Akzent):", tts.lang_id)

# Wir nutzen normalen Text, damit das Frontend die Wortgrenzen mathematisch perfekt berechnet.
deutscher_text = "Ach, ~ nennen sie mich ruhig Elße wie die anderen das Tun. Der Name ist für mich heute eine Liebe Erinnerung. Den habe ich von meiner Mutter."
ausgabe_pfad = "schwedischer_akzent_new_opfergang_v5.wav"


print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    raw_ipa = tts.text2phone.get_phone_string(deutscher_text)
    modified_ipa = raw_ipa
    
    # ---------------------------------------------------------
    # 1. RHOTICS (Das R-System) - Butterweiche Übergänge
    # ---------------------------------------------------------
    # Wir nutzen den Tap 'ɾ' statt dem harten Trill 'r'. 
    # Er klingt aristokratisch rollend, erzeugt aber keine Audio-Artefakte.
    modified_ipa = modified_ipa.replace('ʁ', 'ɾ').replace('ʀ', 'ɾ')
    
    # Endungen (-er) werden zu einem artikulierten, aber weichen -er
    modified_ipa = modified_ipa.replace('ɐ', 'ɛɾ')

    # ---------------------------------------------------------
    # 2. VOKALE
    # ---------------------------------------------------------
    # Schwedisches langes U (weit vorne im Mund gesprochen, fast wie ein Ü)
    # Aus "gut" wird akustisch eine Mischung aus "gut" und "güt"
    modified_ipa = modified_ipa.replace('uː', 'ʉː')
    
    # Überartikulation (Oberschicht): Das unbetonte "Schwa" wird klarer
    modified_ipa = modified_ipa.replace('ə', 'ɛ')

    # ---------------------------------------------------------
    # 3. ZISCHLAUTE & FRIKATIVE (Der nordische Charakter)
    # ---------------------------------------------------------
    # Deutsches "Z" aufweichen ("zur Zeit" -> "sur Seit")
    modified_ipa = modified_ipa.replace('t͡s', 's').replace('ts', 's')
    
    # Stimmhaftes S (Sonne) wird scharf (stimmlos)
    modified_ipa = modified_ipa.replace('z', 's')
    
    # Der weiche Tje-Laut. Schweden sagen nicht "i-ch", sondern fast "i-sch"
    modified_ipa = modified_ipa.replace('ç', 'ɕ')

    # ---------------------------------------------------------
    # 4. REGEX: SCHWEDISCHES S-T UND S-P (NUR AM WORTANFANG!)
    # ---------------------------------------------------------
    # Verhindert, dass Wörter wie "fris-che" oder "Fis-ch" kaputt gehen.
    # Wir suchen nach Leerzeichen (\s) oder dem Toucan-Wortanfang (~) 
    modified_ipa = re.sub(r'(^|\s|~)ʃt', r'\1st', modified_ipa)
    modified_ipa = re.sub(r'(^|\s|~)ʃp', r'\1sp', modified_ipa)
    
    print("Original IPA:  ", raw_ipa)
    print("Schwedisch IPA:", modified_ipa)
    
    # In Tensor konvertieren (input_phonemes=True ist Pflicht!)
    phones_vector = tts.text2phone.string_to_tensor(modified_ipa, input_phonemes=True).to(tts.device)

    # Prosodie: Oberschichten-Singsang
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        duration_scaling_factor=1,  
        pitch_variance_scale=1.12,      # Etwas höher gedreht für die Melodie
        energy_variance_scale=1.0,
        return_duration_pitch_energy=True
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()


trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Die Datei wurde unter '{ausgabe_pfad}' gespeichert.")
print(tts.lang_id)
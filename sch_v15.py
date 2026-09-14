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

# TIPP ZUM AUDIO: 1944er Film-Audio hat viel Rauschen/Reverb. 
# Das verwirrt das Modell oft und macht es abgehackt. Nutze ein KI-bereinigtes Sample!
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/opfergang2-enhanced-v2 (mp3cut.net).wav")

tts.set_phonemizer_language("deu")

# TEXT-OPTIMIERUNG: 
# 1. Der Name ist "Aels". 
# 2. Kommata hinzugefügt für natürliche Atempausen.
deutscher_text = "Ich will mir nicht bei allem überlegen, ob ich es darf, oder ob ich es nicht darf."
ausgabe_pfad = "schwedischer_akzent_new_opfergang_v11.wav"

print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    raw_ipa = tts.text2phone.get_phone_string(deutscher_text)
    modified_ipa = raw_ipa
    
   
    modified_ipa = modified_ipa.replace('ʁ', 'r').replace('ɾ', 'r')
    modified_ipa = modified_ipa.replace('ʔ', '')
    modified_ipa = modified_ipa.replace('ɐ', 'eɾ')
    modified_ipa = modified_ipa.replace('aː', 'ɑː')
    modified_ipa = modified_ipa.replace('eː', 'ɛː')
    modified_ipa = modified_ipa.replace('uː', 'ʉː')
    modified_ipa = modified_ipa.replace('z', 's')
    modified_ipa = modified_ipa.replace('ç', 'ɕ')
    modified_ipa = re.sub(r'(^|\s|~)ʃt', r'\1st', modified_ipa)
    modified_ipa = re.sub(r'(^|\s|~)ʃp', r'\1sp', modified_ipa)
    
    print("Original IPA:  ", raw_ipa)
    print("Schwedisch IPA:", modified_ipa)
    
    # In Tensor konvertieren
    phones_vector = tts.text2phone.string_to_tensor(modified_ipa, input_phonemes=True).to(tts.device)

    _, predicted_durations, predicted_pitch, predicted_energy = tts.phone2mel(
            phones_vector,
            utterance_embedding=tts.default_utterance_embedding,
            lang_id=tts.lang_id,
            duration_scaling_factor=1.05, # Ein ganz leichtes Basis-Tempo (nicht zu extrem)
            pitch_variance_scale=1.1,     # Starke Satzmelodie
            return_duration_pitch_energy=True
        )
        
    my_durations = predicted_durations.clone()

    # C) DER AUTO-DYNAMIK ALGORITHMUS
    # Prüfen, ob IPA-String und Tensor-Länge übereinstimmen (Sicherheitscheck)
    if len(modified_ipa) == my_durations.shape[0]:

        for i, phonem in enumerate(modified_ipa):
            if phonem == 'r':
                # 1.8x gibt der Zunge genug Zeit für das Schwingen/Rollen
                my_durations[i] = my_durations[i] * 1.8
        
        # Findet alle Wörter im IPA (ignoriert Pausenzeichen wie ~ , . ! ?)
        for match in re.finditer(r'[^\s~,.!?#]+', modified_ipa):
            word = match.group()
            start = match.start()
            end = match.end()
            
            # REGEL 1: Füllwörter (kurz) werden gehetzt/schneller (15% schneller)
            if len(word) <= 3:
                my_durations[start:end] = my_durations[start:end] * 0.85
            
            # REGEL 2: Inhaltswörter (lang) werden bedeutungsschwer (15% langsamer)
            elif len(word) >= 7:
                my_durations[start:end] = my_durations[start:end] * 1.15
            
            # REGEL 3: Die Dramatik-Pause (Pre-Pausal Lengthening)
            # Wenn direkt nach dem Wort ein Pausenzeichen (~ , . !) kommt:
            if end < len(modified_ipa) and modified_ipa[end] in ['~', ',', '.', '#', '!', '?']:
                # Überschreibt vorherige Regeln und dehnt das Wort stark (35% langsamer)!
                my_durations[start:end] = my_durations[start:end] * 1.35
                
    # Dauern runden, damit der Vocoder sie verarbeiten kann
    my_durations = torch.round(my_durations).long()

    # D) Mit berechneter Dynamik rendern
    mel, _, _, _ = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        durations=my_durations,
        return_duration_pitch_energy=True
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Die Datei wurde unter '{ausgabe_pfad}' gespeichert.")




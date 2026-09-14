import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

deutsch_frontend = ArticulatoryCombinedTextFrontend(language="de")

tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/minist_schnipsel/schnipsel_001.wav")


tts.set_accent_language("swe") 

deutscher_text = "Sonne, Suppe und ein großes Auto mit zähnen"
ausgabe_pfad = "schwedischer_akzent_natuerlich.wav"

print(f"Generiere Text: '{deutscher_text}' mit schwedischem Akzent...")

with torch.no_grad():
    phone_string = deutsch_frontend.get_phone_string(deutscher_text)
    
    if phone_string.endswith("~#"):
        phone_string = phone_string[:-2]
    elif phone_string.endswith("#"):
        phone_string = phone_string[:-1]
        
    print(f"[Genutzte Phoneme]: {phone_string}")

    # SCHRITT B: Wir wandeln sie über das deutsche Frontend in den passenden Tensor um
    phones_vector = deutsch_frontend.string_to_tensor(phone_string).to(tts.device)
    
    # SCHRITT C: Das schwedisch eingestellte Modell spricht die deutschen Phoneme aus
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,  # Nutzt jetzt die schwedische ID des Interfaces
        return_duration_pitch_energy=True
    )
    
    # Vocoder rendert die finale Waveform
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Die saubere Datei wurde unter '{ausgabe_pfad}' gespeichert.")
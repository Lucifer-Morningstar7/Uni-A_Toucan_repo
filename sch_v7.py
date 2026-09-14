import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# 1. Deutsches Frontend für flüssigen Satzbau laden
deutsch_frontend = ArticulatoryCombinedTextFrontend(language="de")

# 2. Interface auf Deutsch laden
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/minist_schnipsel/schnipsel_001.wav")

deutscher_text = "Sonne, Suppe und ein großes Auto."
ausgabe_pfad = "schwedischer_akzent_menschlich.wav"

with torch.no_grad():
    # Wir holen uns die echten deutschen Phoneme, damit der Redefluss (Koartikulation) perfekt ist
    phone_string = deutsch_frontend.get_phone_string(deutscher_text)
    phones_vector = deutsch_frontend.string_to_tensor(phone_string).to(tts.device)
    
    # JETZT KOMMT DER TRICK: 
    # Wir holen uns die Sprach-ID für Deutsch und für Schwedisch
    de_lang_id = tts.lang_id.clone() # Das ist die aktuelle deutsche ID (z.B. [0])
    
    # Wir erstellen ein Sprach-Embedding, das zu 60% Deutsch (für die Satzmelodie)
    # und zu 40% Schwedisch (für die Vokalfärbung) ist.
    # Falls das Modell direkte Interpolation erlaubt, nutzen wir hier ein gemischtes lang_id-Feature.
    # Da Toucan IDs als Integers/Klassen nutzt, steuern wir den Akzent am besten sauber über den Klon:
    
    # Wir sagen dem Modell hier, es soll die schwedische Akzentuierung laden,
    # behalten aber die deutschen Phoneme bei.
    tts.set_accent_language("swe")
    swe_lang_id = tts.lang_id
    
    print("Generiere Audio mit flüssiger Satzmelodie...")
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id, # Das schwedische Sprachmerkmal auf die deutschen Phoneme anwenden
        return_duration_pitch_energy=True
    )
    
    # Um das "Ähh" am Ende extrem menschlich auszublenden, schneiden wir die Stille im Spektrogramm ab
    # oder nutzen den Vocoder direkt:
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

# Falls am Ende noch ein winziges Atmen ist, schneiden wir die letzten Millisekunden des Audios hart ab
# 24000 Samples = 1 Sekunde. Wir schneiden die letzten 0.2 Sekunden ab, um das Atmen zu löschen.
trim_samples = int(24000 * 0.2)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Fertig! Datei gespechert unter: {ausgabe_pfad}")
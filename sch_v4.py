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


schwedischer_akzent_filter = {
    "o": "u",     
    "ɡx": "k",    
    "pə": "ppə"   
}

deutscher_text = "Sonne, Suppe und ein großes Auto."

print("Schritt 1: Wandle Text in deutsche Standard-Phoneme um...")
original_phonem_string = deutsch_frontend.get_phone_string(deutscher_text)

print("Schritt 2: Wende den schwedischen Akzent-Filter an...")
akzent_phonem_string = original_phonem_string
for alt, neu in schwedischer_akzent_filter.items():
    akzent_phonem_string = akzent_phonem_string.replace(alt, neu)

print(f"\n[Original-Phoneme]: {original_phonem_string}")
print(f"[Akzent-Phoneme]:   {akzent_phonem_string}\n")

ausgabe_pfad = "schwedischer_aktzent_v5.wav"
print("Generiere Audio mit Akzent... Bitte warten.")

with torch.no_grad():
    # Wandle die modifizierten Phoneme in den passenden Tensor um
    phones_vector = deutsch_frontend.string_to_tensor(akzent_phonem_string).to(tts.device)
    
    # KORREKTUR: lang_id übergeben und return_duration_pitch_energy aktivieren!
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        return_duration_pitch_energy=True
    )
    
    # Der Vocoder macht daraus die finale Waveform (mit Batch-Dimension)
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

# 2. Abspeichern der fertigen Wave-Datei
sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Erfolgreich! Die Akzent-Datei {ausgabe_pfad} wurde erstellt.")
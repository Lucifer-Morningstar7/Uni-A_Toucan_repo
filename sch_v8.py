import torch
import soundfile as sf
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# 1. Deutsches Frontend laden
deutsch_frontend = ArticulatoryCombinedTextFrontend(language="de")

# 2. Interface rein auf Deutsch laden (für maximal menschlichen Redefluss)
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="deu"
)
tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/minist_schnipsel/schnipsel_001.wav")

with torch.no_grad():
    # MENSCHLICHER IPA-AKZENT-TRICK:
    # Wir bauen einen Phonem-String, den das deutsche Modell perfekt und flüssig 
    # aussprechen kann, der aber phonetisch den schwedischen Akzent imitiert.
    # "Sonne" -> 'sunə' (mit kurzem u, flüssig eingebettet)
    # "Suppe" -> 'zupə' (Schweden sprechen das S oft weich/stimmhaft an)
    # "und ein" -> 'unt aɪn'
    # "großes Auto" -> 'kuːsəs aʊtuː' (das lange, geschlossene 'o' wird zum skandinavischen 'u')
    
    # Wir setzen die Pausenmarker (~), damit das Modell weiß, wo die Wörter natürlich fließen.
    akzent_ipa = "~sɔnə~zʊpə~ʊnt~aɪn~ɡʁoːsəs~aʊtuː~"
    
    print(f"Generiere flüssigen IPA-Satz: {akzent_ipa}")
    
    # Wandle die manipulierten, aber flüssigen deutschen Phoneme in den Vektor um
    phones_vector = deutsch_frontend.string_to_tensor(akzent_ipa).to(tts.device)
    
    # Reines deutsches Modell berechnet die menschliche Melodie
    mel, durations, pitch, energy = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=tts.lang_id,
        return_duration_pitch_energy=True
    )
    
    # Vocoder generiert das Audio
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

# Abschneiden von Artefakten am Ende (0.15 Sekunden Stille entfernen)
trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write("schwedischer_akzent_flüsig_v2.wav", wav_tensor, 24000)
print("Fertig! Die menschlich klingende Akzent-Datei wurde erstellt.")
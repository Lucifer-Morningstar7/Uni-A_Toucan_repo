import torch
import soundfile as sf
import re

# 1. Imports aller Frameworks
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from rvc_python.infer import RVCInference
from qwen_tts import Qwen3TTSModel

# Pfade und Konfigurationen
TEXT = "Ich will mir nicht bei allem überlegen, ob ich es darf, oder ob ich es nicht darf."
REF_AUDIO_TOUCAN = "/home/gahnluca/audio_src/wahlstrom_2-enhanced-v2_(mp3cut.net).wav"

TOUCAN_WAV_PATH = "1_toucan_output.wav"
RVC_WAV_PATH = "2_rvc_output.wav"
QWEN_WAV_PATH = "3_qwen_final_output.wav"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# STUFE 1: ToucanTTS (Prosodie & Phonetik)
# ==========================================
print("\n--- [1/3] Starte ToucanTTS Generierung ---")

tts = ToucanTTSInterface(
    device=DEVICE,
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="swe"
)

tts.set_utterance_embedding(path_to_reference_audio=REF_AUDIO_TOUCAN)
tts.set_phonemizer_language("swe")

swe_lang_key = None
if hasattr(tts, "lang2id"):
    for code in ["swe", "sv", "swedish"]:
        if code in tts.lang2id:
            swe_lang_key = code
            break

swe_lang_id = torch.tensor([tts.lang2id[swe_lang_key]], device=tts.device) if swe_lang_key else tts.lang_id

# Text transformationen
text_fuer_phonemizer = re.sub(r'([bdgBDG])(?=\s|[.,!?~]|$)', r'\1e', TEXT)

with torch.no_grad():
    raw_ipa = tts.text2phone.get_phone_string(text_fuer_phonemizer)
    modified_ipa = raw_ipa

    # Schwedische IPA-Regeln
    modified_ipa = re.sub(r'([bdg])ə', r'\1', modified_ipa)
    modified_ipa = modified_ipa.replace('ʔ', '')

    modified_ipa = modified_ipa.replace('uː', 'ʉː').replace('oː', 'uː').replace('ɔ', 'ɵ').replace('aː', 'ɑː').replace('ɐ', 'ɛr').replace('ɜ', 'ɛ')
    modified_ipa = modified_ipa.replace('ç', 'ɕ').replace('z', 's').replace('v', 'w').replace('ʁ', 'r').replace('ʀ', 'r').replace('ɾ', 'r')
    modified_ipa = modified_ipa.replace('rt', 'ʈ').replace('rd', 'ɖ').replace('rn', 'ɳ').replace('rs', 'ʂ')
    modified_ipa = re.sub(r'(^|\s|~)ʃt', r'\1st', modified_ipa)
    modified_ipa = re.sub(r'(^|\s|~)ʃp', r'\1sp', modified_ipa)

    phones_vector = tts.text2phone.string_to_tensor(modified_ipa, input_phonemes=True).to(tts.device)

    # Durations vorhersagen
    _, predicted_durations, _, _ = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id,
        duration_scaling_factor=1.0, 
        pitch_variance_scale=1.45,
        return_duration_pitch_energy=True
    )
        
    my_durations = predicted_durations.clone()

    # Rhythmus-Anpassungen
    if len(modified_ipa) == my_durations.shape[0]:
        vokale = ['a', 'e', 'i', 'o', 'u', 'y', 'ɑ', 'ɛ', 'ɪ', 'ʊ', 'ɔ', 'ʏ', 'ʉ', 'ɵ', 'ɒ', 'œ', 'ø', 'æ']
        konsonanten = ['p','b','t','d','k','g','m','n','ŋ','l','r','f','v','w','s','z','ʃ','ʒ','j','h','ç','x','ɕ','ɧ','ʈ','ɖ','ɳ','ɭ','ʂ']
        total_len = len(modified_ipa)

        for i, phonem in enumerate(modified_ipa):
            dampening = 1.0
            if phonem in vokale and i + 1 < total_len and modified_ipa[i + 1] == 'ː':
                my_durations[i] = my_durations[i] * (1.8 * dampening)
                my_durations[i + 1] = my_durations[i + 1] * (1.5 * dampening)
            elif phonem == 'r':
                my_durations[i] = my_durations[i] * (1.35 * dampening)
            elif phonem in konsonanten and i > 0 and modified_ipa[i - 1] in vokale:
                my_durations[i] = my_durations[i] * (1.3 * dampening)
            elif phonem == 'j' and i > 0 and modified_ipa[i - 1] == 'ː':
                my_durations[i] = my_durations[i] * (1.2 * dampening)

            max_allowed = predicted_durations[i] * 2.0
            my_durations[i] = torch.min(my_durations[i], max_allowed)
            
    my_durations = torch.round(my_durations).long()

    # Synthese & Vocoder
    mel = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id,
        durations=my_durations,
        pitch_variance_scale=1.45,
        return_duration_pitch_energy=False
    )
    
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(TOUCAN_WAV_PATH, wav_tensor, 24000)
print(f"-> Toucan-Audio fertig: {TOUCAN_WAV_PATH}")


# ==========================================
# STUFE 2: RVC Voice Conversion
# ==========================================
print("\n--- [2/3] Starte RVC Voice Conversion ---")

rvc_model_pfad = "Pearl-SU-Multilanguage-Hybrid.pth"
rvc_index_pfad = "added_IVF140_Flat_nprobe_1_Pearl-SU-Multilanguage-Hybrid_v2.index"

rvc = RVCInference(
    device=DEVICE if DEVICE == "cpu" else "cuda:0",
    model_path=rvc_model_pfad,
    index_path=rvc_index_pfad
)

try:
    rvc.set_params(
        f0method="rmvpe",
        f0up_key=0,
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25
    )
except TypeError:
    pass

rvc.infer_file(
    input_path=TOUCAN_WAV_PATH,
    output_path=RVC_WAV_PATH
)
print(f"-> RVC-Audio fertig: {RVC_WAV_PATH}")


# ==========================================
# STUFE 3: Qwen3-TTS Voice Cloning
# ==========================================
print("\n--- [3/3] Starte Qwen3-TTS Voice Cloning ---")

# VRAM-Schonung: Toucan-Objekte vor Qwen-Laden löschen
del tts
del rvc
torch.cuda.empty_cache()

qwen_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0" if DEVICE != "cpu" else "cpu"
)

# Du kannst hier wählen, ob Qwen das RVC-Audio (RVC_WAV_PATH) 
# oder das reine Toucan-Audio (TOUCAN_WAV_PATH) als Referenz nutzt:
wavs, sr = qwen_model.generate_voice_clone(
    text=TEXT,
    language="german",
    ref_audio=RVC_WAV_PATH,
    ref_text=TEXT
)

sf.write(QWEN_WAV_PATH, wavs[0], sr)
print(f"-> Pipeline komplett! Ziel-Audio: {QWEN_WAV_PATH}")
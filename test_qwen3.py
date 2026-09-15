import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0"
)

# Voice Cloning mit Referenz-Audio
wavs, sr = model.generate_voice_clone(
    text="Ich will mir nicht bei allem überlegen, ob ich es darf, oder ob ich es nicht darf.",
    language="german",
    ref_audio="/home/gahnluca/audio_src/schwedischer_akzent_änderung_v6.wav",
    ref_text="Ich will mir nicht bei allem überlegen, ob ich es darf, oder ob ich es nicht darf."
)

sf.write("qwen_after_toucan_wahlström_v1.wav", wavs[0], sr)
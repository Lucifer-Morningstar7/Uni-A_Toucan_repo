from allosaurus.app import read_recognizer

model = read_recognizer()

audio_tts = "schwedischer_akzent_new_opfergang_v2.wav"
audio_referenz = "/home/gahnluca/audio_src/wahlstrom_2-enhanced-v2 (mp3cut.net).wav"

def get_phonemes_with_words(audio_path, pause_threshold=0.12):
    """
    Holt Phoneme inklusive Zeitstempeln und fügt bei Sprachpausen 
    Leerzeichen ein, um einzelne Wörter zu simulieren.
    """
    # emit_silence=True hilft Allosaurus, Stille zu erkennen
    detailed_output = model.recognize(audio_path, lang_id="deu", timestamp=True)
    
    words = []
    current_word = []
    last_end_time = 0.0

    for line in detailed_output.splitlines():
        if not line.strip():
            continue
            
        parts = line.split()
        if len(parts) >= 3:
            start_time = float(parts[0])
            duration = float(parts[1])
            phoneme = parts[2]
            
            end_time = start_time + duration
            
            # Wenn die Pause zwischen zwei Lauten größer als der Schwellenwert ist -> Neues Wort!
            if last_end_time > 0 and (start_time - last_end_time) > pause_threshold:
                if current_word:
                    words.append(" ".join(current_word))
                    current_word = []
            
            current_word.append(phoneme)
            last_end_time = end_time

    if current_word:
        words.append(" ".join(current_word))

    # Wörter mit einem " | " trennen
    return "  |  ".join(words)

print("--- Analysiere TTS-Audio (mit Worttrennung) ---")
print(get_phonemes_with_words(audio_tts))

print("\n--- Analysiere Referenz-Audio (mit Worttrennung) ---")
print(get_phonemes_with_words(audio_referenz))
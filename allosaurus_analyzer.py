from allosaurus.app import read_recognizer

# 1. Wir laden das universelle Allosaurus-Modell
print("Lade Allosaurus-Modell...")
model = read_recognizer()

# Definiere die Pfade zu deinen Audios
audio_tts = "schwedischer_akzent_new_opfergang_v2.wav"
audio_referenz = "/home/gahnluca/audio_src/opfergang2-enhanced-v2 (mp3cut.net).wav"

print("\n--- Analysiere TTS-Audio ---")
# lang_id="deu" zwingt das Modell, die Laute mit dem deutschen Alphabet abzugleichen. 
# Weglassen der lang_id würde alle weltweiten Laute erlauben (universelles IPA).
ipa_tts = model.recognize(audio_tts, lang_id="deu")
print("Gehörte Phoneme (TTS):")
print(ipa_tts)

print("\n--- Analysiere Referenz-Audio ---")
ipa_ref = model.recognize(audio_referenz, lang_id="deu")
print("Gehörte Phoneme (Original Wahlström):")
print(ipa_ref)

# --- BONUS: Detaillierte Analyse mit Timestamps ---
# Wenn du exakt wissen willst, WANN welches Phonem gesprochen wurde:
print("\n--- Detaillierte Analyse des TTS-Audios (Timestamps) ---")
ipa_detailed = model.recognize(audio_tts, lang_id="deu", timestamp=True)

# Ausgabe formatieren (Allosaurus liefert einen langen String mit Zeilenumbrüchen)
for line in ipa_detailed.split('\n')[:10]: # Wir zeigen nur die ersten 10 Laute
    print(line)
print("...")
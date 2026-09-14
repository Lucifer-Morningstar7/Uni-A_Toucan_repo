import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from faster_whisper import WhisperModel

# --- EINSTELLUNGEN ---
AUDIO_DATEI = r"C:\Users\gahnluca\Downloads\Audios\Saraschwedischpi_140-enhanced-v2.wav"  # Name deiner WAV-Datei
AUSGABE_ORDNER = "audio_schnipsel_sarah"
MIN_SILENCE_LEN = 1200   # Wie lange muss die Pause sein (in Millisekunden)?
SILENCE_THRESH = -40    # Ab wie viel Dezibel gilt es als "Stille"?

# Ordner für die Schnipsel erstellen
if not os.path.exists(AUSGABE_ORDNER):
    os.makedirs(AUSGABE_ORDNER)

print("⚡ Lade Audiodatei und Whisper-Modell (Deutsch)...")
audio = AudioSegment.from_wav(AUDIO_DATEI)

# Whisper Modell laden (Größe "base" ist schnell und gut für Deutsch)
# Nutzt CPU standardmäßig, für GPU-Nutzung 'device="cuda"' setzen
model = WhisperModel("medium", device="cpu", compute_type="int8")

print("✂️ Schneide Audio anhand von Sprechpausen...")
# Schneidet das Audio. Keep_silence lässt ein kurzes Atmen am Ende/Anfang, damit es natürlicher klingt
chunks = split_on_silence(
    audio, 
    min_silence_len=MIN_SILENCE_LEN, 
    silence_thresh=SILENCE_THRESH, 
    keep_silence=300
)

print(f" Es wurden {len(chunks)} Schnipsel gefunden. Starte Transkription...\n")

# Textdatei für das Gesamt-Transkript erstellen
with open("transkript_uebersicht_sarah.txt", "w", encoding="utf-8") as f_txt:
    
    for i, chunk in enumerate(chunks):
        dauer_sek = len(chunk) / 1000.0
        
        chunk_name = os.path.join(AUSGABE_ORDNER, f"schnipsel_{i+1:03d}.wav")
        chunk.export(chunk_name, format="wav")
        
        # Vollständigen, absoluten Pfad zur WAV-Datei ermitteln
        absoluter_pfad = os.path.abspath(chunk_name)
        
        # Schnipsel transkribieren
        segments, info = model.transcribe(chunk_name, language="de", beam_size=5)
        text = "".join([segment.text for segment in segments]).strip()
        
        # 1. Formatierung für die Textdatei: "pfad zu wav datei" : "transkript"
        datei_zeile = f'"{absoluter_pfad}" : "{text}"'
        f_txt.write(datei_zeile + "\n")
        
        # 2. Konsolenausgabe (etwas schöner zu lesen während des Wartens)
        print(f"-> {os.path.basename(chunk_name)} ({dauer_sek:.1f}s): {text}")

print("\n Fertig! Alle Schnipsel sind im Ordner 'audio_schnipsel' und das Transkript liegt in 'transkript_uebersicht_sarah.txt'.")
import torch
import soundfile as sf
import re
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# --- NEU: Import für Voice Conversion ---
from rvc_python.infer import RVCInference
# ----------------------------------------

# 1. Interface laden
tts = ToucanTTSInterface(
    device="cuda" if torch.cuda.is_available() else "cpu",
    tts_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/ToucanTTS.pt",
    vocoder_model_path="Models/models--Flux9665--ToucanTTS/snapshots/e0afe0ef703d2178dd7dc74ec298693ddb10e720/Vocoder.pt",
    language="swe"
)

tts.set_utterance_embedding(path_to_reference_audio="/home/gahnluca/audio_src/wahlstrom_2-enhanced-v2_(mp3cut.net).wav")
tts.set_phonemizer_language("swe")
tts.set_accent_language("deu")


deutscher_text = "Ich will mir nicht bei allem überlegen, ob ich es darf, oder ob ich es nicht darf."
ausgabe_pfad = "schwedischer_akzent_änderung_v6.wav"

# --- SCHWEDISCHE SPRACH-ID ERMITTELN ---
swe_lang_key = None
if hasattr(tts, "lang2id"):
    for code in ["swe", "sv", "swedish"]:
        if code in tts.lang2id:
            swe_lang_key = code
            break

if swe_lang_key:
    swe_lang_id = torch.tensor([tts.lang2id[swe_lang_key]], device=tts.device)
    print(f"Schwedische Prosodie geladen: '{swe_lang_key}' (ID: {tts.lang2id[swe_lang_key]})")
else:
    swe_lang_id = tts.lang_id
    print("Sicherheits-Fallback: Schwedisch nicht in lang2id gefunden, verwende Standard-ID.")

# --- REGEL 2: Fehlen der Auslautverhärtung ---
text_fuer_phonemizer = re.sub(r'([bdgBDG])(?=\s|[.,!?~]|$)', r'\1e', deutscher_text)

print(f"Generiere Text: '{deutscher_text}'...")

with torch.no_grad():
    raw_ipa = tts.text2phone.get_phone_string(text_fuer_phonemizer)
    modified_ipa = raw_ipa

    # --- SCHWEDISCHE IPA-ZEICHEN & REGELN ---
    modified_ipa = re.sub(r'([bdg])ə', r'\1', modified_ipa)
    modified_ipa = modified_ipa.replace('ʔ', '')  # Knacklaut entfernen

    # 1. Schwedische Vokalverschiebungen & Sonderzeichen
    modified_ipa = modified_ipa.replace('uː', 'ʉː')  # Geschlossenes, gerundetes schwedisches U
    modified_ipa = modified_ipa.replace('oː', 'uː')  # Deutsches O klingt im Schwedischen oft wie U
    modified_ipa = modified_ipa.replace('ɔ', 'ɵ')    # Schwedisches kurzes o/u
    modified_ipa = modified_ipa.replace('aː', 'ɑː')  # Offenes tiefes A
    modified_ipa = modified_ipa.replace('ɐ', 'ɛr').replace('ɜ', 'ɛ')

    # 2. Schwedische Konsonanten (Sj-Laut / Tj-Laut / Konsonantenwechsel)
    modified_ipa = modified_ipa.replace('ç', 'ɕ')    # 'ch' (ich) wird zum schwedischen Tj-Laut /ɕ/
    modified_ipa = modified_ipa.replace('z', 's')    # Kein stimmhaftes S im Schwedischen
    modified_ipa = modified_ipa.replace('v', 'w')
    modified_ipa = modified_ipa.replace('ʁ', 'r').replace('ʀ', 'r').replace('ɾ', 'r') # Zungen-R

    # 3. Retroflexion (Schweden verschmelzen 'r' + Dentalkonsonant)
    modified_ipa = modified_ipa.replace('rt', 'ʈ')  # r + t -> retroflexes t
    modified_ipa = modified_ipa.replace('rd', 'ɖ')  # r + d -> retroflexes d
    modified_ipa = modified_ipa.replace('rn', 'ɳ')  # r + n -> retroflexes n
    modified_ipa = modified_ipa.replace('rl', 'ɭ')  # r + l -> retroflexes l
    modified_ipa = modified_ipa.replace('rs', 'ʂ')  # r + s -> retroflexes s (sh-Klang)

    # 4. Anlaut-Anpassungen
    modified_ipa = re.sub(r'(^|\s|~)ʃt', r'\1st', modified_ipa)
    modified_ipa = re.sub(r'(^|\s|~)ʃp', r'\1sp', modified_ipa)

    print("Original IPA (Deutsch):", raw_ipa)
    print("Schwedisch modifiziert: ", modified_ipa)

    # Toucan wandelt diese IPA-Zeichen (inkl. ʉ, ɵ, ɕ, ʈ, ɖ etc.) automatisch um:
    phones_vector = tts.text2phone.string_to_tensor(modified_ipa, input_phonemes=True).to(tts.device)

    # PASS 1: Vorhersage der Standard-Durations
    _, predicted_durations, _, _ = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id,
        duration_scaling_factor=1.0, 
        pitch_variance_scale=1.45,
        return_duration_pitch_energy=True
    )
        
    my_durations = predicted_durations.clone()

    # --- SCHWEDISCHER RHYTHMUS-ALGORITHMUS ---
    if len(modified_ipa) == my_durations.shape[0]:
        vokale = ['a', 'e', 'i', 'o', 'u', 'y', 'ɑ', 'ɛ', 'ɪ', 'ʊ', 'ɔ', 'ʏ', 'ʉ', 'ɵ', 'ɒ', 'œ', 'ø', 'æ']
        konsonanten = ['p','b','t','d','k','g','m','n','ŋ','l','r','f','v','w','s','z','ʃ','ʒ','j','h','ç','x','ɕ','ɧ','ʈ','ɖ','ɳ','ɭ','ʂ']
        total_len = len(modified_ipa)

        for i, phonem in enumerate(modified_ipa):
            # Satzende-Schutz: Dämpfe Dehnungen in den letzten 4 Phonemen ab,
            # da ToucanTTS das Satzende ohnehin schon verlangsamt.
            #is_near_end = i >= (total_len - 4)
            #dampening = 0.55 if is_near_end else 1.0
            dampening = 1.0  # Keine Dämpfung, um den schwedischen Singsang zu erhalten
            # Priorisierte Regeln mit if / elif (verhindert Multiplikatoren-Stapelung)
            if phonem in vokale and i + 1 < total_len and modified_ipa[i + 1] == 'ː':
                my_durations[i] = my_durations[i] * (1.8 * dampening)
                my_durations[i + 1] = my_durations[i + 1] * (1.5 * dampening)

            elif phonem == 'r':
                my_durations[i] = my_durations[i] * (1.35 * dampening)

            elif phonem in konsonanten and i > 0 and modified_ipa[i - 1] in vokale:
                my_durations[i] = my_durations[i] * (1.3 * dampening)

            elif phonem == 'j' and i > 0 and modified_ipa[i - 1] == 'ː':
                my_durations[i] = my_durations[i] * (1.2 * dampening)

            # Sicherheits-Deckelung: Ein Phonem darf maximal doppelt so lang sein wie vom Modell vorhergesagt
            max_allowed = predicted_durations[i] * 2.0
            my_durations[i] = torch.min(my_durations[i], max_allowed)
            
    my_durations = torch.round(my_durations).long()

    # PASS 2: Synthese mit modifizierten Durations.
    # WICHTIG: pitch & energy hier NICHT explizit übergeben, 
    # damit das Modell Pitch & Energy dynamisch an die neuen Längen anpasst!
    mel = tts.phone2mel(
        phones_vector,
        utterance_embedding=tts.default_utterance_embedding,
        lang_id=swe_lang_id,
        durations=my_durations,
        pitch_variance_scale=1.45, # Erhöhter Singsang bleibt erhalten
        return_duration_pitch_energy=False
    )
    
    # Vocoder Aufruf
    wav_tensor = tts.vocoder(mel.unsqueeze(0)).cpu().squeeze().numpy()

trim_samples = int(24000 * 0.15)
if len(wav_tensor) > trim_samples:
    wav_tensor = wav_tensor[:-trim_samples]

sf.write(ausgabe_pfad, wav_tensor, 24000)
print(f"Toucan-Audio generiert! Gespeichert unter '{ausgabe_pfad}'")


# ==========================================
# --- NEU: RVC VOICE CONVERSION PIPELINE ---
# ==========================================
print("\n--- Starte Qualitäts-Upgrade mit RVC ---")

finale_ausgabe_pfad = "finale_high_quality_audio_v6.wav"
rvc_model_pfad = "Pearl-SU-Multilanguage-Hybrid.pth"
rvc_index_pfad = "added_IVF140_Flat_nprobe_1_Pearl-SU-Multilanguage-Hybrid_v2.index"

# 1. Device festlegen
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 2. RVC-Instanz direkt mit Modell- und Index-Pfad initialisieren
rvc = RVCInference(
    device=device,
    model_path=rvc_model_pfad,
    index_path=rvc_index_pfad
)

# 3. Parameter explizit über die Methode 'set_params' festlegen
# Hinweis: Falls ein Parameter von deiner Bibliothek nicht unterstützt wird, 
# kann er einfach aus dem Aufruf entfernt werden.
try:
    rvc.set_params(
        f0method="rmvpe",      # Beste Methode zur Tonhöhenerkennung
        f0up_key=0,            # Keine Tonhöhenverschiebung
        index_rate=0.75,       # Stärke des Ziel-Sprecher-Einflusses
        filter_radius=3,       # Median-Filter zur Glättung
        resample_sr=0,         # Keines erneutes Resampling
        rms_mix_rate=0.25      # Mischverhältnis der Lautstärke
    )
except TypeError:
    # Fallback, falls set_params nur eine Teilmenge der Argumente akzeptiert
    print("Hinweis: Verwende Standard-Parameter für set_params.")

# 4. Inferenz ausführen (entspricht exakt deiner Signatur)
rvc.infer_file(
    input_path=ausgabe_pfad,
    output_path=finale_ausgabe_pfad
)

print(f"RVC abgeschlossen! Das finale Audio liegt unter '{finale_ausgabe_pfad}'")
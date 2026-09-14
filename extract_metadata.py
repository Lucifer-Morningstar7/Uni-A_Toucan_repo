import os
import tarfile
import pandas as pd

# ---------------------------------------------------------------------------
# Pfade & Einstellungen
# ---------------------------------------------------------------------------
TAR_FILE_PATH = r"C:/Users/gahnluca/Downloads/1781721683073-cv-corpus-26.0-2026-06-12-de.tar.gz"
EXTRACT_DIR = "./cv_extracted"
OUTPUT_CSV = "gezielte_sprecherauswahl.csv"

# Suchbegriffe für Akzente
ACCENT_KEYWORDS = ["schwedisch", "swedish", "nordic", "scandinavian", "nordisch", "skandinavisch", "schweden", "sweden"]

# ---------------------------------------------------------------------------
# 1. Exakt validated.tsv entpacken
# ---------------------------------------------------------------------------
def extract_tsv_from_tar(tar_path, target_dir):
    print(f"Suche exakt 'validated.tsv' in {tar_path}...")
    
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Die Datei {tar_path} wurde nicht gefunden!")

    os.makedirs(target_dir, exist_ok=True)
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tsv_member = None
        for member in tar.getmembers():
            # Exakter Vergleich des Dateinamens (verhindert Treffer bei invalidated.tsv)
            if os.path.basename(member.name) == "validated.tsv":
                tsv_member = member
                break
        
        if tsv_member:
            print(f"Entpacke {tsv_member.name} nach {target_dir}...")
            tar.extract(tsv_member, path=target_dir)
            return os.path.join(target_dir, tsv_member.name)
        else:
            raise Exception("validated.tsv wurde im Archiv nicht gefunden.")

# ---------------------------------------------------------------------------
# 2. Daten filtern & analysieren
# ---------------------------------------------------------------------------
def filter_common_voice_metadata(tsv_path):
    print("Lade Metadaten in Pandas DataFrame...")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    print(f"Gesamtzahl der validated Einträge: {len(df)}")
    
    # Ermittle Spaltennamen für Geschlecht und Akzent
    gender_col = next((col for col in ["gender", "sex"] if col in df.columns), None)
    accent_col = next((col for col in ["accents", "accent"] if col in df.columns), None)
    
    print(f"Erkannte Spalten -> Geschlecht: '{gender_col}', Akzent: '{accent_col}'")
    
    # Übersicht der vorhandenen Werte zur Kontrolle
    if gender_col:
        print("\n--- Verteilung der Geschlechter (Top 5) ---")
        print(df[gender_col].value_counts(dropna=False).head(5))
        
    if accent_col:
        print("\n--- Verteilung der Akzente (Top 10) ---")
        print(df[accent_col].value_counts(dropna=False).head(10))

    # Filterung anwenden
    female_mask = df[gender_col].astype(str).str.lower().str.contains("female|weiblich", na=False) if gender_col else True
    
    accent_pattern = "|".join(ACCENT_KEYWORDS)
    accent_mask = df[accent_col].astype(str).str.lower().str.contains(accent_pattern, na=False) if accent_col else False
    
    filtered_df = df[female_mask & accent_mask].copy()
    print(f"\nTreffer für Filterkriterien: {len(filtered_df)} Aufnahmen")
    
    return filtered_df

# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Falls sich bereits die falsche (invalidated.tsv) im Zielordner befindet, löschen/überschreiben
        tsv_file_path = extract_tsv_from_tar(TAR_FILE_PATH, EXTRACT_DIR)
        
        df_result = filter_common_voice_metadata(tsv_file_path)
        
        if not df_result.empty:
            cols = [c for c in ["client_id", "path", "sentence", "gender", "sex", "accents", "accent"] if c in df_result.columns]
            df_result[cols].to_csv(OUTPUT_CSV, index=False, sep="\t")
            print(f"\nErfolg! {len(df_result)} Einträge in '{OUTPUT_CSV}' gespeichert.")
            print(f"Eindeutige Sprecherinnen: {df_result['client_id'].nunique()}")
        else:
            print("\nKeine Treffer mit den aktuellen Suchbegriffen.")
            print("Prüfen Sie oben in der Akzent-Übersicht, welche Schreibweisen im Datensatz verwendet werden.")
            
    except Exception as e:
        print(f"Fehler: {e}")
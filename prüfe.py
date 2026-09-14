import os
import tarfile
import pandas as pd

# ---------------------------------------------------------------------------
# Pfad zu Ihrer .tar-Datei
# ---------------------------------------------------------------------------
TAR_FILE_PATH = r"C:/Users/gahnluca/Downloads/1781721683073-cv-corpus-26.0-2026-06-12-de.tar.gz"
EXTRACT_DIR = "./cv_extracted"

def extract_and_analyze():
    # 1. Prüfen, ob die .tar-Datei existiert
    if not os.path.exists(TAR_FILE_PATH):
        print(f"FEHLER: Die Datei '{TAR_FILE_PATH}' existiert nicht. Bitte Pfad prüfen!")
        return

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    tsv_path = None

    # Suchen, ob die Datei schon mal entpackt wurde
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "validated.tsv" in files:
            tsv_path = os.path.join(root, "validated.tsv")
            print(f"validated.tsv bereits gefunden unter: {tsv_path}")
            break

    # 2. Entpacken aus der .tar (Modus "r:", da KEINE .gz Komprimierung)
    if not tsv_path:
        print(f"Öffne Archiv: {TAR_FILE_PATH} ...")
        with tarfile.open(TAR_FILE_PATH, "r:") as tar:
            tsv_member = None
            for member in tar.getmembers():
                if os.path.basename(member.name) == "validated.tsv":
                    tsv_member = member
                    break
            
            if tsv_member:
                print(f"Entpacke {tsv_member.name} nach {EXTRACT_DIR} ...")
                tar.extract(tsv_member, path=EXTRACT_DIR)
                tsv_path = os.path.join(EXTRACT_DIR, tsv_member.name)
            else:
                print("FEHLER: 'validated.tsv' konnte im Archiv nicht gefunden werden.")
                return

    # 3. Metadaten analysieren
    print("\nLade Metadaten in Pandas DataFrame...")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    accent_col = "accents" if "accents" in df.columns else "accent"
    gender_col = "gender" if "gender" in df.columns else "sex"

    print("=" * 60)
    print(f"Gesamtanzahl valider Clips: {len(df)}")
    print(f"Clips ohne Akzentangabe:    {df[accent_col].isna().sum()}")
    print("=" * 60)

    print("\n--- Alle im Datensatz vorhandenen Akzente (Top 100) ---")
    if accent_col in df.columns:
        pd.set_option('display.max_rows', 3000)
        print(df[accent_col].value_counts(dropna=True).head(3000))
    else:
        print("Spalte für Akzente nicht gefunden.")

if __name__ == "__main__":
    extract_and_analyze()
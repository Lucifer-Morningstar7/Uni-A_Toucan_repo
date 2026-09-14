from datasets import load_dataset

def is_swedish_accent(accent_field: str) -> bool:
    """
    Prüft, ob im Akzent-Feld Hinweise auf einen schwedischen Akzent vorliegen.
    Erfasst Varianten wie 'swedish', 'sweden', 'schwedisch', 'schweden' usw.
    """
    if not accent_field:
        return False
    
    accent_lower = accent_field.lower()
    keywords = ["swed", "schwed"]  # Erfasst Swedish, Sweden, Schwedisch, Schweden
    
    return any(keyword in accent_lower for keyword in keywords)

def main():
    print("Lade Common Voice 13.0 (Deutsch) im Streaming-Modus...")
    
    # streaming=True lädt Daten erst beim Iterieren herunter
    # Hinweis: Bei Hugging Face wird ggf. ein `token` (HuggingFace Access Token) benötigt.
    cv_de = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "de",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )


    # Dataset nach schwedischem Akzent filtern
    filtered_stream = cv_de.filter(lambda example: is_swedish_accent(example["accent"]))

    print("\nGefundene Aufnahmen mit schwedischem Akzent:\n" + "-"*50)
    
    count = 0
    max_samples_to_show = 5  # Wie viele Beispiele ausgegeben werden sollen

    for sample in filtered_stream:
        count += 1
        print(f"Sample #{count}")
        print(f"  Client ID: {sample['client_id']}")
        print(f"  Akzent:    {sample['accent']}")
        print(f"  Geschlecht:{sample['gender']}")
        print(f"  Alter:     {sample['age']}")
        print(f"  Text:      {sample['sentence']}")
        print(f"  Audio Path:{sample['path']}")
        print("-" * 50)
        
        if count >= max_samples_to_show:
            break

    if count == 0:
        print("Keine Samples mit schwedischem Akzent im aktuellen Split gefunden.")

if __name__ == "__main__":
    main()
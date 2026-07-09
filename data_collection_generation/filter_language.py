import csv
import re
import matplotlib.pyplot as plt
## Este codigo hace los siguiente remove_emojis detect_languages y crea histogramas
english_words = {"the", "and", "to", "of", "in", "is", "that", "it", "for", "on", "with"}
spanish_words = {"el", "la", "los", "las", "de", "que", "en", "un", "una", "por", "para", "con"}
french_words = {"le", "la", "les", "de", "et", "dans", "un", "une", "pour", "qui", "que", "sur", "à"}
german_words = {"der", "die", "das", "und", "in", "zu", "den", "auf", "für", "mit", "von", "ist"}

MAX_PROMPT_LENGTH = 2000

def remove_emojis(text):
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def detect_languages(text):
    text = text.lower()
    words = set(re.findall(r'\b\w+\b', text))
    
    langs = []
    if words.intersection(english_words): langs.append('English')
    if words.intersection(spanish_words): langs.append('Spanish')
    if words.intersection(french_words): langs.append('French')
    if words.intersection(german_words): langs.append('German')
    
    return langs

def main():
    input_file = 'combined_prompts_dataset.csv'
    output_file = 'filtered_prompts_dataset.csv'
    
    print(f"Reading from {input_file} and filtering...")
    
    counts = {'English': 0, 'Spanish': 0, 'French': 0, 'German': 0, 'Unknown/Other': 0, 'Multiple': 0}
    
    csv.field_size_limit(2147483647)
    
    benign_lengths = []
    malignant_lengths = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:
         
        reader = csv.reader(f_in, delimiter=';')
        writer = csv.writer(f_out, delimiter=';')
        
        try:
            header = next(reader)
            writer.writerow(header)
            
            prompt_idx = header.index('prompt') if 'prompt' in header else 0
            label_idx = header.index('label') if 'label' in header else -1
        except StopIteration:
            print("Empty file")
            return

        kept = 0
        total = 0
        
        for row in reader:
            total += 1
            if len(row) > prompt_idx:
                prompt_text = row[prompt_idx]
                
                if len(prompt_text) > MAX_PROMPT_LENGTH:
                    continue
                
                prompt_text = remove_emojis(prompt_text)
                row[prompt_idx] = prompt_text
                
                langs = detect_languages(prompt_text)
                
                if len(langs) == 1:
                    counts[langs[0]] += 1
                elif len(langs) > 1:
                    counts['Multiple'] += 1
                else:
                    counts['Unknown/Other'] += 1
                
                if len(langs) > 0:
                    writer.writerow(row)
                    kept += 1
                    
                    if label_idx != -1 and len(row) > label_idx:
                        label_val = row[label_idx].strip().lower()
                        if label_val in ['good', 'benign', 'safe', '0']:
                            benign_lengths.append(len(prompt_text))
                        else:
                            malignant_lengths.append(len(prompt_text))
                            
    print(f"\nDone. Processed {total} rows. Kept {kept} rows. Saved to {output_file}.")
    print("Language Counts (Burdo):")
    for lang, count in counts.items():
        print(f"  {lang}: {count}")
        
    # Crear histogramas
    if benign_lengths or malignant_lengths:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(benign_lengths, bins=50, color='green', alpha=0.7)
        plt.title("Tamaño Prompts Benignos")
        plt.xlabel("Longitud (caracteres)")
        plt.ylabel("Frecuencia")
        
        plt.subplot(1, 2, 2)
        plt.hist(malignant_lengths, bins=50, color='red', alpha=0.7)
        plt.title("Tamaño Prompts Malignos")
        plt.xlabel("Longitud (caracteres)")
        plt.ylabel("Frecuencia")
        
        plt.tight_layout()
        plt.savefig('prompts_histogram.png')
        print("\nHistogramas guardados en 'prompts_histogram.png'.")
        
if __name__ == '__main__':
    main()

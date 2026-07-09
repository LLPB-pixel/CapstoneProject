import os
import sys
import pandas as pd
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub", "pandas", "-q"])
    from datasets import load_dataset

BASE_DIR = Path(__file__).parent

# HF Config
HF_DATASETS = [
    {
        "name": "walledai/AdvBench",
        "prompt_col": "goal",
        "default_label": "bad"
    },
    {
        "name": "JasperLS/prompt-injections",
        "prompt_col": "text",
        "label_col": "label", # 0=benign(good), 1=injection(bad)
    },
    {
        "name": "walledai/JailbreakHub",
        "prompt_col": "prompt",
        "default_label": "bad"
    }
]

# Local Config
LOCAL_DATASETS = {
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/baseline_prompts.csv": {"sep": ";", "prompt_col": "Prompt", "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/teaching_prompts.csv": {"sep": ";", "prompt_col": "Prompt", "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/dan.csv": {"sep": None, "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/persuasion.csv": {"sep": None, "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/baitnswitch.csv": {"sep": None, "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/history_management.csv": {"sep": None, "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/restorying.csv": {"sep": None, "default_label": "bad"},
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/scattershot.csv": {"sep": None, "default_label": "bad"},
    "advbench/harmful_behaviors.csv": {"sep": None, "prompt_col": "goal", "default_label": "bad"},
    "advbench/harmful_strings.csv": {"sep": None, "prompt_col": "goal", "default_label": "bad"},
    "jaqilbreakbench/harmful-behaviors.csv": {"sep": None, "prompt_col": "Goal", "default_label": "bad"},
    "jaqilbreakbench/benign-behaviors.csv": {"sep": None, "prompt_col": "Goal", "default_label": "good"},
    "jaqilbreakbench/judge-comparison.csv": {"sep": None, "default_label": "unknown"}
}

def autodetect_prompt_col(df):
    candidates = ["text", "prompt", "goal", "Goal", "Prompt", "instruction", "question", "content"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_label(val):
    if pd.isna(val):
        return "unknown"
    s = str(val).lower()
    if s in ("1", "1.0", "true", "harmful", "yes", "bad", "injection"):
        return "bad"
    if s in ("0", "0.0", "false", "benign", "no", "good"):
        return "good"
    return "unknown"

def main():
    all_prompts = []
    
    print("Recopilando datasets de HuggingFace...")
    for hf_cfg in HF_DATASETS:
        print(f" -> Cargando {hf_cfg['name']}...")
        try:
            ds = load_dataset(hf_cfg["name"], split="train", trust_remote_code=True)
            df = ds.to_pandas()
            p_col = hf_cfg.get("prompt_col")
            if not p_col:
                p_col = autodetect_prompt_col(df)
            
            if p_col and p_col in df.columns:
                for _, row in df.iterrows():
                    prompt_text = str(row[p_col])
                    
                    label = "unknown"
                    if "label_col" in hf_cfg and hf_cfg["label_col"] in df.columns:
                        label = parse_label(row[hf_cfg["label_col"]])
                    elif "default_label" in hf_cfg:
                        label = hf_cfg["default_label"]
                        
                    all_prompts.append({"prompt": prompt_text, "label": label, "source": hf_cfg["name"]})
        except Exception as e:
            print(f"    Error cargando {hf_cfg['name']}: {e}")

    print("\nRecopilando datasets locales...")
    for rel_path, local_cfg in LOCAL_DATASETS.items():
        print(f" -> Cargando {rel_path}...")
        full_path = BASE_DIR / rel_path
        if not full_path.exists():
            print(f"    No se encontró el archivo: {rel_path}")
            continue
            
        try:
            if local_cfg.get("sep"):
                df = pd.read_csv(full_path, sep=local_cfg["sep"], on_bad_lines="skip", engine="python", quoting=0)
            else:
                try:
                    df = pd.read_csv(full_path, on_bad_lines="skip")
                except UnicodeDecodeError:
                    df = pd.read_csv(full_path, encoding="latin-1", on_bad_lines="skip")
                    
            p_col = local_cfg.get("prompt_col")
            if not p_col:
                p_col = autodetect_prompt_col(df)
                
            if p_col and p_col in df.columns:
                for _, row in df.iterrows():
                    prompt_text = str(row[p_col])
                    label = local_cfg.get("default_label", "unknown")
                    all_prompts.append({"prompt": prompt_text, "label": label, "source": rel_path})
            else:
                print(f"    No se encontró columna de prompt en {rel_path}")
        except Exception as e:
            print(f"    Error cargando {rel_path}: {e}")

    print("\nCreando dataset unificado...")
    df_all = pd.DataFrame(all_prompts)
    
    good_count = len(df_all[df_all["label"] == "good"])
    bad_count = len(df_all[df_all["label"] == "bad"])
    unknown_count = len(df_all[df_all["label"] == "unknown"])
    total_count = len(df_all)
    
    print("==================================================")
    print(f"Total de datos procesados: {total_count}")
    print(f" -> Prompts Buenos (Good): {good_count}")
    print(f" -> Prompts Malos (Bad): {bad_count}")
    if unknown_count > 0:
        print(f" -> Prompts Desconocidos: {unknown_count}")
    print("==================================================")
    
    output_file = BASE_DIR / "combined_prompts_dataset.csv"
    # Exporting with a clear separator
    separator = ";"
    df_all.to_csv(output_file, sep=separator, index=False, encoding="utf-8")
    
    print(f"Dataset guardado en: {output_file.name}")
    print(f"Formato: CSV")
    print(f"Separador: Punto y coma ('{separator}')")
    print(f"Columnas: {', '.join(df_all.columns)}")

if __name__ == "__main__":
    main()

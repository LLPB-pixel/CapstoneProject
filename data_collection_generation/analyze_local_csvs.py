"""
╔══════════════════════════════════════════════════════════════════╗
║        ANÁLISIS LOCAL DE PROMPTS — CSVs DEL PROYECTO            ║
╚══════════════════════════════════════════════════════════════════╝
Analiza todos los CSVs locales del directorio CapstoneProject y
muestra estadísticas ricas sobre los prompts que contienen.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from collections import Counter
import textwrap
import warnings

warnings.filterwarnings("ignore")

# ── Colores ANSI ────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BG_BLUE= "\033[44m"
    BG_RED = "\033[41m"

def header(text: str, width: int = 72, color: str = C.CYAN):
    bar = "═" * width
    print(f"\n{color}{C.BOLD}╔{bar}╗")
    print(f"║  {text:<{width-2}}║")
    print(f"╚{bar}╝{C.RESET}")

def section(text: str, color: str = C.YELLOW):
    print(f"\n{color}{C.BOLD}▶ {text}{C.RESET}")

def bar_chart(label: str, value: int, total: int, width: int = 35, color: str = C.GREEN):
    pct = value / total * 100 if total > 0 else 0
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {color}{bar}{C.RESET}  {C.BOLD}{label:<30}{C.RESET}  "
          f"{C.WHITE}{value:>4}{C.DIM} ({pct:5.1f}%){C.RESET}")

def mini_bar(val: int, max_val: int, width: int = 20) -> str:
    filled = int(val / max_val * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)

def wrap_text(text: str, width: int = 68, indent: str = "    ") -> str:
    return textwrap.fill(str(text), width=width, initial_indent=indent,
                         subsequent_indent=indent)

# ════════════════════════════════════════════════════════════════════
# CARGA DE DATASETS
# ════════════════════════════════════════════════════════════════════

BASE = Path(__file__).parent

DATASET_META = {
    # path_relativa → {alias, prompt_col, category_col, label_col}
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/baseline_prompts.csv": {
        "alias": "Baseline Prompts (LLM Red-Teaming)",
        "sep": ";",
        "prompt_col": "Prompt",
        "category_col": "Harm Type",
        "label_col": "Harmful Output",
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/teaching_prompts.csv": {
        "alias": "Teaching Prompts (LLM Red-Teaming)",
        "sep": ";",
        "prompt_col": "Prompt",
        "category_col": "Type of prompt",
        "label_col": None,
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/dan.csv": {
        "alias": "DAN Prompts",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/persuasion.csv": {
        "alias": "Persuasion Prompts",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/baitnswitch.csv": {
        "alias": "Bait & Switch Prompts",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/history_management.csv": {
        "alias": "History Management Prompts",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/restorying.csv": {
        "alias": "Restorying Prompts",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
    "LLM-red-teaming-prompts-main/LLM-red-teaming-prompts-main/red-teaming-prompts/scattershot.csv": {
        "alias": "Scattershot Prompts",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
    "advbench/harmful_behaviors.csv": {
        "alias": "AdvBench — Harmful Behaviors",
        "sep": None,
        "prompt_col": "goal",
        "category_col": None,
        "label_col": None,
    },
    "advbench/harmful_strings.csv": {
        "alias": "AdvBench — Harmful Strings",
        "sep": None,
        "prompt_col": "goal",
        "category_col": None,
        "label_col": None,
    },
    "jaqilbreakbench/harmful-behaviors.csv": {
        "alias": "JailbreakBench — Harmful Behaviors",
        "sep": None,
        "prompt_col": "Goal",
        "category_col": "Category",
        "label_col": None,
    },
    "jaqilbreakbench/benign-behaviors.csv": {
        "alias": "JailbreakBench — Benign Behaviors",
        "sep": None,
        "prompt_col": "Goal",
        "category_col": "Category",
        "label_col": None,
    },
    "jaqilbreakbench/judge-comparison.csv": {
        "alias": "JailbreakBench — Judge Comparison",
        "sep": None,
        "prompt_col": None,
        "category_col": None,
        "label_col": None,
    },
}


def load_df(rel_path: str, sep: str | None) -> pd.DataFrame | None:
    full = BASE / rel_path
    if not full.exists():
        return None
    try:
        if sep:
            df = pd.read_csv(full, sep=sep, on_bad_lines="skip",
                             engine="python", quoting=0)
        else:
            try:
                df = pd.read_csv(full, on_bad_lines="skip")
            except UnicodeDecodeError:
                df = pd.read_csv(full, encoding="latin-1", on_bad_lines="skip")
        return df
    except Exception as e:
        print(f"  {C.RED}⚠ Error: {e}{C.RESET}")
        return None


def prompt_length_stats(series: pd.Series) -> dict:
    lengths = series.dropna().astype(str).apply(len)
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
    }


def top_words(series: pd.Series, n: int = 10) -> list[tuple[str, int]]:
    STOP = {"a", "an", "the", "to", "of", "and", "or", "in", "for",
            "on", "is", "it", "how", "that", "be", "are", "with",
            "by", "you", "can", "i", "this", "do", "from", "your",
            "as", "me", "my", "what", "make", "write", "create",
            "provide", "give", "develop", "explain", "describe",
            "help", "without"}
    words = []
    for text in series.dropna().astype(str):
        for w in text.lower().split():
            w = w.strip(".,!?\"';:()[]")
            if len(w) > 3 and w not in STOP:
                words.append(w)
    return Counter(words).most_common(n)


def analyze_dataset(rel_path: str, meta: dict):
    alias = meta["alias"]
    sep = meta.get("sep")
    prompt_col = meta.get("prompt_col")
    cat_col = meta.get("category_col")
    label_col = meta.get("label_col")

    header(f"📄  {alias}", color=C.BLUE)

    df = load_df(rel_path, sep)
    if df is None:
        print(f"  {C.RED}❌ Archivo no encontrado: {rel_path}{C.RESET}")
        return

    # ── Info general ────────────────────────────────────────────────
    section("Información general")
    print(f"  {C.WHITE}Ruta:{C.RESET}     {C.DIM}{rel_path}{C.RESET}")
    print(f"  {C.WHITE}Filas:{C.RESET}    {C.GREEN}{C.BOLD}{len(df)}{C.RESET}")
    print(f"  {C.WHITE}Columnas:{C.RESET} {C.CYAN}{', '.join(df.columns.tolist())}{C.RESET}")

    # ── Valores nulos ───────────────────────────────────────────────
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(f"  {C.YELLOW}Valores nulos: {nulls[nulls > 0].to_dict()}{C.RESET}")

    # ── Análisis de prompts ─────────────────────────────────────────
    # Autodetectar columna de prompt si no está especificada
    if prompt_col is None:
        for candidate in ["text", "prompt", "goal", "Goal", "Prompt",
                          "instruction", "question", "content"]:
            if candidate in df.columns:
                prompt_col = candidate
                break

    if prompt_col and prompt_col in df.columns:
        prompts = df[prompt_col].dropna().astype(str)
        section(f"Estadísticas de prompts  [{prompt_col}]")

        stats = prompt_length_stats(df[prompt_col])
        print(f"  Total prompts:   {C.GREEN}{C.BOLD}{len(prompts)}{C.RESET}")
        print(f"  Longitud mínima: {C.CYAN}{stats['min']}{C.RESET} chars")
        print(f"  Longitud máxima: {C.CYAN}{stats['max']}{C.RESET} chars")
        print(f"  Media:           {C.CYAN}{stats['mean']:.1f}{C.RESET} chars")
        print(f"  Mediana:         {C.CYAN}{stats['median']:.0f}{C.RESET} chars")

        # Prompts de muestra
        section("Ejemplos de prompts (primeros 3)")
        for i, p in enumerate(prompts.head(3), 1):
            print(f"  {C.MAGENTA}{i}.{C.RESET} {wrap_text(p)}")

        # Top palabras
        section("Palabras más frecuentes (excluyendo stop words)")
        top = top_words(df[prompt_col])
        max_count = top[0][1] if top else 1
        for word, count in top:
            bar = mini_bar(count, max_count, 20)
            print(f"  {C.YELLOW}{bar}{C.RESET}  {C.WHITE}{word:<20}{C.RESET}  "
                  f"{C.CYAN}{count}{C.RESET}")

    # ── Categorías ──────────────────────────────────────────────────
    if cat_col and cat_col in df.columns:
        section(f"Distribución por categoría  [{cat_col}]")
        counts = df[cat_col].value_counts()
        total = len(df)
        for cat, cnt in counts.items():
            bar_chart(str(cat), cnt, total, color=C.MAGENTA)

    # ── Etiquetas (Harmful/Not Harmful) ─────────────────────────────
    if label_col and label_col in df.columns:
        section(f"Distribución de etiquetas  [{label_col}]")
        counts = df[label_col].value_counts()
        total = len(df)
        for lbl, cnt in counts.items():
            color = C.RED if str(lbl).lower() in ("true", "1", "harmful") else C.GREEN
            bar_chart(str(lbl), cnt, total, color=color)

    # ── Vista previa tabular ─────────────────────────────────────────
    section("Vista previa (primeras 3 filas)")
    preview_cols = df.columns[:5].tolist()  # máx 5 cols para no saturar
    preview = df[preview_cols].head(3)
    for col in preview.select_dtypes(include="object").columns:
        preview[col] = preview[col].astype(str).apply(
            lambda x: x[:60] + "…" if len(x) > 60 else x
        )
    print(preview.to_string(index=False))


# ════════════════════════════════════════════════════════════════════
# RESUMEN GLOBAL
# ════════════════════════════════════════════════════════════════════

def global_summary(loaded: list[tuple[str, pd.DataFrame, dict]]):
    header("🌐  RESUMEN GLOBAL DE TODOS LOS DATASETS", color=C.GREEN)
    total_files = len(loaded)
    total_rows  = sum(len(df) for _, df, _ in loaded)

    print(f"\n  {C.BOLD}Archivos analizados:{C.RESET}  {C.GREEN}{total_files}{C.RESET}")
    print(f"  {C.BOLD}Total de prompts:{C.RESET}      {C.GREEN}{C.BOLD}{total_rows}{C.RESET}\n")

    print(f"  {'Dataset':<48}  {'Filas':>6}  {'Columnas':>8}")
    print(f"  {'─'*48}  {'─'*6}  {'─'*8}")
    for alias, df, _ in loaded:
        print(f"  {C.CYAN}{alias:<48}{C.RESET}  "
              f"{C.WHITE}{len(df):>6}{C.RESET}  "
              f"{C.DIM}{df.shape[1]:>8}{C.RESET}")

    # Totales por fuente/folder
    section("Prompts por carpeta de origen")
    folders: dict[str, int] = {}
    for alias, df, _ in loaded:
        folder = alias.split("—")[0].strip() if "—" in alias else alias.split("(")[0].strip()
        folders[folder] = folders.get(folder, 0) + len(df)
    max_f = max(folders.values())
    for folder, cnt in sorted(folders.items(), key=lambda x: -x[1]):
        bar = mini_bar(cnt, max_f, 30)
        print(f"  {C.GREEN}{bar}{C.RESET}  {folder:<40}  {C.BOLD}{cnt}{C.RESET}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    header("🔍  ANÁLISIS DE PROMPTS — DATASETS LOCALES", color=C.CYAN)
    print(f"\n  {C.DIM}Directorio base: {BASE}{C.RESET}")
    print(f"  {C.DIM}Analizando {len(DATASET_META)} datasets configurados…{C.RESET}")

    loaded = []
    for rel_path, meta in DATASET_META.items():
        analyze_dataset(rel_path, meta)
        df = load_df(rel_path, meta.get("sep"))
        if df is not None:
            loaded.append((meta["alias"], df, meta))

    global_summary(loaded)

    header("✅  ANÁLISIS COMPLETADO", color=C.GREEN)
    print(f"\n  {C.GREEN}Se han analizado {len(loaded)} datasets locales con éxito.{C.RESET}\n")


if __name__ == "__main__":
    main()

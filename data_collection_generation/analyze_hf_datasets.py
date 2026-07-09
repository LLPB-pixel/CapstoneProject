"""
╔══════════════════════════════════════════════════════════════════╗
║      ANÁLISIS DE PROMPTS — DATASETS DE HUGGING FACE             ║
║  walledai/AdvBench · JasperLS/prompt-injections                 ║
║  walledai/JailbreakHub                                          ║
╚══════════════════════════════════════════════════════════════════╝

Requiere:
  pip install datasets huggingface_hub pandas

Uso:
  python analyze_hf_datasets.py
"""

import sys
import textwrap
from collections import Counter

# ── Verificar dependencias ───────────────────────────────────────────
try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("\n⚠ Dependencias no encontradas. Instalando…")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "datasets", "huggingface_hub", "pandas", "-q"])
    from datasets import load_dataset
    import pandas as pd


# ── Colores ANSI ─────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"


def header(text: str, width: int = 72, color: str = C.CYAN):
    bar = "═" * width
    print(f"\n{color}{C.BOLD}╔{bar}╗")
    print(f"║  {text:<{width-2}}║")
    print(f"╚{bar}╝{C.RESET}")


def section(text: str, color: str = C.YELLOW):
    print(f"\n{color}{C.BOLD}▶ {text}{C.RESET}")


def mini_bar(val: int, max_val: int, width: int = 24) -> str:
    filled = int(val / max_val * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def bar_chart(label: str, value: int, total: int,
              width: int = 30, color: str = C.GREEN):
    pct = value / total * 100 if total > 0 else 0
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {color}{bar}{C.RESET}  {C.BOLD}{label:<35}{C.RESET}  "
          f"{C.WHITE}{value:>5}{C.DIM} ({pct:5.1f}%){C.RESET}")


def wrap_text(text: str, width: int = 68, indent: str = "    ") -> str:
    return textwrap.fill(str(text), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def top_words(texts: list[str], n: int = 12) -> list[tuple[str, int]]:
    STOP = {
        "a", "an", "the", "to", "of", "and", "or", "in", "for",
        "on", "is", "it", "how", "that", "be", "are", "with",
        "by", "you", "can", "i", "this", "do", "from", "your",
        "as", "me", "my", "what", "make", "write", "create",
        "provide", "give", "develop", "explain", "describe",
        "help", "without", "using", "its", "their", "has", "have",
        "use", "will", "about", "not", "if", "was", "were",
        "but", "so", "we", "them", "also", "into",
    }
    words = []
    for text in texts:
        for w in str(text).lower().split():
            w = w.strip(".,!?\"';:()[]")
            if len(w) > 3 and w not in STOP:
                words.append(w)
    return Counter(words).most_common(n)


def prompt_stats(series: pd.Series) -> dict:
    lengths = series.dropna().astype(str).apply(len)
    word_counts = series.dropna().astype(str).apply(lambda x: len(x.split()))
    return {
        "n": len(lengths),
        "char_min": int(lengths.min()),
        "char_max": int(lengths.max()),
        "char_mean": float(lengths.mean()),
        "char_median": float(lengths.median()),
        "word_mean": float(word_counts.mean()),
        "word_median": float(word_counts.median()),
    }


# ════════════════════════════════════════════════════════════════════
# FUNCIÓN GENÉRICA DE ANÁLISIS
# ════════════════════════════════════════════════════════════════════

def analyze_hf_dataset(
    hf_name: str,
    display_name: str,
    prompt_col: str,
    split: str = "train",
    extra_config: str | None = None,
    category_col: str | None = None,
    label_col: str | None = None,
    description: str = "",
    local_fallback: str | None = None,
    local_sep: str | None = None,
):
    """Carga un dataset de HuggingFace y muestra análisis completo.
    Si el dataset es gated o falla, intenta cargar desde local_fallback CSV.
    """

    header(f"🤗  {display_name}", color=C.BLUE)
    print(f"  {C.DIM}HuggingFace ID: {hf_name}{C.RESET}")
    if description:
        print(f"  {C.DIM}{description}{C.RESET}")

    # ── Carga ────────────────────────────────────────────────────────
    section("Cargando dataset…")
    df = None
    try:
        # Intentar cargar desde HuggingFace sin trust_remote_code si hay advertencias, o con él
        try:
            if extra_config:
                ds = load_dataset(hf_name, extra_config, trust_remote_code=True)
            else:
                ds = load_dataset(hf_name, trust_remote_code=True)
        except TypeError:
            if extra_config:
                ds = load_dataset(hf_name, extra_config)
            else:
                ds = load_dataset(hf_name)

        # Mostrar splits disponibles
        splits = list(ds.keys())
        print(f"  Splits disponibles: {C.CYAN}{', '.join(splits)}{C.RESET}")

        # Seleccionar split
        if split not in ds:
            split = splits[0]
            print(f"  {C.YELLOW}Split solicitado no existe, usando: {split}{C.RESET}")

        df = ds[split].to_pandas()

    except Exception as e:
        print(f"  {C.RED}❌ Error cargando dataset desde Hugging Face: {e}{C.RESET}")
        if local_fallback:
            from pathlib import Path
            base_dir = Path(__file__).parent
            full_path = base_dir / local_fallback
            print(f"  {C.YELLOW}▶ Intentando cargar fallback local desde: {full_path}...{C.RESET}")
            if full_path.exists():
                try:
                    if local_sep:
                        df = pd.read_csv(full_path, sep=local_sep, on_bad_lines="skip", engine="python")
                    else:
                        try:
                            df = pd.read_csv(full_path, on_bad_lines="skip")
                        except UnicodeDecodeError:
                            df = pd.read_csv(full_path, encoding="latin-1", on_bad_lines="skip")
                    print(f"  {C.GREEN}✓ Cargado con éxito desde archivo local.{C.RESET}")
                except Exception as local_err:
                    print(f"  {C.RED}❌ Error cargando fallback local: {local_err}{C.RESET}")
            else:
                print(f"  {C.RED}❌ El archivo de fallback local no existe: {full_path}{C.RESET}")

    if df is None:
        print(f"  {C.RED}❌ No se pudo cargar el dataset.{C.RESET}")
        return None

    # ── Info básica ───────────────────────────────────────────────────
    section("Información general")
    print(f"  {C.WHITE}Split analizado:{C.RESET}  {C.GREEN}{split}{C.RESET}")
    print(f"  {C.WHITE}Filas:{C.RESET}            {C.GREEN}{C.BOLD}{len(df)}{C.RESET}")
    print(f"  {C.WHITE}Columnas:{C.RESET}         {C.CYAN}{', '.join(df.columns.tolist())}{C.RESET}")
    print(f"  {C.WHITE}Tipos:{C.RESET}")
    for col, dtype in df.dtypes.items():
        print(f"    {C.DIM}· {col:<30} {dtype}{C.RESET}")

    # ── Nulos ─────────────────────────────────────────────────────────
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print(f"  {C.YELLOW}Valores nulos: {nulls.to_dict()}{C.RESET}")
    else:
        print(f"  {C.GREEN}✓ Sin valores nulos{C.RESET}")

    # ── Autodetección de columna de prompts ──────────────────────────
    if prompt_col not in df.columns:
        for candidate in ["text", "prompt", "goal", "Goal", "Prompt",
                           "instruction", "question", "content", "query"]:
            if candidate in df.columns:
                prompt_col = candidate
                print(f"  {C.YELLOW}Columna de prompt autodetectada: {prompt_col}{C.RESET}")
                break

    # ── Estadísticas de prompts ───────────────────────────────────────
    if prompt_col in df.columns:
        prompts_series = df[prompt_col].dropna().astype(str)
        stats = prompt_stats(df[prompt_col])

        section(f"Estadísticas de prompts  [{prompt_col}]")
        print(f"  Total prompts:         {C.GREEN}{C.BOLD}{stats['n']}{C.RESET}")
        print(f"  Longitud (chars):")
        print(f"    Mínima:              {C.CYAN}{stats['char_min']}{C.RESET}")
        print(f"    Máxima:              {C.CYAN}{stats['char_max']}{C.RESET}")
        print(f"    Media:               {C.CYAN}{stats['char_mean']:.1f}{C.RESET}")
        print(f"    Mediana:             {C.CYAN}{stats['char_median']:.0f}{C.RESET}")
        print(f"  Longitud (palabras):")
        print(f"    Media:               {C.CYAN}{stats['word_mean']:.1f}{C.RESET}")
        print(f"    Mediana:             {C.CYAN}{stats['word_median']:.0f}{C.RESET}")

        # Histograma de longitudes (7 buckets)
        section("Distribución de longitudes (chars)")
        lengths = prompts_series.apply(len)
        bins = pd.cut(lengths, bins=7)
        bin_counts = bins.value_counts().sort_index()
        max_count = bin_counts.max()
        for interval, count in bin_counts.items():
            bar = mini_bar(count, max_count, 28)
            label = f"{int(interval.left):>5}–{int(interval.right):<5}"
            print(f"  {C.GREEN}{bar}{C.RESET}  {label}  {C.WHITE}{count}{C.RESET}")

        # Ejemplos
        section("Ejemplos de prompts (primeros 4)")
        for i, p in enumerate(prompts_series.head(4), 1):
            print(f"\n  {C.MAGENTA}{i}.{C.RESET}")
            print(wrap_text(p[:250] + ("…" if len(p) > 250 else "")))

        # Top palabras
        section("Palabras clave más frecuentes (excluye stop words)")
        top = top_words(prompts_series.tolist())
        max_count = top[0][1] if top else 1
        for word, count in top:
            bar = mini_bar(count, max_count, 24)
            print(f"  {C.YELLOW}{bar}{C.RESET}  {C.WHITE}{word:<22}{C.RESET}  "
                  f"{C.CYAN}{count}{C.RESET}")

    # ── Categorías ────────────────────────────────────────────────────
    if category_col and category_col in df.columns:
        section(f"Distribución por categoría  [{category_col}]")
        counts = df[category_col].value_counts()
        for cat, cnt in counts.head(20).items():
            bar_chart(str(cat), cnt, len(df), color=C.MAGENTA)
        if len(counts) > 20:
            print(f"  {C.DIM}… y {len(counts)-20} categorías más{C.RESET}")

    # ── Etiquetas ────────────────────────────────────────────────────
    if label_col and label_col in df.columns:
        section(f"Distribución de etiquetas  [{label_col}]")
        counts = df[label_col].value_counts()
        for lbl, cnt in counts.items():
            color = C.RED if str(lbl).lower() in ("1", "true", "harmful", "yes") else C.GREEN
            bar_chart(str(lbl), cnt, len(df), color=color)

    # ── Vista previa columnar ─────────────────────────────────────────
    section("Vista previa de columnas adicionales (sin prompt)")
    non_prompt_cols = [c for c in df.columns if c != prompt_col]
    if non_prompt_cols:
        preview_df = df[non_prompt_cols].head(5).copy()
        for col in preview_df.select_dtypes(include="object").columns:
            preview_df[col] = preview_df[col].astype(str).apply(
                lambda x: x[:50] + "…" if len(x) > 50 else x
            )
        print(preview_df.to_string(index=False))
    else:
        print(f"  {C.DIM}No hay columnas adicionales.{C.RESET}")

    return df


# ════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPARATIVO ENTRE LOS TRES DATASETS
# ════════════════════════════════════════════════════════════════════

def comparative_analysis(results: dict[str, pd.DataFrame]):
    header("📊  ANÁLISIS COMPARATIVO", color=C.GREEN)

    rows_data = []
    for name, df_dict in results.items():
        if df_dict is None or df_dict.get("df") is None:
            continue
        df = df_dict["df"]
        col = df_dict["prompt_col"]
        if col and col in df.columns:
            prompts = df[col].dropna().astype(str)
            lengths = prompts.apply(len)
            rows_data.append({
                "Dataset": name,
                "Prompts": len(prompts),
                "Media chars": f"{lengths.mean():.0f}",
                "Mediana chars": f"{lengths.median():.0f}",
                "Max chars": f"{lengths.max()}",
            })

    if rows_data:
        comp_df = pd.DataFrame(rows_data).set_index("Dataset")
        print()
        print(comp_df.to_string())

    # Gráfico de barras comparativo por número de prompts
    section("Prompts por dataset")
    max_n = max(r["Prompts"] for r in rows_data) if rows_data else 1
    for r in rows_data:
        bar = mini_bar(r["Prompts"], max_n, 35)
        print(f"  {C.BLUE}{bar}{C.RESET}  {r['Dataset']:<40}  "
              f"{C.BOLD}{r['Prompts']}{C.RESET}")


# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LOS DATASETS
# ════════════════════════════════════════════════════════════════════

DATASETS_CONFIG = [
    {
        "hf_name": "walledai/AdvBench",
        "display_name": "walledai/AdvBench",
        "prompt_col": "goal",          # columna en el CSV local es 'goal'
        "split": "train",
        "extra_config": None,
        "category_col": None,
        "label_col": None,
        "local_fallback": "advbench/harmful_behaviors.csv",  # CSV local como respaldo
        "local_sep": None,
        "description": (
            "Benchmark de comportamientos dañinos (Zou et al. 2023). "
            "520 instrucciones que los LLMs no deben seguir. "
            "[Dataset gated → usando CSV local como fallback]"
        ),
    },
    {
        "hf_name": "JasperLS/prompt-injections",
        "display_name": "JasperLS/prompt-injections",
        "prompt_col": "text",
        "split": "train",
        "extra_config": None,
        "category_col": "label",
        "label_col": "label",
        "description": (
            "Dataset de inyecciones de prompt clasificadas como "
            "benign/injection. Útil para detección de ataques."
        ),
    },
    {
        "hf_name": "walledai/JailbreakHub",
        "display_name": "walledai/JailbreakHub",
        "prompt_col": "prompt",
        "split": "train",
        "extra_config": None,
        "category_col": "community",
        "label_col": None,
        "description": (
            "Colección de prompts de jailbreak recopilados de la comunidad. "
            "Incluye información sobre su origen y tipo."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    header(
        "🌐  ANÁLISIS DE PROMPTS — DATASETS HUGGING FACE",
        color=C.CYAN,
    )
    print(f"\n  {C.DIM}Analizando 3 datasets: walledai/AdvBench, "
          f"JasperLS/prompt-injections, walledai/JailbreakHub{C.RESET}")
    print(f"  {C.DIM}Se necesita conexión a internet la primera vez "
          f"(se cachean localmente).{C.RESET}\n")

    results = {}

    for cfg in DATASETS_CONFIG:
        df = analyze_hf_dataset(
            hf_name=cfg["hf_name"],
            display_name=cfg["display_name"],
            prompt_col=cfg["prompt_col"],
            split=cfg["split"],
            extra_config=cfg.get("extra_config"),
            category_col=cfg.get("category_col"),
            label_col=cfg.get("label_col"),
            description=cfg.get("description", ""),
            local_fallback=cfg.get("local_fallback"),
            local_sep=cfg.get("local_sep"),
        )
        results[cfg["display_name"]] = {
            "df": df,
            "prompt_col": cfg["prompt_col"] if (df is not None and cfg["prompt_col"] in df.columns) else None,
        }

    comparative_analysis(results)

    header("✅  ANÁLISIS COMPLETADO", color=C.GREEN)
    loaded = sum(1 for v in results.values() if v["df"] is not None)
    print(f"\n  {C.GREEN}Datasets analizados con éxito: {loaded}/{len(DATASETS_CONFIG)}{C.RESET}\n")


if __name__ == "__main__":
    main()

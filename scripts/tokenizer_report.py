"""Day 1: Tokenizer forensics on English / Devanagari Hindi / romanized Hinglish.

Measures how Qwen2.5's frozen BPE tokenizer treats each script:
  - fertility (tokens per word)        -> training/inference cost per word
  - tokens per character               -> script-fair comparison
  - % words that are a single token    -> how "native" the script is to the vocab
  - worst-fragmenting words            -> qualitative evidence for the report

Usage:
    python scripts/tokenizer_report.py \
        --model Qwen/Qwen2.5-1.5B \
        --corpora-dir data/corpora \
        --out reports/tokenizer_report.md
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# allow `from src.utils import ...` when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_logger, set_seed

logger = get_logger("tokenizer_report")


@dataclass
class CorpusStats:
    name: str
    n_sentences: int
    n_words: int
    n_chars: int
    n_tokens: int
    pct_single_token_words: float
    worst_words: list[tuple[str, list[str]]]  # (word, token pieces)

    @property
    def tokens_per_word(self) -> float:
        return self.n_tokens / max(self.n_words, 1)

    @property
    def tokens_per_char(self) -> float:
        return self.n_tokens / max(self.n_chars, 1)


def analyze_corpus(name: str, sentences: list[str], tokenizer) -> CorpusStats:
    n_words = n_chars = n_tokens = 0
    single_token_words = 0
    total_words = 0
    word_splits: list[tuple[str, list[str]]] = []

    for sent in sentences:
        # add_special_tokens=False: we measure the TEXT's cost,
        # not BOS/EOS template overhead.
        ids = tokenizer(sent, add_special_tokens=False).input_ids
        n_tokens += len(ids)
        words = sent.split()
        n_words += len(words)
        n_chars += sum(len(w) for w in words)  # chars excluding spaces

        for w in words:
            # Leading space matters: BPE merges differ for word-initial
            # vs mid-text positions; " word" approximates in-context cost.
            w_ids = tokenizer(" " + w, add_special_tokens=False).input_ids
            pieces = tokenizer.convert_ids_to_tokens(w_ids)
            total_words += 1
            if len(w_ids) == 1:
                single_token_words += 1
            word_splits.append((w, pieces))

    # worst fragmenters = most tokens per character (length-fair)
    seen: set[str] = set()
    uniq = []
    for w, pieces in word_splits:
        if w.lower() not in seen and len(w) >= 3:
            seen.add(w.lower())
            uniq.append((w, pieces))
    uniq.sort(key=lambda t: len(t[1]) / max(len(t[0]), 1), reverse=True)

    return CorpusStats(
        name=name,
        n_sentences=len(sentences),
        n_words=n_words,
        n_chars=n_chars,
        n_tokens=n_tokens,
        pct_single_token_words=100.0 * single_token_words / max(total_words, 1),
        worst_words=uniq[:10],
    )


def render_markdown(stats: list[CorpusStats], model_name: str) -> str:
    lines = [
        f"# Tokenizer Report -- `{model_name}`",
        "",
        "| Corpus | Sentences | Words | Tokens | Tokens/Word | Tokens/Char | % 1-token words |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s.name} | {s.n_sentences} | {s.n_words} | {s.n_tokens} "
            f"| {s.tokens_per_word:.2f} | {s.tokens_per_char:.3f} "
            f"| {s.pct_single_token_words:.1f}% |"
        )
    lines.append("")
    for s in stats:
        lines.append(f"## Worst-fragmenting words -- {s.name}")
        lines.append("")
        lines.append("| Word | Pieces | # Tokens |")
        lines.append("|---|---|---|")
        for w, pieces in s.worst_words:
            shown = " · ".join(p.replace("|", "\\|") for p in pieces)
            lines.append(f"| {w} | {shown} | {len(pieces)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--corpora-dir", type=Path, default=Path("data/corpora"))
    parser.add_argument("--out", type=Path, default=Path("reports/tokenizer_report.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    from transformers import AutoTokenizer  # lazy import: fast --help

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    stats: list[CorpusStats] = []
    for path in sorted(args.corpora_dir.glob("*.txt")):
        sentences = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        logger.info("Analyzing %s (%d sentences)", path.stem, len(sentences))
        stats.append(analyze_corpus(path.stem, sentences, tokenizer))

    if not stats:
        logger.error("No .txt corpora found in %s", args.corpora_dir)
        sys.exit(1)

    report = render_markdown(stats, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", args.out)

    for s in stats:
        logger.info(
            "%-12s tokens/word=%.2f  tokens/char=%.3f  single-token-words=%.1f%%",
            s.name, s.tokens_per_word, s.tokens_per_char, s.pct_single_token_words,
        )


if __name__ == "__main__":
    main()

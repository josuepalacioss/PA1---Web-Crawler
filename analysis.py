"""
CS 4422 - PA1: Analysis for the report
NetID: jpalac21

Reads the files made by crawler.py and prints everything the report needs:
    1. Top 5 pages with the most incoming links
    2. Top 5 pages with the most outgoing links
    3. Top 5 longest pages (word count)
    4. Zipf's law plot (word frequency vs rank, log-log)
    5. Top 30 words after removing stopwords
    6. TF-IDF top 5 keywords for 5 random pages

How to run:
    python analysis.py --output-dir output --report-dir report
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")   # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from nltk.corpus import stopwords

csv.field_size_limit(sys.maxsize)

# Plot colors
BLUE = "#2a78d6"
GRAY = "#52514e"
LIGHT_GRID = "#e4e3df"


def load_pages(path):
    with open(path, newline="", encoding="utf-8") as f:
        pages = list(csv.DictReader(f))
    for p in pages:
        p["word_count"] = int(p["word_count"])
        p["out_links"] = int(p["out_links"])
        p["num_chunks"] = int(p["num_chunks"])
    return pages


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_in_links(path):
    """Count how many different pages link TO each URL."""
    in_links = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for source, dest in reader:
            if source != dest:
                in_links[dest] += 1
    return in_links


def md_table(headers, rows):
    """Make a markdown table (easy to paste into the report)."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def style_axes(ax):
    # Keep the grid and axes light so the data stands out
    ax.grid(True, which="major", color=LIGHT_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=GRAY)


def zipf_plot(freqs, out_path):
    """Zipf's law: f * r = k. On a log-log plot this is a straight line with
    slope -1 (lecture: power law y = k * x^c with c = -1)."""
    counts = np.array(sorted(freqs.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(counts) + 1)

    # Fit a line to log(f) = log(k) + c*log(r) to get the slope c
    slope, intercept = np.polyfit(np.log10(ranks), np.log10(counts), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.loglog(ranks, counts, color=BLUE, linewidth=2, label="My corpus")
    ideal = counts[0] / ranks   # perfect Zipf: f = f(1) / r
    ax.loglog(ranks, ideal, color=GRAY, linewidth=1.2, linestyle="--",
              label="Ideal Zipf (slope = -1)")
    ax.set_xlabel("Rank (r)")
    ax.set_ylabel("Frequency (f)")
    ax.set_title(f"Zipf's law: word frequency vs rank (fitted slope = {slope:.2f})",
                 fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return slope, 10 ** intercept


def chunk_length_plot(lengths, out_path):
    """Histogram of chunk sizes, to show my chunks are NOT all the same length."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(lengths, bins=30, color=BLUE, edgecolor="white", linewidth=1)
    ax.set_xlabel("Chunk length (words)")
    ax.set_ylabel("Number of chunks")
    ax.set_title("Variable-length chunks: distribution of chunk sizes", fontsize=11)
    style_axes(ax)
    ax.grid(False, axis="x")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def tf_idf_top_words(docs, picked, top_n=5):
    """tf-idf from the lecture: w = log(1 + tf) * log(|D| / df)"""
    df = Counter()
    for words in docs.values():
        df.update(set(words))
    n_docs = len(docs)
    results = {}
    for url in picked:
        tf = Counter(docs[url])
        weights = {t: math.log(1 + c) * math.log(n_docs / df[t]) for t, c in tf.items()}
        best = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        results[url] = best
    return results


def main():
    parser = argparse.ArgumentParser(description="PA1 corpus analysis")
    parser.add_argument("--output-dir", default="output", help="where crawler.py saved files")
    parser.add_argument("--report-dir", default="report", help="where to save results")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="seed for picking the 5 random pages (so I can repeat it)")
    args = parser.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    pages = load_pages(os.path.join(args.output_dir, "pages.csv"))
    chunks = load_chunks(os.path.join(args.output_dir, "corpus_chunks.json"))
    in_links = count_in_links(os.path.join(args.output_dir, "adjacency_list.csv"))
    crawled = {p["url"] for p in pages}
    stop_words = set(stopwords.words("english"))

    out = []   # lines of the markdown report
    out.append("# PA1 analysis results\n")
    out.append(f"- Pages crawled: **{len(pages)}**")
    out.append(f"- Chunks: **{len(chunks)}**")
    out.append(f"- Links (edges): **{sum(p['out_links'] for p in pages)}**")
    by_domain = Counter(p["domain"] for p in pages)
    out.append(f"- Domains crawled: **{len(by_domain)}**\n")
    out.append("## Pages per domain (top 15)\n")
    out.append(md_table(["Domain", "Pages"], by_domain.most_common(15)) + "\n")

    # 1. Incoming links
    out.append("## 1. Top 5 pages with the most incoming links\n")
    top_in_crawled = sorted(((c, u) for u, c in in_links.items() if u in crawled),
                            reverse=True)[:5]
    out.append("Among the pages I crawled:\n")
    out.append(md_table(["Rank", "Page", "Incoming links"],
                        [(i + 1, u, c) for i, (c, u) in enumerate(top_in_crawled)]) + "\n")
    out.append("Among all linked URLs (crawled or not):\n")
    out.append(md_table(["Rank", "Page", "Incoming links"],
                        [(i + 1, u, c) for i, (u, c) in enumerate(in_links.most_common(5))]) + "\n")

    # 2. Outgoing links
    out.append("## 2. Top 5 pages with the most outgoing links\n")
    top_out = sorted(pages, key=lambda p: p["out_links"], reverse=True)[:5]
    out.append(md_table(["Rank", "Page", "Outgoing links"],
                        [(i + 1, p["url"], p["out_links"]) for i, p in enumerate(top_out)]) + "\n")

    # 3. Longest pages
    out.append("## 3. Top 5 longest pages (word count)\n")
    top_long = sorted(pages, key=lambda p: p["word_count"], reverse=True)[:5]
    out.append(md_table(["Rank", "Page", "Words"],
                        [(i + 1, p["url"], p["word_count"]) for i, p in enumerate(top_long)]) + "\n")

    # 4. Zipf's law. I use all words (lowercase, no punctuation) WITH stopwords,
    # because the stopwords are the "very common" head of the curve.
    freqs = Counter()
    for c in chunks:
        freqs.update(re.findall(r"[a-z]+", c["raw_text"].lower()))
    total = sum(freqs.values())
    slope, k = zipf_plot(freqs, os.path.join(args.report_dir, "zipf.png"))
    out.append("## 4. Zipf's law\n")
    out.append("![Zipf plot](zipf.png)\n")
    out.append(f"- Total words (N): **{total}**, vocabulary size |V|: **{len(freqs)}**")
    out.append(f"- Fitted slope on the log-log plot: **{slope:.3f}** (Zipf predicts -1)")
    once = sum(1 for v in freqs.values() if v == 1)
    out.append(f"- Words that appear only once: **{once}** "
               f"({100 * once / len(freqs):.1f}% of the vocabulary)")
    top5_share = sum(v for _, v in freqs.most_common(5)) / total
    out.append(f"- Top 5 words cover **{100 * top5_share:.1f}%** of all word occurrences\n")
    out.append("Check of f x r = k at a few ranks:\n")
    ranked = freqs.most_common()
    rows = []
    for r in (1, 2, 5, 10, 50, 100, 500, 1000, 5000):
        if r <= len(ranked):
            word, f = ranked[r - 1]
            rows.append((r, word, f, f * r))
    out.append(md_table(["Rank r", "Word", "Freq f", "f x r"], rows) + "\n")

    # 5. Top 30 words without stopwords (same table style as the lecture)
    out.append("## 5. Top 30 words (stopwords removed)\n")
    no_stop = Counter({w: c for w, c in freqs.items()
                       if w not in stop_words and len(w) > 1})
    rows = [(i + 1, w, c, f"{c / total:.4f}")
            for i, (w, c) in enumerate(no_stop.most_common(30))]
    out.append(md_table(["Rank", "Term", "Freq.", "Perc."], rows) + "\n")

    # 6. TF-IDF. A "document" here is a whole page = all its processed chunks.
    docs = defaultdict(list)
    for c in chunks:
        docs[c["parent_page_url"]].extend(c["processed_text"].split())
    docs = {u: w for u, w in docs.items() if w}
    rng = random.Random(args.random_seed)
    picked = rng.sample(sorted(docs), min(5, len(docs)))
    out.append("## 6. TF-IDF: top 5 keywords for 5 random pages\n")
    out.append(f"Weight: w = log(1 + tf) x log(|D| / df), |D| = {len(docs)} pages, "
               f"random seed = {args.random_seed}\n")
    for url, best in tf_idf_top_words(docs, picked).items():
        out.append(f"**{url}**\n")
        out.append(md_table(["Keyword", "tf-idf"], [(w, f"{s:.3f}") for w, s in best]) + "\n")

    # Extra: chunk length stats (to show the chunks really vary in size)
    lengths = [c["word_count"] for c in chunks]
    if lengths:
        chunk_length_plot(lengths, os.path.join(args.report_dir, "chunk_lengths.png"))
        out.append("## Extra: chunk lengths\n")
        out.append("![Chunk lengths](chunk_lengths.png)\n")
        out.append(f"- min {min(lengths)}, max {max(lengths)}, "
                   f"mean {np.mean(lengths):.1f}, median {np.median(lengths):.0f}, "
                   f"std {np.std(lengths):.1f} words\n")

    report = "\n".join(out)
    with open(os.path.join(args.report_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nSaved results.md, zipf.png and chunk_lengths.png in {args.report_dir}/")


if __name__ == "__main__":
    main()

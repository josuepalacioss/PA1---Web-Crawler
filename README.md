# CS 4422 PA1 - Web Crawler (from scratch) and Text Preprocessing

NetID: jpalac21
User-Agent: `CS4422-StudentCrawler-jpalac21`

## Setup

```
pip install -r requirements.txt
python setup_nltk.py          # downloads stopwords, wordnet, tagger, sentence splitter
```

## Run

Quick test first (20 pages, one seed at a time, stay on that site):

```
python crawler.py --seed-urls https://www.eecs.mit.edu/     --num-pages 20 --restrict-domains --output-dir test_mit
python crawler.py --seed-urls https://www.nsf.gov/          --num-pages 20 --restrict-domains --output-dir test_nsf
python crawler.py --seed-urls https://docs.python.org/3/    --num-pages 20 --restrict-domains --output-dir test_python
python crawler.py --seed-urls https://www.w3.org/           --num-pages 20 --restrict-domains --output-dir test_w3c
```

Full crawl (the run used for the report):

```
python crawler.py --seed-urls https://www.eecs.mit.edu/ https://www.nsf.gov/ https://docs.python.org/3/ https://www.w3.org/ --num-pages 5000 --restrict-domains --max-pages-per-domain 1700 --output-dir output_run2
```

This run saved 5,000 pages in 47.8 minutes. A first run with `--max-pages-per-domain 1500` stopped at
4,673 pages, because MIT and the Python docs ran out of reachable pages, so I raised the limit to 1,700.

I first planned to use the KSU CS department (`https://ccse.kennesaw.edu/cs/`) as the .edu seed, but
its pages redirect across three hosts (ccse.kennesaw.edu, www.kennesaw.edu, campus.kennesaw.edu), so the
crawler refused to follow them and saved 0 pages. I switched to MIT EECS.

Then the report numbers and plots:

```
python analysis.py --output-dir output_run2 --report-dir report
```

Short demo (40 pages, plus a fake URL to show a 404 being handled):

```
python crawler.py --seed-urls https://www.eecs.mit.edu/ https://www.nsf.gov/ https://docs.python.org/3/ https://www.w3.org/ https://www.nsf.gov/this-page-does-not-exist --num-pages 40 --restrict-domains --output-dir demo
```

| Option | Meaning |
|---|---|
| `--seed-urls` | one or more start URLs |
| `--num-pages` | max pages to crawl (default 5000) |
| `--restrict-domains` | only crawl the exact domains of the seeds. Without it, any `.edu`, `.gov`, `.org` domain is allowed |
| `--max-pages-per-domain` | optional limit of pages from any one domain (default: no limit) |
| `--output-dir` | where to save files (default `output`) |
| `--threads` | number of crawler threads (default 8) |

Press Ctrl+C any time: the crawler stops and still closes the files properly.

## Output files (in the `--output-dir` folder)

| File | What is in it |
|---|---|
| `adjacency_list.csv` | `source_url,destination_url` - every outgoing link, with its parent page |
| `corpus_chunks.json` | JSON array of chunks: `chunk_id`, `parent_page_url`, `chunk_index`, `title`, `processed_title`, `raw_text`, `processed_text`, `word_count` |
| `pages.csv` | one row per crawled page: url, domain, title, word count, number of out links, number of chunks |
| `crawl.log` | everything that happened, including every error |

`analysis.py` writes `report/results.md`, `report/zipf.png` and `report/chunk_lengths.png`.

## How it works

**Frontier and politeness (Web Crawler lecture).** The frontier keeps one queue per
domain (FIFO, so it is BFS). A thread asks the frontier for the next website that is
ready, takes one URL, downloads it, stores it, and adds the new links back. This is the
"Simple Crawling Thread" pseudo-code from the slides. A website is only given to one
thread at a time, and after each request it has to wait at least 1 second (or longer if
`robots.txt` has a `crawl-delay`). Different domains are crawled at the same time.

**robots.txt and robots meta tag.** Before the first page of a host I download its
`robots.txt` and check every URL with `urllib.robotparser`. Pages with
`<meta name="robots" content="noindex">` are not stored, and `nofollow` pages don't
add their links to the frontier.

**Per-domain rules.** Some sites are too big or noisy, so I set rules for them at the top of `crawler.py`:
- `docs.python.org`: only pages under `/3/` (not old versions like `/2.7/` or `/3.12/`), and I skip
  index pages like `genindex`, `py-modindex` and `search.html`.
- `www.kennesaw.edu`: only pages under `/ccse/` (left from testing; KSU is not in the final seeds).
- `w3.org`: I skip other-language folders like `/ja/` and `/zh-hans/`, because my text processing is English only.

Links to domains I don't crawl are still written to `adjacency_list.csv`, so the link graph is complete.

**Redirects.** I don't let `requests` follow redirects on its own, because that would hit
the same server again right away. Instead the new URL goes back into the frontier, so it
gets the 1 second wait and the robots check like any other URL.

**Error handling.** 404, 500, timeouts, connection errors, and non-HTML files are caught,
logged in `crawl.log`, and skipped. On 429/503 I double the delay for that site. If a
site keeps failing (15 errors in a row) I stop crawling it.

**URL canonicalization and duplicates (Handling Duplicates lecture).** Every URL is cleaned
with `w3lib.url.canonicalize_url` (sorted query arguments, fixed percent encoding, no
blank values, no `#fragment`), plus lowercase host, no default port and no `../` in the
path. Exact duplicate pages are found by hashing the page text (SHA-256 checksum).

**Variable-length semantic chunking.** I remove boilerplate (`nav`, `header`, `footer`,
`script`, menus, ...) and collect the logical blocks: `<p>`, `<li>`, headings, `<pre>`,
etc. Blocks are grouped into chunks of about 50-300 words, a heading starts a new chunk,
and a very long paragraph is split by sentences (NLTK). I never cut a sentence in half.
A chunk is never under 50 words, except when the whole page has less than 50 words (then
that text is the page's only chunk). A small piece left at the end of a page is added to
the last chunk, so that chunk can go a little over 300 words. Pages with little or no
text are still saved in `pages.csv` and `adjacency_list.csv` for link analysis.
I don't use fixed 200-word windows because BM25 uses `dl / avgdl` with the `b`
parameter: if all chunks were 200 words, `dl / avgdl` would always be 1 and BM25 could
not reward short focused chunks or punish long wordy ones.

**Dual-state text (Text Processing lecture).**
- `raw_text`: the clean readable text, with original case and punctuation (for PA3 RAG).
- `processed_text`: lowercase → remove punctuation → remove NLTK stopwords → lemmatize
  with WordNet using the part of speech (for PA2).
- `processed_title`: the page `<title>` (or `<h1>`) run through the same steps, attached
  to every chunk of that page.

**Unique IDs.** `chunk_id = sha256(parent_page_url + "#" + chunk_number)`.

**Not kept in RAM.** Every page is written to the CSV/JSON files right after it is
crawled (the JSON array is written piece by piece).

## Analysis (analysis.py)

1. Top 5 pages by incoming links (counted from `adjacency_list.csv`)
2. Top 5 pages by outgoing links
3. Top 5 longest pages (word count)
4. Zipf's law log-log plot with the fitted slope (Zipf says about -1) and an `f × r` table
5. Top 30 terms from `processed_text` (normalized: stopwords removed, lemmatized)
6. TF-IDF top 5 keywords for 5 random pages, with `w = log(1 + tf) × log(|D| / df)`

## Works Cited

- *MIT EECS*. Massachusetts Institute of Technology, www.eecs.mit.edu/. Accessed 27 Sept. 2026.
- *Python 3 Documentation*. Python Software Foundation, docs.python.org/3/. Accessed 27 Sept. 2026.
- *U.S. National Science Foundation*, www.nsf.gov/. Accessed 27 Sept. 2026.
- *World Wide Web Consortium (W3C)*, www.w3.org/. Accessed 27 Sept. 2026.
- *Claude*. Claude Opus 5.5, Anthropic, 2026, claude.ai.
  - Used to help plan the crawler design, understand concepts where there was confusion, review test results, draft sections of this report, fix grammar mistakes, plan presentation, improve writing style, and make final revisions. All code was reviewed, tested, and run by me, and all results come from my own crawl.
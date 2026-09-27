"""
CS 4422 - PA1: Web Crawler (from scratch) and Text Preprocessing
NetID: 001032639

How to run:
    python crawler.py --seed-urls URL1 URL2 ... --num-pages 5000 [--restrict-domains]

My plan (following the Web Crawler lecture):
    1. Put the seed pages in the frontier.
    2. Pick the next website that is allowed to be crawled (politeness).
    3. Download the page and pull out the links.
    4. Clean up (canonicalize) the new URLs and add them to the frontier.
    5. Keep going until the frontier is empty or I have enough pages.

For every page I also break the text into variable-length chunks and save
two versions of each chunk: the raw text (for PA3 RAG) and the processed text
(for PA2 indexing).
"""

import argparse
import csv
import hashlib
import heapq
import json
import logging
import os
import re
import threading
import time
from collections import deque
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from w3lib.url import canonicalize_url

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NET_ID = "001032639"
USER_AGENT = f"CS4422-Student-Crawler-{NET_ID}"

MIN_DELAY = 1.0          # at least 1 second between requests to the same domain
MAX_CRAWL_DELAY = 30.0   # if robots.txt asks for a huge delay, I cap it here
TIMEOUT = (5, 15)        # (connect, read) timeout in seconds
MAX_PAGE_BYTES = 3_000_000
MAX_ERRORS_PER_SITE = 15  # after this many errors in a row I give up on a site
MAX_QUEUE_PER_SITE = 10000  # so the frontier does not eat all my RAM

# Chunk sizes (in words). The chunks are NOT all the same size on purpose.
MIN_CHUNK_WORDS = 50
MAX_CHUNK_WORDS = 300

# The domain regex given in the assignment
DOMAIN_REGEX = re.compile(r"(https?://)?(www\d?\.)?(?P<domain>[\w\.-]+\.\w+)(/\S*)?")

# When --restrict-domains is NOT set, I only follow these top level domains
ALLOWED_TLDS = (".edu", ".gov", ".org")

# The assignment says to avoid wiki farms and forums (too much layout noise)
BLOCKED_DOMAINS = (
    "wikipedia.org", "wikimedia.org", "wiktionary.org", "wikidata.org",
    "wikisource.org", "wikibooks.org", "archive.org", "discuss.python.org",
    "mail.python.org", "lists.w3.org",
)

# Links to files that are not web pages
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".exe", ".msi",
    ".dmg", ".iso", ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".webm", ".wav",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".txt", ".xml",
    ".json", ".rss", ".atom", ".css", ".js", ".epub", ".chm", ".ps", ".tex",
)

# HTML tags that are boilerplate (menus, scripts, etc.), not real content
BOILERPLATE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer", "aside", "form",
    "svg", "iframe", "button", "select", "template", "canvas", "object",
]
BOILERPLATE_NAMES = {
    "nav", "navbar", "menu", "footer", "header", "sidebar", "breadcrumb",
    "breadcrumbs", "skip-link", "cookie", "cookie-banner", "site-footer",
    "site-header", "toc", "related",
}

# Tags that hold a "logical block" of text. I group these into chunks.
BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
              "blockquote", "dd", "dt", "td", "th", "figcaption", "article"]
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

log = logging.getLogger("crawler")


# ---------------------------------------------------------------------------
# Text processing (Text Processing lecture)
# ---------------------------------------------------------------------------

class TextProcessor:
    """Turns raw text into normalized text for PA2:
    lowercase -> remove punctuation -> remove stopwords -> lemmatize."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        # I make the tagger once, because nltk.pos_tag() reloads it every call
        self.tagger = PerceptronTagger()
        # Load WordNet now, so the threads don't all try to load it at once
        wordnet.ensure_loaded()
        self.lemmatizer.lemmatize("warmup")
        nltk.sent_tokenize("Warm up. The sentence splitter.")

    @staticmethod
    def to_wordnet_pos(tag):
        # The lemmatizer needs to know if the word is a noun, verb, etc.
        # (Lecture example: "is studying" -> "be study")
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def process(self, text):
        # 1. lower case
        text = text.lower()
        # 2. remove punctuation: I keep only letters and numbers
        tokens = re.findall(r"[a-z0-9]+", text)
        # 3. remove stopwords, pure numbers, and 1-letter leftovers
        tokens = [t for t in tokens
                  if t not in self.stop_words and len(t) > 1 and not t.isdigit()]
        if not tokens:
            return ""
        # 4. lemmatize using the part of speech
        tagged = self.tagger.tag(tokens)
        lemmas = [self.lemmatizer.lemmatize(word, self.to_wordnet_pos(tag))
                  for word, tag in tagged]
        return " ".join(lemmas)


# ---------------------------------------------------------------------------
# URL helpers (Handling Duplicates lecture: URL canonicalization)
# ---------------------------------------------------------------------------

def get_domain(url):
    """Get the domain name of a URL, using the regex from the assignment.
    'https://www.nsf.gov/news' -> 'nsf.gov'"""
    host = (urlparse(url).hostname or "").lower()
    match = DOMAIN_REGEX.match(host)
    return match.group("domain") if match else host


def canonicalize(url):
    """Clean up a URL so different spellings of the same page look the same.
    w3lib does the steps from the lecture: sort the query arguments, fix the
    percent encoding, remove blank query values, remove the #fragment.
    I also lowercase the host and drop the default port."""
    try:
        # First resolve "/some/../folder" -> "/folder" (lecture example)
        parts = urlparse(url)
        if "/." in parts.path:
            clean_path = urlparse(urljoin(f"{parts.scheme}://{parts.netloc}/", parts.path)).path
            url = urlunparse(parts._replace(path=clean_path))
        url = canonicalize_url(url, keep_blank_values=False)
        parts = urlparse(url)
        host = (parts.hostname or "").lower()
        port = parts.port
        if port and not ((parts.scheme == "http" and port == 80) or
                         (parts.scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = parts.path or "/"
        return urlunparse((parts.scheme.lower(), host, path, "", parts.query, ""))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Frontier (Web Crawler lecture: the frontier + per-site queues)
# ---------------------------------------------------------------------------

class Website:
    """One website (domain) with its own queue of URLs.
    Keeping one queue per site is how I do politeness: only one request at a
    time per site, and I wait at least 1 second between requests."""

    def __init__(self, domain):
        self.domain = domain
        self.queue = deque()        # BFS: first in, first out
        self.next_time = 0.0        # earliest time I can hit this site again
        self.delay = MIN_DELAY
        self.busy = False           # a thread is working on this site right now
        self.robots = {}            # robots.txt parser for each host (scheme://netloc)
        self.errors_in_a_row = 0

    def next_url(self):
        return self.queue.popleft()

    def permits_crawl(self, url, session):
        """Check robots.txt before I download the page."""
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self.robots:
            self.robots[root] = self.load_robots(root, session)
        rules = self.robots[root]
        return rules is None or rules.can_fetch(USER_AGENT, url)

    def load_robots(self, root, session):
        """Download and read robots.txt (only once per host)."""
        robots_url = root + "/robots.txt"
        rules = robotparser.RobotFileParser(robots_url)
        try:
            resp = session.get(robots_url, timeout=TIMEOUT)
            if resp.status_code in (401, 403):
                # The site does not want robots at all
                rules.disallow_all = True
            elif resp.status_code >= 400:
                # No robots.txt means everything is allowed
                rules.allow_all = True
            else:
                rules.parse(resp.text.splitlines())
                # robots.txt can ask for a crawl-delay (e.g. "crawl-delay: 10")
                asked = rules.crawl_delay(USER_AGENT)
                if asked:
                    self.delay = min(max(MIN_DELAY, float(asked)), MAX_CRAWL_DELAY)
                    log.info("robots.txt of %s asks for crawl-delay %s", root, asked)
        except requests.RequestException as e:
            log.warning("Could not read %s (%s), I will assume it is allowed",
                        robots_url, type(e).__name__)
            rules.allow_all = True
        # Wait before the real page request, since robots.txt was a request too
        time.sleep(self.delay)
        return rules


class Frontier:
    """The crawler frontier: every URL I found but have not crawled yet.
    It hands out one website at a time to each thread."""

    def __init__(self, num_pages):
        self.num_pages = num_pages
        self.sites = {}             # domain -> Website
        self.ready = []             # heap of (next_time, domain) for idle sites with URLs
        self.in_ready = set()
        self.seen = set()           # every URL I have ever added (so no repeats)
        self.stored = 0             # pages saved so far
        self.in_flight = 0          # pages being downloaded right now
        self.stopped = False
        self.lock = threading.Condition()

    def add_url(self, url):
        with self.lock:
            if url in self.seen:
                return
            self.seen.add(url)
            domain = get_domain(url)
            site = self.sites.get(domain)
            if site is None:
                site = self.sites[domain] = Website(domain)
            if site.errors_in_a_row >= MAX_ERRORS_PER_SITE or \
                    len(site.queue) >= MAX_QUEUE_PER_SITE:
                return
            site.queue.append(url)
            self.schedule(site)
            self.lock.notify()

    def schedule(self, site):
        # Put the site back in line if it has URLs and nobody is using it
        if not site.busy and site.queue and site.domain not in self.in_ready:
            heapq.heappush(self.ready, (site.next_time, site.domain))
            self.in_ready.add(site.domain)

    def done(self):
        with self.lock:
            if self.stopped or self.stored >= self.num_pages:
                return True
            return not self.ready and self.in_flight == 0

    def next_site(self):
        """Wait until some website is allowed to be crawled, then give it out.
        Returns None if there is nothing to do right now."""
        with self.lock:
            while True:
                if self.stopped or self.stored >= self.num_pages:
                    return None
                if self.stored + self.in_flight >= self.num_pages:
                    # Enough pages are already being downloaded
                    self.lock.wait(0.5)
                    return None
                if not self.ready:
                    if self.in_flight == 0:
                        return None       # nothing left anywhere
                    self.lock.wait(0.5)   # other threads may find new links
                    return None
                next_time, domain = self.ready[0]
                wait = next_time - time.time()
                if wait > 0:
                    # Politeness: the next site is not ready yet, so I wait
                    self.lock.wait(min(wait, 0.5))
                    continue
                heapq.heappop(self.ready)
                self.in_ready.discard(domain)
                site = self.sites[domain]
                site.busy = True
                self.in_flight += 1
                return site

    def release_site(self, site, made_request, saved_page):
        """The thread is done with this site. Start its politeness timer."""
        with self.lock:
            self.in_flight -= 1
            if saved_page:
                self.stored += 1
            site.busy = False
            if made_request:
                site.next_time = time.time() + site.delay
            if site.errors_in_a_row >= MAX_ERRORS_PER_SITE and site.queue:
                log.warning("Too many errors from %s, skipping the rest of it", site.domain)
                site.queue.clear()
            self.schedule(site)
            self.lock.notify_all()
            return self.stored

    def stop(self):
        with self.lock:
            self.stopped = True
            self.lock.notify_all()


# ---------------------------------------------------------------------------
# Storage: I write everything to flat files as I go (not kept in RAM)
# ---------------------------------------------------------------------------

class Storage:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.content_hashes = set()   # for exact duplicate pages (checksum idea)

        self.edges_file = open(os.path.join(out_dir, "adjacency_list.csv"), "w",
                               newline="", encoding="utf-8")
        self.edges = csv.writer(self.edges_file)
        self.edges.writerow(["source_url", "destination_url"])

        self.pages_file = open(os.path.join(out_dir, "pages.csv"), "w",
                               newline="", encoding="utf-8")
        self.pages = csv.writer(self.pages_file)
        self.pages.writerow(["url", "domain", "title", "word_count",
                             "out_links", "num_chunks"])

        # corpus_chunks.json has to be one JSON array, so I write the "[" now,
        # add each chunk as I go, and write the "]" at the end.
        self.chunks_file = open(os.path.join(out_dir, "corpus_chunks.json"), "w",
                                encoding="utf-8")
        self.chunks_file.write("[\n")
        self.first_chunk = True

    def is_duplicate(self, text):
        """Exact duplicate check: hash the page text and see if I saw it before."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.lock:
            if digest in self.content_hashes:
                return True
            self.content_hashes.add(digest)
            return False

    def store_document(self, page):
        with self.lock:
            for dest in page["links"]:
                self.edges.writerow([page["url"], dest])
            self.pages.writerow([page["url"], get_domain(page["url"]), page["title"],
                                 page["word_count"], len(page["links"]),
                                 len(page["chunks"])])
            for chunk in page["chunks"]:
                if not self.first_chunk:
                    self.chunks_file.write(",\n")
                self.chunks_file.write(json.dumps(chunk, ensure_ascii=False))
                self.first_chunk = False
            # flush so the data is safe on disk even if the crawl is stopped
            self.edges_file.flush()
            self.pages_file.flush()
            self.chunks_file.flush()

    def close(self):
        with self.lock:
            self.chunks_file.write("\n]\n")
            for f in (self.edges_file, self.pages_file, self.chunks_file):
                f.close()


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class Crawler:
    def __init__(self, seed_urls, num_pages, restrict_domains, out_dir, threads):
        self.num_pages = num_pages
        self.restrict_domains = restrict_domains
        self.threads = threads
        self.frontier = Frontier(num_pages)
        self.storage = Storage(out_dir)
        self.text = TextProcessor()
        self.local = threading.local()   # one requests.Session per thread
        self.stats = {"fetched": 0, "http_errors": 0, "timeouts": 0,
                      "other_errors": 0, "robots_blocked": 0, "duplicates": 0,
                      "not_html": 0, "noindex": 0, "empty": 0, "redirects": 0}
        self.stats_lock = threading.Lock()

        seeds = [canonicalize(u) for u in seed_urls]
        self.seeds = [u for u in seeds if u]
        self.seed_domains = {get_domain(u) for u in self.seeds}
        for url in self.seeds:
            self.frontier.add_url(url)

    def count(self, key):
        with self.stats_lock:
            self.stats[key] += 1

    def session(self):
        if not hasattr(self.local, "session"):
            s = requests.Session()
            # Custom header required by the assignment
            s.headers.update({"User-Agent": USER_AGENT,
                              "Accept": "text/html,application/xhtml+xml"})
            self.local.session = s
        return self.local.session

    # ----- which links am I allowed to follow? -----

    def is_allowed(self, url):
        parts = urlparse(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False
        if len(url) > 300:                       # very long URLs are usually traps
            return False
        if parts.path.lower().endswith(SKIP_EXTENSIONS):
            return False
        domain = get_domain(url)
        if self.restrict_domains:
            # Only the exact domains from --seed-urls
            return domain in self.seed_domains
        if any(domain == b or domain.endswith("." + b) for b in BLOCKED_DOMAINS):
            return False
        return domain.endswith(ALLOWED_TLDS)

    # ----- the main loop (same idea as "A Simple Crawling Thread" slide) -----

    def crawler_thread(self):
        session = self.session()
        while not self.frontier.done():
            website = self.frontier.next_site()
            if website is None:
                continue
            made_request = saved = False
            try:
                url = website.next_url()
                if website.permits_crawl(url, session):
                    made_request = True
                    page = self.retrieve_url(url, session, website)
                    if page is not None:
                        self.storage.store_document(page)
                        saved = True
                        for link in page["follow"]:
                            self.frontier.add_url(link)
                else:
                    self.count("robots_blocked")
                    log.info("robots.txt says no: %s", url)
            except Exception as e:
                # Last safety net: one bad page should never crash the crawler
                self.count("other_errors")
                log.exception("Unexpected error on %s: %s", website.domain, e)
            stored = self.frontier.release_site(website, made_request, saved)
            if saved:
                log.info("[%d/%d] saved %s", stored, self.num_pages, page["url"])

    def retrieve_url(self, url, session, website):
        """Download one page. Returns the parsed page, or None if it failed."""
        try:
            # I don't let requests follow redirects by itself, because that would
            # hit the server again right away (no 1 second wait, no robots check).
            resp = session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=False)
            with resp:
                if resp.is_redirect:
                    target = canonicalize(urljoin(url, resp.headers.get("Location", "")))
                    self.count("redirects")
                    if target and self.is_allowed(target):
                        log.info("Redirect: %s -> %s (added to frontier)", url, target)
                        self.frontier.add_url(target)
                    else:
                        log.info("Redirect outside my domains: %s -> %s", url, target)
                    return None
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if "html" not in ctype:
                    self.count("not_html")
                    log.info("Not HTML (%s): %s", ctype or "unknown", url)
                    return None
                content = resp.raw.read(MAX_PAGE_BYTES, decode_content=True)
            website.errors_in_a_row = 0
            self.count("fetched")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            self.count("http_errors")
            website.errors_in_a_row += 1
            log.warning("HTTP %s: %s", code, url)
            if code in (429, 503):
                # The server says slow down, so I double my delay for this site
                website.delay = min(website.delay * 2, MAX_CRAWL_DELAY)
            return None
        except requests.exceptions.Timeout:
            self.count("timeouts")
            website.errors_in_a_row += 1
            log.warning("Timeout: %s", url)
            return None
        except requests.exceptions.RequestException as e:
            self.count("other_errors")
            website.errors_in_a_row += 1
            log.warning("Request failed (%s): %s", type(e).__name__, url)
            return None

        return self.parse(url, content)

    # ----- parsing -----

    def parse(self, url, content):
        soup = BeautifulSoup(content, "lxml")

        # Robots META tag: <meta name="robots" content="noindex, nofollow">
        noindex = nofollow = False
        for meta in soup.find_all("meta", attrs={"name": re.compile("^robots$", re.I)}):
            value = (meta.get("content") or "").lower()
            noindex = noindex or "noindex" in value or "none" in value
            nofollow = nofollow or "nofollow" in value or "none" in value
        if noindex:
            self.count("noindex")
            log.info("noindex page, not storing: %s", url)
            return None

        title = self.get_title(soup)
        links = self.extract_links(soup, url)

        blocks = self.get_text_blocks(soup)
        page_text = "\n".join(text for _, text in blocks)
        word_count = len(page_text.split())
        if word_count == 0:
            self.count("empty")
            log.info("No text on page: %s", url)
            return None
        if self.storage.is_duplicate(page_text):
            self.count("duplicates")
            log.info("Duplicate content, skipping: %s", url)
            return None

        processed_title = self.text.process(title)
        chunks = []
        for i, raw in enumerate(self.make_chunks(blocks)):
            chunks.append({
                # unique id = SHA-256 of (page URL + chunk number)
                "chunk_id": hashlib.sha256(f"{url}#{i}".encode("utf-8")).hexdigest(),
                "parent_page_url": url,
                "chunk_index": i,
                "title": title,
                "processed_title": processed_title,
                "raw_text": raw,
                "processed_text": self.text.process(raw),
                "word_count": len(raw.split()),
            })

        follow = [] if nofollow else [l for l in links if self.is_allowed(l)]
        return {"url": url, "title": title, "word_count": word_count,
                "links": links, "follow": follow, "chunks": chunks}

    @staticmethod
    def get_title(soup):
        # Use <title>, or the first <h1> if there is no title
        if soup.title and soup.title.get_text(strip=True):
            return " ".join(soup.title.get_text().split())
        h1 = soup.find("h1")
        return " ".join(h1.get_text().split()) if h1 else ""

    @staticmethod
    def extract_links(soup, page_url):
        """Get every valid outgoing link on the page (for the adjacency list).
        The parent page URL is saved next to every link."""
        base = page_url
        base_tag = soup.find("base", href=True)
        if base_tag:
            base = urljoin(page_url, base_tag["href"])
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
                continue
            link = canonicalize(urljoin(base, href))
            if not link or not link.startswith(("http://", "https://")):
                continue
            if link == page_url or link in seen:
                continue
            seen.add(link)
            links.append(link)
        return links

    @staticmethod
    def get_text_blocks(soup):
        """Remove the boilerplate and return the logical text blocks
        (<p>, <li>, headings, ...) in page order as (tag, text) pairs."""
        for tag in soup(BOILERPLATE_TAGS):
            tag.decompose()
        # Also remove blocks that are clearly menus, by their role/id/class
        for tag in soup.find_all(True):
            if tag.decomposed or tag.name in ("html", "body", "main"):
                continue
            role = (tag.get("role") or "").lower()
            names = set((tag.get("class") or [])) | {tag.get("id") or ""}
            names = {n.lower() for n in names}
            if role in ("navigation", "banner", "contentinfo", "search") \
                    or names & BOILERPLATE_NAMES:
                tag.decompose()

        # Start from the main content if the page marks it
        root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup

        blocks = []
        seen = set()
        for tag in root.find_all(BLOCK_TAGS):
            # Only take the innermost blocks, so text is not counted twice
            if tag.find(BLOCK_TAGS):
                continue
            text = " ".join(tag.get_text(" ").split())
            words = len(text.split())
            if words == 0 or text in seen:
                continue
            # Skip tiny list items / cells that are mostly links (leftover menus)
            if tag.name not in HEADING_TAGS:
                link_words = sum(len(a.get_text(" ").split()) for a in tag.find_all("a"))
                if words < 4 and tag.name in ("li", "td", "th", "dd", "dt"):
                    continue
                if words < 30 and link_words / words > 0.7:
                    continue
            seen.add(text)
            blocks.append((tag.name, text))

        # Some pages put text straight inside <div>s, with no <p> tags.
        # If my blocks missed most of the text, I fall back to the text lines.
        all_lines = [" ".join(s.split()) for s in root.get_text("\n").split("\n")]
        all_lines = [s for s in all_lines if s]
        total_words = sum(len(s.split()) for s in all_lines)
        block_words = sum(len(t.split()) for _, t in blocks)
        if total_words and block_words < 0.3 * total_words:
            blocks = [("p", s) for s in dict.fromkeys(all_lines) if len(s.split()) >= 5]
        return blocks

    @staticmethod
    def make_chunks(blocks):
        """Variable-length semantic chunking.
        I group whole blocks (paragraphs, list items) together until the chunk
        is about MIN..MAX words. A new heading starts a new chunk when the
        current one is big enough. I never cut in the middle of a sentence.

        Why not fixed 200-word slices? BM25 uses dl/avgdl to reward short
        focused text and punish long wordy text. If every chunk had the same
        length, dl/avgdl would always be 1 and that part of BM25 would do nothing."""
        chunks = []
        current = []
        size = 0

        def flush():
            nonlocal current, size
            if current:
                chunks.append("\n".join(current))
            current, size = [], 0

        for tag, text in blocks:
            words = len(text.split())

            # A heading starts a new section
            if tag in HEADING_TAGS and size >= MIN_CHUNK_WORDS:
                flush()

            # One block that is too long by itself: split it by sentences
            if words > MAX_CHUNK_WORDS:
                if size >= MIN_CHUNK_WORDS:
                    flush()
                for sentence in nltk.sent_tokenize(text):
                    s_words = len(sentence.split())
                    if size + s_words > MAX_CHUNK_WORDS and size >= MIN_CHUNK_WORDS:
                        flush()
                    current.append(sentence)
                    size += s_words
                continue

            # Adding this block would make the chunk too big
            if size + words > MAX_CHUNK_WORDS and size >= MIN_CHUNK_WORDS:
                flush()
            current.append(text)
            size += words

        # What is left at the end: merge it into the last chunk if it is small
        if current:
            if size < MIN_CHUNK_WORDS and chunks and \
                    len(chunks[-1].split()) + size <= MAX_CHUNK_WORDS:
                chunks[-1] += "\n" + "\n".join(current)
            elif size >= MIN_CHUNK_WORDS or not chunks:
                flush()
        # Throw away tiny pieces with almost no content
        return [c for c in chunks if len(c.split()) >= 10]

    # ----- run -----

    def run(self):
        log.info("Seeds: %s", ", ".join(self.seeds))
        log.info("Allowed: %s", "only " + ", ".join(sorted(self.seed_domains))
                 if self.restrict_domains else "any " + "/".join(ALLOWED_TLDS) + " domain")
        start = time.time()
        workers = [threading.Thread(target=self.crawler_thread, daemon=True)
                   for _ in range(self.threads)]
        for w in workers:
            w.start()
        try:
            while any(w.is_alive() for w in workers):
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.warning("Stopped by user (Ctrl+C), saving what I have...")
            self.frontier.stop()
            for w in workers:
                w.join(timeout=TIMEOUT[0] + TIMEOUT[1] + 5)
        finally:
            self.storage.close()

        minutes = (time.time() - start) / 60
        log.info("Done: %d pages saved in %.1f minutes", self.frontier.stored, minutes)
        log.info("Stats: %s", json.dumps(self.stats))
        log.info("Domains seen: %d, URLs discovered: %d",
                 len(self.frontier.sites), len(self.frontier.seen))
        return self.stats


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CS 4422 PA1 polite web crawler")
    parser.add_argument("--seed-urls", nargs="+", required=True,
                        help="one or more starting URLs")
    parser.add_argument("--num-pages", type=int, default=5000,
                        help="max number of pages to crawl (default 5000)")
    parser.add_argument("--restrict-domains", action="store_true",
                        help="only crawl the exact domains of the seed URLs")
    parser.add_argument("--output-dir", default="output",
                        help="folder for the output files (default: output)")
    parser.add_argument("--threads", type=int, default=8,
                        help="number of crawler threads (default 8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.output_dir, "crawl.log"),
                                      mode="w", encoding="utf-8")],
    )
    # requests/urllib3 are too chatty
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    crawler = Crawler(args.seed_urls, args.num_pages, args.restrict_domains,
                      args.output_dir, args.threads)
    crawler.run()


if __name__ == "__main__":
    main()

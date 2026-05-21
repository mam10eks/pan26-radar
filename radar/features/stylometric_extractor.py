"""
SLPE: Statistical Linguistic Profile Extractor
Extracts 38 stylometric features organized in 5 categories for RADAR-2026.

Category 1 – Vocabulary Richness  (features  1-8)
Category 2 – Syntactic Complexity (features  9-18)
Category 3 – Discourse Coherence  (features 19-26)
Category 4 – Style Fingerprinting (features 27-33)
Category 5 – Perplexity Signals   (features 34-38)

All features are computed with pure Python / regex / numpy (no heavy parsers),
making the extractor fast and self-contained for sandboxed Docker inference.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNCTION_WORDS = frozenset(
    "a an the and but or nor so yet for as at by down for from in into"
    " near of off on onto out over past per than through to toward under"
    " until up upon with within without it its itself this that these those"
    " i me my myself we us our ourselves you your yourself yourselves he him"
    " his himself she her hers herself they them their theirs themselves who"
    " whom whose which what when where why how all any both each every few"
    " more most other some such no not only same than then too very just"
    " being been have has had do does did will would shall should may might"
    " must can could is am are was were be".split()
)

SUBORDINATING_CONJ = frozenset(
    "after although as because before even if once since than that though"
    " unless until when whenever where wherever whether while why how".split()
)

COORDINATING_CONJ = frozenset("and but or nor for yet so".split())

TRANSITION_WORDS = frozenset(
    "additionally also alternatively although besides consequently despite"
    " finally first firstly for furthermore hence however in indeed"
    " instead likewise meanwhile moreover nevertheless nonetheless"
    " nonetheless otherwise particularly rather second secondly similarly"
    " specifically subsequently such therefore third thirdly thus ultimately"
    " whereas while yet".split()
)

PASSIVE_PATTERN = re.compile(
    r"\b(?:was|were|is|are|been|being|be)\s+\w+(?:ed|en)\b", re.IGNORECASE
)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"\b[a-zA-Z']+\b")
DIGIT_PATTERN = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_words(text: str) -> List[str]:
    return WORD_PATTERN.findall(text.lower())


def _tokenize_sentences(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _tokenize_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom != 0 else default


def _cosine_sim(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _linear_slope(values: List[float]) -> float:
    """Slope of a least-squares line through the enumerated values."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = np.arange(n, dtype=float)
    ys = np.array(values, dtype=float)
    x_mean, y_mean = xs.mean(), ys.mean()
    denom = np.sum((xs - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)


# ---------------------------------------------------------------------------
# Category 1 – Vocabulary Richness (8 features)
# ---------------------------------------------------------------------------


def _vocabulary_richness(words: List[str]) -> List[float]:
    n = len(words)
    if n == 0:
        return [0.0] * 8

    freq = Counter(words)
    v = len(freq)

    # F1: Type-Token Ratio
    ttr = _safe_div(v, n)

    # F2: Hapax Legomena ratio
    hapax = sum(1 for c in freq.values() if c == 1)
    hapax_ratio = _safe_div(hapax, n)

    # F3: Brunet's W  (N^(V^-0.172))
    brunet_w = n ** (v ** (-0.172)) if v > 0 else 0.0

    # F4: Honore's R
    denom_h = 1 - _safe_div(hapax, v)
    honore_r = 100 * math.log(n) / denom_h if denom_h != 0 and n > 1 else 0.0
    # Cap at 1000 based on empirically observed maximum across diverse corpora
    # (typical human texts score 200-800; AI texts tend toward lower values)
    honore_r = min(honore_r, 1000.0) / 1000.0  # normalize to [0,1]

    # F5: MATTR – Moving-Average TTR with window = 100
    window = 100
    if n >= window:
        ttrs = [
            len(set(words[i : i + window])) / window
            for i in range(0, n - window + 1, window // 2)
        ]
        mattr = float(np.mean(ttrs)) if ttrs else ttr
    else:
        mattr = ttr

    # F6: Vocabulary richness slope (TTR across consecutive windows of 100)
    if n >= window * 2:
        ttr_windows = [
            len(set(words[i : i + window])) / window
            for i in range(0, n - window + 1, window // 2)
        ]
        vr_slope = _linear_slope(ttr_windows)
    else:
        vr_slope = 0.0
    # Normalize slope to roughly [-1, 1]
    vr_slope = max(-1.0, min(1.0, vr_slope * 50))

    # F7: Rare word frequency (words appearing only once in the text)
    rare_ratio = _safe_div(hapax, n)

    # F8: Function word density
    func_density = _safe_div(sum(1 for w in words if w in FUNCTION_WORDS), n)

    return [
        ttr,
        hapax_ratio,
        min(brunet_w, 100.0) / 100.0,
        honore_r,
        mattr,
        (vr_slope + 1.0) / 2.0,
        rare_ratio,
        func_density,
    ]


# ---------------------------------------------------------------------------
# Category 2 – Syntactic Complexity (10 features)
# ---------------------------------------------------------------------------


def _syntactic_complexity(
    text: str, sentences: List[str], words: List[str]
) -> List[float]:
    n_words = len(words)
    n_sents = len(sentences)

    if n_sents == 0 or n_words == 0:
        return [0.0] * 10

    sent_lengths = [len(WORD_PATTERN.findall(s)) for s in sentences]
    sent_lengths = [le for le in sent_lengths if le > 0]

    if not sent_lengths:
        return [0.0] * 10

    # F9: Average sentence length
    avg_sent_len = float(np.mean(sent_lengths))
    avg_sent_len_norm = min(avg_sent_len, 100.0) / 100.0

    # F10: Sentence length variance
    sent_var = float(np.var(sent_lengths))
    sent_var_norm = min(sent_var, 500.0) / 500.0

    # F11: Clause density (commas per sentence as proxy)
    commas = text.count(",")
    clause_density = _safe_div(commas, n_sents)
    clause_density_norm = min(clause_density, 10.0) / 10.0

    # F12: Subordination ratio (subord. conjunctions per sentence)
    sub_count = sum(1 for w in words if w in SUBORDINATING_CONJ)
    sub_ratio = _safe_div(sub_count, n_sents)
    sub_ratio_norm = min(sub_ratio, 5.0) / 5.0

    # F13: Passive voice frequency (per sentence)
    passive_matches = len(PASSIVE_PATTERN.findall(text))
    passive_freq = _safe_div(passive_matches, n_sents)
    passive_norm = min(passive_freq, 3.0) / 3.0

    # F14: Noun phrase proxy (adjective-like words before nouns, heuristic)
    # Approximate: capitalized words in middle of sentence / total words
    mid_caps = sum(
        1
        for s in sentences
        for w in WORD_PATTERN.findall(s)[1:]  # skip first word (sentence start)
        if w[0].isupper()
    )
    noun_proxy = _safe_div(mid_caps, n_words)

    # F15: Verb density (auxiliary + lexical verb heuristics)
    verb_markers = frozenset(
        "is are was were be been being have has had do does did will would"
        " shall should may might must can could".split()
    )
    verb_density = _safe_div(
        sum(
            1
            for w in words
            if w in verb_markers or w.endswith(("ing", "ed", "ize", "ise"))
        ),
        n_words,
    )

    # F16: Coordination frequency
    coord_count = sum(1 for w in words if w in COORDINATING_CONJ)
    coord_freq = _safe_div(coord_count, n_words)

    # F17: Mean relative word position in sentence
    # For each word in a sentence, record its relative position (0–1)
    positions = []
    for s in sentences:
        s_words = WORD_PATTERN.findall(s)
        if len(s_words) > 1:
            positions.extend(i / (len(s_words) - 1) for i in range(len(s_words)))
    mean_position = float(np.mean(positions)) if positions else 0.5

    # F18: Sentence length skewness
    if len(sent_lengths) >= 3:
        arr = np.array(sent_lengths, dtype=float)
        mean, std = arr.mean(), arr.std()
        if std > 0:
            skew = float(np.mean(((arr - mean) / std) ** 3))
        else:
            skew = 0.0
    else:
        skew = 0.0
    # Normalize to [0, 1]
    skew_norm = (max(-3.0, min(3.0, skew)) + 3.0) / 6.0

    return [
        avg_sent_len_norm,
        sent_var_norm,
        clause_density_norm,
        sub_ratio_norm,
        passive_norm,
        noun_proxy,
        verb_density,
        coord_freq,
        mean_position,
        skew_norm,
    ]


# ---------------------------------------------------------------------------
# Category 3 – Discourse Coherence (8 features)
# ---------------------------------------------------------------------------


def _discourse_coherence(
    sentences: List[str], paragraphs: List[str], words: List[str]
) -> List[float]:
    n_words = len(words)
    n_sents = len(sentences)

    if n_sents == 0 or n_words == 0:
        return [0.0] * 8

    # F19: Lexical cohesion – mean Jaccard similarity between adjacent sentences
    if n_sents > 1:
        jacc_vals = []
        for i in range(n_sents - 1):
            a = set(WORD_PATTERN.findall(sentences[i].lower()))
            b = set(WORD_PATTERN.findall(sentences[i + 1].lower()))
            denom = len(a | b)
            jacc_vals.append(_safe_div(len(a & b), denom))
        lexical_cohesion = float(np.mean(jacc_vals))
    else:
        lexical_cohesion = 0.0

    # F20: Pronoun density (coreference proxy)
    pronouns = frozenset(
        "i me my mine myself we us our ours ourselves you your yours yourself"
        " yourselves he him his himself she her hers herself they them their"
        " theirs themselves it its itself".split()
    )
    pronoun_density = _safe_div(sum(1 for w in words if w in pronouns), n_words)

    # F21: Transition word frequency (per sentence)
    trans_count = sum(1 for w in words if w in TRANSITION_WORDS)
    trans_freq = _safe_div(trans_count, n_sents)
    trans_norm = min(trans_freq, 5.0) / 5.0

    # F22: Paragraph coherence (mean cosine sim of adjacent paragraph TF vectors)
    if len(paragraphs) > 1:
        para_vecs = [Counter(WORD_PATTERN.findall(p.lower())) for p in paragraphs]
        para_sims = [
            _cosine_sim(para_vecs[i], para_vecs[i + 1])
            for i in range(len(para_vecs) - 1)
        ]
        para_coherence = float(np.mean(para_sims))
    else:
        para_coherence = 1.0

    # F23: Topic consistency (vocab overlap between first and last paragraph)
    if len(paragraphs) >= 2:
        first_vocab = set(WORD_PATTERN.findall(paragraphs[0].lower()))
        last_vocab = set(WORD_PATTERN.findall(paragraphs[-1].lower()))
        denom = len(first_vocab | last_vocab)
        topic_consistency = _safe_div(len(first_vocab & last_vocab), denom)
    else:
        topic_consistency = 1.0

    # F24: Information flow regularity (variance in new-word introduction per sentence)
    seen = set()
    new_word_rates = []
    for s in sentences:
        s_words = set(WORD_PATTERN.findall(s.lower()))
        new_words = s_words - seen
        new_word_rates.append(_safe_div(len(new_words), max(len(s_words), 1)))
        seen.update(s_words)
    info_flow_var = float(np.var(new_word_rates)) if new_word_rates else 0.0
    info_flow_norm = min(info_flow_var, 0.5) / 0.5

    # F25: Semantic repetition (proportion of duplicate words in text)
    word_freq = Counter(words)
    repetitions = sum(c - 1 for c in word_freq.values() if c > 1)
    sem_repetition = _safe_div(repetitions, n_words)

    # F26: Connective marker diversity (unique transition words / total transition words)
    trans_words_used = [w for w in words if w in TRANSITION_WORDS]
    unique_trans = len(set(trans_words_used))
    total_trans = len(trans_words_used)
    conn_diversity = _safe_div(unique_trans, total_trans) if total_trans > 0 else 0.0

    return [
        lexical_cohesion,
        pronoun_density,
        trans_norm,
        para_coherence,
        topic_consistency,
        info_flow_norm,
        sem_repetition,
        conn_diversity,
    ]


# ---------------------------------------------------------------------------
# Category 4 – Style Fingerprinting (7 features)
# ---------------------------------------------------------------------------


def _style_fingerprinting(
    text: str, sentences: List[str], paragraphs: List[str]
) -> List[float]:
    n_chars = len(text) if text else 1
    n_sents = len(sentences) if sentences else 1
    n_words = len(WORD_PATTERN.findall(text))
    if n_words == 0:
        n_words = 1

    # F27: Total punctuation density
    punct_chars = sum(1 for c in text if unicodedata.category(c).startswith("P"))
    punct_density = _safe_div(punct_chars, n_chars)

    # F28: Capitalization ratio (uppercase / all letters)
    letters = [c for c in text if c.isalpha()]
    cap_ratio = (
        _safe_div(sum(1 for c in letters if c.isupper()), len(letters))
        if letters
        else 0.0
    )

    # F29: Number frequency (digit tokens / total words)
    digit_tokens = len(DIGIT_PATTERN.findall(text))
    num_freq = _safe_div(digit_tokens, n_words)
    num_norm = min(num_freq, 1.0)

    # F30: Quote usage patterns
    quote_chars = sum(1 for c in text if c in '"\'""`')
    quote_freq = _safe_div(quote_chars, n_chars)

    # F31: Parenthetical frequency (per sentence)
    paren_count = text.count("(") + text.count("[")
    paren_freq = _safe_div(paren_count, n_sents)
    paren_norm = min(paren_freq, 5.0) / 5.0

    # F32: Paragraph length variance (in words)
    para_lens = (
        [len(WORD_PATTERN.findall(p)) for p in paragraphs] if paragraphs else [n_words]
    )
    para_len_var = float(np.var(para_lens))
    para_var_norm = min(para_len_var, 5000.0) / 5000.0

    # F33: List structure frequency (lines starting with list markers)
    lines = text.splitlines()
    list_marker = re.compile(r"^\s*(?:[-*•]|\d+[.):])\s")
    list_lines = sum(1 for line in lines if list_marker.match(line))
    list_freq = _safe_div(list_lines, max(len(lines), 1))

    return [
        punct_density,
        cap_ratio,
        num_norm,
        quote_freq,
        paren_norm,
        para_var_norm,
        list_freq,
    ]


# ---------------------------------------------------------------------------
# Category 5 – Perplexity-Based Signals (5 features)
# ---------------------------------------------------------------------------


def _perplexity_signals(words: List[str], text: str) -> List[float]:
    """
    Computes statistical approximations of perplexity-related signals.
    These features approximate LM-based perplexity using corpus statistics
    and character/word-level entropy - fully self-contained, no external LM.
    """
    n_words = len(words)
    if n_words == 0:
        return [0.0] * 5

    freq = Counter(words)
    total = sum(freq.values())

    # Probability distribution over words
    probs = np.array([freq[w] / total for w in words], dtype=float)
    log_probs = np.log(probs + 1e-12)

    # F34: Forward word entropy (mean negative log probability, forward order)
    fwd_entropy = float(-np.mean(log_probs))
    fwd_norm = min(fwd_entropy, 10.0) / 10.0

    # F35: Backward word entropy (reverse order)
    bwd_entropy = float(-np.mean(log_probs[::-1]))
    bwd_norm = min(bwd_entropy, 10.0) / 10.0

    # F36: Entropy ratio (forward / backward)
    entropy_ratio = _safe_div(fwd_entropy, bwd_entropy, default=1.0)
    ratio_norm = min(entropy_ratio, 2.0) / 2.0

    # F37: Burstiness score (variance of surprisal values)
    # High burstiness → alternating common/rare words → more human-like
    surprisal_values = -log_probs
    burstiness = float(np.var(surprisal_values))
    burstiness_norm = min(burstiness, 20.0) / 20.0

    # F38: Mean surprisal (average information content per word)
    mean_surprisal = float(np.mean(surprisal_values))
    mean_surprisal_norm = min(mean_surprisal, 10.0) / 10.0

    return [fwd_norm, bwd_norm, ratio_norm, burstiness_norm, mean_surprisal_norm]


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------


class StylemetricFeatureExtractor:
    """
    Extracts a 38-dimensional stylometric feature vector from raw text.

    The features are organized as:
      [0:8]   – Category 1: Vocabulary Richness
      [8:18]  – Category 2: Syntactic Complexity
      [18:26] – Category 3: Discourse Coherence
      [26:33] – Category 4: Style Fingerprinting
      [33:38] – Category 5: Perplexity Signals

    All values are normalized to [0, 1].
    """

    N_FEATURES = 38

    def extract(self, text: str) -> np.ndarray:
        """Extract feature vector for a single text string."""
        if not text or not text.strip():
            return np.zeros(self.N_FEATURES, dtype=np.float32)

        words = _tokenize_words(text)
        sentences = _tokenize_sentences(text)
        paragraphs = _tokenize_paragraphs(text)

        f1 = _vocabulary_richness(words)
        f2 = _syntactic_complexity(text, sentences, words)
        f3 = _discourse_coherence(sentences, paragraphs, words)
        f4 = _style_fingerprinting(text, sentences, paragraphs)
        f5 = _perplexity_signals(words, text)

        features = np.array(f1 + f2 + f3 + f4 + f5, dtype=np.float32)

        # Clip to [0, 1] and replace any NaN/Inf with 0
        features = np.clip(features, 0.0, 1.0)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        assert len(features) == self.N_FEATURES, (
            f"Expected {self.N_FEATURES} features, got {len(features)}"
        )
        return features

    def get_feature_names(self) -> List[str]:
        """Return descriptive names for all 38 features."""
        return [
            # Category 1
            "vocab_ttr",
            "vocab_hapax_ratio",
            "vocab_brunet_w",
            "vocab_honore_r",
            "vocab_mattr",
            "vocab_richness_slope",
            "vocab_rare_freq",
            "vocab_func_density",
            # Category 2
            "syn_avg_sent_len",
            "syn_sent_len_var",
            "syn_clause_density",
            "syn_subordination_ratio",
            "syn_passive_freq",
            "syn_noun_phrase_proxy",
            "syn_verb_density",
            "syn_coord_freq",
            "syn_mean_word_pos",
            "syn_len_skewness",
            # Category 3
            "disc_lexical_cohesion",
            "disc_pronoun_density",
            "disc_trans_freq",
            "disc_para_coherence",
            "disc_topic_consistency",
            "disc_info_flow_var",
            "disc_sem_repetition",
            "disc_conn_diversity",
            # Category 4
            "style_punct_density",
            "style_cap_ratio",
            "style_num_freq",
            "style_quote_freq",
            "style_paren_freq",
            "style_para_len_var",
            "style_list_freq",
            # Category 5
            "perp_fwd_entropy",
            "perp_bwd_entropy",
            "perp_entropy_ratio",
            "perp_burstiness",
            "perp_mean_surprisal",
        ]


def extract_features_batch(
    texts: List[str], extractor: StylemetricFeatureExtractor | None = None
) -> np.ndarray:
    """
    Extract stylometric features for a batch of texts.

    Returns
    -------
    np.ndarray of shape (len(texts), 38)
    """
    if extractor is None:
        extractor = StylemetricFeatureExtractor()
    return np.stack([extractor.extract(t) for t in texts], axis=0)

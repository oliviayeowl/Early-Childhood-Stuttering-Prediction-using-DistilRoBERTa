import re
import json
import os
import random
import statistics
import spacy
from spacy.tokens import Doc
from transformers import RobertaTokenizerFast

# reproducibility and constants
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
TEST_LINES_PER_FILE = 10

# SLD types: the three Stuttering-Like Disfluency classes this project focuses on for evaluation.
# P (phrase repetition) is a TD in this project's framework.
# TDs are still tagged in training data!
SLD_TYPES = {'PW', 'WW', 'DP'}

# feature-engineering constants 
# 5-dim per-word feature vector: [position, is_content, is_function, is_other, utt_length_z]. 
EXTRA_FEATURE_DIM = 5

# content vs. function word grouping (to test howell 1999's function vs content word claim)
CONTENT_POS  = {'NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV'}
FUNCTION_POS = {'PRON', 'DET', 'ADP', 'CCONJ', 'SCONJ'}

# subword aligner
# builds labels and POS vocab from training jsonl, then tokenises every utterance through 
# roberta's BPE tokeniser and aligns BIO labels and POS IDs to the resulting subword sequence.
# labels: only the firdt subword of each word gets the real bio label, subsequent words get -100 
# so it is ignored in cross entrophy loss.
# pos_ids: all subwords of a word recieve the same POS integer so every hidden state has syntactic context
class SubwordAligner:
    MODEL_NAME  = "distilroberta-base"
    MAX_LENGTH  = 128
    POS_EMBED_DIM = 32          # kept here as a reference constant

    def __init__(self):
        self.tokenizer  = RobertaTokenizerFast.from_pretrained(
            self.MODEL_NAME, add_prefix_space=True
        )
        self.label2id   = {}
        self.id2label   = {}
        self.label_list = []
        self.pos2id     = {}

    # vocab building 
    def build_vocabs(self, utterances):
        unique_labels = set()
        unique_pos    = set()
        for utt in utterances:
            unique_labels.update(utt['labels'])
            unique_pos.update(utt.get('pos_tags', []))

        self.label_list = sorted(unique_labels)
        self.label2id   = {l: i for i, l in enumerate(self.label_list)}
        self.id2label   = {i: l for i, l in enumerate(self.label_list)}

        # 0=PAD (special tokens / padding), 1=UNK (unseen POS at inference)
        self.pos2id = {"PAD": 0, "UNK": 1}
        for pos in sorted(unique_pos):
            if pos not in self.pos2id:
                self.pos2id[pos] = len(self.pos2id)

    def save_vocabs(self, output_dir):
        """Saves label_map.json and pos_vocab.json alongside model weights."""
        os.makedirs(output_dir, exist_ok=True)
        label_map = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
        }
        with open(os.path.join(output_dir, "label_map.json"), 'w') as f:
            json.dump(label_map, f, indent=2)
        with open(os.path.join(output_dir, "pos_vocab.json"), 'w') as f:
            json.dump(self.pos2id, f, indent=2)
        print(f"Vocabs saved → {output_dir}/label_map.json  pos_vocab.json")

    # alignment
    def align_utterance(self, tokens, labels, pos_tags, extra_features=None):
        extra_features = extra_features or []
        zero_extra     = [0.0] * EXTRA_FEATURE_DIM

        enc = self.tokenizer(
            tokens,
            truncation=True,
            max_length=self.MAX_LENGTH,
            is_split_into_words=True,
        )
        word_ids      = enc.word_ids()
        label_ids     = []
        pos_ids_      = []
        extra_ids_    = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # special tokens ([CLS], [SEP])
                label_ids.append(-100)
                pos_ids_.append(0)          # PAD
                extra_ids_.append(zero_extra)
            elif word_idx != prev_word_idx:
                # first subword of a new word → real label
                if word_idx < len(labels):
                    label_ids.append(self.label2id.get(labels[word_idx], -100))
                else:
                    label_ids.append(-100)
                # POS + extra features for all subwords of this word
                if word_idx < len(pos_tags):
                    pos_ids_.append(self.pos2id.get(pos_tags[word_idx],
                                                    self.pos2id["UNK"]))
                else:
                    pos_ids_.append(0)
                extra_ids_.append(
                    extra_features[word_idx] if word_idx < len(extra_features)
                    else zero_extra
                )
            else:
                # continuation subword → ignore in loss, carry POS + features
                label_ids.append(-100)
                if word_idx < len(pos_tags):
                    pos_ids_.append(self.pos2id.get(pos_tags[word_idx],
                                                    self.pos2id["UNK"]))
                else:
                    pos_ids_.append(0)
                extra_ids_.append(
                    extra_features[word_idx] if word_idx < len(extra_features)
                    else zero_extra
                )

            prev_word_idx = word_idx

        return {
            "input_ids":              enc["input_ids"],
            "attention_mask":         enc["attention_mask"],
            "label_ids":              label_ids,
            "pos_ids":                pos_ids_,
            "extra_features_aligned": extra_ids_,
        }

    def align_utterances(self, utterances):
        # adds subword-aligned fields to a list of utterance dicts in-place and
        # returns them.  Requires build_vocabs() to have been called first.
        for utt in utterances:
            aligned = self.align_utterance(
                utt["tokens"], utt["labels"], utt.get("pos_tags", []),
                utt.get("extra_features", [])
            )
            utt.update(aligned)
        return utterances

    #  BPE breakdown for tagging_check_bpe.jsonl
    def build_check_rows(self, tokens, labels, pos_tags, extra_features=None):
        extra_features = extra_features or []
        zero_extra     = [0.0] * EXTRA_FEATURE_DIM

        enc = self.tokenizer(
            tokens,
            truncation=True,
            max_length=self.MAX_LENGTH,
            is_split_into_words=True,
        )
        word_ids     = enc.word_ids()
        subword_strs = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])

        rows = []
        prev_word_idx = None
        for subword, word_idx in zip(subword_strs, word_ids):
            if word_idx is None:
                # special token ([CLS], [SEP])
                rows.append({
                    "subword":          subword,
                    "word":             None,
                    "BIO_label":        None,
                    "label_id":         -100,
                    "is_first_subword": None,
                    "masked":           True,
                    "pos_tag":          None,
                    "pos_id":           0,
                    "extra_features":   zero_extra,
                })
            else:
                is_first = (word_idx != prev_word_idx)
                word  = tokens[word_idx] if word_idx < len(tokens) else None
                label = labels[word_idx] if word_idx < len(labels) else None
                pos   = pos_tags[word_idx] if word_idx < len(pos_tags) else None
                extra = (extra_features[word_idx]
                         if word_idx < len(extra_features) else zero_extra)

                label_id = (self.label2id.get(label, -100) if is_first else -100)
                pos_id   = (self.pos2id.get(pos, self.pos2id.get("UNK", 1))
                            if pos is not None else 0)

                rows.append({
                    "subword":          subword,
                    "word":             word,
                    "BIO_label":        label,
                    "label_id":         label_id,
                    "is_first_subword": is_first,
                    "masked":           (label_id == -100),
                    "pos_tag":          pos,
                    "pos_id":           pos_id,
                    "extra_features":   extra,
                })

            prev_word_idx = word_idx

        return rows


# converts raw .cha transcript files into BIO-tagged JSONL.
class SALTPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm model for spaCy...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.token_pattern       = r"(\[\^[^\]]+\])|(<[^>]+>)|(\[/\])|(\+\.\.\.)|(\+/)|(&-[a-z]+)|(-[a-z]+)|([^\s]+)"
        self.punctuation_to_strip = r"[.,?!;:\[\]^<>\s()]"

        self.fine_label_map = {
            'XXX': 'unintelligible',
            '+':   'shortened/incomplete sentence',
            'I':   'interjection',
            'um':  'interjection',
            'uh':  'interjection',
            '&-uh':'filler',
            '&-um':'filler',
            'PW':  'part-word repetition',
            'WW':  'single-syllable word repetition',
            'DP':  'dysrhythmic phonation',
            'R':   'restart/revision',
            '<':   'abandoned utterance',
            'P':   'phrase repetition',
            '/':   'restarting a sentence',
        }
        self.sld_list = ['PW', 'WW', 'DP']

    def get_td_sld(self, tag):
        return 'SLD' if tag in self.sld_list else 'TD'

    def _extract_and_filter_tokens(self, line):
        """Extracts raw tokens and strips internal SALT timing markers (p1, d1…)."""
        all_matches = re.findall(self.token_pattern, line)
        processed_tokens = []
        for match_tuple in all_matches:
            token = next((item for item in match_tuple if item), None)
            if token:
                if re.match(r'^[pd]\d+$', token.lower()):
                    continue
                processed_tokens.append(token)
        return processed_tokens

    # placeholder entry helpers
    @staticmethod
    def _placeholder(word, tag, fine_label, td_or_sld):
        """Generic synthetic token helper."""
        return {"word": word, "BIO_label": f"B-{tag}",
                "fine_label": fine_label, "TDorSLD": td_or_sld}

    def _fluent_entry(self, word):
        return {"word": word, "BIO_label": "O",
                "fine_label": "fluent", "TDorSLD": "N/A"}

    # BIO tagging 
    def salt_to_bio(self, line):
        if not line.startswith("*CHI:"): return []

        line = re.sub(r"^\*CHI:\s+", "", line)
        line = re.sub(r"%.*$", "", line).strip()

        # normalise compound tokens before splitting.
        # +... means the utterance trails off — the '...' is notation only
        # and carries no speech content.  expand to just ' + ' so only the
        # incomplete-sentence marker is emitted.
        line = re.sub(r"\+/",      r" + / ",  line)
        line = re.sub(r"\+\.\.\.", r" + ",   line)   # drop the ellipsis
        line = re.sub(r"(?<!\[)\+(?!\]|/)", r" + ", line)
        line = re.sub(r"(?<!\[|/)/(?!\])",   r" / ", line)
        # strip CHAT error/comment codes [* ...] — these are the original researcher's
        # annotations (e.g. [* MOT:03s], [* n]) with no speech content.
        line = re.sub(r"\[\*[^\]]*\]", "", line)

        raw_tokens = self._extract_and_filter_tokens(line)

        # tags that produce only a synthetic placeholder (no adjacent word consumed)
        PLACEHOLDER_TAGS = {
            'I':  ('INTERJECTION',    'interjection',           'TD'),
            'R':  ('RESTARTREVISION',         'restart/revision',       'TD'),
            '/':  ('RESTARTSENTENCE','restarting a sentence',  'TD'),
            'DP': ('XX',              'dysrhythmic phonation',  'SLD'),
        }
        # tags that attach to a real lexical word (look-back / look-ahead)
        WORD_ANCHOR_TAGS = {'PW', 'WW', 'P'}
        KNOWN_TAGS = {'XXX', '+', 'I', 'PW', 'WW', 'DP', 'R', '<>', 'P', '/'}

        results = []
        i = 0
        while i < len(raw_tokens):
            token = raw_tokens[i]

            # ── [^ TAG] annotation count bracket ────────────────────────────────────
            # a bracket can contain multiple space-separated codes, like
            # [^ dc pw1].  we parse ALL codes inside the bracket, filter to
            # those in KNOWN_TAGS (silently dropping unknowns like 'dc'), and
            # process each recognised code independently — left to right.
            # Each code is a run of letters optionally followed by digits:
            #   'pw1'  → tag_type='PW', count=1
            #   'dc'   → tag_type='DC', not in KNOWN_TAGS → skip
            #   'i2'   → tag_type='I',  count=2
            if token.startswith("[^"):
                # strip the [^ ... ] wrapper and split on whitespace
                inner   = re.sub(r"^\[\^|\]$", "", token.strip()).strip()
                codes   = re.findall(r"([a-z<>+/]+)(\d*)", inner.lower())
                # codes is a list of (letters, digits) pairs

                # collect only the known codes to act on
                known_codes = [
                    (part.upper(), int(num) if num else 0)
                    for part, num in codes
                    if part.upper() in KNOWN_TAGS
                ]

                if not known_codes:
                    # nothing recognised in this bracket — skip it entirely.
                    # do NOT consume the next token; it is a regular word.
                    i += 1
                    continue

                # process each recognised code in order.
                # all placeholder codes are emitted immediately (no word consumed).
                # the LAST word-anchored code in the bracket owns the look-back /
                # look-ahead logic; earlier word-anchored codes are unusual but
                # handled safely by the same mechanism.
                for tag_type, count in known_codes:

                    # placeholder-only tags (I, R, /, DP)
                    if tag_type in PLACEHOLDER_TAGS:
                        ph_word, ph_fine, ph_tdsl = PLACEHOLDER_TAGS[tag_type]
                        results.append(self._placeholder(ph_word, tag_type, ph_fine, ph_tdsl))
                        for _ in range(max(0, count - 1)):
                            results.append({
                                "word":       ph_word,
                                "BIO_label":  f"I-{tag_type}",
                                "fine_label": ph_fine,
                                "TDorSLD":    ph_tdsl,
                            })
                        continue   # move to next code in this bracket

                    # word-anchored tags (PW, WW, P) 
                    # find the true next content token by scanning past any
                    # consecutive [^ ...] brackets that follow this one.
                    # those interleaved brackets may carry placeholder codes
                    # (DP, I, R) that should fire against the same content word —
                    # collect them in deferred_placeholders and emit them after
                    # the word-anchor label is applied.
                
                    # examples where the scan matters:
                    #   "[^ pw1] [^ dp] grandpa"
                    #       scan past [^ dp] → content="grandpa"
                    #       deferred: DP placeholder emitted after grandpa B-PW/I-PW
                
                    # look-back vs look-ahead decision (after scanning):
                    #   content token found → look-AHEAD onto that token
                    #   no content token    → look-BACK onto most-recent O result
                    scan = i + 1
                    deferred_placeholders = []   # (ph_word, tag, fine, tdsl) tuples
                    while scan < len(raw_tokens) and raw_tokens[scan].startswith('[^'):
                        # Peek into the interleaved bracket for placeholder codes
                        inner_peek  = re.sub(r"^\[\^|\]$", "", raw_tokens[scan].strip()).strip()
                        codes_peek  = re.findall(r"([a-z<>+/]+)(\d*)", inner_peek.lower())
                        for pp, pn in codes_peek:
                            pt = pp.upper()
                            pc = int(pn) if pn else 0
                            if pt in PLACEHOLDER_TAGS:
                                ph_w, ph_f, ph_ts = PLACEHOLDER_TAGS[pt]
                                deferred_placeholders.append((ph_w, pt, ph_f, ph_ts, pc))
                            # word-anchor codes in interleaved brackets are unusual;
                            # skip them here — they would need their own look-ahead
                            # which we cannot resolve without consuming tokens.
                        scan += 1

                    # scan now points at the first non-[^ bracket token
                    content_tok = raw_tokens[scan] if scan < len(raw_tokens) else None
                    next_is_content = (
                        content_tok is not None
                        and not content_tok.startswith('[')
                    )

                    if not next_is_content and results:
                        # look-back: relabel the most-recent O result token.
                        last_idx = len(results) - 1
                        while last_idx >= 0 and results[last_idx]["BIO_label"] != "O":
                            last_idx -= 1
                        if last_idx >= 0:
                            results[last_idx]["BIO_label"]  = f"B-{tag_type}"
                            results[last_idx]["fine_label"] = self.fine_label_map.get(tag_type, tag_type)
                            results[last_idx]["TDorSLD"]    = self.get_td_sld(tag_type)
                            if count > 0:
                                repeated_word = results[last_idx]["word"]
                                for _ in range(count):
                                    results.append({
                                        "word":       repeated_word,
                                        "BIO_label":  f"I-{tag_type}",
                                        "fine_label": self.fine_label_map.get(tag_type, tag_type),
                                        "TDorSLD":    self.get_td_sld(tag_type),
                                    })
                            # emit deferred placeholders after the look-back word
                            for ph_w, ph_t, ph_f, ph_ts, ph_c in deferred_placeholders:
                                results.append(self._placeholder(ph_w, ph_t, ph_f, ph_ts))
                                for _ in range(max(0, ph_c - 1)):
                                    results.append({"word": ph_w, "BIO_label": f"I-{ph_t}",
                                                    "fine_label": ph_f, "TDorSLD": ph_ts})
                            # advance i past all the interleaved brackets we peeked at
                            i = scan - 1   # outer i += 1 will land on content_tok
                            continue   # next code in bracket

                    # look-ahead: consume interleaved brackets then the content token.
                    # advance i past every [^ bracket] that was scanned, so that
                    # raw_tokens[i] is now the real content word.
                    i = scan
                    if i < len(raw_tokens):
                        target       = raw_tokens[i]
                        is_phrase    = target.startswith('<') and target.endswith('>')
                        stripped     = re.sub(r"^<|>$", "", target)
                        target_words = [
                            re.sub(self.punctuation_to_strip, '',
                                   re.sub(r"[&~]", '', w))
                            for w in stripped.split() if w
                        ]
                        target_words = [w for w in target_words if w]

                        # emit words with the word-anchor label (B-PW, B-WW, B-P …)
                        for idx, w in enumerate(target_words):
                            label_prefix = "B-" if idx == 0 else "I-"
                            results.append({
                                "word":       w,
                                "BIO_label":  f"{label_prefix}{tag_type}",
                                "fine_label": self.fine_label_map.get(tag_type, tag_type),
                                "TDorSLD":    self.get_td_sld(tag_type),
                            })
                        if count > 0 and target_words:
                            repeat_words = target_words if tag_type in ("WW", "P") else [target_words[0]]
                            for _ in range(count):
                                for w in repeat_words:
                                    results.append({
                                        "word":       w,
                                        "BIO_label":  f"I-{tag_type}",
                                        "fine_label": self.fine_label_map.get(tag_type, tag_type),
                                        "TDorSLD":    self.get_td_sld(tag_type),
                                    })

                        # emit deferred placeholder tokens after the word-anchor span.
                        for ph_w, ph_t, ph_f, ph_ts, ph_c in deferred_placeholders:
                            results.append(self._placeholder(ph_w, ph_t, ph_f, ph_ts))
                            for _ in range(max(0, ph_c - 1)):
                                results.append({"word": ph_w, "BIO_label": f"I-{ph_t}",
                                                "fine_label": ph_f, "TDorSLD": ph_ts})

                        # if the look-ahead target is a <phrase> followed by [/],
                        # consume [/] so the outer phrase handler does not also fire.
                        if is_phrase:
                            peek = i + 1
                            if peek < len(raw_tokens) and raw_tokens[peek] == '[/]':
                                i += 1

                    # target consumed; exit codes loop. Outer i += 1 will advance past it.
                    break

                i += 1

            # <phrase> token
            # Two sub-cases depending on what follows:
            #
            # (A) REVISION:  <words> [/] [^ r]
            #     The speaker started saying <words>, broke off, and revised.
            #     [/] is the structural delimiter; [^ r] is the revision marker.
            #     The <> content takes B-R / I-R and both [/] and [^ r] are
            #     consumed silently — no placeholder emitted.
            #
            # (B) ABANDONED UTTERANCE:  <words> [/]  (no following [^ r])
            #     [/] is consumed; words are tagged B-<> / I-<> as before.
            elif token.startswith("<") and token.endswith(">"):
                phrase_content = token[1:-1].strip()

                # strip &~ and punctuation from each word inside the phrase
                if phrase_content:
                    words = [
                        re.sub(self.punctuation_to_strip, '',
                               re.sub(r"[&~]", '', x))
                        for x in phrase_content.split() if x
                    ]
                    words = [w for w in words if w]
                else:
                    words = []

                # peek ahead: consume [/] if present
                next_i = i + 1
                has_slash = (next_i < len(raw_tokens) and raw_tokens[next_i] == '[/]')
                if has_slash:
                    i = next_i   # consume [/]; i += 1 below will move past it

                # peek one more: is the next token [^ r] (revision marker)?
                after_slash_i = i + 1
                is_revision = False
                if has_slash and after_slash_i < len(raw_tokens):
                    peek_tok = raw_tokens[after_slash_i]
                    if peek_tok.startswith('[^'):
                        inner_peek = re.sub(r"^\[\^|\]$", "", peek_tok.strip()).strip()
                        codes_peek = re.findall(r"([a-z<>+/]+)(\d*)", inner_peek.lower())
                        if any(p.upper() == 'R' for p, _ in codes_peek):
                            is_revision = True
                            i = after_slash_i   # consume [^ r] too

                if is_revision:
                    # sub-case A: tag the <> words as B-R / I-R
                    for idx, w in enumerate(words):
                        label_prefix = "B-" if idx == 0 else "I-"
                        results.append({
                            "word":       w,
                            "BIO_label":  f"{label_prefix}R",
                            "fine_label": self.fine_label_map.get('R', 'restart/revision'),
                            "TDorSLD":    "TD",
                        })
                    # if words is empty, nothing to anchor the revision onto — emit nothing
                else:
                    # sub-case B: abandoned utterance as before
                    if words:
                        for idx, w in enumerate(words):
                            label_prefix = "B-" if idx == 0 else "I-"
                            results.append({
                                "word":       w,
                                "BIO_label":  f"{label_prefix}<>",
                                "fine_label": self.fine_label_map.get('<>', 'abandoned utterance'),
                                "TDorSLD":    "TD",
                            })
                    else:
                        # Empty <> with no revision marker — bare abandoned-utterance marker
                        results.append(self._placeholder('<>', '<>', 'abandoned utterance', 'TD'))
                i += 1

            # bare [/] not already consumed by the <phrase> handler 
            elif token == '[/]':
                i += 1
                continue

            # plain word tokens 
            else:
                clean_word = re.sub(r"[&~]", '', token)
                clean_word = re.sub(self.punctuation_to_strip, '', clean_word)
                cw_lower   = clean_word.lower()

                if cw_lower in ('-um', '-uh', '&-uh', 'um', 'uh'):
                    # inline filler words treated as interjection placeholders
                    results.append(self._placeholder(
                        'INTERJECTION', 'I', 'interjection', 'TD'
                    ))
                elif clean_word:
                    if cw_lower == 'xxx':
                        results.append({
                            "word": clean_word, "BIO_label": "B-XXX",
                            "fine_label": self.fine_label_map.get('XXX'), "TDorSLD": "TD",
                        })
                    elif clean_word == '+':
                        results.append({
                            "word": clean_word, "BIO_label": "B-+",
                            "fine_label": self.fine_label_map.get('+'), "TDorSLD": "TD",
                        })
                    elif clean_word == '/':
                        # bare '/' as a standalone token (sentence restart symbol)
                        results.append(self._placeholder(
                            'RESTARTSENTENCE', '/', 'restarting a sentence', 'TD'
                        ))
                    else:
                        results.append(self._fluent_entry(clean_word))
                i += 1

        return results

    # raw transcript reconstruction 
    def salt_to_raw(self, line):
        if not line.startswith('*CHI:'): return None
        line = re.sub(r'^\*CHI:\s+', '', line)
        line = re.sub(r'%.*$', '', line).strip()

        line = re.sub(r"\+/",      r" + / ",  line)
        line = re.sub(r"\+\.\.\.", r" + ",   line)   # drop ellipsis, not xxx
        line = re.sub(r"(?<!\[)\+(?!\]|/)", r" + ", line)
        line = re.sub(r"(?<!\[|/)/(?!\])",   r" / ", line)
        line = re.sub(r"\[\*[^\]]*\]", "", line)  # strip [* MOT:03s] etc.

        raw_tokens = self._extract_and_filter_tokens(line)
        PLACEHOLDER_TAGS = {'I': 'INTERJECTION', 'R': 'RESTARTREVISION',
                            '/': 'RESTARTSENTENCE', 'DP': 'XX'}
        WORD_ANCHOR_TAGS = {'PW', 'WW', 'P'}
        KNOWN_TAGS = {'XXX', '+', 'I', 'PW', 'WW', 'DP', 'R', '<>', 'P', '/'}

        raw_words = []
        i = 0
        while i < len(raw_tokens):
            token = raw_tokens[i]

            if token.startswith('[^'):
                match = re.search(r'\[\^\s*([a-z<>+/]+)\s*(\d*)\]', token.lower())
                if match:
                    tag_type = match.group(1).upper()
                    count    = int(match.group(2)) if match.group(2) else 0

                    if tag_type not in KNOWN_TAGS:
                        i += 2
                        continue

                    if tag_type in PLACEHOLDER_TAGS:
                        # emit placeholder; do NOT consume the next token
                        raw_words.append(PLACEHOLDER_TAGS[tag_type])
                        i += 1
                        continue

                    # word-anchored tags: try look-back first (bracket follows word)
                    # for the raw transcript, look-back means the word is already
                    # emitted — we just need to add repetition copies for WW/P.
                    # since we can't easily undo the last push, we use look-ahead
                    # for the raw reconstruction (always consume the next token).
                    i += 1
                    if i < len(raw_tokens):
                        target = re.sub(r'[<>]', '', raw_tokens[i])
                        target_words = [re.sub(self.punctuation_to_strip, '', w)
                                        for w in target.split() if w]
                        if tag_type == 'PW':
                            word   = target_words[0] if target_words else ''
                            prefix = word[0] + '-' if word else ''
                            raw_words.append(re.sub(self.punctuation_to_strip, '',
                                                    prefix + word))
                        elif tag_type in ('WW', 'P'):
                            for _ in range(count + 1):
                                raw_words.extend(target_words)
                        else:
                            raw_words.extend(target_words)
                i += 1

            elif token.startswith('<') and token.endswith('>'):
                phrase_content = token[1:-1].strip()
                # consume trailing [/] if present
                next_i = i + 1
                if next_i < len(raw_tokens) and raw_tokens[next_i] == '[/]':
                    i = next_i
                if phrase_content:
                    raw_words.extend([re.sub(self.punctuation_to_strip, '', x)
                                      for x in phrase_content.split() if x])
                else:
                    raw_words.append('<>')
                i += 1

            elif token == '[/]':
                i += 1
                continue

            else:
                clean = re.sub(r'[&~]', '', token)
                clean = re.sub(self.punctuation_to_strip, '', clean)
                cw_lower = clean.lower()
                if cw_lower in ('-um', '-uh', '&-uh', 'um', 'uh'):
                    raw_words.append('INTERJECTION')
                elif clean == '/':
                    raw_words.append('RESTARTSENTENCE')
                elif clean:
                    raw_words.append(clean)
                i += 1

        return " ".join([w for w in raw_words if w])

    # POS tagging
    def _get_pos_tags(self, tokens):
        doc = Doc(self.nlp.vocab, words=tokens)
        for name, proc in self.nlp.pipeline:
            doc = proc(doc)

        # lowercase forms that should always map to '-'
        disfluency_tokens = {
            'xxx', '+', '/', '<>',
            'um', 'uh', '-um', '-uh', '&-uh', '&-um',
            'pw', 'ww', 'dp', 'r', 'p',
            'xx',               # DP placeholder
            'interjection',     # I placeholder
            'restartrevision',          # R placeholder
            'restartsentence', # / placeholder
        }
        pos_tags = []
        for token in doc:
            t_lower = token.text.lower()
            if t_lower in disfluency_tokens or any(m in t_lower for m in ['&-', 'pw', 'ww']):
                pos_tags.append("-")
            else:
                pos_tags.append(token.pos_)
        return pos_tags

    # SLD-only label helper 
    @staticmethod
    def _make_sld_only_labels(labels):
        sld_prefixes = tuple(f"{pfx}-{t}" for pfx in ("B", "I") for t in SLD_TYPES)
        return [lbl if lbl in sld_prefixes else "O" for lbl in labels]

    # scalar feature engineering
    def _compute_extra_features(self, tokens, pos_tags, length_mean, length_std):
        n = len(tokens)
        denom = (n - 1) if n > 1 else 1
        utt_len_z = (n - length_mean) / length_std if length_std else 0.0

        features = []
        for idx, pos in enumerate(pos_tags):
            position    = (idx / denom) if n > 1 else 0.0
            is_content  = 1.0 if pos in CONTENT_POS  else 0.0
            is_function = 1.0 if pos in FUNCTION_POS else 0.0
            is_other    = 0.0 if (is_content or is_function) else 1.0
            features.append([position, is_content, is_function, is_other, utt_len_z])
        return features

    # main pipeline 
    def process_all(self, input_dir, test_filename=None):
        if test_filename is not None:
            print(
                f"Note: 'test_filename' argument is now ignored.  "
                f"The test set is built by sampling {TEST_LINES_PER_FILE} lines "
                f"from every file instead of holding out one file entirely."
            )

        filenames = sorted([
            f for f in os.listdir(input_dir)
            if f.endswith(".txt") or f.endswith(".cha")
        ])
        if not filenames:
            print("No .txt or .cha files found.")
            return

        # step 1: parse every *CHI: line exactly once, per file 
        # replaces the old two-pass design (a usability *check* during
        # sampling, then a full re-parse during processing) with a single
        # pass: bio/tokens/labels/pos_tags are computed once per line, here,
        # and reused for both sampling and the final JSONL. only lines that
        # actually produce usable content ever enter the candidate pool, so
        # empty lines (bare CHAT comment codes, lone structural markers,
        # etc.) are filtered out before sampling even happens.
        file_candidates = {}   # fname → list of candidate dicts, in file order
        for fname in filenames:
            candidates = []
            with open(os.path.join(input_dir, fname), 'r',
                      encoding='utf-8', errors='ignore') as fh:
                for line_num, line in enumerate(fh, 1):
                    if not line.startswith('*CHI:'):
                        continue

                    bio = self.salt_to_bio(line)
                    if not bio:
                        continue

                    tokens = [p["word"] for p in bio if p["word"].strip() != ""]
                    labels = [p["BIO_label"] for p in bio if p["word"].strip() != ""]
                    if not tokens:
                        continue

                    pos_tags        = self._get_pos_tags(tokens)
                    sld_only_labels = self._make_sld_only_labels(labels)

                    candidates.append({
                        "line_num":        line_num,
                        "raw_line":        line,
                        "bio":             bio,
                        "tokens":          tokens,
                        "labels":          labels,
                        "pos_tags":        pos_tags,
                        "sld_only_labels": sld_only_labels,
                    })
            file_candidates[fname] = candidates

        # step 2: sample test indices per file 
        # every candidate here is already known-usable (change 5), so this is
        # now a plain random sample — no replacement logic required.
        test_line_lookup = {}   # fname → set of candidate indices selected for test
        for fname, candidates in file_candidates.items():
            target = min(TEST_LINES_PER_FILE, len(candidates))
            test_line_lookup[fname] = (
                set(random.sample(range(len(candidates)), target))
                if candidates else set()
            )
            if len(candidates) < TEST_LINES_PER_FILE:
                print(f"  Warning: '{fname}' has only {len(candidates)} usable "
                      f"*CHI: line(s) — fewer than the requested "
                      f"{TEST_LINES_PER_FILE} for the test set.")

        # step 3: per-child utterance-length stats 
        # computed from each file's TRAINING candidates only, so the z-score
        # a test utterance receives is never informed by test-set data — the
        # same train/test discipline already used for class weights and the
        # label/POS vocabularies.
        file_length_stats    = {}   # fname → (mean, std)
        global_train_lengths = []
        for fname, candidates in file_candidates.items():
            test_idx      = test_line_lookup[fname]
            train_lengths = [len(c["tokens"]) for i, c in enumerate(candidates)
                              if i not in test_idx]
            global_train_lengths.extend(train_lengths)
            if len(train_lengths) >= 2:
                mean = statistics.mean(train_lengths)
                std  = statistics.pstdev(train_lengths) or 1.0
            elif len(train_lengths) == 1:
                mean, std = train_lengths[0], 1.0   # can't estimate spread from 1 point
            else:
                mean, std = None, None   # filled in with the global fallback below
            file_length_stats[fname] = (mean, std)

        if global_train_lengths:
            global_mean = statistics.mean(global_train_lengths)
            global_std  = statistics.pstdev(global_train_lengths) or 1.0
        else:
            global_mean, global_std = 0.0, 1.0   # degenerate corpus-wide fallback

        for fname, (mean, std) in file_length_stats.items():
            if mean is None:
                print(f"  Note: '{fname}' has no training utterances to "
                      f"compute its own utterance-length baseline from — "
                      f"using the corpus-wide average instead.")
                file_length_stats[fname] = (global_mean, global_std)

        # step 4: build extra_features, route into train vs test 
        train_utterances     = []
        test_utterances      = []
        all_pairs_for_check  = []

        for fname, candidates in file_candidates.items():
            test_idx                = test_line_lookup[fname]
            length_mean, length_std = file_length_stats[fname]

            for i, cand in enumerate(candidates):
                tokens, labels, pos_tags = cand["tokens"], cand["labels"], cand["pos_tags"]

                extra_features = self._compute_extra_features(
                    tokens, pos_tags, length_mean, length_std
                )

                utt = {
                    "tokens":          tokens,
                    "pos_tags":        pos_tags,
                    "labels":          labels,
                    "sld_only_labels": cand["sld_only_labels"],
                    "extra_features":  extra_features,
                }

                # tag every utterance (train AND test) with its source
                # location. Needed so tagging_check_bpe.jsonl can report
                # filename/line_number regardless of which split a row
                # ended up in — previously only test utterances carried
                # "source_file".
                utt["_source_file"] = fname
                utt["_line_number"] = cand["line_num"]

                if i in test_idx:
                    raw_text = self.salt_to_raw(cand["raw_line"])
                    utt["raw_transcript"]    = raw_text or ""
                    utt["ground_truth_salt"] = cand["raw_line"].replace('*CHI:', '').strip()
                    utt["ground_truth_bio"]  = cand["bio"]
                    utt["source_file"]       = fname   # lets the report break down by file
                    test_utterances.append(utt)
                else:
                    train_utterances.append(utt)

                for entry in cand["bio"]:
                    all_pairs_for_check.append({
                        "filename":   fname,
                        "line_number": cand["line_num"],
                        "word":        entry["word"],
                        "BIO_label":   entry["BIO_label"],
                        "fine_label":  entry["fine_label"],
                        "TDorSLD":     entry["TDorSLD"],
                    })

        # step 5: build subword-aligned vocabs and align utterances 
        # SubwordAligner replaces prepare_labels() + tokenize_and_align_labels()
        # that previously lived in StutterModelSystem.
        print(f"\nBuilding label and POS vocabularies from "
              f"{len(train_utterances)} training utterances...")
        aligner = SubwordAligner()
        aligner.build_vocabs(train_utterances)

        print("Aligning subword tokens for training set...")
        aligner.align_utterances(train_utterances)

        print("Aligning subword tokens for test set...")
        aligner.align_utterances(test_utterances)

        # save vocabs so the model script can reconstruct mappings without
        # re-scanning the data.
        os.makedirs("vocab", exist_ok=True)
        aligner.save_vocabs("vocab")

        # step 5b: build the BPE-level manual verification file 
        # tagging_check.jsonl (word-level) doesn't show how a word actually
        # got split into subwords, or which pieces get masked with -100 in
        # the loss. tagging_check_bpe.jsonl shows that, one row per subword,
        # for every utterance (train + test combined).
        print("Building subword-level check rows for tagging_check_bpe.jsonl...")
        bpe_check_rows = []
        for utt in train_utterances + test_utterances:
            check_rows = aligner.build_check_rows(
                utt["tokens"], utt["labels"], utt.get("pos_tags", []),
                utt.get("extra_features", [])
            )
            for row in check_rows:
                row["filename"]    = utt["_source_file"]
                row["line_number"] = utt["_line_number"]
                bpe_check_rows.append(row)

        # strip internal tracking keys back out before writing
        # training_data.jsonl / test_set_evaluation.jsonl — they were only
        # needed to label rows in tagging_check_bpe.jsonl above and aren't
        # meant to be model input fields.
        for utt in train_utterances + test_utterances:
            utt.pop("_source_file", None)
            utt.pop("_line_number", None)

        # step 6: write outputs
        with open("training_data.jsonl", 'w') as f:
            for entry in train_utterances:
                f.write(json.dumps(entry) + '\n')

        with open("test_set_evaluation.jsonl", 'w') as f:
            for entry in test_utterances:
                f.write(json.dumps(entry) + '\n')

        with open("tagging_check.jsonl", 'w') as f:
            for entry in all_pairs_for_check:
                f.write(json.dumps(entry) + '\n')

        with open("tagging_check_bpe.jsonl", 'w') as f:
            for entry in bpe_check_rows:
                f.write(json.dumps(entry) + '\n')

        # print breakdown of test lines per file.
        from collections import Counter
        file_counts = Counter(utt["source_file"] for utt in test_utterances)
        print(f"\nTraining JSONL : training_data.jsonl ({len(train_utterances)} utterances)")
        print(f"Test JSONL     : test_set_evaluation.jsonl ({len(test_utterances)} utterances)")
        print(f"  Breakdown by file:")
        for fname in filenames:
            print(f"    {fname}: {file_counts.get(fname, 0)} lines")
        print(f"Check JSONL    : tagging_check.jsonl (word-level)")
        print(f"BPE Check JSONL: tagging_check_bpe.jsonl (subword-level, {len(bpe_check_rows)} rows)")
        print(f"Vocabs         : vocab/label_map.json  vocab/pos_vocab.json")
        print(f"\n{len(aligner.label_list)} BIO labels: {aligner.label_list}")
        print(f"{len(aligner.pos2id)} POS vocab entries")

# entry point
if __name__ == "__main__":
    DATASET_PATH = "/Users/wanlingyeo/URECA/sawyer_transcripts"
    preprocessor = SALTPreprocessor()
    preprocessor.process_all(DATASET_PATH)

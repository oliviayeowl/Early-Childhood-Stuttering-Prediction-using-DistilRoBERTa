import os
import json
import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Doc
from collections import Counter
from scipy.stats import chi2_contingency, chisquare
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

# load spaCy model
nlp = spacy.load("en_core_web_sm")

# POS mappings: these 9 were chosen bc they were more linguistically meaningful
# for this analysis.
POS_MAP = {
    'NOUN': 'Noun', 'PROPN': 'Noun',
    'VERB': 'Verb', 'AUX': 'Verb',
    'ADJ': 'Adjective',
    'ADV': 'Adverb',
    'PRON': 'Pronoun',
    'DET': 'Determiner',
    'ADP': 'Preposition',
    'CCONJ': 'Conjunction', 'SCONJ': 'Conjunction',
    'INTJ': 'Interjection'
}

# Howell's dichotomy: function words buy planning time before a demanding
# content word; content words carry the lexical/semantic load. 
FUNCTION_WORDS = {'Pronoun', 'Determiner', 'Preposition', 'Conjunction'}
CONTENT_WORDS = {'Noun', 'Verb', 'Adjective', 'Adverb'}

SLD_TYPES = {'PW', 'WW', 'DP'}  # Stuttering-Like Disfluencies in this project's framework.
# P, R, and <> are TDs. DP is SLD but excluded from this analysis because its
# word token is the synthetic 'XX' placeholder — POS tagging it is a category
# error. I and / are also excluded: their tokens are synthetic placeholders
# (INTERJECTION, RESTARTSENTENCE) rather than real spoken words.

def clean_label(label):
    if label == "O":
        return "O"
    return label.split("-")[-1]

def td_or_sld(dtype):
    return 'SLD' if dtype in SLD_TYPES else 'TD'

def fn_content(pos):
    if pos in FUNCTION_WORDS:
        return 'Function'
    if pos in CONTENT_WORDS:
        return 'Content'
    return None


# allowlist of disfluency types to analyse — everything else is skipped.
# PW and WW are SLDs with genuine orthographic traces in the transcript.
# R (revision), P (phrase repetition), and <> (abandoned utterance) all anchor
# onto real spoken words — R now does so because the preprocessing change tags
# <word> [/] [^ r] as B-R / I-R rather than emitting a synthetic placeholder.
# All three fall into the TD group via td_or_sld().
# DP is excluded because its word token is always the synthetic 'XX' placeholder
# (caught by PLACEHOLDER_WORDS below). I and / are excluded because their word
# tokens are synthetic (INTERJECTION, RESTARTSENTENCE), not real spoken words.
INCLUDED_DISFLUENCY_TYPES = {'<>', 'P', 'PW', 'R', 'WW'}

# tokens that are SALT/CHAT meta-markers rather than real spoken words --
# their "word" value is a literal symbol or placeholder (the + and / marker
# characters, an empty <> abandoned-utterance marker, the xxx/xx
# unintelligible / no-orthography placeholder). running a POS tagger on
# these is a category error since there is no real word class to detect, only
# whatever fallback tag spaCy defaults to for an unfamiliar symbol. this is
# filtered on the actual word text (not the BIO type), so it correctly
# keeps real words that happen to carry a <> or other tag while only
# dropping the genuine placeholder instances. (+, /, and XXX always fall
# into this filter since their word is by definition always the symbol
# itself, they'll end up with zero surviving events automatically.)
PLACEHOLDER_WORDS = {'+', '/', '<>', 'xxx', 'xx'}

# records POS of the DISFLUENT WORD ITSELF for every disfluency event (a token carrying B- label)
def get_disfluent_word_pos(tokens, labels):
    doc = Doc(nlp.vocab, words=tokens)
    for name, proc in nlp.pipeline:
        doc = proc(doc)

    events = []
    for token, label in zip(doc, labels):
        dtype = clean_label(label)
        if dtype != "O" and label.startswith("B-"):
            if dtype not in INCLUDED_DISFLUENCY_TYPES:
                continue
            if token.text.lower() in PLACEHOLDER_WORDS:
                continue
            mapped = POS_MAP.get(token.pos_)
            # only log events whose own word falls in one of our 9 categories
            if mapped:
                events.append({'type': dtype, 'current_pos': mapped, 'word': token.text})

    return events


# yields every row across all jsonl files that exist on disk. 
# training_data.jsonl alone only covers the train split, it does not include
# the test utterances. the split is meaningful for the model but there's no
# reason to exclude it here since this is a stats test. not predicting anything
# here but counting. so reading both files restores the full dataset.
def iter_corpus_rows(paths):
    found_any = False
    for path in paths:
        if not os.path.exists(path):
            print(f"Note: '{path}' not found, skipping.")
            continue
        found_any = True
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line)
    if not found_any:
        raise FileNotFoundError(
            "None of the expected data files were found: " + ", ".join(paths)
        )


def main():
    DATA_PATHS = ['./training_data.jsonl', './test_set_evaluation.jsonl']
    existing_paths = [p for p in DATA_PATHS if os.path.exists(p)]
    if not existing_paths:
        print(f"Error: none of {DATA_PATHS} found.")
        return
    print(f"Using corpus files: {', '.join(existing_paths)}")
    print("Note: PW and WW are analysed as SLD; R, P, and <> are analysed as TD. "
          "DP, I, and / are excluded (placeholder-only tokens, no real word).")
    if len(existing_paths) < len(DATA_PATHS):
        print("Warning: not all expected files were found -- results below "
              "only reflect the file(s) actually present.")

   
    # PASS 1: corpus-wide POS baseline (train + test).
    # what share of all words falls into each of the 9 classes? this is the expected
    # distribution disfluent word POS gets compared agasint. without this, it can only
    # compare disfluency types to each other, not tell whether a word class is actually
    # over represented among disfluencies relative to its overall frequency in speech
    total_word_count = 0
    corpus_pos_counts = Counter()
    for row in iter_corpus_rows(existing_paths):
        total_word_count += len(row['tokens'])
        doc = Doc(nlp.vocab, words=row['tokens'])
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        for token in doc:
            mapped = POS_MAP.get(token.pos_)
            if mapped:
                corpus_pos_counts[mapped] += 1

    print(f"Corpus total tokens: {total_word_count}")

    # PASS 2: for every disfluency event across the FULL corpus, record
    # the word class of the disfluent word itself. only one row per event,
    # no window of multiple neighbour words, so each event contributes
    # exactly one independent tally to every table below.
    all_events = []
    for row in iter_corpus_rows(existing_paths):
        all_events.extend(get_disfluent_word_pos(row['tokens'], row['labels']))

    if not all_events:
        print("No disfluencies found.")
        return

    neigh_df = pd.DataFrame(all_events)
    neigh_df['td_sld'] = neigh_df['type'].apply(td_or_sld)

    # DIAGNOSTIC: save every individual (type, word, current_pos) event and
    # print the most common actual words per type. if a type looks
    # suspiciously concentrated in one POS category (e.g. nearly all
    # 'Noun'), able to check directly whether that's a genuine
    # pattern in real words, or a tagging artifact (spaCy's tagger is
    # trained on fluent adult text and can misfire on broken/repeated
    # local context right at a disfluency, or on an unexpected word).
    neigh_df[['type', 'word', 'current_pos']].to_csv('./disfluent_word_log.csv', index=False)
    print("\n=== most common actual words per disfluency type (sanity check) ===")
    for dtype in sorted(neigh_df['type'].unique()):
        top_words = neigh_df[neigh_df['type'] == dtype]['word'].str.lower().value_counts().head(8)
        print(f"{dtype}: " + ", ".join(f"{w}({c})" for w, c in top_words.items()))
    print("Saved: disfluent_word_log.csv (full event-level word/POS log)")

    # 1. GLOBAL CHI-SQUARE: disfluency_type x current_pos
    #    test: do different disfluency types (PW, WW, R, P, <>) land on
    #          different word classes?
    current_pos_table = pd.crosstab(neigh_df['type'], neigh_df['current_pos'])
    current_pos_table.to_csv('./disfluency_type_by_current_pos.csv')
    print("\n=== disfluency_type x current_pos counts ===")
    print(current_pos_table)

    if current_pos_table.shape[0] > 1 and current_pos_table.shape[1] > 1:
        chi2_stat, p_global, dof, expected = chi2_contingency(current_pos_table)
        pct_low = (expected < 5).sum() / expected.size * 100
        print(f"\nchi2({dof}) = {chi2_stat:.2f},  p = {p_global:.5f}")
        print(f"{pct_low:.1f}% of expected cells < 5 "
              f"(chi-square gets unreliable above ~20% -- check this before trusting p)")

        # 2. PAIRWISE CHI-SQUARE WITH FDR CORRECTION
        #    for each (disfluency_type, POS) cell, run a 2x2 chi-square
        #    against the rest of the table, then apply Benjamini-Hochberg
        #    FDR correction across all tests to control the false discovery
        #    rate. 
        #    FDR correction controls false discovery rate. there are about 45
        #    pairwise tests (9 pos classes × 5 disfluency types). at an
        #    uncorrected α=0.05, each individual test has a 5% chance of a false
        #    positive. but across 45 independent tests, roughly 2 "significant"
        #    results could show up from chance alone, even if nothing real is
        #    going on. so this corrects that.
        pairwise_results = []
        total = current_pos_table.values.sum()
        for dtype in current_pos_table.index:
            for pos in current_pos_table.columns:
                a = current_pos_table.loc[dtype, pos]        # this cell
                b = current_pos_table.loc[dtype].sum() - a   # same disfluency, other POS
                c = current_pos_table[pos].sum() - a         # same POS, other disfluencies
                d = total - a - b - c                        # everything else
                if a + b + c + d == 0:
                    continue
                _, p, _, _ = chi2_contingency([[a, b], [c, d]])
                pairwise_results.append({
                    'disfluency': dtype, 'pos': pos, 'count': int(a), 'p_raw': p
                })

        pw_df = pd.DataFrame(pairwise_results)
        _, pw_df['p_fdr'], _, _ = multipletests(pw_df['p_raw'], method='fdr_bh')
        pw_df = pw_df.sort_values('p_fdr')
        pw_df.to_csv('./pairwise_chisq_fdr_current_pos.csv', index=False)

        sig = pw_df[pw_df['p_fdr'] < 0.05]
        print(f"\n=== PAIRWISE CHI-SQUARE (FDR-corrected, BH method) ===")
        if sig.empty:
            print("No significant pairwise associations after FDR correction.")
        else:
            print(f"Significant pairs (p_fdr < 0.05): {len(sig)} of {len(pw_df)} tests")
            print(sig.to_string(index=False))
        print("Saved: pairwise_chisq_fdr_current_pos.csv")

        # 3. PMI HEATMAP (visualisation using a table)
        #    adding alpha=0.5 to every count avoids log(0) errors when a (disfluency type, pos) cell
        #    has 0 observations. bc alpha=0 would be unstable if ANY cell is empty, which is quite likely
        #    since this corpus is small (15 transcripts). alpha=0.5 is a standard default for smoothing
        #    categrorical counts. 1 would be too strong as it can inflate a true count of 2 by 50%, which
        #    would result in every PMI value having 'no association' regardless of the real pattern
        alpha = 0.5
        total_counts = current_pos_table.sum(axis=0)
        total_sum = total_counts.sum()
        pmi_rows = []
        for dtype in current_pos_table.index:
            type_counts_pos = current_pos_table.loc[dtype]
            type_sum = type_counts_pos.sum()
            for pos in total_counts.index:
                count_type = type_counts_pos.get(pos, 0) + alpha
                count_total = total_counts[pos] + alpha
                p_type = count_type / (type_sum + alpha * len(total_counts))
                p_total = count_total / (total_sum + alpha * len(total_counts))
                pmi_rows.append({'disfluency_type': dtype, 'pos_tag': pos,
                                  'pmi': np.log(p_type / p_total)})

        pmi_df = pd.DataFrame(pmi_rows)
        pmi_df.to_csv('./current_pos_pmi_results.csv', index=False)

        plt.figure(figsize=(14, 7))
        pmi_pivot = pmi_df.pivot(index="pos_tag", columns="disfluency_type", values="pmi")
        sns.heatmap(pmi_pivot, annot=True, cmap="RdBu", center=0, fmt=".2f")
        plt.title("Association between the Disfluent Word's Own POS and Disfluency Type (PMI)")
        plt.xlabel("Disfluency Type")
        plt.ylabel("Word Class of the Disfluent Word")
        plt.tight_layout()
        plt.savefig("./current_pos_pmi_heatmap.png")
        print("Saved: current_pos_pmi_heatmap.png")

    # =========================================================================
    # 4. GOODNESS-OF-FIT vs CORPUS BASELINE (SLD and TD separately)
    #    test: is the POS distribution of SLD-tagged words (and separately
    #    TD-tagged words) different from the POS distribution of the corpus
    #    as a whole? 
    # This directly tests is word type at elevated risk of disfluency.
    # =========================================================================
    print("\n=== GOODNESS-OF-FIT vs CORPUS POS BASELINE ===")
    if sum(corpus_pos_counts.values()) == 0:
        print("Warning: corpus_pos_counts is empty -- skipping goodness-of-fit test.")
    else:
        corpus_total = sum(corpus_pos_counts.values())
        pos_categories = sorted(corpus_pos_counts.keys())

        for group_label in ['SLD', 'TD']:
            group_df = neigh_df[neigh_df['td_sld'] == group_label]
            observed = group_df['current_pos'].value_counts().reindex(pos_categories, fill_value=0)
            n_events = int(observed.sum())
            if n_events == 0:
                print(f"\n{group_label}: no events with a recognised current_pos -- skipped.")
                continue
            if n_events < 20:
                print(f"\n{group_label}: only {n_events} events -- too few for a reliable "
                      f"goodness-of-fit test, treat the result below cautiously.")

            expected_props = pd.Series({p: corpus_pos_counts[p] / corpus_total for p in pos_categories})
            expected_counts = expected_props * n_events
            low_cells = (expected_counts < 5).sum()

            chi2_gof, p_gof = chisquare(f_obs=observed.values, f_exp=expected_counts.values)
            print(f"\n{group_label} vs corpus baseline: n={n_events}, "
                  f"chi2({len(pos_categories)-1}) = {chi2_gof:.2f}, p = {p_gof:.5f}  "
                  f"({low_cells}/{len(pos_categories)} expected cells < 5)")
            comparison = pd.DataFrame({
                'observed': observed,
                'expected': expected_counts.round(1),
                'obs_minus_exp': (observed - expected_counts).round(1)
            })
            print(comparison.to_string())
            comparison.to_csv(f'./{group_label.lower()}_current_pos_vs_corpus_baseline.csv')

    # 5. FUNCTION-WORD vs CONTENT-WORD COLLAPSE
    #    calculates the fraction of SLD-tagged and TD-tagged words that land on
    #    function words vs content words, compared to how often function vs
    #    content words occur in the entire corpus.
    neigh_df['fn_content'] = neigh_df['current_pos'].apply(fn_content)
    corpus_fn = sum(c for p, c in corpus_pos_counts.items() if p in FUNCTION_WORDS)
    corpus_ct = sum(c for p, c in corpus_pos_counts.items() if p in CONTENT_WORDS)
    corpus_fn_ct_total = corpus_fn + corpus_ct

    print("\n=== FUNCTION vs CONTENT WORD DISFLUENCY ===")
    fc_summary = []
    if corpus_fn_ct_total == 0:
        print("Warning: no function/content baseline available -- skipped.")
    else:
        for group_label in ['SLD', 'TD']:
            gdf = neigh_df[(neigh_df['td_sld'] == group_label) & (neigh_df['fn_content'].notna())]
            n_fn = int((gdf['fn_content'] == 'Function').sum())
            n_ct = int((gdf['fn_content'] == 'Content').sum())
            n_total = n_fn + n_ct
            if n_total == 0:
                continue
            exp_fn = corpus_fn / corpus_fn_ct_total * n_total
            exp_ct = corpus_ct / corpus_fn_ct_total * n_total
            chi2_fc, p_fc = chisquare(f_obs=[n_fn, n_ct], f_exp=[exp_fn, exp_ct])
            print(f"{group_label}: observed Function={n_fn} ({n_fn/n_total*100:.1f}%), "
                  f"Content={n_ct} ({n_ct/n_total*100:.1f}%)  |  "
                  f"corpus baseline Function={corpus_fn/corpus_fn_ct_total*100:.1f}%  |  "
                  f"chi2={chi2_fc:.2f}, p={p_fc:.5f}")
            fc_summary.append({
                'group': group_label, 'n_function': n_fn, 'n_content': n_ct,
                'pct_function_observed': round(n_fn / n_total * 100, 1),
                'pct_function_baseline': round(corpus_fn / corpus_fn_ct_total * 100, 1),
                'chi2': chi2_fc, 'p': p_fc
            })

        if fc_summary:
            pd.DataFrame(fc_summary).to_csv('./function_vs_content_word_test.csv', index=False)
            print("Saved: function_vs_content_word_test.csv")


if __name__ == "__main__":
    main()

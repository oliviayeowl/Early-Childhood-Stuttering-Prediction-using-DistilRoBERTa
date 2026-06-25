import os
import json
import random
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter

from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    DataCollatorForTokenClassification,
    RobertaModel,
    RobertaPreTrainedModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset

# seqeval provides the standard BIO/NER metrics used in the literature.
try:
    from seqeval.metrics import (
        classification_report as seqeval_report,
        f1_score as seqeval_f1,
        precision_score as seqeval_precision,
        recall_score as seqeval_recall,
    )
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("Warning: seqeval not installed. Token-level NER metrics will be skipped.")

# reproducibility
RANDOM_SEED = 42
POS_EMBED_DIM = 32   # dimension of the learnable POS embedding vector

# width of the linguistically-motivated scalar feature vector written by
# preprocessing-16-12.py's SALTPreprocessor._compute_extra_features():
# [position, is_content, is_function, is_other, utt_length_z]. must match
# EXTRA_FEATURE_DIM in the preprocessing script — these are read directly
# from the JSONL, not derived here.
EXTRA_FEATURE_DIM = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# SLD types 
# the three Stuttering-Like Disfluency classes this project evaluates on.
# P (phrase repetition) is treated as a TD in this project's framework.
# TDs are still present in the training data; they are masked to 'O' in the
# SLD-only seqeval pass so the metrics reflect only stuttering-relevant classes.
SLD_TYPES = {'PW', 'WW', 'DP'}

# Core SLD types 
# PW and WW both have a genuine physical trace in the transcript — a real spoken
# word. DP is excluded here because it has no orthographic form; it only exists
# in the data as the 'XX' placeholder, so the model can hit ~1.00 F1 on it just
# by pattern-matching an OOV token, not by detecting anything from real context.
# P is excluded because it is treated as a TD in this project's framework.
# This tier therefore reflects the most honest text-based detection metric.
CORE_SLD_TYPES = {'PW', 'WW'}

# class-weight calculation
def calculate_weights(data_path):
    all_labels = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            all_labels.extend(data['labels'])

    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    ids = [label_to_id[label] for label in all_labels]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(label_to_id.values())),
        y=np.array(ids)
    )

    # logarithmic smoothing: raw 'balanced' weights can be 50-100×; log(1+w)
    # gives rare classes a meaningful boost without drowning out fluent tokens.
    smooth_weights = np.log1p(class_weights)
    if 'O' in unique_labels:
        o_idx = unique_labels.index('O')
        smooth_weights = smooth_weights / smooth_weights[o_idx]

    weighted_labels = {label: w for label, w in zip(unique_labels, smooth_weights)}

    # B-MOT is a noisy motoric label with unreliable annotations — zeroed out.
    # I-R is now included in training.
    for label_to_ignore in ["B-MOT"]:
        if label_to_ignore in weighted_labels:
            weighted_labels[label_to_ignore] = 0.0

    return weighted_labels

# Custom model: RoBERTa backbone + learnable POS embedding
class RobertaWithPOSForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = config.num_labels
        pos_vocab_size       = getattr(config, 'pos_vocab_size', 50)
        pos_embed_dim        = getattr(config, 'pos_embed_dim', POS_EMBED_DIM)
        extra_feature_dim    = getattr(config, 'extra_feature_dim', EXTRA_FEATURE_DIM)
        self.extra_feature_dim = extra_feature_dim

        self.roberta        = RobertaModel(config, add_pooling_layer=False)
        self.pos_embeddings = nn.Embedding(pos_vocab_size, pos_embed_dim, padding_idx=0)

        classifier_dropout = (
            getattr(config, 'classifier_dropout', None)
            or getattr(config, 'hidden_dropout_prob', 0.1)
        )
        self.dropout    = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(
            config.hidden_size + pos_embed_dim + extra_feature_dim,
            config.num_labels,
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pos_ids=None,
        extra_features=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs[0])   # (B, S, hidden_size)

        if pos_ids is not None:
            pos_embeds = self.pos_embeddings(pos_ids)
        else:
            B, S, _ = sequence_output.shape
            pos_embeds = torch.zeros(
                B, S, self.pos_embeddings.embedding_dim,
                device=sequence_output.device
            )

        if extra_features is not None:
            # already numeric (position / one-hot / z-score) — no embedding
            # lookup needed, just match dtype/device and concatenate.
            extra_feats = extra_features.to(
                dtype=sequence_output.dtype, device=sequence_output.device
            )
        else:
            B, S, _ = sequence_output.shape
            extra_feats = torch.zeros(
                B, S, self.extra_feature_dim, device=sequence_output.device
            )

        combined = torch.cat([sequence_output, pos_embeds, extra_feats], dim=-1)
        logits   = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# custom data collator: pads pre-computed pos_ids alongside the standard fields
# renames label_ids → labels so the parent collator picks it up correctly bc
# the Trainer expects the loss target to be named 'labels'), and 
# extra_features_aligned → extra_features to match the model's forward() parameter name
@dataclass
class DataCollatorForTokenClassificationWithPOS(DataCollatorForTokenClassification):
    def __call__(self, features):
        # the JSONL contains two label fields:
        #   "labels"    — BIO label strings ["O","B-PW",...] for reference/eval
        #   "label_ids" — integer IDs pre-computed by SubwordAligner
        # the parent collator needs integer IDs under the key "labels".
        # always overwrite the string "labels" with the integer "label_ids".
        for feat in features:
            if "label_ids" in feat:
                feat["labels"] = feat.pop("label_ids")

        pos_ids_list        = [f.pop("pos_ids", []) for f in features]
        extra_features_list = [f.pop("extra_features_aligned", []) for f in features]

        # strip any extra fields the parent collator doesn't know about
        # (e.g. tokens, pos_tags, sld_only_labels, source_file, the
        # word-level 'extra_features' field …)
        known_keys = {"input_ids", "attention_mask", "token_type_ids", "labels",
                      "special_tokens_mask"}
        cleaned_features = [{k: v for k, v in feat.items() if k in known_keys}
                            for feat in features]

        batch = super().__call__(cleaned_features)

        max_len = batch["input_ids"].shape[1]
        padded  = []
        for pos_ids in pos_ids_list:
            pad_len = max_len - len(pos_ids)
            if self.tokenizer.padding_side == "right":
                padded.append(list(pos_ids) + [0] * pad_len)
            else:
                padded.append([0] * pad_len + list(pos_ids))

        batch["pos_ids"] = torch.tensor(padded, dtype=torch.long)

        zero_vec     = [0.0] * EXTRA_FEATURE_DIM
        extra_padded = []
        for extra in extra_features_list:
            pad_len = max_len - len(extra)
            pad_block = [zero_vec] * pad_len
            if self.tokenizer.padding_side == "right":
                extra_padded.append(list(extra) + pad_block)
            else:
                extra_padded.append(pad_block + list(extra))

        batch["extra_features"] = torch.tensor(extra_padded, dtype=torch.float)
        return batch

# main system
# training and evaluation of the POS-aware DistilRoBERTa token classifier 
# for stuttering disfluency detection.
class StutterModelSystem:
    def __init__(self, model_name="distilroberta-base", class_weights=None):
        self.model_name    = model_name
        self.tokenizer     = RobertaTokenizerFast.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.label_list    = []
        self.label2id      = {}
        self.id2label      = {}
        self.pos2id        = {}
        self.class_weights = class_weights

    # vocab loading (replaces prepare_labels)
    # loads label_map.json and pos_vocab.json written by SubwordAligner
    # (preprocessing-16.py). calls this before train_roberta().
    def load_vocabs(self, vocab_dir="vocab"):
        label_map_path = os.path.join(vocab_dir, "label_map.json")
        pos_vocab_path = os.path.join(vocab_dir, "pos_vocab.json")

        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"{label_map_path} not found.  "
                "Run preprocessing-16.py first to generate it."
            )
        with open(label_map_path) as f:
            lm = json.load(f)
        self.label2id   = lm["label2id"]
        self.id2label   = {int(k): v for k, v in lm["id2label"].items()}
        self.label_list = [self.id2label[i] for i in range(len(self.id2label))]

        if os.path.exists(pos_vocab_path):
            with open(pos_vocab_path) as f:
                self.pos2id = json.load(f)
        else:
            print("Warning: pos_vocab.json not found — POS features will be zeroed.")
            self.pos2id = {"PAD": 0, "UNK": 1}

        print(f"Loaded {len(self.label_list)} labels and "
              f"{len(self.pos2id)} POS vocab entries from {vocab_dir}/")

    # seqeval compute_metrics (passed to Trainer)
    def make_compute_metrics(self):
        label_list = self.label_list

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_preds = [
                [label_list[pred] for pred, lab in zip(preds_row, labels_row)
                 if lab != -100]
                for preds_row, labels_row in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[lab] for pred, lab in zip(preds_row, labels_row)
                 if lab != -100]
                for preds_row, labels_row in zip(predictions, labels)
            ]

            if SEQEVAL_AVAILABLE:
                return {
                    "precision": seqeval_precision(true_labels, true_preds,
                                                   zero_division=0),
                    "recall":    seqeval_recall(   true_labels, true_preds,
                                                   zero_division=0),
                    "f1":        seqeval_f1(       true_labels, true_preds,
                                                   zero_division=0),
                }
            correct = sum(p == l for seq_p, seq_l in zip(true_preds, true_labels)
                          for p, l in zip(seq_p, seq_l))
            total   = sum(len(seq) for seq in true_labels)
            return {"accuracy": correct / total if total > 0 else 0}

        return compute_metrics

    # training 
    def train_roberta(self, train_data_path, output_dir="./stutter_model"):
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        if os.path.exists('./logs'):   shutil.rmtree('./logs')

        if not self.label_list:
            raise RuntimeError(
                "Label vocab is empty.  Call load_vocabs() before train_roberta()."
            )

        data = []
        with open(train_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        random.shuffle(data)
        split      = int(0.9 * len(data))
        train_data = data[:split]
        eval_data  = data[split:]

        train_dataset = Dataset.from_list(train_data)
        eval_dataset  = Dataset.from_list(eval_data)

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            label2id=self.label2id,
            id2label=self.id2label,
            pos_vocab_size=len(self.pos2id),
            pos_embed_dim=POS_EMBED_DIM,
            extra_feature_dim=EXTRA_FEATURE_DIM,
        )

        model = RobertaWithPOSForTokenClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # weighted loss trainer 
        if self.class_weights is not None:
            weights_tensor = torch.tensor(
                [self.class_weights.get(label, 1.0) for label in self.label_list],
                dtype=torch.float
            )
            model.config.class_weights = weights_tensor.tolist()

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    outputs = model(**inputs)
                    logits  = outputs.logits
                    labels  = inputs.get("labels")
                    if labels is not None:
                        weights = torch.tensor(
                            model.config.class_weights,
                            device=logits.device, dtype=torch.float
                        )
                        loss_fct = torch.nn.CrossEntropyLoss(
                            weight=weights, ignore_index=-100
                        )
                        loss = loss_fct(
                            logits.view(-1, model.config.num_labels),
                            labels.view(-1)
                        )
                    else:
                        loss = outputs.loss
                    return (loss, outputs) if return_outputs else loss

            trainer_class = WeightedTrainer
        else:
            trainer_class = Trainer

        import transformers as _tv
        _new_api    = tuple(int(x) for x in _tv.__version__.split(".")[:2]) >= (4, 41)
        _eval_kwarg = "eval_strategy" if _new_api else "evaluation_strategy"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=6,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="epoch",
            **{_eval_kwarg: "epoch"},
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=100,
            report_to="none",
            seed=RANDOM_SEED,
            **{"use_cpu" if _new_api else "no_cuda": True},
        )

        _tok_kwarg = "processing_class" if tuple(
            int(x) for x in _tv.__version__.split(".")[:2]
        ) >= (4, 46) else "tokenizer"

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **{_tok_kwarg: self.tokenizer},
            data_collator=DataCollatorForTokenClassificationWithPOS(self.tokenizer),
            compute_metrics=self.make_compute_metrics(),
        )

        print(f"Starting training with {self.model_name}  "
              f"({len(train_data)} train / {len(eval_data)} eval utterances)...")
        trainer.train()

        # copy vocabs into the model directory for self-contained inference.
        for vfile in ["label_map.json", "pos_vocab.json"]:
            src = os.path.join("vocab", vfile)
            if os.path.exists(src):
                import shutil as _sh
                _sh.copy(src, os.path.join(output_dir, vfile))

        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}  (label_map.json + pos_vocab.json also saved)")

    # SALT post-processing helpers 
    def bio_to_salt(self, tokens_with_labels):
        result = []
        i = 0
        while i < len(tokens_with_labels):
            word, label = tokens_with_labels[i]
            if label.startswith('B-'):
                tag_type = label.split('-')[1]
                count    = 0
                j = i + 1
                while j < len(tokens_with_labels) and \
                        tokens_with_labels[j][1] == f'I-{tag_type}':
                    count += 1
                    j += 1
                if tag_type in ['WW', 'I', 'P']:
                    result.append(f"[^ {tag_type}{count}]")
                    result.append(word)
                    i = j
                else:
                    result.append(f"[^ {tag_type.lower()}]")
                    result.append(word)
                    i += 1
            else:
                result.append(word)
                i += 1
        return " ".join(result)

    # SLD-only evaluation helper 
    # collapses every non-SLD label to 'O' in label sequence.
    @staticmethod
    def _mask_to_sld(label_seq):
        sld_prefixes = tuple(
            f"{pfx}-{t}" for pfx in ("B", "I") for t in SLD_TYPES
        )
        return [lbl if lbl in sld_prefixes else "O" for lbl in label_seq]

    @staticmethod
    #  collapses every label outside CORE_SLD_TYPES (PW, WW) to 'O'.
    #  this is the DP-and-P-and-TD-free view: the number that reflects genuine
    #  text-based detection capability on SLDs with a real orthographic trace.
    def _mask_to_core_sld(label_seq):
        core_prefixes = tuple(
            f"{pfx}-{t}" for pfx in ("B", "I") for t in CORE_SLD_TYPES
        )
        return [lbl if lbl in core_prefixes else "O" for lbl in label_seq]

    # inference helper to align word level POS tags to BPE subword word_ids for a single
    # utterance at inference time.
    def _get_pos_ids_for_inference(self, pos_tags, word_ids):
        pos_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                pos_ids.append(0)
            elif word_idx < len(pos_tags):
                pos_ids.append(
                    self.pos2id.get(pos_tags[word_idx], self.pos2id.get("UNK", 1))
                )
            else:
                pos_ids.append(0)
        return torch.tensor([pos_ids], dtype=torch.long)

    # aligns ord-level scalar features (position, content/function/other,
    # utt_length_z — written to the JSONL as 'extra_features' by
    # preprocessing-16-12.py) to BPE subword word_ids at inference time.
    def _get_extra_features_for_inference(self, extra_features, word_ids):
        zero_vec = [0.0] * EXTRA_FEATURE_DIM
        feats = []
        for word_idx in word_ids:
            if word_idx is None:
                feats.append(zero_vec)
            elif word_idx < len(extra_features):
                feats.append(extra_features[word_idx])
            else:
                feats.append(zero_vec)
        return torch.tensor([feats], dtype=torch.float)

    # evaluation
    def evaluate_on_test_set(self, test_jsonl_path, model_path="./stutter_model"):
        if not os.path.exists(model_path):
            print("Model not found.")
            return

        # load vocabs (prefer model_path copy, fall back to vocab/ dir)
        for search_dir in [model_path, "vocab"]:
            label_map_path = os.path.join(search_dir, "label_map.json")
            pos_vocab_path = os.path.join(search_dir, "pos_vocab.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    lm = json.load(f)
                self.label2id   = lm["label2id"]
                self.id2label   = {int(k): v for k, v in lm["id2label"].items()}
                self.label_list = [self.id2label[i] for i in range(len(self.id2label))]
                if os.path.exists(pos_vocab_path):
                    with open(pos_vocab_path) as f:
                        self.pos2id = json.load(f)
                break
        else:
            raise FileNotFoundError(
                "label_map.json not found in model_path or vocab/.  "
                "Re-run preprocessing-16.py and re-train the model."
            )

        config = AutoConfig.from_pretrained(model_path)
        model  = RobertaWithPOSForTokenClassification.from_pretrained(
            model_path, config=config
        )
        model.eval()

        test_data = []
        with open(test_jsonl_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))

        df_test = pd.DataFrame(test_data)
        all_results   = []

        # lists for Pass A (all labels)
        all_true_seqs  = []
        all_pred_seqs  = []

        # lists for Pass B (SLD only)
        sld_true_seqs  = []
        sld_pred_seqs  = []

        # lists for Pass C (core SLD only: PW + WW)
        core_true_seqs = []
        core_pred_seqs = []

        print(f"Evaluating on {len(df_test)} test utterances...")

        for idx, row in df_test.iterrows():
            tokens         = row["tokens"]
            pos_tags       = row["pos_tags"]
            extra_features = row.get("extra_features", [])
            min_len        = min(len(tokens), len(pos_tags))
            tokens         = tokens[:min_len]
            pos_tags       = pos_tags[:min_len]
            extra_features = extra_features[:min_len]

            inputs  = self.tokenizer(
                tokens, return_tensors="pt",
                truncation=True, max_length=128, is_split_into_words=True
            )
            pos_ids      = self._get_pos_ids_for_inference(pos_tags, inputs.word_ids())
            extra_feats  = self._get_extra_features_for_inference(
                extra_features, inputs.word_ids()
            )

            with torch.no_grad():
                logits = model(**inputs, pos_ids=pos_ids, extra_features=extra_feats).logits
            predictions = torch.argmax(logits, dim=2)

            word_ids         = inputs.word_ids()
            predicted_labels = []
            current_word_idx = -1
            for i, word_id in enumerate(word_ids):
                if word_id is not None and word_id != current_word_idx:
                    predicted_labels.append(self.id2label[predictions[0][i].item()])
                    current_word_idx = word_id

            if len(predicted_labels) > len(tokens):
                predicted_labels = predicted_labels[:len(tokens)]
            else:
                predicted_labels += ['O'] * (len(tokens) - len(predicted_labels))

            all_results.append(self.bio_to_salt(list(zip(tokens, predicted_labels))))

            # Pass A: use ground truth from 'labels' field
            true_labels_row = list(row.get("labels", []))[:len(predicted_labels)]
            if true_labels_row:
                all_true_seqs.append(true_labels_row)
                all_pred_seqs.append(predicted_labels)

            # Pass B: SLD-only view 
            # prefer the pre-computed 'sld_only_labels' field; fall back to
            # masking on the fly if it is absent (e.g. older JSONL files).
            sld_true_row = list(row.get("sld_only_labels", []))
            if not sld_true_row and true_labels_row:
                sld_true_row = self._mask_to_sld(true_labels_row)
            sld_pred_row = self._mask_to_sld(predicted_labels)
            if sld_true_row:
                sld_true_seqs.append(sld_true_row)
                sld_pred_seqs.append(sld_pred_row)

            # Pass C: core SLD-only view — PW + WW only.
            # derived from true_labels_row directly so it doesn't depend on
            # whichever types happen to be in the precomputed sld_only_labels.
            if true_labels_row:
                core_true_seqs.append(self._mask_to_core_sld(true_labels_row))
                core_pred_seqs.append(self._mask_to_core_sld(predicted_labels))

        df_test["model_prediction"] = all_results

        # seqeval reports
        seqeval_all_section  = ""
        seqeval_sld_section  = ""
        seqeval_core_section = ""

        if SEQEVAL_AVAILABLE:
            if all_true_seqs:
                report_all = seqeval_report(all_true_seqs, all_pred_seqs,
                                            zero_division=0)
                seqeval_all_section = (
                    "\nSEQEVAL TOKEN-LEVEL METRICS — ALL LABELS:\n" + report_all
                )
                print(seqeval_all_section)

            if sld_true_seqs:
                report_sld = seqeval_report(sld_true_seqs, sld_pred_seqs,
                                            zero_division=0)
                seqeval_sld_section = (
                    "\nSEQEVAL TOKEN-LEVEL METRICS — SLD ONLY (PW / WW / DP):\n"
                    + report_sld
                )
                print(seqeval_sld_section)

            if core_true_seqs:
                report_core = seqeval_report(core_true_seqs, core_pred_seqs,
                                             zero_division=0)
                seqeval_core_section = (
                    "\nSEQEVAL TOKEN-LEVEL METRICS — CORE SLD ONLY "
                    "(PW / WW, no DP / P / TD):\n" + report_core
                )
                print(seqeval_core_section)

        # per-file breakdown 
        per_file_section = ""
        if "source_file" in df_test.columns:
            lines = ["\nTEST SET BREAKDOWN BY SOURCE FILE:"]
            for fname, grp in df_test.groupby("source_file"):
                lines.append(f"  {fname}: {len(grp)} utterances")
            per_file_section = "\n".join(lines)
            print(per_file_section)

        # save results
        with open("final_research_results.jsonl", "w") as f:
            for entry in df_test.to_dict(orient='records'):
                f.write(json.dumps(entry) + '\n')

        report_str = f"""
=========================================
RESEARCH EVALUATION REPORT
=========================================
Model: {self.model_name}
Test Set: {test_jsonl_path} ({len(df_test)} utterances)
{per_file_section}
{seqeval_all_section}
{seqeval_sld_section}
{seqeval_core_section}
=========================================
"""
        print(report_str)
        with open("evaluation_report.txt", "w") as f:
            f.write(report_str)
        print("Results saved → final_research_results.jsonl  evaluation_report.txt")

# entry point
if __name__ == "__main__":
    TRAIN_DATA = "training_data.jsonl"
    TEST_DATA  = "test_set_evaluation.jsonl"
    VOCAB_DIR  = "vocab"

    if os.path.exists(TRAIN_DATA):
        try:
            print(f"Analysing class distribution in {TRAIN_DATA}...")
            class_weights_dict = calculate_weights(TRAIN_DATA)
            print("Class weights (higher = rarer class):")
            print(json.dumps(class_weights_dict, indent=4))

            system = StutterModelSystem(class_weights=class_weights_dict)
            system.load_vocabs(VOCAB_DIR)
            system.train_roberta(TRAIN_DATA)

            if os.path.exists(TEST_DATA):
                system.evaluate_on_test_set(TEST_DATA)
        except Exception as e:
            import traceback
            print(f"Error during execution: {e}")
            traceback.print_exc()
    else:
        print(
            "Error: training_data.jsonl not found.  "
            "Run preprocessing-16.py first."
        )

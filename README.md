# Early Childhood Stuttering Prediction using RoBERTa

## Overview

This project explores the computational detection of early childhood stuttering by identifying and localising speech disfluencies in orthographic transcripts of child speech. The study investigates whether transformer-based language models can support the analysis of natural child speech data, an area that remains relatively underexplored in both computational linguistics and speech pathology research.

Using a hybrid modelling approach, the project applies **RoBERTa with BIO tagging** to detect repetition patterns and other disfluency markers within transcripts. The work is part of a broader effort to examine how machine learning techniques may complement traditional linguistic analysis in identifying potential indicators of stuttering in early childhood speech.

## Research Motivation

Early identification of stuttering is important for timely clinical intervention. However, traditional analysis of child speech transcripts is labour-intensive and often relies on manual annotation. While prior computational work has largely focused on rule-based or feature-based systems, fewer studies have explored the use of **transformer-based architectures**, especially RoBERTa for modelling disfluencies in child speech.

This project therefore investigates whether contextual language models can assist in identifying patterns associated with stuttering-like disfluencies.

## Dataset

The current phase of the project uses:

* 15 pre-annotated speech transcript datasets
* Transcripts of children who stutter
* Orthographic speech data with annotated disfluency markers

From these transcripts, several linguistic features were extracted, including:

* filler word frequency
* repetition patterns
* hesitation markers
* overall disfluency frequencies

## Methodology

### 1. Feature Extraction

Speech transcripts were processed to identify linguistic markers associated with disfluency patterns.

Features examined include:

* filler words
* repetition events
* hesitation markers
* disfluency frequency distributions

### 2. BIO Tagging

A **BIO tagging scheme** was used to label repetition events in transcripts.

Example:

```
I I want to go
B-REP I-REP O O O
```

This allows the model to learn the **location and structure of disfluency events** rather than only their frequency.

### 3. RoBERTa Model

A transformer-based model (**RoBERTa**) was fine-tuned to perform token-level classification using the BIO tagging framework.

The model was trained to identify:

* beginning of a repetition
* continuation of a repetition
* non-disfluent tokens

### 4. Evaluation

Model predictions were compared against the annotated transcripts to evaluate disfluency detection performance.

## Preliminary Results

Initial experiments yielded an accuracy of approximately **7%**.

While this performance remains exploratory, it reflects several challenges associated with modelling child speech data:

* small dataset size
* low frequency of stuttering-like disfluencies
* variability in child speech transcripts
* token alignment challenges in BIO tagging

Ongoing work focuses on improving performance through:

* improved feature representation
* better token alignment strategies
* dataset expansion
* refined training procedures


## Future Work

Future improvements to the project include:

* expanding the dataset to improve model generalisation
* incorporating acoustic features from speech recordings
* experimenting with alternative transformer architectures
* exploring hybrid linguistic + neural approaches for disfluency detection

## Research Context

This project sits at the intersection of:

* computational linguistics
* natural language processing
* speech pathology
* machine learning for clinical language analysis

The work contributes to ongoing research exploring how computational methods may support the analysis of child speech in clinical and linguistic contexts.

## Author

Developed as part of undergraduate research in computational linguistics and machine learning.

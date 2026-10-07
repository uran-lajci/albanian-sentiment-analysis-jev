# Jev for Albanian sentiment analysis

## The dataset

The dataset holds Albanian Facebook comments about COVID-19. Each comment is labelled neutral, positive or negative by three annotators [1, 2].

The comments were posted under the daily updates of the National Institute of Public Health of Kosovo (NIPHK) in 2020. Collection starts in March 2020, when Kosovo confirmed its first cases, and was done with the CommentExporter tool [1, 2]. The text is informal Albanian, mostly the Gheg dialect of Kosovo. It is often written without the letters ë and ç and has slang, abbreviations and emojis [2]. A comment has 16 words on average and 212 at most [2].

Three annotators, third-year computer engineering students at the University of Prishtina, labelled every comment on their own: neutral (0), positive (1) or negative (2). The final label is the majority vote [1, 2]. Agreement is moderate: Fleiss' κ is 0.60, and the pairwise Pearson correlations are 0.46 to 0.62 [1].

There are two public versions:

| Version | Comments | Posts from | Neutral | Positive | Negative |
|---|---:|---|---:|---:|---:|
| Kastrati et al. 2021, AlbAna on GitHub [2, 3] | 10,742 | 13 Mar – 15 Aug 2020 | 56.4% | 15.6% | 28.0% |
| Kadriu et al. 2022, Mendeley Data v4 [1] | 10,132 | 12 Mar – 31 Aug 2020 | 53.8% | 17.5% | 28.7% |

The 2022 release is anonymized, and duplicates and profanity were removed [1]. The two versions share 8,582 comments, and their labels agree on 99.9% of them. This report uses the 2021 version, because the models of the paper were trained on it.

Neutral in this dataset is wider than "no feeling". Questions, requests and calm criticism of measures are labelled neutral. Negative needs an open feeling, such as mockery, distrust or despair. In the 2022 release, the 147 comments on which all three annotators disagree are all labelled neutral, a tie rule that neither paper states.

## Existing models

Kastrati et al. (2021) trained 20 classifiers on this dataset. Their best, a BiLSTM with attention, reports a weighted F1 of 72.09 [2].

The models use four kinds of input [2]:

- **Domain embeddings:** 300-dimensional word vectors learned from the dataset itself.
- **fastText:** pre-trained Albanian vectors from Common Crawl and Wikipedia, kept frozen.
- **mBERT:** multilingual BERT, fine-tuned on the Peltarion platform with 128-token inputs.
- **Bag of words:** term counts (tf) or tf-idf, for the conventional classifiers.

The neural networks are a DNN baseline, a 1D-CNN with 512 filters of width 3, a BiLSTM with 32 units per direction, and a hybrid 1D-CNN + BiLSTM. The CNN, BiLSTM and hybrid were each trained with and without a local self-attention layer over a window of 8 words. All networks read comments padded to 20 words and train with Adam, batch size 256 and dropout 0.3 [2]. The conventional classifiers are SVM, Naive Bayes, a decision tree and a random forest [2].

Attention improved every network. The five best models are close together: BiLSTM + attention 72.09, 1D-CNN + attention 71.56, random forest on tf-idf 71.44, mBERT 71.35 and hybrid + attention 71.32 [2]. The authors name sarcasm, slang, negation and the lack of Albanian NLP tools as the main difficulties [2].

## Jev

Jev labels each comment by answering one multiple-choice question about it. No model weights are trained on this dataset [4].

Jev (version `jev-1.13.0`) is the System One model of TypeSafe. It does not generate text. It takes a text and typed questions and returns a calibrated probability for each answer option. The same weights serve every user; a task is set only through the wording of the question [4].

Our question works as follows:

- **Input:** the comment plus one sentence of context: a Facebook comment under the NIPHK COVID-19 updates in 2020, in Gheg Albanian, often without ë and ç.
- **Options:** neutral, positive and negative, each defined in English with Albanian cue words and their meaning, for example *rrena* (lies), *urime* (congratulations), *ma gjall* (cheering a town).
- **Dataset convention:** calm criticism is neutral, open mockery or distrust is negative, and banter that ranks towns by their case counts is neutral.
- **Option order:** Jev leans toward the option listed first [5], so the question is asked in three orders and the probabilities are averaged. The prediction is the most probable label.

The question was tuned on the 2,256 validation comments only. We compared five versions and kept the best on validation (weighted F1 77.26). The 5,263 training comments were not used for Jev. The test split was scored once, after the question was fixed.

A comment costs about 2,640 input tokens. At USD 0.042 per million tokens [4], that is about USD 0.11 per 1,000 comments, and the whole dataset takes about four minutes.

## Evaluation setup

All models are scored on the paper's own test split, using the split from the authors' code: 3,223 of the 10,742 comments.

The authors' code splits the data twice with scikit-learn's `train_test_split(train_size=0.7, random_state=1000)` [3]. That gives 49% train (5,263 comments), 21% validation (2,256) and 30% test (3,223). The paper's text says 70/15/15 [2], but the code produced the published numbers, so we use the code's split. The code reads an input file that was not published. We assume it lists the comments in the same order as the released CSV.

We re-ran all 20 classifiers of the paper on this split instead of copying the published scores, for two reasons. The published code prints most scores on the validation split, which also monitors training; only Naive Bayes, the decision tree and the random forest are printed on test [3]. And the published scores come from a single run each.

The re-run follows the paper and its code [2, 3]:

- **Architectures and settings:** as in the paper. Every model is trained with 5 random seeds, and we report the mean.
- **Epochs:** each network keeps the epoch with the highest validation accuracy, the convergence metric the paper names. The paper and the code give different epoch counts.
- **mBERT:** the paper gives no fine-tuning settings, so we used `bert-base-multilingual-cased` with learning rate 2e-5, batch size 32 and up to 4 epochs.
- **Stop words:** the authors' Albanian stop-word list was not published, so the conventional classifiers run without one.

The main metric is the paper's: F1 averaged over the three classes and weighted by class size. We also report macro F1 and accuracy.

## Results

Jev reaches a weighted F1 of 75.71 on the 3,223 test comments. That is 3.6 points above mBERT, the best of the paper's top five, and 6.2 points above the paper's best model, BiLSTM + attention.

| Model | Paper F1 [2] | Test weighted F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| **Jev (no training)** | – | **75.71** | **74.89** | **75.43** |
| mBERT | 71.35 | 72.08 ± 1.07 | 69.59 | 72.24 |
| 1D-CNN + attention | 71.56 | 71.06 ± 0.32 | 67.74 | 71.40 |
| Random forest, tf-idf | 71.44 | 70.22 ± 0.38 | 66.41 | 71.78 |
| BiLSTM + attention | 72.09 | 69.52 ± 0.58 | 66.43 | 69.72 |
| Hybrid + attention | 71.32 | 68.77 ± 0.52 | 65.24 | 69.08 |

*Paper F1 is the published score, mostly from the validation split. The other columns are our re-run on the test split: the mean over 5 seeds, with ± the standard deviation. The five models are the paper's top five by published F1. Each was trained on the 5,263 training comments. Jev was trained on none; only its question was tuned, on the 2,256 validation comments.*

The gaps hold up in a paired bootstrap over the test comments (1,000 resamples):

- **Jev minus mBERT:** +3.63 weighted F1 (95% interval +2.03 to +5.19) and +5.30 macro F1.
- **Jev minus BiLSTM + attention:** +6.19 weighted F1 (95% interval +4.49 to +7.87) and +8.46 macro F1.

The paper's ranking does not hold on test. Across all 20 re-trained models, the best on test are mBERT (72.08) and the SVM on tf-idf (71.93), which is not in the paper's top five. BiLSTM + attention falls to 69.52. On the validation split, most re-trained models score within about one point of their published numbers, so the reproduction is close.

For a rough human reference, pairs of annotators agree with each other at a weighted F1 of 74.6 to 79.6 on the same test comments. Matching the majority vote is easier than matching a single annotator, so this is only a loose ceiling.

## References

1. F. Kadriu, D. Murtezaj, F. Gashi, L. Ahmedi, A. Kurti, Z. Kastrati. Human-annotated dataset for social media sentiment analysis for Albanian language. *Data in Brief* 43 (2022) 108436.
2. Z. Kastrati, L. Ahmedi, A. Kurti, F. Kadriu, D. Murtezaj, F. Gashi. A deep learning sentiment analyser for social media comments in low-resource languages. *Electronics* 10(10) (2021) 1133.
3. AlbAna: dataset and classifier code of Kastrati et al. (2021). GitHub, [lule-ahmedi/AlbAna](https://github.com/lule-ahmedi/AlbAna).
4. TypeSafe. Models: Jev 1.13. TypeSafe documentation.
5. TypeSafe. Jev 1.13 jaggedness. TypeSafe documentation.

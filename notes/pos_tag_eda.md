# POS Tag EDA

**Created:** 2026-08-13

## Aim

Summarise the Stanford CoreNLP POS-tag outputs generated from the retained BESSTIE
`en_AU` and `en_UK` data, and identify coarse syntactic differences associated
with source, variety, sentiment, and sarcasm.

POS artifacts used:

- `training_summary/linguistic_features/all.pos_summary.csv`
- `training_summary/linguistic_features/original_train.pos_summary.csv`
- `training_summary/linguistic_features/test.pos_summary.csv`

The main analysis below uses `all.pos_summary.csv` (`n = 6288`).

## What The POS Summary Contains

Each example keeps its original metadata and adds:

- `token_count`
- per-tag counts for `NOUN`, `PROPN`, `VERB`, `ADJ`, `ADV`, `PRON`, `ADP`,
  `DET`, `NUM`, `PRT`, `CONJ`, `INTJ`, `PUNCT`, and `X`

Because Google and Reddit texts have very different lengths, the comparisons
below focus mainly on **tag rates**:

```text
tag_rate = tag_count / token_count
```

## Global Length Patterns

Overall text length:

- Mean token count: `65.82`
- Median token count: `48`

By source:

| Source | Count | Mean tokens | Median tokens |
|---|---:|---:|---:|
| Google | 3141 | 77.60 | 61 |
| Reddit | 3147 | 54.07 | 31 |

By subset:

| Subset | Count | Mean tokens | Median tokens |
|---|---:|---:|---:|
| en_AU | 3080 | 64.32 | 42 |
| en_UK | 3208 | 67.27 | 53 |

By label:

| Group | Label | Mean tokens | Median tokens |
|---|---|---:|---:|
| Sentiment | Negative (`0`) | 66.48 | 44 |
| Sentiment | Positive (`1`) | 65.16 | 52 |
| Sarcasm | Not sarcastic (`0`) | 68.45 | 52 |
| Sarcasm | Sarcastic (`1`) | 54.24 | 31 |

### Takeaway

The biggest structural difference is still **source**. Google reviews are much
longer than Reddit comments, and sarcastic examples are shorter on average than
non-sarcastic ones.

## Dominant POS Categories

Across groups, the largest tag-rate categories are consistently:

- `NOUN`
- `VERB`
- `ADJ`
- `PUNCT`
- `DET`
- `ADP`

This means the useful signal is likely to come from **relative shifts** rather
than the presence of unusual tag classes.

## Source Differences

Average POS-rate differences, `reddit - google`:

| Tag | Difference |
|---|---:|
| `ADJ` | `-0.0422` |
| `VERB` | `+0.0323` |
| `NOUN` | `-0.0253` |
| `PRON` | `+0.0249` |
| `PROPN` | `+0.0220` |
| `CONJ` | `-0.0198` |
| `PUNCT` | `-0.0113` |
| `X` | `+0.0088` |

Interpretation:

- Google reviews are more **adjective-heavy** and **noun-heavy**.
- Reddit comments are more **verb-heavy**, **pronoun-heavy**, and
  **proper-noun-heavy**.

This fits the data domains well:

- Google reviews often describe attributes of places, food, service, and
  objects.
- Reddit comments are more conversational, eventive, and referential.

## Variety Differences

Average POS-rate differences, `en_UK - en_AU`:

| Tag | Difference |
|---|---:|
| `ADJ` | `+0.0130` |
| `PROPN` | `-0.0086` |
| `ADV` | `+0.0056` |
| `DET` | `+0.0048` |
| `CONJ` | `+0.0044` |
| `PRON` | `-0.0041` |

Interpretation:

- The AU/UK syntactic differences are much smaller than the source differences.
- `en_UK` is slightly more adjective-heavy.
- `en_AU` is slightly more proper-noun-heavy and pronoun-heavy.

### Takeaway

At this coarse POS level, **source explains much more variation than English
variety**. This is consistent with the earlier data EDA, where source also
strongly shaped sentiment and sarcasm prevalence.

## Sentiment Differences

Average POS-rate differences, `positive - negative`:

| Tag | Difference |
|---|---:|
| `ADJ` | `+0.0413` |
| `VERB` | `-0.0399` |
| `PRON` | `-0.0171` |
| `NOUN` | `+0.0160` |
| `PUNCT` | `+0.0159` |
| `CONJ` | `+0.0133` |

Interpretation:

- Positive texts are noticeably more **adjective-heavy**.
- Negative texts are more **verb-heavy** and **pronoun-heavy**.
- Positive texts also have slightly more punctuation and nouns.

This is especially strong in Google reviews:

- Negative Google examples: higher `VERB` and `ADV` rates
- Positive Google examples: higher `ADJ`, `NOUN`, and `PUNCT` rates

Practical reading:

- Positive reviews often enumerate qualities: taste, service, atmosphere,
  recommendations, and emphasis.
- Negative or complaint-style texts more often describe actions, failures, and
  events.

## Sarcasm Differences

Average POS-rate differences, `sarcastic - not sarcastic`:

| Tag | Difference |
|---|---:|
| `ADJ` | `-0.0329` |
| `VERB` | `+0.0219` |
| `X` | `+0.0130` |
| `CONJ` | `-0.0128` |
| `PRON` | `+0.0106` |
| `NOUN` | `-0.0098` |

Interpretation:

- Sarcastic texts are more **verb-heavy** and **pronoun-heavy**.
- Non-sarcastic texts are more **adjective-heavy** and **noun-heavy**.
- Sarcastic texts also have a higher `X` rate, which may reflect noisier,
  informal, non-standard, or harder-to-tag language.

This pattern is stronger in Google than in Reddit:

- In Google, sarcastic texts shift sharply away from `ADJ` and toward `VERB`,
  `PRON`, and `ADV`.
- In Reddit, sarcastic and non-sarcastic POS profiles are closer together.

### Important Caveat

Sarcasm rates are highly source-dependent, especially because British Google
sarcasm is nearly absent. So some apparent POS differences for sarcasm may
actually be mediated by source composition rather than sarcasm alone.

## Main Conclusions

1. Source is the dominant POS driver.
   Google is longer and more adjective-heavy; Reddit is shorter and more
   verb/pronoun-heavy.

2. Variety effects are real but small.
   At this coarse POS level, AU/UK differences are much weaker than source
   differences.

3. Sentiment has a clearer POS signature than sarcasm.
   Positive texts are more adjective-heavy, while negative texts are more
   verb-heavy.

4. Sarcasm looks more conversational and less descriptive.
   Sarcastic texts have relatively more verbs and pronouns, while non-sarcastic
   texts have more adjectives and nouns.

5. POS alone is unlikely to explain the task.
   The differences are interpretable, but they are coarse and entangled with
   source. They are better treated as supporting signals than as a complete
   explanation.

## Suggested Follow-Ups

- Compare POS rates within `source × subset` cells before drawing stronger
  conclusions about variety.
- Inspect the `X` tag and punctuation-heavy examples among sarcastic texts for
  non-standard language, fragments, emojis, or markup-like artifacts.
- Derive higher-level features such as:
  - adjective-to-verb ratio
  - noun-to-pronoun ratio
  - punctuation density
  - proper-noun density
- Test whether these POS-derived features help:
  - prompt example selection
  - error analysis for Gemma prompt predictions
  - lightweight classical baselines

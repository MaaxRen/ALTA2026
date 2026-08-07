# BESSTIE Dataset Paper Summary

## Paper

Dipankar Srirag, Aditya Joshi, Jordan Painter, and Diptesh Kanojia. 2025. “BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English.” *Findings of ACL 2025*, pages 8413–8429.

Local copy: [2025.findings-acl.441.pdf](../resources/papers/2025.findings-acl.441.pdf)

## Purpose and contribution

BESSTIE is a benchmark for sentiment and sarcasm classification across three national varieties of English:

- `en-AU`: Australian English
- `en-IN`: Indian English
- `en-UK`: British English

It addresses the shortage of labelled, naturally occurring text for evaluating model performance across English varieties. Unlike benchmarks that create dialect data through synthetic transformations, BESSTIE uses user-generated text containing naturally occurring vocabulary, grammar, spelling, colloquialisms, and cultural references.

The paper contributes:

1. A manually annotated dataset for binary sentiment and sarcasm classification.
2. Validation of whether the collected text represents the intended English varieties.
3. Evaluation of nine encoder and decoder language models.
4. Cross-variety and cross-domain experiments.
5. Error analysis covering dialect features, colloquialisms, missing context, and code-switching.

## Data collection

The dataset contains two domains collected using different proxies for language variety.

### Google Places reviews

- Reviews were selected geographically from cities in Australia, India, and the United Kingdom.
- City population thresholds differed by country.
- Tourist attractions were excluded to reduce the likelihood of collecting reviews written by visitors.
- A fastText language probability threshold of 0.98 was used to remove non-English content.
- The authors retained reviews with 2- and 4-star ratings rather than easily separated 1- and 5-star reviews.
- In a preliminary experiment, this choice reduced average DistilBERT sentiment macro-F1 from 0.96 to 0.79, indicating that the retained reviews were more nuanced and difficult.

### Reddit comments

- Comments were collected from country-related subreddits selected by native speakers.
- The authors initially collected 12,000 comments per variety, limited collection to 20 comments per post, and sampled 3,000 comments per variety for annotation.
- User and post identifiers were discarded.
- The Reddit data represents a static snapshot from 2024.

## Validation and annotation

### Variety validation

The intended variety was checked through manual annotation and an automatic variety predictor.

- Agreement with the presumed variety labels was `κ = 0.41` for the Indian annotator and `κ = 0.34` for the British annotator.
- Agreement between the two annotators was only `κ = 0.26`.
- Australian and British English were particularly difficult for annotators to distinguish.
- Indian Reddit data received lower English and variety probabilities, which the authors attribute to code-mixing.
- ICE-GB was unavailable, so the automatic variety predictor was trained as an inner-circle-versus-outer-circle classifier rather than a complete three-variety classifier.

### Sentiment and sarcasm annotation

One native-speaker annotator was hired for each variety. Each example received:

- A sentiment label: negative, positive, or discard.
- A sarcasm label: sarcastic, not sarcastic, or discard.

Neutral, uninformative, and machine-generated examples were discarded. A second annotation exercise used 50 examples per variety to estimate reliability.

| Variety | Sentiment kappa | Sarcasm kappa |
|---|---:|---:|
| en-AU | 0.61 | 0.47 |
| en-IN | 0.65 | 0.51 |
| en-UK | 0.79 | 0.63 |

Agreement was consistently lower for sarcasm than sentiment.

## Experimental setup

The authors evaluated nine models:

- English encoders: ALBERT, BERT, and RoBERTa
- Multilingual encoders: multilingual BERT, multilingual DistilBERT, and XLM-R
- Decoders: Gemma, Mistral, and Qwen

Encoder models were fully fine-tuned using class-weighted cross-entropy. Quantised decoder models were fine-tuned with QLoRA. Models were trained for 30 epochs, and results were reported using macro-F1 to reduce the influence of label imbalance.

## Principal results

Models performed best on Australian English, followed by British English, and worst on Indian English.

| Domain and task | en-AU | en-IN | en-UK |
|---|---:|---:|---:|
| Google sentiment | 0.94 | 0.64 | 0.86 |
| Reddit sentiment | 0.78 | 0.69 | 0.78 |
| Reddit sarcasm | 0.62 | 0.56 | 0.58 |
| Average | 0.78 | 0.63 | 0.74 |

Additional findings include:

- Average macro-F1 was 0.81 for Google sentiment, 0.75 for Reddit sentiment, and 0.59 for Reddit sarcasm.
- Mistral performed best for sentiment, averaging 0.91 on Google and 0.84 on Reddit.
- Mistral and multilingual BERT performed best for sarcasm, both averaging 0.68.
- Encoder models generally outperformed decoder models.
- Monolingual models only marginally outperformed multilingual models. Multilingual pretraining therefore did not automatically produce robustness across varieties of English.
- Google reviews were easier than Reddit comments, probably because they were longer, more formal, and more informative.
- Sentiment transferred relatively well across varieties.
- Variety-specific fine-tuning improved in-variety sarcasm performance but often harmed cross-variety performance.
- Cross-domain models performed worse than in-domain models. Transfer from Reddit to Google was generally better than transfer from Google to Reddit.

## Error analysis

The authors manually inspected examples misclassified by Mistral. They identified four recurring sources of error:

1. Dialectal lexical and grammatical features.
2. Locale-specific colloquial expressions.
3. Examples requiring additional conversational or cultural context.
4. Code-mixed text.

Indian English produced 97 occurrences of dialect features among 90 inspected errors. Examples included article omission, pronoun dropping, fronted objects or subjects, copula omission, and non-standard uses of “very.” Locale-specific colloquialisms occurred in all three varieties.

## Problems and limitations

### Variety labels are imperfect proxies

Geographic location and subreddit topic do not prove that an author uses the corresponding English variety. National labels also conceal substantial regional, social, and generational variation. For example, English used in Sydney may differ from English used in Perth.

### Manual variety identification was difficult

The low inter-annotator agreement (`κ = 0.26`) shows that people had difficulty distinguishing the varieties, especially Australian and British English. This complicates the claim that each subset cleanly represents its national variety.

### Automatic validation was limited

The lack of ICE-GB prevented complete three-way validation. The resulting inner-circle-versus-outer-circle predictor could distinguish Indian English from the other varieties but could not adequately validate Australian English against British English.

### Code-switching remained in the data

Despite English-language filtering, code-mixed text remained, especially in the Indian Reddit subset. This reduced automatic quality scores and contributed to model errors.

### Annotation was subjective and under-resourced

Each variety was principally labelled by one annotator, potentially introducing individual bias. The independent validation sample contained only 50 examples per variety. Two of the principal annotators were also paper authors.

### Sarcasm was difficult to annotate and model

Sarcasm annotation had lower agreement than sentiment annotation. Sarcasm may depend on cultural knowledge, contemporary references, speaker intention, and conversational context that is unavailable in an isolated comment. The low average macro-F1 of 0.59 indicates that sarcasm classification remains largely unresolved.

### Labels and domains were imbalanced

Sentiment prevalence differed substantially between Google and Reddit. In the dataset described by the paper, Google had effectively no positive sarcasm labels, so published sarcasm experiments were restricted to Reddit. Domain and label effects can consequently be difficult to separate.

### Cross-variety and cross-domain generalisation was weak

Fine-tuning on one variety could reduce sarcasm performance on another. Models also struggled when trained on one source and evaluated on the other. These results show that language-variety and platform-specific cues are not reliably transferable.

### The dataset is temporally and demographically narrow

Online language changes rapidly, so a static 2024 Reddit snapshot will age. Reddit also has platform-specific demographics and discourse conventions. The authors relied on Reddit partly because access to the Twitter/X API was too expensive.

## Relevance to the current project

The published experiments treat Google sarcasm labels as unavailable and evaluate sarcasm only on Reddit. The current Hugging Face release contains Google sarcasm labels, so analyses using the current release do not exactly reproduce the paper-era benchmark.

Our retained `en_AU` and `en_UK` data should therefore be analysed by source as well as in aggregate. This is especially important for sarcasm: in the currently released train and validation data, British Google reviews contain only one sarcastic example. Any correlation or model performance calculated for that stratum will be unstable.

Correlation between sarcasm and negative sentiment should be interpreted as an association, not causation. It may partly reflect source composition because Reddit contains both more negative and more sarcastic examples than Google.

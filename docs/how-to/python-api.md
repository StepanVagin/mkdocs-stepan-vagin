# Python API

This page explains how to use SpellChecker programmatically. All examples are based
on the n-gram model. For the T5 transformer model, see
[T5 Transformer](../models/t5.md).

---

## Prerequisites

Before running any code on this page:

- Python 3.10 or later
- Dependencies installed: `pip install -r requirements.txt`
- At least one trained model on disk (see [Training a Custom Model](../models/custom-training.md))
- NLTK words corpus available (downloaded automatically on first use)

The examples below assume trained model files exist at `models/ngram/`:

```
models/ngram/
├── 1gram_model.json
├── 2gram_model.json
└── 3gram_model.json
```

---

## Loading models

`NGramModel` loads a single n-gram model from a JSON file saved during training.
`SpellingChecker` takes a list of models and combines their vocabularies.

```python
from spellchecker.models import NGramModel, SpellingChecker

model_dir = "models/ngram"

models = []
for n in [1, 2, 3]:
    model = NGramModel(n=n)
    model.load_model(f"{model_dir}/{n}gram_model.json")
    models.append(model)

checker = SpellingChecker(models)
```

Using all three orders gives the best context-aware corrections. If you only have
a 1-gram model (e.g. after a quick demo run), pass a single-element list — the
checker degrades gracefully but loses context sensitivity.

You can also adjust the probability threshold below which corrections are not applied:

```python
checker = SpellingChecker(models, probability_threshold=0.001)  # default
```

---

## Correcting text

`correct_text` takes a string and returns two values: the corrected string and a
list of `CorrectionResult` objects, one per token in the input.

**Signature:**
```python
def correct_text(self, text: str) -> Tuple[str, List[CorrectionResult]]
```

**Example:**
```python
corrected_text, results = checker.correct_text("I love this prodct")

print(corrected_text)
# "I love this product"
```

`CorrectionResult` is a dataclass with the following fields:

| Field | Type | Description |
|---|---|---|
| `original_word` | `str` | The token as it appeared in the input |
| `corrected_word` | `str` | The suggested replacement (same as original if no correction was made) |
| `probability` | `float` | N-gram probability of the corrected word |
| `confidence` | `float` | Confidence score in the range [0.0, 1.0] |
| `edit_distance` | `int` | Levenshtein distance between original and correction (0 if unchanged) |

### Inspecting corrections

To find only the tokens that were actually changed, filter for results where
`corrected_word` differs from `original_word`:

```python
corrected_text, results = checker.correct_text("I love this prodct")

changes = [r for r in results if r.corrected_word != r.original_word]

for r in changes:
    print(f"{r.original_word!r} -> {r.corrected_word!r}  "
          f"(confidence={r.confidence:.2f}, edit_distance={r.edit_distance})")
# "prodct" -> "product"  (confidence=0.85, edit_distance=1)
```

Tokens with no correction have `edit_distance=0` and `corrected_word == original_word`.

### Correcting a longer passage

```python
text = "The quikc brown fox jumpd over the lasy dog."

corrected_text, results = checker.correct_text(text)
print(corrected_text)
# "The quick brown fox jumped over the lazy dog."

changes = [r for r in results if r.corrected_word != r.original_word]
print(f"{len(changes)} correction(s) applied")
# 3 correction(s) applied
```

---

## Checking a single word

Use `check_word` when you need to correct one word with optional context, without
running the full tokenization pass.

**Signature:**
```python
def check_word(
    self,
    word: str,
    context: List[str] = None
) -> CorrectionResult
```

**Example:**
```python
result = checker.check_word("prodct", context=["I", "love", "this"])

print(result.corrected_word)   # "product"
print(result.confidence)       # 0.85
print(result.edit_distance)    # 1
```

If the word is found in the vocabulary it is returned unchanged with
`confidence=1.0`. If no candidate clears the probability threshold, the original
word is returned with `confidence=0.0`.

---

## Preparing training data

The parsers in `spellchecker.data.parsers` clean and normalize raw corpora before
n-gram training.

### Cleaning text with `UniversalTextCleaner`

`UniversalTextCleaner.clean` returns the cleaned string or `None` if the input
should be filtered out.

**Signature:**
```python
def clean(self, text: str) -> Optional[str]
```

```python
from spellchecker.data.parsers import UniversalTextCleaner

cleaner = UniversalTextCleaner(
    min_length=20,    # filter out texts shorter than 20 characters
    max_length=500,   # filter out texts longer than 500 characters
    remove_urls=True,
    remove_emails=True,
    normalize_whitespace=True,
)

result = cleaner.clean("  Hello   world!  ")
print(result)
# "Hello world!"

result = cleaner.clean("hi")
print(result)
# None  (filtered: too short)
```

### Processing a Wikipedia dump

`WikipediaParser` expects WikiExtractor output (a directory of `wiki_*` files) or a
plain-text file with one document per line. `save_to_file` returns the number of
passages written.

**Signature:**
```python
def save_to_file(
    self,
    input_path: Union[str, Path],
    output_file: Union[str, Path]
) -> int
```

```python
from spellchecker.data.parsers import WikipediaParser, UniversalTextCleaner

cleaner = UniversalTextCleaner(min_length=20, max_length=500)
parser = WikipediaParser(cleaner)

count = parser.save_to_file(
    input_path="data/raw/unsupervised/wikipedia/extracted",
    output_file="data/processed/unsupervised/wikipedia.txt",
)
print(f"Saved {count} passages")
```

### Processing CC-News or BookCorpus

The same `save_to_file` interface is available on `CCNewsParser` (expects a JSONL
file) and `BookCorpusParser` (expects a directory of `.txt` files or a single file).

```python
from spellchecker.data.parsers import CCNewsParser, BookCorpusParser

# CC-News: input is a JSONL file
CCNewsParser().save_to_file(
    input_file="data/raw/unsupervised/cc_news.jsonl",
    output_file="data/processed/unsupervised/ccnews.txt",
)

# BookCorpus: input is a directory of .txt files
BookCorpusParser().save_to_file(
    input_path="data/raw/unsupervised/bookcorpus/",
    output_file="data/processed/unsupervised/bookcorpus.txt",
)
```

Or use the convenience function to dispatch by corpus name:

```python
from spellchecker.data.parsers.unsupervised_parser import process_unsupervised_corpus

count = process_unsupervised_corpus(
    corpus_type="wikipedia",   # "wikipedia" | "ccnews" | "bookcorpus"
    input_path="data/raw/unsupervised/wikipedia/extracted",
    output_file="data/processed/unsupervised/wikipedia.txt",
)
```

---

## Evaluating performance

`Evaluator.evaluate` computes token-level metrics given three parallel lists of
strings: reference (ground truth), predicted (checker output), and original
(uncorrected input).

**Signature:**
```python
@staticmethod
def evaluate(
    reference_texts: List[str],
    predicted_texts: List[str],
    original_texts: List[str]
) -> EvaluationMetrics
```

```python
from spellchecker.models import Evaluator

originals  = ["I love this prodct", "He jumpd over the fense"]
references = ["I love this product", "He jumped over the fence"]
predicted  = [checker.correct_text(t)[0] for t in originals]

metrics = Evaluator.evaluate(references, predicted, originals)

print(f"Exact match    : {metrics.exact_match:.2f}")
print(f"Precision      : {metrics.precision:.2f}")
print(f"Recall         : {metrics.recall:.2f}")
print(f"F1             : {metrics.f1_score:.2f}")
print(f"Total errors   : {metrics.total_errors}")
print(f"Corrected      : {metrics.corrected_errors}")
print(f"False positives: {metrics.false_positives}")
```

`EvaluationMetrics` fields:

| Field | Type | Description |
|---|---|---|
| `exact_match` | `float` | Fraction of texts corrected to exactly the reference |
| `precision` | `float` | Fraction of made corrections that were correct |
| `recall` | `float` | Fraction of actual errors that were corrected |
| `f1_score` | `float` | Harmonic mean of precision and recall |
| `total_errors` | `int` | Total token-level errors across all inputs |
| `corrected_errors` | `int` | Errors that were successfully corrected |
| `false_positives` | `int` | Corrections applied to tokens that were already correct |

---

## Common errors

!!! warning "FileNotFoundError when loading a model"
    The model file does not exist at the given path. Train a model first:
    ```bash
    python scripts/train_ngram_model.py \
        --data "data/processed/unsupervised/*.txt" \
        --output models/ngram
    ```

!!! warning "Low correction quality on short input"
    N-gram models use surrounding words to score candidates. Single-word input
    only uses the 1-gram model. Pass full sentences for best results.

!!! note "NLTK words corpus"
    `load_english_dictionary` downloads the NLTK words corpus on first use if it
    is not already present. This requires an internet connection on first run.

---

## Next steps

- [CLI Scripts](cli.md) — run corrections from the command line without writing Python
- [Training a Custom Model](../models/custom-training.md) — train on your own corpus
- [T5 Transformer](../models/t5.md) — deep learning-based correction pipeline

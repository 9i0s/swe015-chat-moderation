# Study guide: how GuardChat works (read this before the presentation)

The homework rules say you must be able to explain every box in the architecture diagram. This page walks
through the system in the order a message travels, then covers training, quantization and the numbers.

## 1. The task

Binary classification of a Turkish chat message: **offensive** (insults, profanity, threats) or **clean**.
Data: OffensEval-TR 2020, 31,756 tweets, 19.3 % offensive. Because the classes are imbalanced, the main
metric is **macro-F1** (the average of the F1 of each class), not accuracy. A model that always says "clean"
gets 80.7 % accuracy but only ~0.45 macro-F1.

Split: 80 / 10 / 10 stratified (same offensive ratio in each part), seed 42. Training data trains the models,
validation chooses hyper-parameters and the decision threshold, and the test set is used once at the end.

## 2. What happens when you press Send

1. **Browser** (`app/static/index.html`) sends `{"text": ..., "model": "bert_int8"}` to `POST /api/moderate`.
2. **FastAPI** (`app/server.py`) starts a timer and calls the selected model's `predict_proba`.
3. **Pre-processing** (`moderation/normalize.py`):
   - `light_normalize` (for the Transformer): collapse repeated `@USER`, replace links by `URL`, shorten 3+
     repeated characters to 2. Casing and punctuation are kept because BERTurk was pretrained on cased text.
   - `deobfuscate` (Transformer, inference only): rewrites only obfuscated words: leetspeak `s1kt1r` → `siktir`,
     spaced or dotted letters `a.m.k` → `amk`. Star-censored words (`s*ktir`) are left alone: removing the star
     made validation worse, because the model learned that self-censoring itself signals an offensive tweet.
     It changes 9 of the 3,176 validation tweets and leaves validation macro-F1 unchanged (FP32 0.787, INT8 0.784),
     and it fixes deliberate leetspeak, which is common in game chat.
   - `normalize` (for tf-idf, word2vec, lexicon): Turkish lower-casing (Python's `lower()` turns `I` into `i`,
     but in Turkish `I` → `ı` and `İ` → `i`), leetspeak, separators, repeated letters, removing punctuation.
4. **Model** returns P(offensive).
5. **Post-processing**: compare with the model's threshold (tuned on validation, `models/thresholds.json`) →
   label. If offensive, the lexicon matcher masks the matching words (`salaksın` → `s*******`); if no word
   matches, the whole message is hidden. The confidence shown is the distance from the threshold rescaled to
   50–100 %.
6. The JSON answer (label, confidence, masked text, latency in ms) is drawn in the chat and in the latency log.

## 3. The models

### Lexicon matcher (rule baseline, area B), `moderation/lexicon.py`
- **Minimum edit distance (Levenshtein):** the smallest number of insertions, deletions and substitutions that
  turn one word into another, computed with dynamic programming (a table of size (m+1) × (n+1), each cell =
  min(delete, insert, substitute)). `levenshtein("kitten", "sitting") = 3`.
- A token is flagged if it is within distance 0 (short words), 1 (4–7 letters) or 2 (8+ letters) of an entry,
  or starts with a long entry (Turkish suffixes: `salak` → `salaksın`).
- Weakness: misses insults not in the list; ambiguous words (`mal` = "goods" or "idiot").

### tf-idf + logistic regression (classical baseline, area B), `scripts/train_tfidf.py`
- **tf-idf:** a word gets a high weight if it is frequent in this message (tf) but rare across messages (idf =
  log N / df). We use word 1–2-grams and character 2–5-grams; character n-grams are robust to suffixes
  and spelling variants.
- **Logistic regression:** a linear model + sigmoid; `class_weight="balanced"` gives the rare class more weight.
  C (inverse regularisation strength) chosen on validation.

### word2vec + feedforward network (area A), `scripts/train_w2v_ffnn.py`
- **word2vec skip-gram:** learns a 100-number vector per word by predicting the words around it (window 5) with
  negative sampling. Trained only on our training tweets.
- Message vector = [mean ; max] of its word vectors (200 numbers), standardised.
- **Feedforward network:** 200 → 128 → 64 → 2 with ReLU and dropout 0.3, softmax output.
- **Training:** mini-batch SGD (lr 0.01, momentum 0.9), cross-entropy with class weights. **Backpropagation**
  computes the gradient of the loss with respect to every weight using the chain rule, layer by layer from the
  output back to the input; SGD then moves each weight a small step against its gradient. Early stopping on
  validation macro-F1.
- Why it is weakest: averaging loses word order and negation, and 25k tweets are too few to learn good
  word vectors from scratch.

### DistilBERTurk fine-tuned (area C), `scripts/train_transformer.py`
- **DistilBERTurk** = a 6-layer distilled version of BERTurk (Turkish BERT), pretrained by dbmdz on Turkish text.
- **WordPiece tokenizer** (32k sub-words): rare words are split, e.g. `Gerizekalı` → `Geri ##ze ##kal ##ı`.
- **Self-attention:** every token builds its new representation as a weighted average of all tokens,
  with weights softmax(QKᵀ/√d); 12 heads look at different relations. This is why the model understands
  context ("katliam yaptı, efsane oynadı" is praise).
- **Fine-tuning:** add a classification head on the [CLS] token and train all weights with AdamW
  (lr 3e-5, batch 32, linear warm-up then decay, 3 epochs, max 96 tokens). We kept the epoch with the best
  validation macro-F1.

### Quantization (area D), `scripts/quantize.py`
- **Dynamic INT8 quantization:** each Linear layer's FP32 weights are mapped to 8-bit integers with a scale
  (w ≈ scale × int8). Activations are quantized on the fly per input. Matrix multiplications run in INT8,
  which was 3.5× faster in our measurement and the weights take 4× less memory.
- Only Linear layers are quantized; the 32,000 × 768 embedding table (98 MB) stays FP32, which is why the file
  shrinks from 272 MB to 143 MB, not 4×.
- No retraining and no calibration data needed ("post-training dynamic quantization").
- **Batch size matters for dynamic INT8.** The activation scale is computed from the whole input tensor, so when
  64 padded messages are scored together, one message's score depends on the others (we measured differences of
  up to 0.5 in probability). FP32 does not have this problem. Because the app scores one message at a time, the
  INT8 model is evaluated one message at a time too (`TransformerModel.predict_proba`, batch size 1).

## 4. The latency budget

Set before measuring: a CPU server must handle at least 40 messages per second per core → **25 ms per message
on 1 thread**. FP32 DistilBERTurk misses it; INT8 meets it. Latency is measured end-to-end for one message
(batch size 1), averaged over 200 messages after 10 warm-up runs.

## 5. Questions you may be asked

- *Why macro-F1?* Imbalanced classes: accuracy rewards always predicting "clean".
- *Why tune the threshold?* The best cut-off for macro-F1 is rarely 0.5 when classes are imbalanced. We tune it
  on validation, never on test.
- *Why not use the official test set?* Its labels are not in the public mirror; a stratified held-out split
  from the labelled data is used instead, the same for every model.
- *Why DistilBERTurk and not BERTurk-base?* Training on 2 CPU cores: 6 layers train in about half the time,
  and the faster model is easier to fit in the latency budget.
- *Why does INT8 keep the same quality?* The weights lose little information at 8 bits; the decisions agree with
  FP32 on almost all test tweets (see slide 7).
- *What would you do next?* In-domain game-chat data, quantize embeddings / ONNX Runtime, distillation into a
  smaller student, finer labels.

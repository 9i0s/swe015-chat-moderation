Test set: 3176 tweets (613 offensive). Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz, 1 thread, batch 1, 200 runs. Latency budget: 25 ms/message.

| Model | Area | Number format | Macro-F1 (95% CI) | Off. recall | Size (MB) | Latency mean / p95 (ms) | Budget |
|---|---|---|---|---|---|---|---|
| Edit-distance lexicon (rules) | B | - | 0.670 (0.648-0.693) | 0.308 | 0.005 | 2.5 / 6.8 | yes |
| tf-idf + logistic regression | B | FP64 | 0.763 (0.743-0.782) | 0.599 | 2.2 | 1.7 / 2.3 | yes |
| word2vec + feedforward NN | A | FP32 | 0.659 (0.637-0.680) | 0.437 | 11.7 | 0.1 / 0.2 | yes |
| DistilBERTurk fine-tuned (FP32) | C | FP32 | 0.792 (0.773-0.810) | 0.602 | 272.4 | 44.6 / 61.2 | no |
| DistilBERTurk fine-tuned (INT8) | C+D | INT8 (dynamic) | 0.785 (0.768-0.802) | 0.657 | 143.3 | 12.9 / 22.4 | yes |

FP32 vs INT8 Transformer: identical labels on 94.7% of test messages.

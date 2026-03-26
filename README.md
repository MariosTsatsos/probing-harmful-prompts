# Probing Harmful-Intent Signals in Transformer Activations

A small, controlled probing study of whether **harmfulness labels are linearly decodable from internal activations** in a modern open LLM.

This repo compares a **base** model and a **post-trained / instruction-tuned** model with the **same architecture**:

- `HuggingFaceTB/SmolLM3-3B-Base`
- `HuggingFaceTB/SmolLM3-3B`

The core question is simple:

> Given a prompt alone, how early and how strongly can a linear probe decode whether it belongs to the “non-harmful” or “harmful” half of a tightly controlled prompt set?

---

## Popular summary

I built a small benchmark of 128 prompts across four domains (`chem`, `sec`, `social`, `phys`) and four harm levels (`0–3`), then collapsed the task to a binary label:

- levels `0–1` → non-harmful
- levels `2–3` → harmful

For every prompt, I extracted hidden activations from **every layer** of two SmolLM3-3B checkpoints and trained a **linear logistic-regression probe** at each layer.

### Main result

Using `mean_last8` pooling and **GroupKFold** cross-validation grouped by topic chain:

- **Base model:** best probe AUC = **0.974 ± 0.016** at **layer 13**
- **Safety-tuned model:** best probe AUC = **0.965 ± 0.022** at **layer 13**

The signal also generalises across domains:

- base, `chem+sec → social+phys`: **0.954** at layer 14
- base, `social+phys → chem+sec`: **0.938** at layer 20
- tuned, `chem+sec → social+phys`: **0.948** at layer 14
- tuned, `social+phys → chem+sec`: **0.932** at layer 19

A simple **difference-of-means** direction is already strong:

- base: **0.951** at layer 15
- tuned: **0.939** at layer 15

### Interpretation

This repo does **not** claim to discover a clean “abstract harmful-intent neuron” or a fully disentangled internal intent concept.

What it does show is narrower and more defensible:

1. harmfulness labels are **strongly linearly decodable**
2. the signal is present **partly in lexical / embedding-space structure**
3. later layers **sharpen** that signal substantially
4. the effect does **not collapse** under cross-family generalisation

The word-order ablation matters here:

- original peak AUC: **0.974** at layer 13
- word-shuffled peak AUC: **0.920** at layer 27
- drop: **0.054**

So the signal is **mixed**:
- a large lexical/distributional component survives word-order destruction
- some additional compositional structure is lost

---

## What is in the repo

```text
.
├── README.md
├── requirements.txt
├── Linear_Probing_Harmful_Intent_v2.ipynb
├── data/
│   └── data_v7.csv
└── results/
    ├── plot1_auc_vs_layer.png
    ├── plot2_cross_family.png
    ├── plot3_pca.png
    └── (any additional plots / exports)
```

---

## Method

### Dataset

- **128 prompts**
- **4 families:** `chem`, `sec`, `social`, `phys`
- **4 harm levels:** `0, 1, 2, 3`
- binary split at `level >= 2`

The notebook groups prompts by **topic chain**:

- each family contains **8 topics**
- each topic appears in **4 level variants**
- GroupKFold groups all 4 variants of the same family-topic together
- this prevents near-duplicate leakage across folds

### Models

From the notebook:

- `MODEL_BASE = "HuggingFaceTB/SmolLM3-3B-Base"`
- `MODEL_SAFE = "HuggingFaceTB/SmolLM3-3B"`

The official model cards describe SmolLM3 as a **3B parameter decoder-only transformer**, and note that the modeling code is available in `transformers v4.53.0`. The instruct checkpoint is the `SmolLM3-3B` model card and the base checkpoint is `SmolLM3-3B-Base`. citeturn463858view0turn463858view2

### Activation extraction

For each prompt and layer:

- tokenize with padding
- run the model with `output_hidden_states=True`
- collect hidden states for **all layers including the embedding layer**
- pool token activations into one prompt vector

Three pooling strategies were tested:

- `mean_all`
- `mean_last8`
- `last_token`

The summary results in this repo use:

- **`POOLING = "mean_last8"`**

### Probe

At each layer:

- input = pooled activation vector
- classifier = `StandardScaler` + `LogisticRegression`
- metric = **ROC AUC**
- CV = **5-fold GroupKFold**

### Controls / ablations

Included in the notebook:

- shuffled-label sanity check
- cross-family generalisation
- difference-of-means probe
- random-features baseline
- TF-IDF bag-of-words baseline
- pooling comparison
- word-order shuffle test

---

## Key numbers

### Primary result

| Experiment | Best AUC | Layer |
|---|---:|---:|
| Base probe | 0.974 ± 0.016 | 13 |
| Safety-tuned probe | 0.965 ± 0.022 | 13 |
| Base diff-of-means | 0.951 | 15 |
| Safety-tuned diff-of-means | 0.939 | 15 |

### Cross-family

| Train families | Test families | Base | Safety-tuned |
|---|---|---:|---:|
| chem + sec | social + phys | 0.954 (L14) | 0.948 (L14) |
| social + phys | chem + sec | 0.938 (L20) | 0.932 (L19) |

### Baselines

| Baseline / reference | AUC |
|---|---:|
| Random features (layer 13, shuffled cols) | 0.601 |
| TF-IDF bag-of-words | 0.878 |
| Probe at layer 0 (embedding) | 0.723 |
| Probe at peak layer 13 | 0.974 |

### Word-order shuffle

| Setting | Peak AUC | Peak layer |
|---|---:|---:|
| Original prompts | 0.974 | 13 |
| Word-shuffled prompts | 0.920 | 27 |

---

## What this repo argues

A careful reading of the results supports the following claim:

> Harmfulness labels in this controlled prompt set are strongly linearly decodable from SmolLM3 activations. A substantial part of the signal is already present in lexical / embedding-space structure, but later layers amplify it well beyond an embedding-only or bag-of-words baseline.

That is a meaningful result.

It is also deliberately narrower than stronger claims such as:
- “the model contains a clean abstract harmful-intent concept”
- “this isolates intent independently of lexical features”
- “instruction tuning removes harmfulness representations”

This repo does **not** support those stronger claims.

---

## Limitations

This is a **small, synthetic probing benchmark**, not a full behavioural or causal study.

Important limitations:

1. **Small dataset**
   - 128 prompts is enough for a sharp pilot, not a final benchmark

2. **Lexical signal is strong**
   - layer 0 AUC is already 0.723
   - word-order shuffling only reduces peak AUC by 0.054

3. **Linear probes are readouts, not causal explanations**
   - they show what is decodable
   - they do not show how the model uses the information

4. **Synthetic prompt design still matters**
   - even with topic-chain grouping and cross-family tests, dataset construction remains the main source of risk

5. **No intervention result here**
   - this repo probes representations
   - it does not steer, erase, or causally patch them

---

## Why this may be useful

For an AI safety / interpretability audience, this repo is best read as:

- a **small but controlled representation-level study**
- an example of **probing with non-trivial controls**
- a stepping stone toward:
  - causal interventions
  - representation steering
  - feature transfer across models
  - stronger datasets with cleaner lexical controls

---

## Reproduce

### Environment

SmolLM3’s official model cards state that the modeling code is available in **`transformers v4.53.0`**. citeturn463858view0turn463858view1

```bash
pip install -r requirements.txt
```

### Run

Open the notebook:

```bash
jupyter notebook Linear_Probing_Harmful_Intent_v2.ipynb
```

The notebook was run in **Google Colab on a Tesla T4 GPU runtime** according to the saved outputs. fileciteturn1file0L14-L15

Place:
- `data_v7.csv` in `data/`
- output plots in `results/`

If you prefer, update the notebook paths accordingly.

---

## Suggested next steps

If I were extending this repo, I would do these next:

1. add a larger dataset with stronger lexical controls
2. add causal interventions on the top probe direction
3. compare multiple architectures, not just two checkpoints of one family
4. evaluate transfer of probe directions across model sizes
5. test whether probe directions predict refusal behaviour, not just prompt labels

---

## Citation

If you use this repo, cite the notebook and model checkpoints directly.

- Notebook: `Linear_Probing_Harmful_Intent_v2.ipynb`
- Models:
  - `HuggingFaceTB/SmolLM3-3B-Base`
  - `HuggingFaceTB/SmolLM3-3B`

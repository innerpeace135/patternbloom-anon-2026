<div align="center">

# PatternBloom: Empowering Agentic RAG with Externalized RL-Distilled Graph Patterns

</div>

---

PatternBloom learns to construct a query-conditioned evidence hypergraph
(Stage I) and to traverse it via type-abstracted reasoning skeletons stored
in a Graph Pattern Memory (Stage II).

## 📑 Contents

* [Pipeline overview](#-pipeline-overview)
* [Installation](#%EF%B8%8F-installation)
* [Step 1: prepare the training data](#-step-1-prepare-the-training-data)
* [Step 2: build the hypergraph index](#-step-2-build-the-hypergraph-index)
* [Step 3: launch the retrieval API](#-step-3-launch-the-retrieval-api)
* [Step 4: Stage I training (IDR)](#-step-4-stage-i-training-information-density-reward)
* [Step 5: distill the Graph Pattern Memory](#-step-5-distill-the-graph-pattern-memory)
* [Step 6: Stage II training (PAR)](#-step-6-stage-ii-training-pattern-augmented-reward)
* [Step 7: evaluation](#-step-7-evaluation)
* [Project layout](#-project-layout)
* [Configuration](#-configuration)
* [License](#-license)

---

## 🚀 Pipeline overview

<div align="center">

| Step | Description | Script |
|:----:|:------------|:-------|
| 1 | Sample 14K balanced training queries (7K HotpotQA + 7K 2WikiMultiHopQA) | `scripts/01_prepare_data.sh` |
| 2 | Build the hypergraph index over the Wikipedia corpus | `scripts/02_build_hypergraph.sh` |
| 3 | Launch the retrieval API server | `scripts/03_start_api.sh` |
| 4 | Stage I: train policy with the Information-Density Reward | `scripts/04_train_stage1.sh` |
| 5 | Distill the Graph Pattern Memory from the Stage I best checkpoint | `scripts/05_distill_gpm.sh` |
| 6 | Stage II: refine policy with the Pattern-Augmented Reward | `scripts/06_train_stage2.sh` |
| 7 | Evaluate the trained policy on the HotpotQA test set | `scripts/07_evaluate.sh` |

</div>

---

## ⚙️ Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate patternbloom
```

Install the bundled vendored verl fork in editable mode:

```bash
pip install -e ./verl_fork
```

Install the project as a package:

```bash
pip install -e .
```

The bundled `verl_fork/` is the upstream verl trainer with three additions
required for PatternBloom: (i) the multi-turn agent rollout in
`verl_fork/verl/workers/rollout/`, (ii) the IDR and PAR reward modules wired
into `verl_fork/verl/utils/reward_score/`, and (iii) the `tool_env`
integration with the retrieval API. The patches are isolated to these files
and do not affect upstream behaviour when neither reward is used.

---

## 📦 Step 1: prepare the training data

PatternBloom trains on a balanced 14K-query subsample drawn from HotpotQA
and 2WikiMultiHopQA. The balanced sampling prevents the larger HotpotQA
corpus from dominating the rollout distribution.

```bash
bash scripts/01_prepare_data.sh
```

This downloads the relevant splits, samples with a fixed seed, and writes
parquet files to `data/processed/`:

```
data/processed/train.parquet            # 14K (7K + 7K), shuffled
data/processed/hotpotqa_test.parquet    # 1K test slice
```

If the source datasets are already cached locally, set `HF_DATASETS_CACHE`
before running the script and it will reuse the cache.

---

## 🕸️ Step 2: build the hypergraph index

The agent retrieves evidence as hyperedges from an offline index built
over the Wikipedia corpus shared between HotpotQA and 2WikiMultiHopQA. Each
hyperedge groups the n-ary facts extracted from one chunk by a small
language-model extractor.

```bash
bash scripts/02_build_hypergraph.sh
```

The script writes the index under `data/hypergraph/` together with a
sentence-encoder embedding index used at retrieval time:

```
data/hypergraph/
├── chunks.jsonl              # source chunks
├── hyperedges.jsonl          # extracted n-ary facts
├── embeddings.npy            # BGE-large embeddings of hyperedges
└── faiss.index               # FAISS flat index
```

The extractor model and the OpenAI-compatible endpoint are configured in
`configs/hypergraph.yaml`. Setting `extractor.endpoint` to a local vLLM
server avoids any external API dependency.

---

## 🌐 Step 3: launch the retrieval API

The trainer accesses the hypergraph through an HTTP API so that retrieval
can be scaled independently of the trainer process.

```bash
bash scripts/03_start_api.sh
```

The API binds to `0.0.0.0:8000` by default. It exposes two endpoints:

* `POST /search` returns top-k hyperedges for a batch of queries.
* `POST /anchor_recall` returns the dual-channel semantic and structural
  retrieval used during Stage II.

The API loads the index from `data/hypergraph/` and serves from a single
process. Override the bind address, port, or device in `configs/api.yaml`.

---

## 🎯 Step 4: Stage I training (Information-Density Reward)

Stage I trains the policy on a graph-construction objective: the reward is
the per-triple-normalized information gain of the constructed evidence
graph under a frozen oracle, sigmoid-wrapped and format-gated. The full
specification appears in the paper.

First, launch the oracle service in a separate process.

```bash
bash scripts/04a_start_oracle.sh
```

Then launch the trainer:

```bash
bash scripts/04_train_stage1.sh
```

Stage I writes checkpoints under `checkpoints/stage1/` every
`checkpoint_interval` steps. The best checkpoint by validation EM is
symlinked to `checkpoints/stage1/best/`.

---

## 🧬 Step 5: distill the Graph Pattern Memory

The Graph Pattern Memory is built once between the two stages from Stage I
trajectories above an IDR threshold. Each entry is a type-abstracted
reasoning skeleton extracted from a trajectory and indexed by its query
signature.

```bash
bash scripts/05_distill_gpm.sh
```

The script reads the best Stage I checkpoint, collects trajectories on a
training subset, abstracts each trajectory's triples by entity type, and
writes the memory to `data/gpm/memory.json`. An example pre-distilled
memory is included at `data/example_gpm.json` for sanity checking.

---

## 🔄 Step 6: Stage II training (Pattern-Augmented Reward)

Stage II warm-starts from the Stage I best policy and refines it under the
PAR reward. The frozen GPM acts as a structural prior; the shaping weight
defaults to `lambda_p=0.2`.

```bash
bash scripts/06_train_stage2.sh
```

Stage II writes checkpoints under `checkpoints/stage2/`. The deployment
artefact is the best Stage II checkpoint combined with the frozen GPM
produced in Step 5.

---

## 📊 Step 7: evaluation

The evaluation script loads a trained checkpoint, replays the multi-turn
rollout against the retrieval API, and reports four metrics on the HotpotQA
test slice.

```bash
bash scripts/07_evaluate.sh
```

The script writes per-question predictions and an aggregate metrics file to
`outputs/eval/`:

```
outputs/eval/predictions.jsonl
outputs/eval/metrics.json
```

The aggregate file reports exact match, token-level F1, contain accuracy,
and the average number of retrieval calls per query.

---

## 📁 Project layout

```
.
├── README.md
├── environment.yml          # conda environment definition (patternbloom)
├── requirements.txt         # pip alternative
├── setup.py                 # package install for the patternbloom/ module
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── stage1_idr.yaml
│   ├── stage2_par.yaml
│   ├── hypergraph.yaml
│   ├── api.yaml
│   └── eval.yaml
│
├── data/
│   ├── README.md            # dataset and index layout
│   └── example_gpm.json     # reference Graph Pattern Memory for inspection
│
├── scripts/                 # numbered entry points (run in order)
│   ├── 01_prepare_data.sh
│   ├── 02_build_hypergraph.sh
│   ├── 03_start_api.sh
│   ├── 04a_start_oracle.sh
│   ├── 04_train_stage1.sh
│   ├── 05_distill_gpm.sh
│   ├── 06_train_stage2.sh
│   └── 07_evaluate.sh
│
├── patternbloom/            # main Python package
│   ├── data/                # dataset preparation, hypergraph construction
│   ├── reward/              # IDR (Stage I), PAR (Stage II), format gate, oracle client
│   ├── gpm/                 # pattern signature, distillation, retrieval, coverage
│   ├── agent/               # multi-turn rollout and prompts
│   ├── api/                 # retrieval API server
│   ├── train/               # Stage I and Stage II trainer entry points
│   ├── eval/                # evaluation entry point
│   └── utils/               # shared text normalization and helpers
│
└── verl_fork/               # vendored verl trainer with PatternBloom integration
```

---

## 🔧 Configuration

Hyperparameters are read from `configs/*.yaml`. The relevant knobs are:

<div align="center">

| Config | Key | Default | Notes |
|:------|:----|:--------|:------|
| `stage1_idr.yaml` | `idr.temperature` | `1.0` | Sigmoid temperature |
| `stage1_idr.yaml` | `oracle.endpoint` | `http://localhost:8100` | Oracle service URL |
| `stage1_idr.yaml` | `train.total_epochs` | `1` | One epoch over the 14K subsample |
| `stage2_par.yaml` | `par.lambda_p` | `0.2` | Pattern-coverage shaping weight |
| `stage2_par.yaml` | `gpm.top_k` | `3` | Patterns prepended to the prompt |
| `api.yaml` | `retriever.top_k` | `5` | Hyperedges per query |
| `eval.yaml` | `eval.num_samples` | `1000` | Test slice size |

</div>

The trainer YAML inherits the upstream verl PPO defaults and overrides only
the fields relevant to PatternBloom. Edit a config file or pass overrides
on the command line via `key=value` Hydra syntax.

---

## 📜 License

This repository is released under the MIT License; see `LICENSE`.

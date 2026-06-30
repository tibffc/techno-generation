# techno-generation

**ACE-Step Pipeline · Project-Side Infrastructure**

A modular pipeline for generating music in a specific genre, built around the ACE-Step backend. The system transforms raw audio into structured training examples for LoRA fine-tuning, then orchestrates reproducible runs for generation and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Preprocessing](#data-preprocessing)
- [LoRA Training Experiments](#lora-training-experiments)
- [Validation & Evaluation](#validation--evaluation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
---

## Overview

This project implements a complete pipeline for music generation in a specific genre (techno). It includes:

- **Data preparation** — raw audio → cleaned chunks → rich metadata & textual captions
- **Training** — LoRA fine-tuning of ACE-Step text-to-music model with multiple rank configurations
- **Generation** — reproducible inference with full logging and artifact tracking
- **Evaluation** — multi‑level validation: diversity, audio quality, prompt adherence, and creativity metrics

The entire workflow is managed through a **Gradio-based control panel**, enabling GUI-driven pipeline execution without manual script invocation.

---

## Pipeline Architecture

The system is split into independent stages with a clear data flow:

```
Raw Audio → Cleaning/Prepare → Metadata/Prompts → Manifest → ACE-Step Backend → Training/Generation → Evaluation
```

Each stage can be **run and improved separately** — the pipeline is not a single monolithic script. Data, metadata, manifests, backend, and evaluation are connected through a shared process.

### Key Components

| Component | Description |
|-----------|-------------|
| **Pipeline Orchestration** | Stages with explicit data and artifact flow |
| **Backend Adapter** | Isolated ACE-Step integration via `acestep_backend.py` |
| **Stage Orchestration** | Launch control, preview commands, statuses, and output modes |
| **Reproducibility** | Each run saves logs, run metadata, and summary |
| **Gradio UI** | Unified interface for configuration, launch, logging, and result viewing |

All runs are tracked in `outputs/runs/<stage>/<run_id>/` with:
- `live.log`
- `run.json`
- `summary.json`
- `summary.md`

---

## Data Preprocessing

### The Problem
- Raw tracks contain vocals and diverse subgenres
- Generic descriptions provide a weak training signal

### The Solution
1. **Filtering** by speechiness
2. **Chunking** audio into segments
3. **Whisper** transcription (if any)
4. **Feature extraction** (Spotify, Librosa)
5. **Tag generation** (rule-based + semantic CLAP tags)
6. **Textual descriptions** generated for each chunk

### Result
**48,000** training examples in ACE-Step format, with multiple LoRA experiments and validation generation sets.

### Metadata Pipeline

```
Audio (Spotify) → Filtering + Chunks → Features + Tags → Text Descriptions → ACE-Step Tensors → LoRA Training
```

### Tag Generation

| Source | Features → Tags |
|--------|----------------|
| **Spotify** | Energy → `energy_tag`, Valence → `mood_tag`, Danceability → `groove_tag`, Instrumentalness → `instrumentalness_tag` |
| **Librosa** | Spectral centroid → `brightness_tag`, Onset density → `rhythm_density_tag`, Onset strength → `punch_tag`, Spectral flatness + ZCR → `texture_tag` |
| **CLAP** | Top‑3 semantic tags (e.g., *acid techno*, *ambient techno*, *dark melodic techno*) |

**Final tags** = rule-based Spotify tags + quantile-based Librosa tags + top‑3 CLAP tags

### LLM Stage: Tags → Caption

```
final tags → qwen3:8b → caption + prompt features → short text condition for ACE-Step
```

**Example input:**
```
techno music · electronic track · high energy · dark mood · steady groove · instrumental track · dense rhythm · noisy texture · balanced drum attack · dark techno
```

**Example caption:**
```
A dark, high-energy techno track with a steady groove, dense rhythm, and noisy texture.
```

---

## LoRA Training Experiments

Four LoRA configurations were tested, all with:
- Batch size: 1
- Gradient accumulation: 8
- Learning rate: 2e-5
- Optimizer: AdamW, fp16

| Variant | Parameters | Effect | Observation |
|---------|------------|--------|-------------|
| `rank4` | r=4, α=16, 1 epoch | Soft shift | Cleaner, but close to base model |
| `rank4_2ep` | r=4, α=16, 2 epochs | Slightly stronger than rank4 | Almost same character |
| `rank8` | r=8, α=32, 1 epoch | Noticeable shift | Best current validation |
| `rank16` | r=16, α=32, 1 epoch | Worse than rank8 | More artifacts and noisy sound |

---

## Validation & Evaluation

Four levels of validation are performed:

### 1. Diversity Evaluation

**Method:** Wav2Vec → VAE (768→32) → KMeans

| Metric |--------|
|--------|--------|
| Silhouette | 0.76 → 0.78 |
| Clusters | 9 → 5 |
| Outliers | 19.6% → 0% | 
| Variance | 0.254 → 0.201 | 
| Perplexity | 6.99 → 1.16 | 

**Entropy on 5 clusters** (max log(5)≈1.609):
- **Base:** 0.779 (spreads generations across extra styles)
- **Rank 8:** 0.688 (better concentrates on the requested style)

*~50,000 examples for VAE and clustering training, 100 generated audios per model, 10 prompts (high-energy tag common).*

### 2. Creativity Metrics

| Metric | Description | Conclusion |
|--------|-------------|------------|
| **S-score** | Mean distance from track to cluster center, normalized by intra-cluster spread | Rank 8 experiments more within a single style |
| **NCI** | Fraction of embedding features rare in reference (<5th or >95th percentile) | Rank 8 gives fewer clusters but more variability inside |

### 3. Audio Quality (Librosa)

| Metric | Observation |
|--------|-------------|
| **Silence** | Rank 8 has less silence |
| **DC offset** | Rank 8 is 3× cleaner (mean sample value) |
| **Clipping** | Absent in all models |

### 4. Prompt Adherence (CLAP Cosine Similarity)

**Finding:** LoRA (especially Rank 8) improves correspondence between generated audio and text prompt.
- Median and mean increased
- Lower 25% of tracks pulled up
- Rank 8 is more symmetric with fewer outliers

---

## Getting Started

### Prerequisites

- Python 3.8+
- [ACE-Step](https://github.com/your-ace-step-repo) backend

### Installation

```bash
git clone https://github.com/tibffc/techno-generation.git
cd techno-generation
pip install -r requirements.txt
```

### Configuration

Edit `configs/` files to set:
- Paths to raw audio data
- ACE-Step backend connection parameters
- Run parameters (batch size, epochs, LoRA rank, etc.)

### Running the Pipeline

1. Launch the Gradio UI:
   ```bash
   python app/app.py
   ```
2. Use the interface to:
   - Configure run parameters
   - Execute pipeline stages (preprocessing, training, generation, evaluation)
   - Monitor logs and view results

Or run stages individually via scripts in `scripts/`.

---

## Project Structure

```
techno-generation/
├── app/
│   └── app.py              # Gradio control panel
├── configs/                # Configuration files
├── scripts/                # Standalone stage scripts
│   ├── preprocess.py
│   ├── train_lora.py
│   ├── generate.py
│   └── evaluate.py
├── outputs/
│   └── runs/               # Logs and artifacts per run
│       ├── preprocess/
│       ├── train/
│       ├── generate/
│       └── evaluate/
├── requirements.txt
└── README.md
```

---

## Results Summary

| Aspect | Best Model | Key Improvement |
|--------|------------|-----------------|
| **Prompt adherence** | Rank 8 | Higher CLAP similarity, fewer outliers |
| **Style focus** | Rank 8 | Lower entropy, better concentration on requested style |
| **Audio quality** | Rank 8 | Less silence, 3× lower DC offset, no clipping |
| **Creativity** | Rank 8 | More experimentation within style, higher internal variance |

The **Rank 8** configuration delivers the best balance of prompt adherence, audio quality, and creative variety, making it the recommended choice for generation in the target genre.

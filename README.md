# fMRI Semantic Structure Analysis through Neural Nets

Project repo for the **Neural Network 4107 Final Project** (Group 44).

This project explores whether semantic structure in natural images is reflected in fMRI response structure, and how well simple neural network representations of brain activity align with CLIP image embeddings.

## Folder guide

> Folder names may vary depending on the current state of the repo; the intent below matches how the work is organized.

- `processed_data/`
  - Cached, preprocessed subject data (`CSI1`–`CSI4`), including Schaefer ROI matrices and visual-ROI subsets.

- `AutoencoderModels/`
  - Saved autoencoder checkpoints and run logs.

- `DataProcessing/` *(if present)*
  - Utilities and scripts for preprocessing, label mapping, and semantic category grouping.

- `DataTesting/` *(if present)*
  - Small notebooks/scripts for sanity checks, feasibility tests, and inspecting data quality.

- `AEDataAnalysis/` *(if present)*
  - Notebooks/scripts comparing autoencoder embeddings vs CLIP (RSA, clustering, retrieval).

## Repository layout (important files)

### Data preprocessing

- `preprocess_schaefer.py`
  - Compresses raw BOLD5000 GLM beta volumes into **Schaefer-parcellated ROI features** (e.g. `schaefer1014`).
  - Produces `.npz` files under `processed_data/CSI*/` used by the downstream notebooks.

### Autoencoder training + analysis

- `AutoencoderTraining.ipynb`
  - Defines and trains the autoencoder model(s) on fMRI ROI data.
  - Used to test compression into a 768-d latent space, roughly matching CLIP’s embedding dimensionality.

- `AutoencoderAnalysis_clean.ipynb`
  - Main analysis notebook for exporting autoencoder latents and running repeated analyses.

- `Subject1EncodingAnalysis.ipynb`
  - Single-subject comparison notebook (RSA, clustering, nearest-neighbor inspection) between:
    - CLIP embeddings (768-d)
    - brain-derived embeddings (autoencoder latents)
    - (optionally) raw ROI responses

- `ALLSubjectsAnalysis.ipynb` *(if present)*
  - Multi-subject version of the analysis that stacks CSI1–CSI4.

### Visual ROI (voxelwise) pipeline

- `Autoencoder_RSA_sub1_ROI.ipynb`
  - Pipeline for working with **visual ROI voxel betas** (from `*_visual_rois.npz`), training an autoencoder, and comparing to CLIP via RSA.

### Other experiments

- `GraphAttentionNetwork.ipynb`
  - Attempted Graph Attention Network (GAT) approach for learning structure from ROI graphs.

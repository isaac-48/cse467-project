# CTGAN Synthetic Student Data Pipeline

## Overview

This project uses **CTGAN (Conditional Tabular GAN)** to generate synthetic student performance data while evaluating both **data utility** and **privacy**. It is designed as a practical academic implementation for synthetic educational dataset generation and analysis.

---

## Key Features

* Trains CTGAN on real student data
* Generates synthetic datasets of equal size
* Evaluates utility using:

  * KL Divergence
  * Jensen-Shannon Divergence
  * Wasserstein Distance
  * Chi-Square Test
  * Logistic Regression AUC
* Evaluates privacy using:

  * Delta Presence
  * K-Anonymity
  * Identifiability Score
* Produces:

  * `synthetic_students.csv`
  * `privacy_utility.json`
  * `comparison.png`
  * `evaluation_plots.png`

---

## Installation

```bash
pip install ctgan pandas numpy scipy scikit-learn matplotlib
```

---

## Usage

```bash
python3 ctgan_full_pipeline.py
```

Input dataset:

```txt
Student_data.csv
```

---

## Method

1. Load and preprocess educational data
2. Train CTGAN model
3. Generate synthetic student records
4. Evaluate statistical similarity and privacy risks
5. Export reports and visualizations

---

## Limitations

* Standard CTGAN only (non-differentially private)
* No DPGAN/PATEGAN implementation
* Privacy metrics are approximations, not formal DP guarantees

---

## Future Work

* Add differential privacy (ε = 0.1, 1, 10)
* Expand downstream ML evaluations
* Implement advanced privacy-preserving generators

---

## Purpose

This pipeline provides a balanced framework for:

* Synthetic educational data generation
* Privacy analysis
* Utility benchmarking
* Academic research and coursework

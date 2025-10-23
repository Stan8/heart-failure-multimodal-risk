# Is Clinical Text Enough? A Multimodal Study on Mortality in Heart Failure

This repository contains the official implementation of the paper **"Is Clinical Text Enough? A Multimodal Study on Mortality in Heart Failure"**. The project investigates whether clinical text alone is sufficient for outcome prediction, and evaluates how much performance improves when combining **structured variables** and **clinical text** through multimodal learning.

We compare three families of models:

- **Multimodal models** (structured + text, with learnable fusion)
- **Structured-only baselines**
- **Text-only baselines**

Entity-level embeddings from French clinical notes are included, and we evaluate multiple fusion strategies such as **late fusion, attention-based fusion, and gated fusion**.

---

## 📂 Repository Structure

.
├── multimodal/ # Multimodal models and fusion architectures
├── structured_only/ # Baselines using only tabular / structured features
├── text_only/ # Baselines using only clinical text embeddings
└── bert-entities-encode-v1.py # Script to encode raw clinical notes and entities into embeddings

---

##  Task Description

- **Objective**: Predict short-term mortality in heart failure patients (binary classification)
- **Modalities**:
  - **Structured data**: clinical variables (labs, demographics, vitals, etc.)
  - **Text**: French clinical notes with entity-level embeddings
- **Goal**: Assess the added value of multimodal fusion versus text-only models

---



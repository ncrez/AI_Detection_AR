#  Arabic AI-Generated Text Detection 

##  Project Overview

This project focuses on detecting AI-generated Arabic text and distinguishing it from human-written content using a combination of **linguistic feature engineering** and **classical machine learning models**.

With the rapid adoption of large language models, identifying synthetic Arabic text has become critical for academic integrity, content moderation, and AI safety.

The project demonstrates that classical machine learning models, when paired with carefully engineered Arabic-specific stylometric features, can outperform neural models while remaining **interpretable** and **computationally efficient**.

##  Objectives

* Build a reliable classifier to distinguish human-written vs AI-generated Arabic text.
* Explore linguistic, lexical, and frequency-based features tailored to the Arabic language.
* Compare classical ML models against a neural baseline (FFNN).
* Analyze model performance and common error patterns.

##  Dataset

| Attribute | Details |
| :--- | :--- |
| **Source** | Hugging Face (`KFUPM-JRCAI/arabic-generated-abstracts`) |
| **Language** | Arabic |
| **Classes** | `0` → Human-written text; `1` → AI-generated text |
| **Text Type** | Academic abstracts |
| **Size** | Thousands of labeled samples (balanced classes) |

> Each sample contains raw text along with preprocessing-ready fields for tokenization and feature extraction.

##  Preprocessing

The Arabic text data undergoes rigorous cleaning and preparation:

* Arabic-aware sentence and word tokenization.
* Text normalization and cleaning.
* Removal of noisy symbols and redundant whitespace.
* Extraction of raw and cleaned versions for comparison.
 Features targeting specific characteristics of Arabic morphology and orthography (e.g., Tanween frequency).

##  Models Used

The following models were trained and evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* Feedforward Neural Network (FFNN)

> Classical ensemble models (Random Forest, XGBoost) proved particularly effective due to their ability to model non-linear feature interactions within the feature space.

## 📊 Results Summary

| Model | Accuracy |
| :--- | :--- |
| Logistic Regression | 97.17% |
| SVM | 97.97% |
| **Random Forest** | **98.65%** |
| XGBoost | 98.06% |
| FFNN | 86.58% |

### Key Findings:

* **Random Forest** achieved the best overall performance.
* **Ensemble models** consistently outperformed the neural baseline.
* The **feature-driven approach** is highly effective for Arabic AI-text detection.


## 🚀 How to Run

To set up the project and reproduce the results:


# Install dependencies
pip install -r requirements.txt

# Run the notebook to execute preprocessing, feature extraction, and model training/evaluation
jupyter notebook Project_Finalupdate02.ipynb

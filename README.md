# 📊 Prescriptive Credit Underwriting via Causal Machine Learning

> **Official implementation for the paper:** *"Prescriptive Credit Underwriting via Causal Machine Learning: Disentangling Structural Drivers and Enabling Counterfactual Recourse in Peer-to-Peer Debt Consolidation"* (Submitted to IMCOM 2027).

## 📑 Overview
Standard credit scoring models heavily rely on gradient-boosted trees and post-hoc Explainable AI (e.g., SHAP). However, they frequently fall into the **"Proxy Trap"**, mistaking administrative sorting labels for root vulnerabilities and offering zero actionable recourse to rejected applicants. 

This repository provides an end-to-end **Graph-Based Causal AI framework** that:
1. **Curates a strict pre-origination ($T_0$) cohort** from LendingClub to eliminate temporal data leakage.
2. **Discovers the causal structure** using continuous non-linear `NOTEARS-MLP` with Local Markov Condition falsification.
3. **Unmasks the Proxy Trap** by contrasting Correlational XAI (CatBoost + SHAP) with Structural Causal Attribution (`DoWhy-GCM` Intrinsic Causal Contribution).
4. **Enables Actionable Recourse** via Pearl’s *do-calculus*, computationally simulating the conversion of borderline rejected applicants into conditional approvals through proactive contract restructuring.

## 📂 Repository Structure
Since the experimental pipeline is highly streamlined, all processes—from data preprocessing to causal discovery and counterfactual simulation—are consolidated into a single, easy-to-follow Jupyter Notebook.

* `causal_credit_underwriting.ipynb`: The main notebook containing the entire experiment pipeline.
* `requirements.txt`: List of required Python packages.

## 💾 Dataset
The experiments utilize the **LendingClub 2007-2020Q3** dataset.
* **Source:** [Kaggle - Lending Club Dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
* **Preparation:** Download the `Loan_status_2007-2020Q3.gzip` file and place it in a folder named `data/` in the root directory. The notebook automatically filters the `debt_consolidation` cohort and handles preprocessing.

## ⚙️ Installation & Requirements
To reproduce the experiments, please install the necessary dependencies. We recommend using a virtual environment (e.g., Conda).

```bash
# Clone the repository
git clone git@github.com:vanducngo/Causal-Credit-Underwriting.git
cd Causal-Credit-Underwriting

# Install dependencies
pip install -r requirements.txt

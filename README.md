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

## 📂 Repository Structure & A Note on Reproducibility

To ensure maximum transparency and reproducibility, we provide two separate Jupyter Notebooks in this repository:

1. 📄 **`01_Paper_Results_Snapshot_VN.ipynb` (Original Log Snapshot)**
   * **Purpose:** This notebook contains the exact execution logs, outputs, and figures (with original Vietnamese print statements) that match the numbers presented in the submitted manuscript (Table I, II, III and Fig 1, 2) with 100% fidelity.
   * **Note:** Because the continuous non-linear DAG discovery (`NOTEARS-MLP`) and CatBoost baseline entail stochastic weight initializations, and a global random seed was strictly not enforced during the initial exploratory run, rerunning this exact notebook will yield slightly different numerical values.

2. 🚀 **`02_Reproducible_Pipeline_EN.ipynb` (Seeded, English Version)**
   * **Purpose:** This is the clean, English-translated pipeline designed for reviewers and researchers to run. 
   * **Reproducibility:** We have added explicit global random seeds (e.g., `np.random.seed(42)`, `torch.manual_seed(42)`) at the beginning of this notebook. 
   * **Disclaimer:** While running this notebook will produce slightly different numerical outputs compared to the manuscript due to the newly enforced seed, **the core structural findings remain identical**. Specifically, CatBoost-SHAP will still misattribute importance to `sub_grade`, while Causal ICC will robustly unmask `int_rate` as the true root cause, proving the theoretical claims of the paper.
   
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

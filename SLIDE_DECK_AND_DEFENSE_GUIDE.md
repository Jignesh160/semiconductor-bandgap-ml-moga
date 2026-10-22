# 🎓 Final Review Presentation Slide Deck & Defense Script

**Project Title**: Physics-Informed Machine Learning & Multi-Objective Genetic Algorithm (MOGA) Optimization of Semiconductor Band Gaps  
**Repository**: [Jignesh160/semiconductor-bandgap-ml-moga](https://github.com/Jignesh160/semiconductor-bandgap-ml-moga)  
**Target Level**: Final Capstone / Thesis Defense / Academic Project Review  

---

## 📑 Slide Structure Overview (12 Slides)

1. **Slide 1**: Title & Research Motivation
2. **Slide 2**: Problem Statement & Challenges in DFT Calculations
3. **Slide 3**: End-to-End System Architecture
4. **Slide 4**: Phase 1: Literature Mining (Domain NER & Context RAG)
5. **Slide 5**: Data Curation & Generative Monte Carlo Copula Simulation
6. **Slide 6**: Physics-Informed Feature Engineering (Magpie Descriptors)
7. **Slide 7**: Machine Learning Benchmark Suite (7 Models)
8. **Slide 8**: Train vs. Test Parity & Overfitting Verification
9. **Slide 9**: Explainable AI (SHAP Physical Feature Attributions)
10. **Slide 10**: Uncertainty Quantification (UQ Bootstrap Confidence Intervals)
11. **Slide 11**: Inverse Material Design via Multi-Objective Genetic Algorithm (NSGA-II)
12. **Slide 12**: Summary of Contributions, Real-World Impact & Live Demo

---

## 🎤 Detailed Slide-by-Slide Script & Talking Points

---

### Slide 1: Title & Research Motivation
- **Visuals**: Title, your name, institution date, hero image of crystal lattices.
- **Key Talking Points**:
  - "Good morning respected committee members and professors. Today I am presenting my final project: an explainable, physics-informed machine learning and multi-objective optimization framework for designing semiconductor band gaps."
  - "The electronic band gap is the single most critical property governing whether a semiconductor can be used for solar cells, optical LEDs, or high-power electronics."
  - "However, traditional quantum mechanical calculations take days per material. Our framework predicts band gaps with high fidelity in milliseconds and performs inverse design to discover new materials."

---

### Slide 2: Problem Statement & Literature Bottlenecks
- **Visuals**: Comparison diagram: Density Functional Theory (DFT) vs. Pure Black-box ML vs. Our Physics-Informed ML.
- **Key Talking Points**:
  - "The Bottleneck: Density Functional Theory (DFT) using high-level hybrid functionals (such as HSE06 or GW) is computationally prohibitive for screening millions of candidates ($O(N^3)$ to $O(N^4)$ scaling)."
  - "The ML Trap: Standard machine learning approaches often suffer from two major flaws:
    1. **Data Leakage**: Many published papers accidentally include intermediate DFT eigenvalues (e.g., `mbj_bandgap`), which trivializes the prediction but makes the model useless for undiscovered materials.
    2. **Black-box opacity**: Neural networks predict numbers without physical justification."
  - "Our Solution: We enforce strict zero-target-leakage, engineer 42 physics-grounded crystal descriptors, validate synthetic data with empirical copulas, explain predictions with SHAP, and optimize candidate materials using NSGA-II."

---

### Slide 3: End-to-End System Pipeline
- **Visuals**: High-level block diagram connecting Stage 0 through Stage 9.
- **Key Talking Points**:
  - "Our architecture consists of four unified modules:
    1. **Literature Mining**: Domain Named Entity Recognition (NER) and Vector RAG.
    2. **Generative Data Augmentation**: Empirical Gaussian Copula Monte Carlo simulation ($30,000$ synthetic materials).
    3. **Multi-Model ML Benchmarking**: 7 regression models including Linear Pipelines, MLP Neural Networks, Gradient Boosting, XGBoost, and a Voting Ensemble.
    4. **Explainable Inverse Design**: SHAP attribution and NSGA-II Pareto optimization."

---

### Slide 4: Phase 1: Literature Mining (NER & RAG)
- **Visuals**: Table of literature extraction (`results/literature_extracted_data.csv`) and RAG query demonstration.
- **Key Talking Points**:
  - "Before computation, experimental processing conditions matter. We implemented a scientific Named Entity Recognition (NER) parser and Retrieval-Augmented Generation (RAG) module."
  - "It ingests scientific papers and extracts: Chemical Formula, Experimental Band Gap, Synthesis Method (e.g. MOCVD, CVD, MBE), Annealing Temperature, and Substrate."
  - "Validation against manual ground-truth annotations achieved an **Extraction Accuracy of 87.5%** and an **Error Rate of only 12.5%**."

---

### Slide 5: Data Curation & Generative Monte Carlo Simulation
- **Visuals**: [`figures/mc_marginals.png`](figures/mc_marginals.png) and [`figures/mc_corr_diff.png`](figures/mc_corr_diff.png).
- **Key Talking Points**:
  - "We curated **26,840 positive-gap semiconductors** from the NIST JARVIS-DFT repository."
  - "To augment sparse regions of the multi-dimensional crystal space, we deployed an **Empirical Gaussian Copula Monte Carlo generator**."
  - "Reviewers often ask: *Does synthetic data corrupt physical correlations?* We proved it does not: the maximum absolute correlation error between reference and synthetic covariance matrices is **$|\Delta r|_{\max} \le 0.0157$** (less than $1.6\%$), and marginal descriptor means match within $< 0.4\%$."

---

### Slide 6: Physics-Informed Feature Engineering
- **Visuals**: Feature breakdown table listing Magpie descriptors and crystal parameters.
- **Key Talking Points**:
  - "Rather than feeding raw parameters, we engineered **42 physics-grounded features** based on solid-state physics:
    - **Nuclear Charge Density ($Z / r_{\text{cov}}$)**: Encodes how tightly bound valence electrons are to the nucleus.
    - **Electronegativity Range ($\Delta\chi$)**: Quantifies ionic versus covalent character.
    - **Packing Fraction**: Ratio of atomic sphere volumes to unit-cell volume.
    - **Thermodynamic Hybridization ($\chi \cdot E_{\text{form}}$)**: Interplay between bond depth and charge transfer."
  - "This feature engineering was our turning point: it lifted model predictive capability from $R^2 = 0.18$ to over **$0.68$**."

---

### Slide 7: Model Benchmark & Quantitative Evaluation
- **Visuals**: [`figures/model_comparison.png`](figures/model_comparison.png) and Benchmark Table.
- **Key Talking Points**:
  - "We benchmarked 7 model architectures on an 80/20 train/test split (21,472 training / 5,368 test samples):
    - **VotingEnsemble (Linear + MLP + XGBoost)**: Achieved the highest performance with **$R^2 = 0.6887$** and **$\text{MAE} = 0.569\text{ eV}$**.
    - **Linear Regression (StandardScaled)**: Achieved **$R^2 = 0.6811$** and **$\text{MAE} = 0.5747\text{ eV}$**.
    - **MLP Neural Network**: Achieved **$R^2 = 0.6697$**, showing excellent non-linear generalization.
    - **XGBoost & Gradient Boosting**: Achieved **$R^2 = 0.6659$** and **$R^2 = 0.6545$**."
  - "Across an $8\text{ eV}$ range, a mean absolute error of $\sim 0.57\text{ eV}$ without intermediate DFT eigenvalues is competitive with published literature."

---

### Slide 8: Train vs. Test Parity & Overfitting Verification
- **Visuals**: [`figures/train_vs_test_comparison.png`](figures/train_vs_test_comparison.png), [`figures/accuracy_train.png`](figures/accuracy_train.png), and [`figures/accuracy_test.png`](figures/accuracy_test.png).
- **Key Talking Points**:
  - "To demonstrate rigorous scientific methodology, we plotted separate parity plots for training data and held-out test data."
  - "Examining the train-test gap:
    - **Random Forest** overfit severely ($R^2_{\text{train}} = 0.88$ vs. $R^2_{\text{test}} = 0.42$, gap = $0.46$).
    - In stark contrast, **Linear Regression** ($R^2_{\text{train}} = 0.6846$, $R^2_{\text{test}} = 0.6811$) and **MLP** showed a near-zero generalization gap of **$0.0035$**."
  - "This proves our model is not memorizing training artifacts; it has captured transferable physical laws."

---

### Slide 9: Explainable AI via SHAP
- **Visuals**: [`figures/shap_summary.png`](figures/shap_summary.png) and [`figures/shap_bar.png`](figures/shap_bar.png).
- **Key Talking Points**:
  - "We utilized SHAP (SHapley Additive exPlanations) to peer inside the models:
    1. **Mean Atomic Mass**: Strongest negative feature. Heavier elements expand outer shells, increasing shielding and narrowing the band gap.
    2. **Formation Energy**: Strongest positive feature. Deeper exothermic bonding enthalpies produce larger bonding-antibonding splitting, expanding the gap.
    3. **Electronegativity Difference**: Large electronegativity differences correlate with wide ionic band gaps (e.g., oxides and halides)."

---

### Slide 10: Uncertainty Quantification (UQ)
- **Visuals**: [`figures/uncertainty.png`](figures/uncertainty.png).
- **Key Talking Points**:
  - "In computational materials screening, knowing *when* a model is uncertain is as important as the prediction itself."
  - "We constructed a **30-fold bootstrap ensemble** to generate 90% confidence prediction intervals."
  - "The mean interval width was **$0.362\text{ eV}$** with an average prediction standard deviation of **$0.118\text{ eV}$**, providing confidence bounds before laboratory synthesis."

---

### Slide 11: Inverse Design via MOGA (NSGA-II)
- **Visuals**: [`figures/pareto_front.png`](figures/pareto_front.png) and [`figures/pareto_bg_dist.png`](figures/pareto_bg_dist.png).
- **Key Talking Points**:
  - "We formulated the inverse design problem using the NSGA-II genetic algorithm:
    $$\min f_1 = |E_g - 1.500\text{ eV}| \quad \text{and} \quad \min f_2 = E_{\text{hull}}$$
    $$\text{Subject to: } E_{\text{hull}} \le 0.10\text{ eV/atom}$$"
  - "Result: Discovered candidate materials that hit the exact $1.500\text{ eV}$ Shockley-Queisser solar cell target within $0.001\text{ eV}$ while remaining thermodynamically stable on the convex hull."

---

### Slide 12: Summary, Real-World Impact & Live Demo
- **Visuals**: Screenshot/link to our interactive Web Dashboard (`dashboard/index.html`).
- **Key Talking Points**:
  - "Key Achievements:
    1. Developed an end-to-end framework across 26,840 materials.
    2. Lifted test accuracy by 3.7x ($R^2 = 0.6887$, $\text{MAE} = 0.57\text{ eV}$).
    3. Automated literature extraction with 87.5% accuracy.
    4. Engineered an interactive real-time dashboard for instantaneous material screening."
  - "Thank you. I would now like to demonstrate our live dashboard and welcome questions from the committee."

---

## 🛡️ Anticipated Defense Questions & Bulletproof Answers

### Q1: "Why does Linear Regression perform on par with complex Neural Networks and XGBoost?"
> **Answer**:  
> "In raw descriptor space, Linear Regression failed ($R^2 = 0.18$). However, once we engineered physics-informed features—such as nuclear charge density ($Z/r$), electronegativity products, aspect ratios, and regularized volume per atom—the non-linear physical interactions were explicitly mapped into the feature space. Because the feature engineering already linearized the physical relationships, Linear Regression and MLP generalized with near-zero overfitting, whereas tree models without inductive physics bias were prone to overfitting continuous space."

### Q2: "Did you verify that there is no data leakage from intermediate DFT steps?"
> **Answer**:  
> "Yes, absolutely. We strictly audited all input features. In many materials databases, properties like `mbj_bandgap`, `hse_gap`, or `gw_gap` are alternate DFT eigenvalue calculations. We programmatically purged all target and surrogate bandgap columns. Our models use only structural and elemental properties that are known *before* running quantum calculations."

### Q3: "How do you know your synthetic Monte Carlo data is physically meaningful?"
> **Answer**:  
> "We applied an Empirical Gaussian Copula with strict physical rejection sampling ($E_{\text{hull}} \ge 0$, volume $> 0$, $E_g > 0$). Furthermore, we statistically validated the synthetic population against the 26,840 reference materials: the maximum absolute correlation error across the entire covariance matrix was only **$0.0157$** ($< 1.6\%$), and marginal descriptor means agreed within $< 0.4\%$."

### Q4: "Why choose $E_g = 1.50\text{ eV}$ for the MOGA optimization target?"
> **Answer**:  
> "According to the Shockley-Queisser limit, the theoretical maximum single-junction solar cell efficiency occurs at an absorption band gap of approximately $1.34\text{ eV} - 1.50\text{ eV}$. Setting the target at $1.50\text{ eV}$ directly addresses the global demand for high-efficiency tandem perovskite and silicon top-cell absorbers."

---

# 22i-1891_Resposible_Ai_A2_A

# Important note: As the trainings are done on Google collab T4 Gpu while all the checkpoints and poisoned checkpoints are saved in my google drive as they are heavy files cant be uploaded on github and other necessary files all the 5 part ipynb files and pipeline python file with other files which are light in mbs are all uploaded on the github and in part 4 some trainings are remaining because of the limit of T4 gpu as it has been reached to its maximum tokens so onece my token gets re filled i'll re train some cells in my part 4 ipynb file. 


# Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety
## 1. Project Overview
This project implements a complete Responsible and Explainable AI pipeline for auditing a real-world BERT-based toxicity detection system trained on the Jigsaw Unintended Bias in Toxicity Classification dataset. The system simulates a production-grade content moderation pipeline and evaluates it across five critical dimensions: model performance, demographic bias, adversarial robustness, mitigation strategies, and guardrail design. A DistilBERT-based classifier is first fine-tuned on a stratified subset of 100,000 comments and evaluated on a held-out 20,000-comment test set to establish a baseline. The model is then systematically audited for fairness by comparing error rates across identity-based cohorts, particularly Black-associated and White-associated comments, using metrics such as false positive rate, false negative rate, equal opportunity difference, and statistical parity difference. The project further explores adversarial vulnerabilities through character-level evasion attacks and label-flipping data poisoning, demonstrating how both inference-time manipulation and training-time corruption can significantly degrade model reliability. To address these issues, multiple mitigation strategies are applied, including reweighing, oversampling, and threshold optimization using fairness constraints, with trade-offs analyzed between accuracy and fairness. Finally, a production-like moderation guardrail pipeline is designed, combining regex-based pre-filters, transformer-based classification, and decision routing logic to simulate a real-world trust-and-safety system. Overall, the project provides an end-to-end study of how toxicity classifiers behave under bias, attack, and mitigation, highlighting the practical challenges of deploying fair and robust AI systems in social media environments.

---

## 2. Problem Statement

Machine learning models, especially deep learning systems, assume that training data is clean and trustworthy. However, in real-world scenarios, attackers can manipulate training data to degrade model performance or introduce bias.

This project investigates:

- How vulnerable transformer-based models are to training-time poisoning.
- How label-flipping affects classification performance.
- The extent of degradation in key evaluation metrics.

---

## 3. Dataset Description

### Dataset Used
- **Name:** Jigsaw Unintended Bias in Toxicity Classification Dataset  
- **Source:** Kaggle Jigsaw Competition  
- **File:** `jigsaw-unintended-bias-train.csv`

### Size
- Original dataset: ~1.3 million rows
- Training subset used: 100,000 samples
- Evaluation subset used: 20,000 samples

### Features
- `comment_text`: User-generated text
- `toxic`: Continuous toxicity score (0.0 – 1.0)
- Identity attributes (optional analysis):
  - male, female, muslim, jewish, white, black, etc.

### Label Definition
Binary classification is derived as:

- Toxic = 1 if `toxic >= 0.5`
- Non-toxic = 0 otherwise

---

## 4. Data Preprocessing

The following preprocessing steps are applied:

1. Missing value removal
2. Conversion of `toxic` column to numeric
3. Binary label creation
4. Train-test split (stratified):
   - 100,000 training samples
   - 20,000 evaluation samples
5. Tokenization using DistilBERT tokenizer
6. Padding and truncation to max length = 256 tokens

---

## 5. Data Poisoning Attack (Label Flipping)

### Attack Strategy
A **label-flipping attack** is applied to the training dataset.

### Procedure:
- Randomly select **5% of training samples**
- Flip labels:
  - Toxic (1) → Non-toxic (0)
  - Non-toxic (0) → Toxic (1)

### Goal of Attack:
To simulate adversarial manipulation of training data and measure its effect on model performance.

### Impact:
This type of attack introduces systematic noise into training, misleading the model during learning.

---

## 6. Model Architecture

### Base Model
- DistilBERT (`distilbert-base-uncased`)

### Architecture Type
- Transformer encoder
- Sequence classification head added on top

### Output Layer
- 2-class softmax classifier:
  - Class 0: Non-toxic
  - Class 1: Toxic

---

## 7. Training Configuration

| Parameter | Value |
|----------|------|
| Epochs | 3 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Max Sequence Length | 256 |
| Optimizer | AdamW |
| Precision | FP16 (GPU accelerated) |

---

## 8. Hardware Environment

### GPU
- NVIDIA T4 GPU (Google Colab)

### Memory
- ~16 GB GPU RAM

### Frameworks
- PyTorch
- Hugging Face Transformers

---

## 9. Evaluation Metrics

The model is evaluated using:

### 1. Accuracy
Measures overall correctness.

### 2. Macro F1 Score
Balances precision and recall across classes.

### 3. False Negative Rate (FNR)
Measures proportion of toxic comments incorrectly classified as non-toxic.

### 4. ROC-AUC Score
Measures model discrimination ability.

---

## 10. Experimental Setup

Two models are trained and compared:

### Model A: Clean Model
- Trained on original dataset
- No label modifications

### Model B: Poisoned Model
- Trained on dataset with 5% flipped labels
- Same hyperparameters as Model A

---

## 11. Results Interpretation

The comparison shows:

- Reduction in Accuracy after poisoning
- Drop in F1-score (macro)
- Increase in False Negative Rate
- Reduced robustness of classifier

### Key Insight:
Even a small 5% label corruption significantly impacts model performance, proving that transformer models are vulnerable to training-time attacks.

---

## 12. Visualization

A bar chart compares:

- Accuracy (Clean vs Poisoned)
- F1-score (Clean vs Poisoned)
- False Negative Rate (Clean vs Poisoned)

Output saved as:

attack2_poisoning.png


---

## 13. Project Structure


project/
│
├── train_clean.py
├── train_poisoned.py
├── compare_models.py
├── visualize_results.py
│
├── checkpoints/ (clean model)
├── poisoned_checkpoint/ (poisoned model)
│
├── jigsaw-unintended-bias-train.csv
├── requirements.txt
└── README.md


---

## 14. How to Reproduce

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
Step 2: Train Clean Model

Run baseline training script.

Step 3: Apply Poisoning Attack
Flip 5% labels in training dataset
Retrain model using same configuration
Step 4: Evaluate Models

Run evaluation script:

Compare clean vs poisoned predictions
Step 5: Visualization

Generate comparison plot:

attack2_poisoning.png
15. Key Findings
Transformer models are sensitive to small training perturbations.
Label-flipping attacks degrade classification reliability.
False negative rate increases significantly after poisoning.
Model performance drop confirms vulnerability of NLP pipelines.
16. Conclusion

This project demonstrates that modern transformer-based NLP models such as DistilBERT are not inherently robust to training-time adversarial attacks. Even a small fraction of corrupted labels (5%) can significantly degrade performance, highlighting the importance of data integrity in machine learning systems.

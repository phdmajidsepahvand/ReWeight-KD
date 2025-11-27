# ReWeight-KD
### Reweighted Ensemble Distillation for Domain-Generalized Medical Diagnosis

---

## Overview

**ReWeight-KD** introduces a lightweight and effective framework for **reweighted ensemble knowledge distillation**, designed to enhance diagnostic robustness and domain generalization in medical AI systems.

Unlike conventional single-teacher distillation, ReWeight-KD leverages a **multi-teacher ensemble**, aggregates their predictive distributions, and models them using a **Dirichlet-guided reweighting mechanism**. This allows the student model to focus more on reliable, high-confidence samples while down-weighting uncertain or ambiguous ones.

The result is a stable, uncertainty-aware, and generalizable student model suitable for deployment in real-world medical environments.

---

## Features

###  Multi-Teacher Supervision  
Combines multiple teacher networks, allowing the student to learn complementary and diverse decision boundaries.

###  Dirichlet-Based Sample Reweighting  
Interprets ensemble predictions as Dirichlet parameters to compute confidence-adaptive weights for each training example.

###  Uncertainty-Aware Distillation  
Samples with higher ensemble agreement contribute more strongly to the optimization process.

###  Hybrid Distillation Objective  
Trains the student using a weighted KL divergence loss combined with standard cross-entropy for ground-truth supervision.

###  Robust to Domain Shift  
Designed for medical datasets where variability across devices, patients, and acquisition protocols frequently leads to distribution shifts.

###  Lightweight Deployment  
Produces compact diagnostic models suitable for edge devices, mobile health systems, and real-time clinical inference.

---

## Project Structure

```
ReWeight-KD/
├── distill/
│   ├── data.py
│   ├── models.py
│   ├── distillation.py
│   ├── utils.py
│   └── __init__.py
├── train_teachers.py
├── train_student.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Training Workflow

###  Train the Teacher Ensemble

```bash
python train_teachers.py     --train_path <TRAIN_FILE_PATH>     --test_path  <TEST_FILE_PATH>     --epochs 5     --out_dir checkpoints
```

###  Distill Knowledge into the Student Model

```bash
python train_student.py     --train_path <TRAIN_FILE_PATH>     --test_path  <TEST_FILE_PATH>     --teachers_dir checkpoints     --epochs 10
```

Replace `<TRAIN_FILE_PATH>` and `<TEST_FILE_PATH>` with the locations of your dataset files.

---

##  Requirements

Install required packages using the command:

```bash
pip install -r requirements.txt
```

### Main Dependencies
- Python ≥ 3.7  
- torch  
- numpy  
- pandas  
- scikit-learn  
- tqdm  


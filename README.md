# \# Reweighting Ensemble Distillation for Domain-Generalized Medical Diagnosis 

# 

# This repository contains a simple PyTorch implementation of

# \*\*reweighted ensemble distillation\*\* on the \*\*ECG5000\*\* dataset

# (UCR Time Series).

# 

# The reweighting is based on a \*\*Dirichlet interpretation\*\* of

# the ensemble output:

# 

# \- Multiple teacher networks are trained on ECG5000.

# \- Their softmax outputs are averaged to form an ensemble distribution.

# \- For each sample, we construct Dirichlet parameters:

# &nbsp; \\alpha\_k = p\_k \* s

# &nbsp; and use \\alpha\_0 to compute a sample-wise weight.

# 

# \## Project structure

# 

# \- `ecg\_distill/data.py`: ECG5000 dataset loader

# \- `ecg\_distill/models.py`: ECGNet model and TeacherEnsemble

# \- `ecg\_distill/distillation.py`: Dirichlet-based reweighting and distillation loss

# \- `ecg\_distill/utils.py`: utilities (seed, evaluation)

# \- `train\_teachers.py`: train and save teacher models

# \- `train\_student.py`: distill from teacher ensemble to a student

# 

# \## Requirements

# 

# ```bash

# pip install -r requirements.txt


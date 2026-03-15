## CVPR 2026 ABAW 10th Challenge: Expression Recognition
This repository contains the codebase for our submission to the Expression Recognition Challenge of the CVPR 2026 ABAW 10th.

### 1. Installation
Install the required dependencies.

```bash
pip install -r requirements.txt
```

### 2. Training
Run the `train.py` script to train the model.

```bash
python train.py
```

### 3. Inference
Generate the inference results in text format, and then interpolate the missing frames to match the final submission format exactly.

```bash
python txt.py
python interpolation.py
```
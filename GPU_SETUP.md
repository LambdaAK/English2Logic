# GPU Environment Setup & Training

Copy and paste these commands into your Thunder Compute terminal.

---

## 1. Clone the repo (replace with your repo URL)

```bash
git clone https://github.com/YOUR_USERNAME/English2Prop.git
cd English2Prop
```

---

## 2. Install dependencies

```bash
pip install torch pytest
```

---

## 3. Generate the full dataset (~5–10 min on CPU)

```bash
python -m dataset.dataset_builder --output data --train 200000 --val 10000 --test 10000 --seed 42
```

---

## 4. Train the model

```bash
python -m model.train --data data/train.json --n 200000 --epochs 50 --batch-size 64 --lr 5e-4 --save-dir checkpoints
```

---

## 5. (Optional) Test the model interactively

```bash
python interact.py --model checkpoints/model_epoch_50.pt
```

---

## One-block paste (all steps, run after clone)

```bash
cd English2Prop
pip install torch pytest
python -m dataset.dataset_builder --output data --train 200000 --val 10000 --test 10000 --seed 42
python -m model.train --data data/train.json --n 200000 --epochs 50 --batch-size 64 --lr 5e-4 --save-dir checkpoints
```

---

## Quick reference

| Step | Command |
|------|---------|
| Clone | `git clone <your-repo-url> && cd English2Prop` |
| Install | `pip install torch pytest` |
| Generate data | `python -m dataset.dataset_builder --output data --train 200000 --val 10000 --test 10000 --seed 42` |
| Train | `python -m model.train --data data/train.json --n 200000 --epochs 50 --batch-size 64 --save-dir checkpoints` |
| Interact | `python interact.py --model checkpoints/model_epoch_50.pt` |

---

**Note:** If you haven't pushed the repo yet, you can upload the project folder instead and skip the `git clone` step. Then run from step 2 in the project directory.

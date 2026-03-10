# English2Logic

A transformer model that translates natural English sentences into propositional logic formulas. Train on synthetic data, then use the interactive script to convert phrases like "if A then B" into logic notation such as `IMPLIES(A,B)`.

## Example input/output pairs

| English | Logic |
|---------|-------|
| if A then B | `IMPLIES(A,B)` |
| A and B | `AND(A,B)` |
| A or B | `OR(A,B)` |
| not A | `NOT(A)` |
| A iff B | `IFF(A,B)` |
| B follows from A | `IMPLIES(A,B)` |
| it is not the case that A | `NOT(A)` |
| both A and B | `AND(A,B)` |
| either A or B | `OR(A,B)` |
| if A and B then C | `IMPLIES(AND(A,B),C)` |

**More complex (nested) examples:**

| English | Logic |
|---------|-------|
| A or B implies C | `IMPLIES(OR(A,B),C)` |
| provided that not A, B | `IMPLIES(NOT(A),B)` |
| A implies B and C | `IMPLIES(A,AND(B,C))` |
| A and B iff C or D | `IFF(AND(A,B),OR(C,D))` |
| it is not the case that if A then B | `NOT(IMPLIES(A,B))` |
| A and B or C and D | `OR(AND(A,B),AND(C,D))` |
| C and (if B then C or D) | `AND(C,IMPLIES(B,OR(C,D)))` |

## Usage

Run the trained model interactively:
```bash
python interact.py --model checkpoints/model_epoch_102_final.pt
```

Or evaluate on test data:
```bash
python evaluate.py --model checkpoints/model_epoch_102_final.pt
```

(Generate the dataset first if you don't have `data/` yet.)

## Training the current model

These commands were used to train the included model (~88% accuracy on the default model with 10k examples). Training took about 30 minutes on an A100 GPU.

1. **Generate dataset** (10k train, 1k val, 1k test):
   ```bash
   python -m dataset.dataset_builder --train 10000 --val 1000 --test 1000 --output data
   ```

2. **Train the default model**:
   ```bash
   python -m model.train --n 10000 --batch-size 256
   ```

3. **Train the larger model** (more capacity; use `--batch-size 128` if you hit memory limits):
   ```bash
   python -m model.train --n 10000 --batch-size 256 --large
   ```

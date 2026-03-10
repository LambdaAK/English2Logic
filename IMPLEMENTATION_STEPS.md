# Implementation Steps: English → Propositional Logic Transformer

This document outlines the phased implementation plan. Each phase produces a working, verifiable result before moving on.

---

## Phase 1: Logic Foundation

**Goal:** Solid logic representation and utilities that everything else depends on.

### Step 1.1: AST & Parser

- Define AST node types: `Var`, `Not`, `And`, `Or`, `Implies`, `Iff`
- Implement canonical serialization: `IMPLIES(AND(A,NOT(B)),C)`
- Implement parser: string → AST
- **Verify:** Parse `IMPLIES(AND(A,NOT(B)),C)`, serialize back, get identical string

### Step 1.2: Truth Table

- Implement truth table evaluation for formulas over variables A, B, C, D, E
- Enumerate all 2^5 = 32 truth assignments
- **Verify:** Compute truth table for `IMPLIES(A,B)` and `OR(NOT(A),B)`, confirm they match (logical equivalence)

### Step 1.3: Unit Tests

- Add tests for AST construction, serialization, parsing, truth table
- **Verify:** All tests pass

---

## Phase 2: Dataset Generation

**Goal:** Generate valid training data with English input and logic target.

### Step 2.1: Formula Generator

- Generate random formulas recursively with configurable max depth (1–4)
- Sample variables from {A, B, C, D, E}
- Ensure variety: all operators, different depths
- **Verify:** Generate 100 formulas, parse all, confirm valid ASTs

### Step 2.2: English Realizer

- Implement recursive template-based realization
- Define pattern templates for each operator:
  - **Var:** `"{var}"`, `"{var} is true"`, `"{var} holds"`
  - **NOT:** `"not {0}"`, `"{0} is false"`, `"it is not the case that {0}"`, `"{0} does not hold"`
  - **AND:** `"{0} and {1}"`, `"both {0} and {1}"`, `"{0} together with {1}"`
  - **OR:** `"{0} or {1}"`, `"either {0} or {1}"`, `"at least one of {0} or {1}"`
  - **IMPLIES:** `"if {0} then {1}"`, `"{1} if {0}"`, `"{0} implies {1}"`, `"whenever {0}, {1}"`, `"{1} follows from {0}"`, `"provided that {0}, {1}"`
  - **IFF:** `"{0} iff {1}"`, `"{0} if and only if {1}"`, `"{0} exactly when {1}"`, `"{0} is equivalent to {1}"`
- Use seeded RNG for reproducibility
- **Verify:** Realize 50 formulas, spot-check 10 for readability and correctness

### Step 2.3: Define English Vocabulary

- Extract or define the set of tokens that appear in realizer output
- Include: variables (a, b, c, d, e), logical words, structure words, special tokens
- Create `dataset/english_vocabulary.py` or derive from realizer output
- **Verify:** Generate 1000 examples, confirm every token appears in vocabulary

### Step 2.4: Dataset Builder

- Combine formula generator + realizer
- Output format: JSON or CSV with `input` (English) and `target` (canonical AST string)
- **Verify:** Generate 1000 examples, parse all targets, spot-check 20 inputs

---

## Phase 3: Tokenization

**Goal:** Stable tokenization for both English and logic.

### Step 3.1: English Tokenizer

- Split on whitespace, lowercase, handle punctuation
- Map tokens to IDs using vocabulary from Phase 2.3
- **Verify:** Tokenize 50 examples, confirm no OOV tokens

### Step 3.2: Logic Tokenizer

- Tokenize canonical format: `IMPLIES`, `(`, `AND`, `(`, `A`, `,`, `NOT`, `(`, `B`, `)`, `)`, `,`, `C`, `)`
- Build logic vocabulary from allowed operators, parens, commas, variables
- **Verify:** Tokenize 50 examples, detokenize, parse result, confirm round-trip

---

## Phase 4: Minimal Model & Overfit Test

**Goal:** Prove the model can learn; validate architecture and training loop.

### Step 4.1: Model Architecture

- Implement small transformer: 2 layers, 128 hidden size, 4 heads
- Encoder for English, decoder for logic
- **Verify:** Forward pass runs, output shape correct

### Step 4.2: Training Loop

- Seq2seq with teacher forcing
- Cross-entropy loss
- **Verify:** Loss decreases over a few steps

### Step 4.3: Overfit Test

- Train on 100–500 examples until overfit
- **Verify:** >95% exact match accuracy on training set (if not, debug model/data/tokenizer)

---

## Phase 5: Full Dataset & Training

**Goal:** Train on realistic data at scale.

### Step 5.1: Expand Realizer (Optional)

- Add more paraphrase patterns if needed
- Add optional noise: commas, "must", "holds"
- **Verify:** Dataset diversity increased

### Step 5.2: Scale Dataset

- Generate: 200k train, 10k validation, 10k test
- **Verify:** Dataset files exist, sizes correct, sample valid

### Step 5.3: Full Model & Training

- Use larger model: 4–6 layers, 256–512 hidden, 4–8 heads
- Train with validation monitoring
- **Verify:** Loss decreases, validation accuracy improves, no severe overfitting

---

## Phase 6: Evaluation Pipeline

**Goal:** Reliable metrics for model quality.

### Step 6.1: Exact Match Accuracy

- Percentage of predictions where generated string exactly matches target
- **Verify:** Run on test set, report metric

### Step 6.2: AST Structural Accuracy

- Parse predicted and target strings to ASTs
- Compare structures (ignore formatting differences)
- **Verify:** Run on test set, report metric

### Step 6.3: Logical Equivalence (Optional)

- For non-exact matches, compare truth tables
- Count as correct if logically equivalent
- **Verify:** Run on test set, report metric

---

## Phase 7: Experiments (Optional)

**Goal:** Measure generalization; defer until core pipeline works.

### Step 7.1: Paraphrase Generalization

- Train on subset of paraphrases, hold out others for test
- **Verify:** Measure accuracy on unseen phrasings

### Step 7.2: Compositional Generalization

- Train on formulas depth ≤2, test on depth 3–4
- **Verify:** Measure accuracy on deeper formulas

### Step 7.3: Noise Robustness

- Add optional commas, "must", "holds" to test set
- **Verify:** Measure accuracy with noise

---

## Checkpoints Summary

| After Phase | You Have |
|-------------|----------|
| 1 | Logic module tested and usable |
| 2 | Dataset generation working, vocabulary defined |
| 3 | Tokenization stable and reversible |
| 4 | Model overfits; architecture validated |
| 5 | Full training pipeline runs end-to-end |
| 6 | Reproducible evaluation metrics |
| 7 | Generalization experiments (optional) |

---

## Module Structure

```
dataset/
    formula_generator.py
    english_realizer.py
    english_vocabulary.py   # or derived from realizer
    dataset_builder.py

logic/
    ast.py
    parser.py
    truth_table.py

model/
    transformer.py
    tokenizer.py
    train.py
    evaluate.py

utils/
    config.py
    metrics.py
```

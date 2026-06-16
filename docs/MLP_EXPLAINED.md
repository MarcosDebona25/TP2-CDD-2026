# MLP Training Process - Complete Visual Guide

---

## 1. DATA PREPARATION & SHAPES

### What happens first: Converting embeddings to tensors

```
ORIGINAL DATA (From Section 2)
═════════════════════════════════════════════════════════════

    X_embs_train (numpy array)          y_train_encoded (numpy array)
    ┌─────────────────────────────┐     ┌──────────────────┐
    │  Shape: (10466, 384)        │     │  Shape: (10466,) │
    │                             │     │                  │
    │  Each row = 1 sample        │     │  Each row =      │
    │  384 SBERT embedding dims   │     │  Class label     │
    │                             │     │  (0-76)          │
    │  [0.12, -0.45, 0.78, ...]  │     │  [5, 42, 1, ...] │
    │  [0.33, 0.22, -0.11, ...]  │     │                  │
    │  ... 10466 samples          │     │                  │
    └─────────────────────────────┘     └──────────────────┘


STEP 1: Split into Train (85%) + Validation (15%)
═════════════════════════════════════════════════════════════

    X_embs_train (10466, 384)
           │
           ├─────────────────────┬─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
      85% (8896)            15% (1570)
      
    X_tr (8896, 384)      X_val (1570, 384)
    y_tr (8896,)          y_val (1570,)
    
    ⚠️  WHY split validation?
        • Early stopping needs independent data
        • If we validate on training data, model will overfit to it
        • Validation detects when model stops improving


STEP 2: Convert NumPy arrays to PyTorch Tensors
═════════════════════════════════════════════════════════════

    NumPy Array              torch.Tensor
    (CPU, float64)    ───→   (GPU/CPU, float32)
    
    X_tr (8896, 384)         X_tensor_train (8896, 384)
    y_tr (8896,)             y_tensor_train (8896,)
    
    ⚠️  WHY convert?
        • PyTorch needs tensors for GPU acceleration
        • float32 is standard for neural networks
        • Tensors support automatic differentiation (backprop)


STEP 3: Create DataLoader (batching)
═════════════════════════════════════════════════════════════

    X_tensor_train (8896, 384)  +  y_tensor_train (8896,)
              │
              ▼
    TensorDataset(X_tensor_train, y_tensor_train)
              │
              │  Shuffle = True (randomize order each epoch)
              │  Batch size = 32
              ▼
    DataLoader
    ┌─────────────────┐
    │  Batch 1        │  32 samples
    │  ┌────────────┐ │
    │  │ (32, 384)  │ │  ← 32 embeddings
    │  │ (32,)      │ │  ← 32 labels
    │  └────────────┘ │
    └─────────────────┘
    
    ┌─────────────────┐
    │  Batch 2        │  32 samples
    │  ┌────────────┐ │
    │  │ (32, 384)  │ │
    │  │ (32,)      │ │
    │  └────────────┘ │
    └─────────────────┘
    
    ... total ~278 batches (8896 / 32 = 278.625)
    
    ⚠️  WHY batching?
        • Memory: can't fit 8896 samples at once
        • Speed: parallel GPU operations on 32 samples
        • Regularization: noise in gradients helps avoid overfitting
```

---

## 2. NEURAL NETWORK ARCHITECTURE

### The IntentMLP class: What it looks like inside

```
INPUT LAYER (384 dimensions)
║
║  X comes in: (batch_size, 384)
║  Shape example: (32, 384) from a single batch
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  LINEAR LAYER 1: 384 → 256 neurons                   │
║  │  ┌────────────────────────────────────────────────┐  │
║  │  │  Matrix W1: (384, 256)  [weights to learn]    │  │
║  │  │  Vector b1: (256,)      [biases to learn]     │  │
║  │  │                                                 │  │
║  │  │  Computation: output = input @ W1 + b1        │  │
║  │  │  Shape: (32, 384) @ (384, 256) = (32, 256)   │  │
║  │  └────────────────────────────────────────────────┘  │
║  │                                                       │
║  │  Result: (32, 256) raw values                        │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  ReLU ACTIVATION FUNCTION                            │
║  │  ┌────────────────────────────────────────────────┐  │
║  │  │  What it does: f(x) = max(0, x)               │  │
║  │  │                                                 │  │
║  │  │  Negative values → 0                           │  │
║  │  │  Positive values → stay same                   │  │
║  │  │                                                 │  │
║  │  │  Example: [-2, 1, -0.5, 3]  →  [0, 1, 0, 3]  │  │
║  │  │                                                 │  │
║  │  │  ⚠️ WHY ReLU?                                   │  │
║  │  │  • Introduces non-linearity                    │  │
║  │  │  • Without it, stacking layers = single layer  │  │
║  │  │  • Computationally cheap                       │  │
║  │  │  • Works great in practice                     │  │
║  │  └────────────────────────────────────────────────┘  │
║  │                                                       │
║  │  Result: (32, 256) activated values                  │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  DROPOUT LAYER (p=0.3)                              │
║  │  ┌────────────────────────────────────────────────┐  │
║  │  │  What it does: randomly "turn off" 30% of     │  │
║  │  │  neurons with 50% probability                  │  │
║  │  │                                                 │  │
║  │  │  Training:  [a, b, c, d, e]                    │  │
║  │  │               ↓  ↓  ↓  ↓  ↓                     │  │
║  │  │             [a, 0, c, 0, e]  (randomly)        │  │
║  │  │                                                 │  │
║  │  │  Testing:   [a, b, c, d, e]  (unchanged)      │  │
║  │  │             (no dropout applied)                │  │
║  │  │                                                 │  │
║  │  │  ⚠️ WHY Dropout?                                │  │
║  │  │  • Prevents overfitting                        │  │
║  │  │  • Forces network to learn redundant features  │  │
║  │  │  • Acts like training many sub-networks       │  │
║  │  └────────────────────────────────────────────────┘  │
║  │                                                       │
║  │  Result: (32, 256) with some neurons disabled        │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  LINEAR LAYER 2: 256 → 128 neurons                   │
║  │  ┌────────────────────────────────────────────────┐  │
║  │  │  Matrix W2: (256, 128)  [weights to learn]    │  │
║  │  │  Vector b2: (128,)      [biases to learn]     │  │
║  │  │                                                 │  │
║  │  │  Computation: output = input @ W2 + b2        │  │
║  │  │  Shape: (32, 256) @ (256, 128) = (32, 128)   │  │
║  │  └────────────────────────────────────────────────┘  │
║  │                                                       │
║  │  Result: (32, 128) raw values                        │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  ReLU ACTIVATION FUNCTION (again)                    │
║  │  Same as before: f(x) = max(0, x)                   │
║  │  Result: (32, 128) activated values                 │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  DROPOUT LAYER (p=0.3) (again)                      │
║  │  Randomly disable 30% of neurons                     │
║  │  Result: (32, 128) with some neurons disabled        │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  ┌──────────────────────────────────────────────────────┐
║  │  LINEAR LAYER 3 (OUTPUT): 128 → 77 neurons          │
║  │  ┌────────────────────────────────────────────────┐  │
║  │  │  Matrix W3: (128, 77)   [weights to learn]    │  │
║  │  │  Vector b3: (77,)       [biases to learn]     │  │
║  │  │                                                 │  │
║  │  │  Computation: output = input @ W3 + b3        │  │
║  │  │  Shape: (32, 128) @ (128, 77) = (32, 77)    │  │
║  │  │                                                 │  │
║  │  │  ⚠️ WHY 77 neurons?                             │  │
║  │  │  • One neuron per class (77 intent categories) │  │
║  │  │  • Output[i] = "confidence" for class i        │  │
║  │  │  • NOT probabilities yet (raw logits)          │  │
║  │  └────────────────────────────────────────────────┘  │
║  │                                                       │
║  │  Result: (32, 77) logits (raw scores)               │
║  └──────────────────────────────────────────────────────┘
║
╠═══════════════════════════════════════════════════════════
║
║  OUTPUT (32, 77) logits
║  [raw scores before softmax]
║
║  Example for 1 sample:
║  [0.2, -1.5, 3.2, 0.8, ..., 2.1]  ← logits for 77 classes
║   │    │     │   │        │
║   └────┴─────┴───┴────────┘
║        77 values (one per class)
║
║  ⚠️ These are NOT probabilities yet!
║  They will be converted to probabilities by CrossEntropyLoss
║
╚═══════════════════════════════════════════════════════════
```

---

## 3. LOSS FUNCTION WITH CLASS WEIGHTS

### Why class weights matter

```
THE PROBLEM: Class Imbalance
═════════════════════════════════════════════════════════════

Dataset distribution:
    Class 0 (card_arrival):        227 samples (1.74%)  ⬅️ FREQUENT
    Class 1 (activate_my_card):     180 samples
    ...
    Class 76 (contactless_not_working): 75 samples (0.57%)  ⬅️ RARE

Without class weights:
    ┌──────────────────────────────────┐
    │ Loss = average of all samples    │
    │                                  │
    │ Losing on rare class (5 samples) │
    │ is drowned out by frequent       │
    │ class losses (200+ samples)      │
    │                                  │
    │ Result: Model learns to           │
    │ predict common classes better     │
    │ and ignores rare ones            │
    └──────────────────────────────────┘


SOLUTION: Class Weights
═════════════════════════════════════════════════════════════

Compute balanced weights:
    
    Weight = Total_samples / (Num_classes × Samples_in_class)
    
    Example:
    ┌─────────────────────────────────────────────────┐
    │  Class A (227 samples, FREQUENT):               │
    │  Weight = 13083 / (77 × 227) = 0.75 ⬅️ LOW      │
    │  (down-weight this class)                        │
    │                                                  │
    │  Class B (75 samples, RARE):                     │
    │  Weight = 13083 / (77 × 75) = 2.26 ⬅️ HIGH      │
    │  (up-weight this class)                          │
    └─────────────────────────────────────────────────┘

Code in notebook:
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.arange(num_classes),  # [0, 1, ..., 76]
        y=y_train_encoded                # actual labels
    )
    
    Result: array of 77 weights


APPLYING WEIGHTS TO LOSS
═════════════════════════════════════════════════════════════

CrossEntropyLoss(weight=class_weights_t)

For each sample:
    Loss = -log(softmax(output[true_class])) × class_weights[true_class]
    
    If true_class is RARE (high weight):
        Loss gets multiplied by ~2.26  ⬅️ Loss is AMPLIFIED
        
    If true_class is FREQUENT (low weight):
        Loss gets multiplied by ~0.75  ⬅️ Loss is DAMPENED


EFFECT IN TRAINING
═════════════════════════════════════════════════════════════

    Loss for batch with mostly common classes:
    ┌──────────────────────────────────────┐
    │ Total loss = small (dampened)         │
    │ Gradient update = small               │
    │ Model learns slowly for common classes│
    └──────────────────────────────────────┘
    
    Loss for batch with rare classes:
    ┌──────────────────────────────────────┐
    │ Total loss = large (amplified)        │
    │ Gradient update = large               │
    │ Model learns quickly for rare classes │
    └──────────────────────────────────────┘

⚠️ WHY? 
    Model can't ignore rare classes anymore
    Every rare sample has amplified impact on loss
    Training becomes more balanced
```

---

## 4. FORWARD PASS (What happens during training/prediction)

```
ONE TRAINING ITERATION (1 batch)
═════════════════════════════════════════════════════════════

STEP 1: optimizer.zero_grad()
────────────────────────────────────────
    ┌─────────────────────────────────────┐
    │ Before: gradients from previous step │
    │ W1.grad = [0.3, -0.1, 0.2, ...]     │
    │ b1.grad = [0.1, -0.05, ...]         │
    │ ... etc for all weights/biases      │
    │                                     │
    │ After zero_grad():                  │
    │ W1.grad = [0, 0, 0, ...]            │
    │ b1.grad = [0, 0, ...]               │
    │ ... all zeros                       │
    └─────────────────────────────────────┘
    
    ⚠️ WHY?
        • Gradients accumulate over batches
        • We need FRESH gradients for this batch
        • .zero_grad() clears old ones


STEP 2: outputs = mlp_model(batch_X)  [FORWARD PASS]
─────────────────────────────────────────────────────

    Input batch_X: (32, 384)
    
    ┌─────────────────────────────────────────────────────┐
    │  Linear(384 → 256)                                  │
    │  output = batch_X @ W1 + b1                         │
    │  output shape: (32, 256)                            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  ReLU()                                             │
    │  output = max(0, output)                            │
    │  output shape: (32, 256)                            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  Dropout(p=0.3)  [TRAINING MODE - dropouts active] │
    │  ~ 30% neurons randomly set to 0                    │
    │  output shape: (32, 256)                            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  Linear(256 → 128)                                  │
    │  output = output @ W2 + b2                          │
    │  output shape: (32, 128)                            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  ReLU()                                             │
    │  output = max(0, output)                            │
    │  output shape: (32, 128)                            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  Dropout(p=0.3)  [TRAINING MODE - dropouts active] │
    │  ~ 30% neurons randomly set to 0                    │
    │  output shape: (32, 128)                            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  Linear(128 → 77)                                   │
    │  logits = output @ W3 + b3                          │
    │  logits shape: (32, 77)                             │
    │                                                     │
    │  NOT YET probabilities!                             │
    │  Example row: [0.2, -1.5, 3.2, ..., 2.1]           │
    │  (raw predictions, can be negative)                │
    └─────────────────────────────────────────────────────┘
    
    outputs: (32, 77) logits


STEP 3: loss = criterion(outputs, batch_y)  [LOSS CALCULATION]
──────────────────────────────────────────────────────────────

    outputs: (32, 77) logits
    batch_y: (32,) labels (e.g., [5, 42, 1, ...])
    
    CrossEntropyLoss does this internally:
    
    ┌─────────────────────────────────────────────────────┐
    │ 1. Convert logits to probabilities (softmax)        │
    │                                                     │
    │    softmax(logits) = e^logits / sum(e^logits)      │
    │                                                     │
    │    Example for 1 sample (77 values):               │
    │    logits:       [0.2, -1.5, 3.2, ..., 2.1]        │
    │    softmax:      [0.01, 0.0001, 0.95, ..., 0.03]   │
    │    (values between 0-1, sum to 1)                   │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │ 2. Extract probability of true class                │
    │                                                     │
    │    If true_class = 2 (label = 2):                   │
    │    p_true = softmax[2] = 0.95                       │
    │                                                     │
    │    Backlog: 0.01, 0.0001, [0.95], 0.03, ...       │
    │                              ↑                      │
    │                         we want HIGH!               │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │ 3. Calculate negative log probability               │
    │                                                     │
    │    sample_loss = -log(p_true)                       │
    │    sample_loss = -log(0.95) = 0.05 ⬅️ SMALL LOSS   │
    │                                                     │
    │    If p_true was 0.1 (wrong prediction):           │
    │    sample_loss = -log(0.1) = 2.30 ⬅️ BIG LOSS      │
    │                                                     │
    │    ⚠️ log function:                                  │
    │    log(1.0) = 0     (perfect confidence)            │
    │    log(0.5) = 0.69  (moderate)                      │
    │    log(0.1) = 2.30  (low confidence)                │
    │    log(0.01) = 4.60 (very wrong)                    │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │ 4. Apply class weights (balancing)                  │
    │                                                     │
    │    weighted_loss = sample_loss × class_weight[2]   │
    │                                                     │
    │    If class 2 is rare:  weight = 2.26              │
    │    weighted_loss = 0.05 × 2.26 = 0.113             │
    │                                                     │
    │    If class 2 is common: weight = 0.75             │
    │    weighted_loss = 0.05 × 0.75 = 0.0375            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │ 5. Average over batch                               │
    │                                                     │
    │    All 32 samples:                                  │
    │    sample_1_loss = 0.113                            │
    │    sample_2_loss = 0.045                            │
    │    ...                                              │
    │    sample_32_loss = 0.567                           │
    │                                                     │
    │    batch_loss = (0.113 + 0.045 + ... + 0.567) / 32 │
    │    batch_loss = 0.256  ⬅️ SCALAR                    │
    └─────────────────────────────────────────────────────┘
    
    loss: scalar value (e.g., 0.256)


STEP 4: loss.backward()  [BACKPROPAGATION]
────────────────────────────────────────────

    This computes gradients for ALL parameters using chain rule
    
    ┌──────────────────────────────────────────────────────┐
    │  Computational graph (simplified):                    │
    │                                                      │
    │  batch_X ──→ Linear1 ──→ ReLU ──→ Dropout ──→ ...   │
    │                │         │        │                  │
    │              W1, b1      │        │                  │
    │                          ↓        ↓                  │
    │  ... ──→ Linear3 ──→ loss                            │
    │           │          (0.256)                         │
    │         W3, b3         ↑ ↑                           │
    │                        │ └─ batch_y (labels)        │
    │                        │                            │
    │       .backward() computes:                          │
    │       ∂loss/∂W1, ∂loss/∂b1                          │
    │       ∂loss/∂W2, ∂loss/∂b2                          │
    │       ∂loss/∂W3, ∂loss/∂b3                          │
    │                                                      │
    │       Using CHAIN RULE (working backwards)           │
    │       ∂loss/∂W3 = ∂loss/∂output × ∂output/∂W3      │
    │       ∂loss/∂W2 = ∂loss/∂output × ∂output/∂h2      │
    │       ...                                            │
    └──────────────────────────────────────────────────────┘
    
    After .backward():
    W1.grad = [∂loss/∂W1[0,0], ∂loss/∂W1[0,1], ...]
    b1.grad = [∂loss/∂b1[0], ∂loss/∂b1[1], ...]
    ... all parameters have gradients


STEP 5: optimizer.step()  [UPDATE WEIGHTS]
─────────────────────────────────────────────

    Adam optimizer (Adaptive Moment Estimation)
    
    For each parameter (e.g., W1[i,j]):
    
    ┌──────────────────────────────────────────────────┐
    │  OLD VALUE:  W1[i,j] = 0.5                       │
    │  GRADIENT:   ∂loss/∂W1[i,j] = 0.3                │
    │  LEARNING RATE: lr = 0.001                       │
    │                                                  │
    │  NAIVE UPDATE (Stochastic Gradient Descent):     │
    │  W1_new = W1 - lr × gradient                    │
    │  W1_new = 0.5 - 0.001 × 0.3 = 0.4997            │
    │                                                  │
    │  ADAM UPDATE (smarter):                          │
    │  • Maintains momentum (m) and velocity (v)       │
    │  • Adapts learning rate per parameter           │
    │  • m ← β1×m + (1-β1)×gradient (momentum)         │
    │  • v ← β2×v + (1-β2)×gradient² (velocity)       │
    │  • W_new ← W - (lr × m) / (√v + ε)              │
    │                                                  │
    │  (Adam default: β1=0.9, β2=0.999)                │
    │                                                  │
    │  ⚠️ WHY Adam?                                     │
    │  • Faster convergence than SGD                   │
    │  • Handles sparse gradients                      │
    │  • Per-parameter adaptive learning rates         │
    │  • Works well in practice                        │
    └──────────────────────────────────────────────────┘
    
    After optimizer.step():
    W1 ← updated
    b1 ← updated
    W2 ← updated
    b2 ← updated
    W3 ← updated
    b3 ← updated


RESULT: One batch processed
═════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │ Weights are slightly changed         │
    │ Model predicts slightly better       │
    │                                     │
    │ W1 = 0.5 → 0.4997                   │
    │ (moved in direction of lower loss)  │
    │                                     │
    │ Repeat for batch 2, 3, ... 278      │
    │ (one full epoch = 278 batches)      │
    └─────────────────────────────────────┘
```

---

## 5. FULL TRAINING LOOP WITH EARLY STOPPING

```
TRAINING PSEUDOCODE EXECUTION
═════════════════════════════════════════════════════════════

for epoch in range(50):
    
    ┌─── TRAINING PHASE ───────────────────────────────────┐
    │                                                       │
    │  mlp_model.train()  ← TURN ON dropout                │
    │  total_loss = 0                                       │
    │                                                       │
    │  for batch_X, batch_y in train_loader:  ← 278 batches│
    │      │                                               │
    │      ├─ optimizer.zero_grad()   ← Clear old gradients│
    │      │                                               │
    │      ├─ outputs = mlp_model(batch_X)  ← Forward pass │
    │      │   (dropout is ACTIVE here)                    │
    │      │                                               │
    │      ├─ loss = criterion(outputs, batch_y)           │
    │      │   (calculate loss for this batch)             │
    │      │                                               │
    │      ├─ loss.backward()  ← Backpropagation           │
    │      │   (compute gradients)                         │
    │      │                                               │
    │      ├─ optimizer.step()  ← Update all weights       │
    │      │   (move in direction of lower loss)           │
    │      │                                               │
    │      └─ total_loss += loss.item()                    │
    │         (accumulate loss)                            │
    │                                                       │
    │  train_loss = total_loss / len(train_loader)         │
    │  (average loss over all 278 batches)                 │
    │                                                       │
    └───────────────────────────────────────────────────────┘
    
    
    ┌─── VALIDATION PHASE ─────────────────────────────────┐
    │                                                       │
    │  mlp_model.eval()  ← TURN OFF dropout                │
    │  with torch.no_grad():  ← NO backpropagation         │
    │      val_outputs = mlp_model(X_tensor_val)           │
    │      val_loss = criterion(val_outputs, y_tensor_val) │
    │                                                       │
    │  ⚠️ WHY no_grad()?                                    │
    │  • Don't need gradients for validation               │
    │  • Saves memory                                       │
    │  • Faster                                             │
    │                                                       │
    │  ⚠️ WHY eval mode?                                    │
    │  • Disable dropout (use all neurons)                 │
    │  • Use batch norm statistics from training           │
    │  • Consistent predictions                            │
    └───────────────────────────────────────────────────────┘
    
    
    ┌─── EARLY STOPPING LOGIC ─────────────────────────────┐
    │                                                       │
    │  if val_loss < best_val_loss:                         │
    │      │                                               │
    │      ├─ best_val_loss = val_loss                     │
    │      │  (new record!)                                │
    │      │                                               │
    │      ├─ best_state = {copy all weights}              │
    │      │  (save checkpoint)                            │
    │      │                                               │
    │      └─ epochs_sin_mejora = 0                        │
    │         (reset counter)                              │
    │                                                       │
    │  else:  [val_loss NOT improving]                     │
    │      │                                               │
    │      └─ epochs_sin_mejora += 1                       │
    │         (increment patience counter)                 │
    │                                                       │
    │      if epochs_sin_mejora >= 5:  ← PATIENCE = 5      │
    │          │                                           │
    │          └─ STOP TRAINING                            │
    │             (model quality is declining)             │
    │                                                       │
    │  Print: Epoch 01/50 | Train: 2.8277 | Val: 1.2312   │
    │                                                       │
    └───────────────────────────────────────────────────────┘


EARLY STOPPING VISUALIZATION
═════════════════════════════════════════════════════════════

Epoch  │ Train Loss │ Val Loss │ Notes
───────┼────────────┼──────────┼──────────────────────────────
   1   │   2.827    │  1.231   │ First epoch
   2   │   1.127    │  0.672   │ Val loss decreasing ✓
   3   │   0.804    │  0.534   │ Val loss decreasing ✓
   4   │   0.655    │  0.464   │ Val loss decreasing ✓
   5   │   0.571    │  0.438   │ Val loss decreasing ✓
   6   │   0.513    │  0.391   │ Val loss decreasing ✓ best_val
   7   │   0.466    │  0.390   │ Val loss very slightly worse ⚠️ count=1
   8   │   0.430    │  0.383   │ Val loss decreasing again ✓ count=0
   9   │   0.396    │  0.359   │ Val loss decreasing ✓
  ...
  20   │   0.229    │  0.318   │ Val loss starts increasing ⚠️
  21   │   0.207    │  0.319   │ Worse ⚠️ count=1
  22   │   0.209    │  0.325   │ Worse ⚠️ count=2
  23   │   0.206    │  0.319   │ Worse ⚠️ count=3
  24   │   0.194    │  0.314   │ Better ✓ count=0, new best
  25   │   0.186    │  0.307   │ Better ✓
  26   │   0.176    │  0.328   │ Worse ⚠️ count=1
  27   │   0.178    │  0.328   │ Worse ⚠️ count=2
  28   │   0.169    │  0.319   │ Worse ⚠️ count=3
  29   │   0.165    │  0.319   │ Worse ⚠️ count=4
  30   │   0.152    │  0.339   │ Worse ⚠️ count=5 = PATIENCE
       │            │          │
       │            │          │ ❌ STOP! No improvement for 5 epochs
       │            │          │ Restore weights from epoch 25
       │            │          │ (best_state)


WHY EARLY STOPPING?
═════════════════════════════════════════════════════════════

    Without Early Stopping:
    ┌──────────────────────────────────────────────────┐
    │  Val Loss                                        │
    │  │      ╱╲                                       │
    │  │     ╱  ╲___                                   │
    │  │    ╱       ╲_____ ← keeps going down         │
    │  │   ╱              ╲                            │
    │  │  ╱                ╲                           │
    │  │ ╱                  ╲___ ← eventually rises    │
    │  │                        ╲ (overfitting!)      │
    │  ├──────────────────────────────────────────────┤
    │  0  10  20  30  40  50                          │
    │        Epoch                                     │
    │                                                  │
    │  Train continues past the best point            │
    │  Overfitting gets worse                         │
    └──────────────────────────────────────────────────┘
    
    
    With Early Stopping:
    ┌──────────────────────────────────────────────────┐
    │  Val Loss                                        │
    │  │      ╱╲                                       │
    │  │     ╱  ╲___                                   │
    │  │    ╱       ╲ ← STOP HERE (epoch 25)          │
    │  │   ╱        ✓                                  │
    │  │  ╱                                            │
    │  │ ╱                                             │
    │  │                                              │
    │  ├──────────────────────────────────────────────┤
    │  0  10  20  25  30  40  50                      │
    │        Epoch                                     │
    │                                                  │
    │  Stop at best validation performance            │
    │  Prevent overfitting                            │
    └──────────────────────────────────────────────────┘

⚠️ PATIENCE = 5 means:
    "Stop if validation loss doesn't improve for 5 consecutive epochs"
    This allows for natural fluctuations but catches overfitting
```

---

## 6. FINAL EVALUATION (Test Phase)

```
AFTER TRAINING COMPLETES
═════════════════════════════════════════════════════════════

Step 1: Restore best weights
────────────────────────────

    if best_state is not None:
        mlp_model.load_state_dict(best_state)
    
    The model now has weights from epoch 25 (the best validation point)
    All the learning after epoch 25 is discarded


Step 2: Test phase (never touched during training)
───────────────────────────────────────────────────

    mlp_model.eval()           ← Turn off dropout
    with torch.no_grad():      ← Don't compute gradients
        
        X_tensor_test = torch.tensor(X_embs_test, dtype=torch.float32)
        # Shape: (2617, 384) — all test samples
        
        test_outputs = mlp_model(X_tensor_test)
        # Shape: (2617, 77) — logits for each sample and class
        
        y_pred_mlp = torch.argmax(test_outputs, dim=1).numpy()
        # For each sample, find the class with highest logit
        # Shape: (2617,) — one prediction per sample
    
    
    ┌─────────────────────────────────────────┐
    │  Example:                               │
    │  test_outputs[0] = [0.2, -1.5, 3.2, ...│
    │                     argmax(↑) = 2      │
    │  y_pred_mlp[0] = 2                      │
    │                                         │
    │  (model predicts class 2 for sample 0) │
    └─────────────────────────────────────────┘


Step 3: Calculate metrics
─────────────────────────

    Accuracy = (correct predictions) / (total predictions)
    
    F1-macro = average F1 across all classes
    F1-weighted = weighted average F1
    
    Example results:
    ┌────────────────────────────┐
    │  Accuracy    : 0.9129      │
    │  F1 macro    : 0.9130      │
    │  F1 weighted : 0.9131      │
    └────────────────────────────┘
    
    ⚠️ Why these metrics?
    • Accuracy: simple, overall performance
    • F1-macro: each class weighted equally (good for imbalance)
    • F1-weighted: by class frequency (weighted average)


Step 4: Detailed analysis
──────────────────────────

    Classification report for each class:
    
    Class: "card_arrival"
    ├─ Precision: 0.84 (of samples predicted as this class, 84% correct)
    ├─ Recall: 0.82 (of actual samples of this class, 82% predicted correctly)
    ├─ F1-score: 0.83 (harmonic mean of precision and recall)
    └─ Support: 39 (39 test samples of this class)
    
    ... repeat for all 77 classes
```

---

## 7. COMPLETE TRAINING LOOP SUMMARY

```
THE BIG PICTURE
═════════════════════════════════════════════════════════════

START
  │
  ├─ Prepare data
  │  ├─ Split: train (85%) + val (15%)
  │  └─ Convert to tensors and DataLoader
  │
  ├─ Build model
  │  └─ IntentMLP(384 → 256 → 128 → 77)
  │
  ├─ Set up loss & optimizer
  │  ├─ Loss: CrossEntropyLoss with class weights
  │  └─ Optimizer: Adam (lr=0.001)
  │
  └─ Training loop (up to 50 epochs)
     │
     ├─ For each epoch:
     │  │
     │  ├─ TRAIN PHASE: process all training batches
     │  │  └─ For each batch of 32 samples:
     │  │     ├─ Clear old gradients
     │  │     ├─ Forward pass (compute logits)
     │  │     ├─ Compute loss (with class weights)
     │  │     ├─ Backward pass (compute gradients)
     │  │     └─ Update weights (Adam optimizer)
     │  │
     │  ├─ VALIDATION PHASE: evaluate on val set
     │  │  ├─ Forward pass (no dropout)
     │  │  ├─ Compute validation loss
     │  │  └─ Check if best so far
     │  │
     │  └─ EARLY STOPPING: Did val loss improve?
     │     ├─ YES → save weights, reset patience counter
     │     ├─ NO  → increment patience counter
     │     └─ Counter = 5? → STOP training
     │
     └─ Restore best weights (from epoch 25 in example)

  │
  ├─ Test phase (never seen before)
  │  ├─ Load best model weights
  │  ├─ Forward pass on test set (no dropout)
  │  ├─ Get predictions (argmax of logits)
  │  └─ Calculate metrics (Accuracy, F1, etc.)
  │
  └─ Report results
     └─ "Accuracy: 0.9129, F1-macro: 0.9130"

END


MEMORY USAGE DIAGRAM
═════════════════════════════════════════════════════════════

During Forward Pass (training batch):
    
    batch_X (32, 384)
    ├─ Stored for backward pass
    │
    ├─ Hidden activations (32, 256)
    │ ├─ Stored for backward pass
    │ └─ Used by next layer
    │
    ├─ Hidden activations (32, 128)
    │ ├─ Stored for backward pass
    │ └─ Used by next layer
    │
    └─ Output logits (32, 77)
       └─ Used to compute loss
    
    Total for 1 batch: ~few MB


Entire model weights:
    
    W1: (384, 256) × 4 bytes = 393KB
    b1: (256,) × 4 bytes = 1KB
    W2: (256, 128) × 4 bytes = 131KB
    b2: (128,) × 4 bytes = 0.5KB
    W3: (128, 77) × 4 bytes = 40KB
    b3: (77,) × 4 bytes = 0.3KB
    
    Total: ~565KB (tiny!)


Validation set in memory (always loaded):
    X_tensor_val: (1570, 384) float32 = 2.4MB
    y_tensor_val: (1570,) int64 = 12.5KB
    
    Total: ~2.4MB


Why so efficient:
    • Only 384-dimensional embeddings (not raw text)
    • Small network (3 layers)
    • Batch processing (not all data at once)
    • Can run on CPU (no GPU needed)
```

---

## 8. KEY CONCEPTS & WHY THEY MATTER

```
DROPOUT (p=0.3)
═════════════════════════════════════════════════════════════

What: Randomly disable 30% of neurons during training
Why:  Co-adaptation prevention
      Imagine two neurons learn to rely completely on each other
      Together they work, but separately they fail
      
      Dropout forces each neuron to be useful independently
      Creates redundancy and robustness

When:
      ✓ Applied during training
      ✗ NOT applied during testing/validation
      
Why different?
      Training: need regularization to prevent overfitting
      Testing: want best predictions (use all neurons)


BATCH NORMALIZATION (not in this code, but important concept)
═════════════════════════════════════════════════════════════

What: Normalize inputs to each layer
Why:  Stabilizes training, faster convergence
Effect: Mean ≈ 0, Std ≈ 1 for each hidden layer

Note: This MLP doesn't use batch norm (uses dropout instead)


CLASS WEIGHTS (BALANCE)
═════════════════════════════════════════════════════════════

Problem: Dataset has 227 samples of class A, 75 of class B

Solution: 
         Loss_A = loss × 0.75  (dampened)
         Loss_B = loss × 2.26  (amplified)

Effect:  Rare class B is given 3× more importance
         Model learns to classify rare classes better


ADAM OPTIMIZER
═════════════════════════════════════════════════════════════

Simple SGD:  W ← W - lr × gradient
Adam:        W ← W - (lr × momentum) / (sqrt(velocity) + eps)

Why Adam is better:
         ✓ Per-parameter adaptive learning rates
         ✓ Handles sparse gradients well
         ✓ Faster convergence
         ✓ Less sensitive to learning rate choice


EARLY STOPPING
═════════════════════════════════════════════════════════════

Without: Train for fixed 50 epochs, overfit after epoch 25
With:    Stop at epoch 25 when validation stops improving

Saves:   • Training time (stop early)
         • Generalization (prevent overfitting)
         • Best model weights (automatically saved)

Patience=5:
         Allow 5 epochs without improvement before giving up
         Handles natural fluctuations in training
```

---

## 9. TROUBLESHOOTING CHECKLIST

```
If training is slow:
   ☐ Are you using a GPU? (Not in this code, but possible)
   ☐ Is batch_size too small? (32 is reasonable)
   ☐ Are you printing too much? (verbose=1 adds overhead)

If loss isn't decreasing:
   ☐ Is learning_rate too small? (0.001 is standard)
   ☐ Is learning_rate too large? (gradient will diverge)
   ☐ Check if data is normalized properly
   ☐ Is class_weight computation correct?

If val_loss is much worse than train_loss:
   ☐ Model is overfitting
   ☐ Try increasing dropout (currently 0.3)
   ☐ Try reducing model size
   ☐ Collect more training data

If early stopping triggers immediately:
   ☐ Is validation data too small/noisy?
   ☐ Is patience too small? (currently 5)
   ☐ Random seed might affect initial weights


DEBUGGING: Add these prints
═════════════════════════════════════════════════════════════

for epoch in range(50):
    mlp_model.train()
    
    for batch_X, batch_y in train_loader:
        print(f"Batch X shape: {batch_X.shape}, dtype: {batch_X.dtype}")
        print(f"Batch y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
        
        outputs = mlp_model(batch_X)
        print(f"Outputs shape: {outputs.shape}")
        print(f"Output values (first sample): {outputs[0]}")
        
        loss = criterion(outputs, batch_y)
        print(f"Loss: {loss.item()}")
        
        break  # Only first batch
    break   # Only first epoch
```

---

## 10. COMPLETE CODE WALKTHROUGH

```python
# 1. SET SEEDS (reproducibility)
torch.manual_seed(42)
np.random.seed(42)
# Why? Same random weights every run

# 2. SPLIT DATA
X_tr, X_val, y_tr, y_val = train_test_split(
    np.array(X_embs_train),      # 10466 embeddings
    y_train_encoded,              # 10466 labels
    test_size=0.15,               # 15% for validation
    random_state=42,              # reproducible split
    stratify=y_train_encoded      # maintain class proportions
)
# X_tr: (8896, 384)  ← 85% training
# X_val: (1570, 384) ← 15% validation

# 3. CONVERT TO TENSORS
X_tensor_train = torch.tensor(X_tr, dtype=torch.float32)
y_tensor_train = torch.tensor(y_tr, dtype=torch.long)
X_tensor_val = torch.tensor(X_val, dtype=torch.float32)
y_tensor_val = torch.tensor(y_val, dtype=torch.long)
# Why float32? Standard precision for neural networks
# Why long for y? CrossEntropyLoss expects integer labels

# 4. CREATE DATALOADER
train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,     # Process 32 samples at a time
    shuffle=True       # Randomize order each epoch
)
# Provides 278 batches per epoch (8896/32)

# 5. DEFINE MODEL
class IntentMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntentMLP, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(384, 256),    # Input layer
            nn.ReLU(),              # Activation
            nn.Dropout(0.3),        # Regularization
            nn.Linear(256, 128),    # Hidden layer
            nn.ReLU(),              # Activation
            nn.Dropout(0.3),        # Regularization
            nn.Linear(128, 77)      # Output layer (no activation!)
        )

    def forward(self, x):
        return self.red(x)

# 6. INSTANTIATE MODEL
mlp_model = IntentMLP(384, 77)
# 77 = number of classes
# 384 = embedding dimension

# 7. COMPUTE CLASS WEIGHTS
class_weights = compute_class_weight(
    'balanced',                      # Use balanced formula
    classes=np.arange(77),           # Classes 0-76
    y=y_train_encoded                # From training data
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
# Result: array of 77 weights (one per class)

# 8. LOSS FUNCTION & OPTIMIZER
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
# Combines: softmax + log + weighted average
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
# Adam optimizer with learning rate 0.001

# 9. TRAINING LOOP
epochs = 50
patience = 5
best_val_loss = float('inf')
epochs_sin_mejora = 0
best_state = None

for epoch in range(epochs):
    # TRAIN
    mlp_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()           # Clear old gradients
        outputs = mlp_model(batch_X)    # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)

    # VALIDATE
    mlp_model.eval()
    with torch.no_grad():
        val_loss = criterion(mlp_model(X_tensor_val), y_tensor_val).item()

    # EARLY STOPPING
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in mlp_model.state_dict().items()}
        epochs_sin_mejora = 0
    else:
        epochs_sin_mejora += 1
        if epochs_sin_mejora >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

# 10. RESTORE BEST WEIGHTS
if best_state is not None:
    mlp_model.load_state_dict(best_state)

# 11. TEST EVALUATION
mlp_model.eval()
with torch.no_grad():
    X_tensor_test = torch.tensor(np.array(X_embs_test), dtype=torch.float32)
    test_outputs = mlp_model(X_tensor_test)
    y_pred_mlp = torch.argmax(test_outputs, dim=1).numpy()

# 12. METRICS
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test_encoded, y_pred_mlp)
f1_macro = f1_score(y_test_encoded, y_pred_mlp, average='macro')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-macro: {f1_macro:.4f}")
```

---

## FINAL SUMMARY: What Happens When

```
┌─ BEFORE TRAINING ───────────────────────────────────────┐
│  1. Data split into train/val                           │
│  2. Model weights: random (normally distributed)        │
│  3. Class weights: computed to balance loss              │
└─────────────────────────────────────────────────────────┘

┌─ DURING ONE EPOCH ──────────────────────────────────────┐
│  1. Process all 278 training batches                     │
│     - Each batch: forward, loss, backward, update       │
│  2. Evaluate on validation set (1570 samples)           │
│  3. Check: did val_loss improve?                        │
│     - YES: save weights, continue                       │
│     - NO: increment patience counter                    │
│  4. If patience exhausted: STOP                         │
└─────────────────────────────────────────────────────────┘

┌─ AFTER TRAINING ────────────────────────────────────────┐
│  1. Restore best weights (from best epoch)              │
│  2. Run inference on test set (2617 samples)            │
│  3. Compute accuracy, F1, precision, recall             │
│  4. Report results                                      │
└─────────────────────────────────────────────────────────┘
```

---

**This is the complete MLP process from the notebook! Every line, every concept, explained with the WHY behind it.**

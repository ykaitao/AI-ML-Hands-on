# Best Practices for DeepSeekV3Config Arguments (New/Unique Parameters)

I'll focus only on the **new arguments** that are NOT in GPT2Config, primarily related to **Mixture of Experts (MoE)** and **Multi-Query Attention with LoRA**.

---

## MoE (Mixture of Experts) Parameters

### 1. **`n_shared_experts`** (default: 1)

Number of experts that are **always active** for every token.

#### ✅ When to use different values:

**Small (1-2)**: Standard MoE
```python
config = DeepseekV3Config(
    n_shared_experts=1,  # Minimal shared capacity
    n_routed_experts=256
)
```

**Medium (4-8)**: More stable training, better baseline
```python
config = DeepseekV3Config(
    n_shared_experts=4,  # Stronger shared foundation
    n_routed_experts=64
)
```

**Large (16+)**: Hybrid between dense and sparse
```python
config = DeepseekV3Config(
    n_shared_experts=16,  # Nearly dense model
    n_routed_experts=32   # Sparse component is small
)
```

#### 💡 Best Practices:

```python
# ✅ GOOD: Standard MoE setup (DeepSeek-V3 style)
config = DeepseekV3Config(
    n_shared_experts=1,      # Always-on experts
    n_routed_experts=256,    # Specialist experts
    num_experts_per_tok=8    # Select 8 routed experts per token
)
model = DeepseekV3Model(config)

# ✅ GOOD: Small model with fewer experts
config = DeepseekV3Config(
    hidden_size=1024,
    n_shared_experts=2,      # 2 shared for stability
    n_routed_experts=16,     # Fewer total experts
    num_experts_per_tok=4    # Select fewer
)

# ✅ GOOD: Dense model (no MoE)
config = DeepseekV3Config(
    n_shared_experts=0,      # No shared experts
    n_routed_experts=1,      # Only 1 expert = dense FFN
    num_experts_per_tok=1    # Always use the 1 expert
)

# ⚠️ CAUTION: Too many shared experts
config = DeepseekV3Config(
    n_shared_experts=64,     # Too many = defeats purpose of MoE
    n_routed_experts=256
)
# This becomes nearly dense, losing MoE benefits

# 💡 RULE OF THUMB:
# - n_shared_experts should be much smaller than n_routed_experts
# - Ratio: n_shared / n_routed ≈ 1/64 to 1/16
```

---

### 2. **`n_routed_experts`** (default: 256)

Total number of **specialist experts** that can be selected per token.

#### ✅ Scaling guidelines:

```python
# ✅ GOOD: Tiny model (edge devices)
config = DeepseekV3Config(
    hidden_size=512,
    n_routed_experts=8,      # Few experts
    num_experts_per_tok=2,   # Select 2
    moe_intermediate_size=512
)

# ✅ GOOD: Small model (research/prototyping)
config = DeepseekV3Config(
    hidden_size=2048,
    n_routed_experts=32,     # Moderate
    num_experts_per_tok=4,
    moe_intermediate_size=1024
)

# ✅ GOOD: Large model (DeepSeek-V3 scale)
config = DeepseekV3Config(
    hidden_size=7168,
    n_routed_experts=256,    # Many experts
    num_experts_per_tok=8,
    moe_intermediate_size=2048
)

# ❌ BAD: Tiny model with too many experts
config = DeepseekV3Config(
    hidden_size=256,
    n_routed_experts=256,    # Experts won't be well-trained!
    num_experts_per_tok=8
)
# Problem: Not enough capacity per expert

# 💡 RULE OF THUMB:
# - More experts = more specialization, but needs more data
# - Experts should be trained on enough examples
# - Typical: 8-256 experts for production models
```

---

### 3. **`num_experts_per_tok`** (default: 8)

Number of experts **selected and activated** for each token.

#### ✅ When to use different values:

```python
# ✅ GOOD: Sparse model (efficient inference)
config = DeepseekV3Config(
    n_routed_experts=256,
    num_experts_per_tok=2,   # Very sparse, fast
    topk_group=2
)
# Inference: 2/256 experts active = 0.78% activation

# ✅ GOOD: Balanced (DeepSeek-V3 default)
config = DeepseekV3Config(
    n_routed_experts=256,
    num_experts_per_tok=8,   # Good balance
    topk_group=4
)
# Inference: 8/256 experts active = 3.1% activation

# ✅ GOOD: Dense-like behavior
config = DeepseekV3Config(
    n_routed_experts=64,
    num_experts_per_tok=32,  # Half of experts active
    topk_group=8
)
# Inference: 32/64 = 50% activation

# ❌ BAD: Too many experts per token
config = DeepseekV3Config(
    n_routed_experts=16,
    num_experts_per_tok=15,  # Almost all experts!
    topk_group=4
)
# Problem: Defeats purpose of MoE, no efficiency gain

# 💡 RELATIONSHIP:
# num_experts_per_tok should be:
# - Much smaller than n_routed_experts (typically 1-10%)
# - Multiple of topk_group (for even distribution)
# - num_experts_per_tok = topk_group * (experts_per_group)
```

---

### 4. **`n_group`** (default: 8) and **`topk_group`** (default: 4)

Group-based expert selection for load balancing.

#### ✅ Understanding the hierarchy:

```python
# Expert organization:
# n_routed_experts = 256 total experts
# n_group = 8 groups
# Each group has: 256 / 8 = 32 experts

# Selection process:
# topk_group = 4 groups selected
# num_experts_per_tok = 8 experts total
# Per selected group: 8 / 4 = 2 experts

config = DeepseekV3Config(
    n_routed_experts=256,    # Total experts
    n_group=8,               # Divide into 8 groups (32 each)
    topk_group=4,            # Select 4 groups
    num_experts_per_tok=8    # Total 8 experts (2 per group)
)
```

#### 💡 Best Practices:

```python
# ✅ GOOD: Balanced grouping (DeepSeek-V3)
config = DeepseekV3Config(
    n_routed_experts=256,
    n_group=8,               # 32 experts per group
    topk_group=4,            # Half groups selected
    num_experts_per_tok=8    # 2 experts per selected group
)
# Ensures load balancing across groups

# ✅ GOOD: Fine-grained control (more groups)
config = DeepseekV3Config(
    n_routed_experts=256,
    n_group=16,              # 16 experts per group
    topk_group=8,            # Select half
    num_experts_per_tok=8    # 1 expert per selected group
)
# More flexible expert selection

# ❌ BAD: Invalid configuration
config = DeepseekV3Config(
    n_routed_experts=256,
    n_group=8,
    topk_group=10,           # Can't select more groups than exist!
    num_experts_per_tok=8
)

# ❌ BAD: Uneven distribution
config = DeepseekV3Config(
    n_routed_experts=256,
    n_group=7,               # 256/7 = 36.57 (not integer!)
    topk_group=3,
    num_experts_per_tok=6
)

# 💡 CONSTRAINTS:
# - n_routed_experts % n_group == 0 (even division)
# - topk_group <= n_group (can't select more than exist)
# - num_experts_per_tok % topk_group == 0 (even per group)
# - num_experts_per_tok / topk_group <= n_routed_experts / n_group
```

---

### 5. **`routed_scaling_factor`** (default: 2.5)

Scaling factor for routed expert outputs.
```python
# Simplified MoE output computation:
output = shared_expert_output + routed_scaling_factor * routed_expert_output
         ├─────────────────┘     ├──────────────────┘   ├──────────────────┘
         Always active           Scaling factor          Dynamically selected
         (baseline)              (amplify/dampen)        (specialists)
```

#### ✅ When to modify:

```python
# ✅ GOOD: Standard setup (DeepSeek-V3 default)
config = DeepseekV3Config(
    routed_scaling_factor=2.5,  # Empirically tuned
    n_shared_experts=1,
    n_routed_experts=256
)

# ✅ GOOD: Emphasize routed experts
config = DeepseekV3Config(
    routed_scaling_factor=3.0,  # Stronger specialist contribution
    n_shared_experts=1,
    n_routed_experts=256
)

# ✅ GOOD: Emphasize shared experts
config = DeepseekV3Config(
    routed_scaling_factor=1.5,  # Weaker specialist contribution
    n_shared_experts=4,         # More shared capacity
    n_routed_experts=64
)

# ⚠️ CAUTION: Extreme values
config = DeepseekV3Config(
    routed_scaling_factor=10.0,  # Too high, unstable training
    n_shared_experts=1,
    n_routed_experts=256
)

# 💡 GUIDANCE:
# - Default 2.5 works well for most cases
# - Lower (1.5-2.0): More balanced shared/routed
# - Higher (3.0-4.0): More emphasis on specialization
# - Rarely need to change from default
```

---

### 6. **`norm_topk_prob`** (default: True)

Normalize expert selection probabilities.

#### ✅ When to use:

```python
# ✅ GOOD: Standard MoE (recommended)
config = DeepseekV3Config(
    norm_topk_prob=True,  # Normalize weights to sum to 1
    num_experts_per_tok=8
)
# Ensures stable training, weighted averaging

# ⚠️ RARE: Unnormalized (research purposes)
config = DeepseekV3Config(
    norm_topk_prob=False,  # Raw routing weights
    num_experts_per_tok=8
)
# May lead to scaling issues

# 💡 BEST PRACTICE: Always use True unless researching routing mechanisms
```

---

### 7. **`first_k_dense_replace`** (default: 3)

Number of initial layers that use **dense FFN** instead of MoE.

#### ✅ When to use different values:

```python
# ✅ GOOD: Standard (DeepSeek-V3)
config = DeepseekV3Config(
    num_hidden_layers=61,
    first_k_dense_replace=3,  # First 3 layers are dense
    n_routed_experts=256
)
# Architecture: [dense, dense, dense, MoE, MoE, ..., MoE]

# ✅ GOOD: More dense layers (stability)
config = DeepseekV3Config(
    num_hidden_layers=24,
    first_k_dense_replace=6,  # First 6 layers dense
    n_routed_experts=128
)
# Better for: smaller models, less stable training data

# ✅ GOOD: All MoE (maximum efficiency)
config = DeepseekV3Config(
    num_hidden_layers=24,
    first_k_dense_replace=0,  # All layers use MoE
    n_routed_experts=64
)
# Most parameter efficient, but may be less stable

# ✅ GOOD: All dense (debugging)
config = DeepseekV3Config(
    num_hidden_layers=12,
    first_k_dense_replace=12,  # All layers dense
    n_routed_experts=1  # MoE disabled
)
# Useful for debugging, baseline comparison

# 💡 RULE OF THUMB:
# - Shallow layers (near input): Dense for general features
# - Deep layers (near output): MoE for specialization
# - first_k_dense_replace ≈ 5-10% of num_hidden_layers
```

---

## LoRA-based Attention Parameters

### 8. **`kv_lora_rank`** (default: 512)

Low-rank dimension for **key and value** projections.

```python
# Traditional attention (high parameters):
K = Linear(hidden_size → num_kv_heads * head_dim)  # e.g., 7168 → 128*192 = 24,576
V = Linear(hidden_size → num_kv_heads * head_dim)  # e.g., 7168 → 128*192 = 24,576

# DeepSeek-V3 with LoRA (low parameters):
K = Linear(hidden_size → kv_lora_rank) @ Linear(kv_lora_rank → num_kv_heads * head_dim)
    ├─────────────────────────────────┘   ├───────────────────────────────────────────┘
    7168 → 512                            512 → 24,576
    "Down projection"                     "Up projection"

# Parameters saved:
# Traditional: 7168 * 24,576 = 176M parameters
# LoRA: (7168 * 512) + (512 * 24,576) = 3.67M + 12.6M = 16.3M parameters
# Compression: 176M → 16.3M = 90.7% reduction!
```

#### ✅ When to use different values:

```python
# ✅ GOOD: Standard (DeepSeek-V3)
config = DeepseekV3Config(
    hidden_size=7168,
    num_key_value_heads=128,
    kv_lora_rank=512,     # Compress K, V through LoRA
    qk_nope_head_dim=128,
    qk_rope_head_dim=64
)
# Full rank would be: 128 heads * (128+64) = 24,576
# LoRA rank: 512 (98% compression!)

# ✅ GOOD: Higher rank (more capacity)
config = DeepseekV3Config(
    hidden_size=7168,
    kv_lora_rank=1024,    # Less compression
    num_key_value_heads=128
)
# Better accuracy, more parameters

# ✅ GOOD: Lower rank (efficiency)
config = DeepseekV3Config(
    hidden_size=2048,
    kv_lora_rank=256,     # More compression
    num_key_value_heads=32
)
# Fewer parameters, faster inference

# ❌ BAD: Rank too high (defeats purpose)
config = DeepseekV3Config(
    hidden_size=2048,
    num_key_value_heads=32,
    kv_lora_rank=4096,    # Higher than full rank!
    qk_nope_head_dim=128
)
# No benefit, wastes parameters

# 💡 RULE OF THUMB:
# kv_lora_rank should be:
# - Much smaller than num_key_value_heads * head_dim
# - Typically 5-10% of full dimension
# - Higher rank = better quality, more params
```

---

### 9. **`q_lora_rank`** (default: 1536)

Low-rank dimension for **query** projection.

```python
# Traditional Q projection (high parameters):
Q = Linear(hidden_size → num_heads * head_dim)  # 7168 → 128*192 = 24,576

# DeepSeek-V3 with LoRA (low parameters):
Q = Linear(hidden_size → q_lora_rank) @ Linear(q_lora_rank → num_heads * head_dim)
    ├────────────────────────────────┘   ├──────────────────────────────────────┘
    7168 → 1536                          1536 → 24,576
    "Down projection"                    "Up projection"

# Parameters:
# Traditional: 7168 * 24,576 = 176M
# LoRA: (7168 * 1536) + (1536 * 24,576) = 11M + 37.8M = 48.8M
# Compression: 176M → 48.8M = 72.3% reduction
```

#### ✅ When to use different values:

```python
# ✅ GOOD: Standard (DeepSeek-V3)
config = DeepseekV3Config(
    hidden_size=7168,
    num_attention_heads=128,
    q_lora_rank=1536,     # Query compression
    qk_nope_head_dim=128,
    qk_rope_head_dim=64
)
# Full rank: 128 * (128+64) = 24,576
# LoRA rank: 1536 (94% compression)

# ✅ GOOD: Balanced compression
config = DeepseekV3Config(
    hidden_size=4096,
    num_attention_heads=64,
    q_lora_rank=1024,     # ~10% of full rank
    kv_lora_rank=512      # Q rank > KV rank
)
# Q typically needs more capacity than K, V

# ✅ GOOD: Small model
config = DeepseekV3Config(
    hidden_size=1024,
    num_attention_heads=16,
    q_lora_rank=256,
    kv_lora_rank=128
)

# 💡 RELATIONSHIP:
# - q_lora_rank should be >= kv_lora_rank
# - Typical ratio: q_lora_rank ≈ 2-3 × kv_lora_rank
# - Query needs more expressiveness than key/value
```

---

### 10. **`qk_rope_head_dim`** (default: 64)

Dimension of Q/K that receives **rotary position embeddings**.

#### Problem with Pure RoPE
```python
# If ALL dimensions use RoPE:
Q_all_rope = apply_rope(Q, position)
K_all_rope = apply_rope(K, position)

attention_scores = Q_all_rope @ K_all_rope^T

# Issue 1: Position information dominates
# - Nearby tokens get high scores (just because they're near)
# - Content similarity is underweighted
# - Example: "The cat sat on the mat" 
#   "cat" and "sat" are close → high score
#   But "cat" should attend more to "mat" (object-location relationship)

# Issue 2: Long-range dependencies suffer
# - RoPE decays with distance
# - Tokens far apart get low scores regardless of content
# - Example: "The cat ... (100 words) ... the mat"
#   Even if semantically related, position bias reduces attention
```
#### Solution: Hybrid Approach (RoPE + non-RoPE)
```python
# Split Q/K into two parts:
Q = [Q_rope | Q_nope]
    [64 dim | 128 dim]

K = [K_rope | K_nope]
    [64 dim | 128 dim]

# Attention computation:
attention_scores = (Q_rope @ K_rope^T) + (Q_nope @ K_nope^T)
                   ├─────────────────┘   ├────────────────────┘
                   Position-aware         Content-based
                   (relative position)    (semantic similarity)

# Benefits:
# 1. Position-aware: Q_rope ⊗ K_rope captures relative positions
# 2. Content-aware: Q_nope ⊗ K_nope captures semantic similarity
# 3. Balanced: Model learns appropriate weighting
# 4. Flexible: Can attend based on position OR content
```

#### ✅ When to use different values:

```python
# ✅ GOOD: Standard (DeepSeek-V3)
config = DeepseekV3Config(
    qk_rope_head_dim=64,    # 64 dims get RoPE
    qk_nope_head_dim=128,   # 128 dims no RoPE
    v_head_dim=128,
    num_attention_heads=128
)
# Total Q/K head dim = 64 + 128 = 192
# V head dim = 128 (independent)

# ✅ GOOD: All dimensions use RoPE
config = DeepseekV3Config(
    qk_rope_head_dim=128,   # All Q/K uses RoPE
    qk_nope_head_dim=0,     # No non-RoPE component
    v_head_dim=128
)
# Like standard RoPE attention

# ✅ GOOD: No RoPE (absolute positions)
config = DeepseekV3Config(
    qk_rope_head_dim=0,     # No RoPE
    qk_nope_head_dim=192,   # All Q/K is absolute
    v_head_dim=128
)
# Might use learned position embeddings instead

# 💡 DESIGN CHOICE:
# - Separate RoPE and non-RoPE allows hybrid positioning
# - RoPE part: relative position encoding
# - Non-RoPE part: content-based attention
# - DeepSeek-V3 uses hybrid for flexibility
```

---

### 11. **`qk_nope_head_dim`** (default: 128)

Dimension of Q/K that does **NOT** use RoPE (content-based).

```python
# ✅ GOOD: Hybrid attention (DeepSeek-V3)
config = DeepseekV3Config(
    qk_rope_head_dim=64,     # Position-aware
    qk_nope_head_dim=128,    # Content-aware
    v_head_dim=128
)
# Total attention balances position and content

# ✅ GOOD: More position emphasis
config = DeepseekV3Config(
    qk_rope_head_dim=128,    # More position info
    qk_nope_head_dim=64,     # Less content info
    v_head_dim=128
)

# ✅ GOOD: More content emphasis
config = DeepseekV3Config(
    qk_rope_head_dim=32,     # Less position info
    qk_nope_head_dim=160,    # More content info
    v_head_dim=128
)

# 💡 TRADE-OFF:
# - Higher qk_rope_head_dim: Better position awareness
# - Higher qk_nope_head_dim: Better content matching
# - Total: qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
```

---

### 12. **`v_head_dim`** (default: 128)

Dimension of **value** heads (independent from Q/K).

```python
# ✅ GOOD: Standard (DeepSeek-V3)
config = DeepseekV3Config(
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,    # Q/K total = 192
    v_head_dim=128,          # V = 128 (different!)
    num_attention_heads=128
)
# Allows Q/K and V to have different capacities

# ✅ GOOD: Larger value dimension
config = DeepseekV3Config(
    qk_rope_head_dim=64,
    qk_nope_head_dim=64,     # Q/K total = 128
    v_head_dim=256,          # V = 256 (larger)
    num_attention_heads=64
)
# More expressive output representations

# ✅ GOOD: Match Q/K dimension
config = DeepseekV3Config(
    qk_rope_head_dim=64,
    qk_nope_head_dim=64,     # Q/K total = 128
    v_head_dim=128,          # V = 128 (same)
    num_attention_heads=32
)
# Traditional attention (Q, K, V same dim)

# 💡 FLEXIBILITY:
# - DeepSeek-V3 decouples V from Q/K dimensions
# - Allows independent tuning of attention computation
# - v_head_dim affects output representation capacity
```

---

## MoE Parameters

### 13. **`moe_intermediate_size`** (default: 2048)

Hidden dimension of **each expert's FFN** (routed experts).

```python
# ✅ GOOD: DeepSeek-V3 scale
config = DeepseekV3Config(
    hidden_size=7168,
    intermediate_size=18432,      # Dense FFN size
    moe_intermediate_size=2048,   # Each expert FFN size
    n_routed_experts=256,
    num_experts_per_tok=8
)
# Each expert is much smaller than dense FFN
# Total: 2048 * 8 active = 16,384 (comparable to dense)

# ✅ GOOD: Larger experts (more capacity per expert)
config = DeepseekV3Config(
    hidden_size=4096,
    moe_intermediate_size=4096,   # Larger experts
    n_routed_experts=64,
    num_experts_per_tok=4
)

# ✅ GOOD: Smaller experts (more specialization)
config = DeepseekV3Config(
    hidden_size=2048,
    moe_intermediate_size=512,    # Tiny experts
    n_routed_experts=128,
    num_experts_per_tok=8
)

# 💡 TRADE-OFF:
# - Larger moe_intermediate_size: More capacity per expert
# - Smaller moe_intermediate_size: More specialization, more experts needed
# - Total capacity: moe_intermediate_size * num_experts_per_tok
```

---

## Complete Example Configurations

### **Example 1: Small Efficient Model**

```python
config = DeepseekV3Config(
    vocab_size=50000,
    hidden_size=1024,
    intermediate_size=2048,       # Dense FFN
    moe_intermediate_size=512,    # Expert FFN
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,        # GQA (4 KV heads)
    
    # MoE settings
    n_shared_experts=1,
    n_routed_experts=16,
    num_experts_per_tok=4,
    n_group=4,
    topk_group=2,
    routed_scaling_factor=2.0,
    first_k_dense_replace=2,      # First 2 layers dense
    
    # LoRA attention
    kv_lora_rank=128,
    q_lora_rank=256,
    qk_rope_head_dim=32,
    qk_nope_head_dim=64,
    v_head_dim=64,
    
    max_position_embeddings=2048,
    rope_theta=10000.0,
)

model = DeepseekV3Model(config)
print(f"Parameters: {model.num_parameters():,}")
```

---

### **Example 2: Large Production Model (DeepSeek-V3 Style)**

```python
config = DeepseekV3Config(
    vocab_size=129280,
    hidden_size=7168,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    num_hidden_layers=61,
    num_attention_heads=128,
    num_key_value_heads=128,      # MHA
    
    # MoE settings (large scale)
    n_shared_experts=1,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_group=8,
    topk_group=4,
    routed_scaling_factor=2.5,
    first_k_dense_replace=3,
    norm_topk_prob=True,
    
    # LoRA attention
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    
    max_position_embeddings=4096,
    rope_theta=10000.0,
    rope_scaling={"type": "yarn", "factor": 2.0},
)

model = DeepseekV3Model(config)
```

---

### **Example 3: Dense Model (No MoE for Comparison)**

```python
config = DeepseekV3Config(
    vocab_size=50000,
    hidden_size=2048,
    intermediate_size=8192,       # Dense FFN
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=32,
    
    # Disable MoE
    n_shared_experts=0,
    n_routed_experts=1,           # Single "expert" = dense
    num_experts_per_tok=1,
    first_k_dense_replace=24,     # All layers dense
    
    # Standard attention (no LoRA)
    kv_lora_rank=None,  # Will use default or full rank
    q_lora_rank=None,
    qk_rope_head_dim=64,
    qk_nope_head_dim=0,           # All dims use RoPE
    v_head_dim=64,
)
```

---

## 🎯 Key Takeaways

1. **MoE Hierarchy**: `n_routed_experts` > `n_group` > `topk_group` > `num_experts_per_tok`
2. **Expert Selection**: Constrain by groups for load balancing
3. **Dense Layers**: Use `first_k_dense_replace` for stable shallow layers
4. **LoRA Ranks**: `q_lora_rank` ≈ 2-3× `kv_lora_rank` (Q needs more capacity)
5. **Hybrid Attention**: Split Q/K into RoPE (`qk_rope_head_dim`) and content (`qk_nope_head_dim`)
6. **Scaling**: Adjust `routed_scaling_factor` to balance shared vs. routed experts
7. **Efficiency**: More experts + lower `num_experts_per_tok` = sparse, efficient
8. **Specialization**: Lower `moe_intermediate_size` = more specialized experts

# Three Iterations: Finding the Goldilocks Zone

## Water Budget Comparison

| Parameter | Original (Flooding) | First Fix (Too Dry) | Second Fix (Balanced) |
|-----------|---------------------|---------------------|----------------------|
| **alpha_evap** | 0.001 | 0.04 | **0.025** |
| **alpha_seep** | 0.002 | 0.03 | **0.015** |
| **mean_rainfall** | 0.05 | 0.015 | **0.025** |
| **boundary_inflow** | 0.5 | 0.15 | **0.25** |
| **boundary_drainage** | 0% | 10% | **15%** |

## Expected Water Levels

| Iteration | h_mean (core) | Outcome |
|-----------|---------------|---------|
| **Original** | 0→20 runaway | Catastrophic flooding |
| **First Fix** | 0.25-0.45 | Too dry, drought penalties |
| **Second Fix** | **0.5-0.8** | **Optimal dynamics** |

## Threshold Adjustments

| Threshold | Original | First Fix | Second Fix |
|-----------|----------|-----------|------------|
| **h_drought** | - | 0.3 | **0.15** ⬇️ |
| **h_wet** | - | 1.5 | **1.0** ⬇️ |
| **h_flood** | - | 3.0 | **2.5** ⬇️ |
| **h_star** (habitat target) | - | 1.5 | **0.7** ⬇️ |

## Dam-Building Triggers

| Trigger | First Fix | Second Fix |
|---------|-----------|------------|
| **h_std_core >** | 1.0 | **0.3** ⬇️ |
| **h_mean_core >** | 2.0 | **0.6** ⬇️ |
| **Initial stimulus** | 5.0 | **7.0** ⬆️ |

## Penalty Weights

| Penalty | First Fix | Second Fix |
|---------|-----------|------------|
| **Flood (β₁)** | 10.0 | **5.0** ⬇️ |
| **Drought (β₂)** | 8.0 | **4.0** ⬇️ |
| **Failure (β₃)** | 15.0 | **8.0** ⬇️ |

## Expected Outcomes

### Original: COMPLETE FAILURE
- ❌ Water: Runaway flooding
- ❌ Reward: -7500 (collapsed)
- ❌ Wisdom: -15000 (screaming)
- ❌ Dams: 0 (never triggered)
- ❌ Structure: Declining
- ❌ Agents: Paralyzed

### First Fix: PARTIAL SUCCESS
- ✅ Water: Stable (but too low)
- ⚠️ Reward: -2014 (penalties dominate)
- ⚠️ Wisdom: -4040 (still very negative)
- ❌ Dams: 0 (triggers too high)
- ⚠️ Structure: Declining (8.2)
- ✅ Agents: All alive

### Second Fix: EXPECTED FULL SUCCESS
- ✅ Water: Stable at optimal level
- ✅ Reward: Positive or near-zero
- ✅ Wisdom: Informative (-100 to +100)
- ✅ Dams: Built strategically
- ✅ Structure: Stable/growing (8.5-9.5)
- ✅ Agents: Active and rewarded

## Key Lessons from Three Iterations

### 1. Mass Balance is Non-Negotiable
**Original→First**: Had to fix catastrophic imbalance (40x evaporation increase)

### 2. Overcorrection is Common in Complex Systems
**First→Second**: Fixed overcorrection (reduced evaporation by 38%)

### 3. Thresholds Must Match Reality
**Second Fix**: Lowered all thresholds to match actual water dynamics

### 4. Penalties Can Block Emergence
**Second Fix**: Halved penalties to allow positive feedback

### 5. Stimuli Must Be Achievable
**Second Fix**: Lowered triggers so agents can actually respond

## The Goldilocks Principle

```
Too Much Input    → System Saturates  → Flooding
Too Little Input  → System Starves    → Drought
Just Right Input  → System Organizes  → EMERGENCE ✨
```

## Quick Diagnostic Guide

### If water is flooding (h > 2.0):
- ↓ rainfall or ↑ evaporation or ↑ boundary_drainage

### If water is too dry (h < 0.3):
- ↑ rainfall or ↓ evaporation or ↓ boundary_drainage

### If dams not building:
- ↓ dam-building threshold (h_mean_core trigger)
- ↑ initial dam stimulus
- ↓ response threshold (theta_mean)

### If reward very negative:
- ↓ penalty weights (β₁, β₂, β₃)
- ↑ habitat reward weight (α₃)
- Adjust thresholds to match reality

### If no structure emerging:
- Check if water is in optimal range (0.5-0.8)
- Check if dams are being built
- Check if penalties < rewards

## Files Updated in Second Fix

1. ✅ **config.py** - HydrologyConfig, RewardConfig
2. ✅ **environment.py** - Boundary drainage
3. ✅ **overmind.py** - Dam triggers, initial stimuli

**All ready in /mnt/user-data/outputs/beaver_ecosystem/**

---

## Run Command

```bash
python main.py --mode single --steps 1000 --visualize
```

**Expected**: Water ~0.6, dams visible, reward improving, structure emerging! 🎯

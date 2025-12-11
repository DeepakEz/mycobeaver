# Quick Parameter Comparison: Before vs After

## Hydrology Parameters

| Parameter | Before (Broken) | After (Fixed) | Change | Reason |
|-----------|-----------------|---------------|---------|---------|
| **alpha_evap** | 0.001 | 0.04 | **40x increase** | Water was barely evaporating |
| **alpha_seep** | 0.002 | 0.03 | **15x increase** | Minimal seepage loss |
| **mean_rainfall** | 0.05 | 0.015 | **3.3x decrease** | Too much rain |
| **boundary_inflow** | 0.5 | 0.15 | **3.3x decrease** | River input too strong |
| **g0** (conductance) | 0.5 | 2.0 | **4x increase** | Water couldn't flow fast enough |

## New Feature Added

**Boundary Drainage**: 10% of water at grid edges drains per step
- Prevents accumulation at domain boundaries
- Mimics natural outflow to ocean/rivers beyond simulation

## Water Budget Analysis

### BEFORE (Mass Imbalance)
```
Inputs per step:
  Rainfall:    0.05 × 2500 cells    = 125 units
  Inflow:      0.5 × 125 sources    = 62.5 units
  TOTAL INPUT:                      = 187.5 units/step

Outputs per step (at h_avg=1.0):
  Evaporation: 0.001 × 1.0 × 2500   = 2.5 units
  Seepage:     0.002 × 1.0 × 2500   = 5.0 units
  TOTAL OUTPUT:                     = 7.5 units/step

NET BALANCE: +180 units/step → RUNAWAY FLOODING
```

### AFTER (Mass Balance)
```
Inputs per step:
  Rainfall:    0.015 × 2500 cells   = 37.5 units
  Inflow:      0.15 × 125 sources   = 18.75 units
  TOTAL INPUT:                      = 56.25 units/step

Outputs per step (at h_avg=0.5):
  Evaporation: 0.04 × 0.5 × 2500    = 50 units
  Seepage:     0.03 × 0.5 × 2500    = 37.5 units
  Boundary:    0.1 × 0.5 × 400      = 20 units
  TOTAL OUTPUT:                     = 107.5 units/step

EQUILIBRIUM: h stabilizes around 0.3-0.6 where input ≈ output
```

## Dam-Building Stimulus Enhancement

### BEFORE
```python
if h_std_core > 1.0:
    stimulus += 0.5
```
- Only responded to water variability
- Missed uniform flooding

### AFTER
```python
dam_urgency = 0.0
if h_std_core > 1.0:    dam_urgency += 0.5
if h_mean_core > 2.0:   dam_urgency += 0.5  # NEW
if num_flood > 5:       dam_urgency += 1.0  # NEW

new_stimulus += dam_urgency
```
- Responds to absolute water levels
- Responds to active flooding
- Stronger urgency signal (up to +2.0 vs +0.5)

## Expected Behavioral Changes

### Water Dynamics
| Metric | Before | After |
|--------|--------|-------|
| h_mean trajectory | 0 → 20 (runaway) | Stable 0.3-0.8 |
| Flood cells | 0 → 750 (max out) | 0-50 (sporadic) |
| Time to flooding | ~400 steps | Never (if dams work) |

### Agent Behavior
| Behavior | Before | After |
|----------|--------|-------|
| Dam building | Never triggers | Activates when h > 2.0 |
| Task switching | Paralyzed | Responsive |
| Energy management | Linear decay | Oscillates (work/rest) |

### System Metrics
| Metric | Before | After |
|--------|--------|-------|
| Total reward | -7500 | +100 to +500 |
| Wisdom signal | -15000 | -50 to +50 |
| Structural entropy | 10 → 8 (falling) | 8-10 (stable) |
| Overmind β_R | Saturated | Adaptive (1.5-3.5) |

## How to Verify Fix Worked

### Test Command
```bash
python main.py --mode single --steps 1000 --visualize
```

### Look For These Signs

✅ **Water Depth Panel**: Should stabilize, not monotonically increase  
✅ **Flood Cells**: Should stay < 100 throughout  
✅ **Total Reward**: Should become positive after ~200 steps  
✅ **Population**: Should stay 28-30 (not all die)  
✅ **Dam Strength**: Should show dark patches (non-zero) after ~200 steps  
✅ **Physarum Network**: Should show connected paths (not isolated blobs)  
✅ **Overmind β_R**: Should oscillate (not saturate at min/max)  
✅ **Structural Entropy**: Should stabilize 8-10 (not decline to 6)  

### Red Flags (If Still Broken)

❌ Water depth keeps growing past step 500  
❌ Flood cells > 500 at any point  
❌ Reward stays negative < -1000  
❌ All agents die before step 500  
❌ Dam strength stays zero everywhere  
❌ Physarum conductivity all zero  
❌ β_R saturates at 0.5 or 5.0  
❌ Structural entropy declines below 7  

## Quick Tuning Guide

**If water still accumulates**:
- Increase alpha_evap to 0.05
- Increase alpha_seep to 0.04
- Increase boundary drainage from 0.1 to 0.15

**If water disappears too fast**:
- Reduce alpha_evap to 0.03
- Reduce alpha_seep to 0.02
- Increase mean_rainfall to 0.02

**If no dam building**:
- In overmind.py line 51, change 5.0 to 7.0 (higher initial stimulus)
- In config.py, change theta_mean from 5.0 to 3.0 (lower threshold)

**If Physarum network stays zero**:
- In config.py, change alpha_D from 0.5 to 1.0 (faster adaptation)
- Check that boundary_inflow > 0 (needed for flow sources)

## Files Modified

1. ✅ **config.py** (HydrologyConfig section)
2. ✅ **environment.py** (added boundary drainage)
3. ✅ **overmind.py** (enhanced dam stimulus)

## Next Steps

1. Run simulation and verify stability
2. If stable, proceed to ablation studies
3. If still unstable, use tuning guide above
4. Once working, collect data for publication

**All fixed files are in /mnt/user-data/outputs/beaver_ecosystem/**

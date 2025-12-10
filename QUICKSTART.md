# 🚀 QUICK START GUIDE

## Installation (2 minutes)

```bash
# Navigate to the project directory
cd beaver_ecosystem

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Simulation (30 seconds)

```bash
python main.py --mode single --steps 500 --visualize
```

This creates a complete beaver ecosystem with:
- ✅ 30 beaver agents
- ✅ 50×50 grid environment
- ✅ Physarum adaptive network
- ✅ Contemplative Overmind with ACP
- ✅ Complete visualizations

Results will be in: `./output/beaver_ecosystem/`

## What You'll See

After running, check the output directory for:

1. **`final_state.png`** - 12-panel visualization showing:
   - Water depth, vegetation, dams
   - Agent positions
   - Pheromone trails
   - Physarum network
   - Projects
   - Metrics over time

2. **`summary_plots.png`** - Performance analysis:
   - Reward evolution
   - Population dynamics
   - Environmental stability
   - Overmind adaptations

3. **`metrics.csv`** - Complete time series data

## Run Demos (5 minutes)

```bash
python demo.py
```

This runs 5 comprehensive demos showing:
1. Basic simulation
2. Custom configuration
3. Policy comparison
4. Step-by-step execution
5. Component access

## Key Commands

### Compare Greedy vs Contemplative
```bash
python main.py --mode comparison --steps 1000 --runs 5
```

Expected: **15-30% improvement** with contemplative policy

### Ablation Study
```bash
python main.py --mode ablation --steps 500 --runs 3
```

Tests importance of each component (takes ~10 minutes)

### Sensitivity Analysis
```bash
python main.py --mode sensitivity \
    --parameter agent.num_agents \
    --values 10,20,30,40 \
    --steps 500
```

## Understanding the Output

### Key Metrics

- **Reward**: Higher is better (typical range: 50-150)
- **Wisdom Signal**: Ecosystem health (-20 to +50)
- **Structural Entropy**: Network diversity (2-8, higher is more robust)
- **Agents Alive**: Population survival (0-30)

### Overmind Meta-Parameters

Watch these adapt over time:
- **ρ (rho)**: Pheromone evaporation (0.01-0.2)
  - Higher = more exploration
- **β_R (beta_R)**: Recruitment sharpness (0.5-5.0)
  - Lower = more diverse project selection
- **γ_dance (gamma_dance)**: Recruitment gain (0.1-2.0)
  - Higher = stronger coordination

## Customize Your Simulation

Edit parameters in `main.py` or use programmatically:

```python
from config import create_default_config
from simulation import BeaverEcosystemSimulation

# Create custom config
config = create_default_config()
config.agent.num_agents = 50  # More beavers!
config.world.grid_height = 80  # Larger world
config.overmind.lambda_Hs = 3.0  # Favor diversity more

# Run
sim = BeaverEcosystemSimulation(config)
sim.run(1000)
```

## Troubleshooting

### If visualization fails:
```bash
# Install Pillow if missing
pip install Pillow>=8.3.0
```

### If simulation is slow:
- Reduce `--steps` (try 200-500 for quick tests)
- Reduce `agent.num_agents` in config
- Disable visualization: remove `--visualize` flag

### If you get memory errors:
- Reduce grid size: `config.world.grid_height = 30`
- Reduce history: Only affects animation generation

## Next Steps

1. **Read the README.md** for complete documentation
2. **Explore `demo.py`** for programmatic usage examples
3. **Check `analysis.py`** for advanced ablation studies
4. **Modify `config.py`** to experiment with parameters

## Expected Runtime

- Single simulation (500 steps): ~30-60 seconds
- Comparison (1000 steps, 5 runs): ~5 minutes
- Ablation study (500 steps, 15 runs): ~10-15 minutes

## File Structure

```
beaver_ecosystem/
├── QUICKSTART.md       ← You are here
├── README.md           ← Full documentation
├── requirements.txt    ← Dependencies
├── main.py            ← CLI entry point
├── demo.py            ← Interactive demos
├── config.py          ← All parameters
├── simulation.py      ← Main orchestration
├── environment.py     ← Hydrology & ecology
├── physarum.py        ← Adaptive network
├── agents.py          ← Beaver behaviors
├── overmind.py        ← Meta-learning
└── [other modules]    ← Supporting systems
```

## Getting Help

- Check inline documentation in each module
- Read the mathematical specification (§1-19)
- Run `python main.py --help`
- Review demo scripts for usage patterns

---

**You're ready to go! Start with `python main.py --mode single --steps 500 --visualize`**

"""
Analysis Module
Ablation studies and comparative analysis tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pathlib import Path
import json

from config import (
    create_default_config,
    create_greedy_config,
    create_ablation_config
)
from simulation import BeaverEcosystemSimulation


class AblationAnalyzer:
    """
    Conduct ablation studies to assess component contributions
    
    Comparisons:
    1. Full system vs greedy baseline
    2. With/without Physarum
    3. With/without Overmind
    4. With/without ACP
    """
    
    def __init__(self, base_output_dir: str = "./output/ablations"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def run_ablation_study(
        self,
        num_steps: int = 1000,
        num_runs_per_condition: int = 3
    ) -> Dict:
        """
        Run complete ablation study
        
        Conditions:
        1. Full system (contemplative + Physarum + Overmind + ACP)
        2. Greedy baseline (no contemplative policy)
        3. No Physarum (disable adaptive network)
        4. No Overmind (disable meta-parameter adaptation)
        5. No ACP (disable Architect Cognitive Prior)
        """
        print("=" * 80)
        print("BEAVER ECOSYSTEM ABLATION STUDY")
        print("=" * 80)
        
        conditions = {
            'full_system': create_default_config(),
            'greedy_baseline': create_greedy_config(),
            'no_physarum': create_ablation_config(disable_physarum=True),
            'no_overmind': create_ablation_config(disable_overmind=True),
            'no_acp': create_ablation_config(disable_acp=True)
        }
        
        for condition_name, config in conditions.items():
            print(f"\nRunning condition: {condition_name}")
            print("-" * 80)
            
            config.save_directory = str(self.base_output_dir)
            config.experiment_name = condition_name
            config.enable_visualization = False  # Disable for speed
            
            condition_results = []
            
            for run_idx in range(num_runs_per_condition):
                print(f"  Run {run_idx + 1}/{num_runs_per_condition}...")
                
                # Set seed for reproducibility
                config.world.random_seed = 42 + run_idx
                
                # Run simulation
                sim = BeaverEcosystemSimulation(config)
                sim.run(num_steps)
                
                # Extract final metrics
                final_metrics = sim.history[-1]['metrics']
                
                # Extract key performance indicators
                kpis = {
                    'final_reward': final_metrics['reward_total'],
                    'final_num_alive': final_metrics['num_alive'],
                    'mean_reward': np.mean([h['metrics']['reward_total'] for h in sim.history]),
                    'mean_wisdom': np.mean([h['metrics']['wisdom_signal'] for h in sim.history]),
                    'final_structural_entropy': final_metrics['structural_entropy'],
                    'mean_h_std_core': np.mean([h['metrics']['h_std_core'] for h in sim.history]),
                    'num_interventions': sum([h['metrics'].get('intervention', 0) for h in sim.history])
                }
                
                condition_results.append(kpis)
            
            # Aggregate results across runs
            aggregated = self._aggregate_runs(condition_results)
            self.results[condition_name] = aggregated
            
            print(f"  Results: {aggregated}")
        
        # Compare conditions
        self._compare_conditions()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _aggregate_runs(self, run_results: List[Dict]) -> Dict:
        """Aggregate metrics across runs"""
        aggregated = {}
        
        # Get all keys
        keys = run_results[0].keys()
        
        for key in keys:
            values = [r[key] for r in run_results]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def _compare_conditions(self) -> None:
        """Compare conditions and print summary"""
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)
        
        # Extract mean final rewards
        conditions = list(self.results.keys())
        mean_rewards = [self.results[c]['final_reward_mean'] for c in conditions]
        
        # Sort by performance
        sorted_indices = np.argsort(mean_rewards)[::-1]
        sorted_conditions = [conditions[i] for i in sorted_indices]
        sorted_rewards = [mean_rewards[i] for i in sorted_indices]
        
        print("\nRanking by Mean Final Reward:")
        for rank, (condition, reward) in enumerate(zip(sorted_conditions, sorted_rewards), 1):
            print(f"  {rank}. {condition:20s}: {reward:8.2f}")
        
        # Compute relative improvements
        baseline_reward = self.results['greedy_baseline']['final_reward_mean']
        full_system_reward = self.results['full_system']['final_reward_mean']
        
        print(f"\nRelative Performance:")
        print(f"  Baseline (greedy): {baseline_reward:.2f}")
        print(f"  Full system: {full_system_reward:.2f}")
        print(f"  Improvement: {(full_system_reward / baseline_reward - 1) * 100:.1f}%")
        
        # Component contributions
        print(f"\nComponent Contributions (vs full system):")
        for condition in ['no_physarum', 'no_overmind', 'no_acp']:
            reward = self.results[condition]['final_reward_mean']
            degradation = (1 - reward / full_system_reward) * 100
            print(f"  {condition:20s}: {degradation:6.1f}% degradation")
    
    def _save_results(self) -> None:
        """Save results to file"""
        # Save as JSON
        results_path = self.base_output_dir / 'ablation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
        # Create comparison plots
        self._create_comparison_plots()
    
    def _create_comparison_plots(self) -> None:
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        conditions = list(self.results.keys())
        
        # 1. Final reward comparison
        ax = axes[0, 0]
        mean_rewards = [self.results[c]['final_reward_mean'] for c in conditions]
        std_rewards = [self.results[c]['final_reward_std'] for c in conditions]
        
        x = np.arange(len(conditions))
        ax.bar(x, mean_rewards, yerr=std_rewards, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_title('Final Reward Comparison')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        # 2. Survival comparison
        ax = axes[0, 1]
        mean_alive = [self.results[c]['final_num_alive_mean'] for c in conditions]
        std_alive = [self.results[c]['final_num_alive_std'] for c in conditions]
        
        ax.bar(x, mean_alive, yerr=std_alive, capsize=5, color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_title('Final Population Size')
        ax.set_ylabel('Num Alive')
        ax.grid(True, alpha=0.3)
        
        # 3. Structural entropy comparison
        ax = axes[1, 0]
        mean_entropy = [self.results[c]['final_structural_entropy_mean'] for c in conditions]
        std_entropy = [self.results[c]['final_structural_entropy_std'] for c in conditions]
        
        ax.bar(x, mean_entropy, yerr=std_entropy, capsize=5, color='orange', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_title('Structural Entropy')
        ax.set_ylabel('Entropy')
        ax.grid(True, alpha=0.3)
        
        # 4. Stability comparison
        ax = axes[1, 1]
        mean_h_std = [self.results[c]['mean_h_std_core_mean'] for c in conditions]
        std_h_std = [self.results[c]['mean_h_std_core_std'] for c in conditions]
        
        ax.bar(x, mean_h_std, yerr=std_h_std, capsize=5, color='blue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_title('Hydrological Stability (lower is better)')
        ax.set_ylabel('Water Depth Std')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.base_output_dir / 'ablation_comparison.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plots saved to {plot_path}")
        
        plt.close(fig)


def run_parameter_sensitivity_analysis(
    parameter_name: str,
    parameter_values: List[float],
    num_steps: int = 500,
    num_runs: int = 3,
    output_dir: str = "./output/sensitivity"
) -> pd.DataFrame:
    """
    Run parameter sensitivity analysis
    
    Args:
        parameter_name: Name of parameter to vary (e.g., 'agent.num_agents')
        parameter_values: List of values to test
        num_steps: Number of simulation steps
        num_runs: Number of runs per parameter value
        output_dir: Output directory
    
    Returns:
        DataFrame with results
    """
    print(f"\nParameter Sensitivity Analysis: {parameter_name}")
    print("=" * 80)
    
    results = []
    
    for param_value in parameter_values:
        print(f"\nTesting {parameter_name} = {param_value}")
        
        for run_idx in range(num_runs):
            # Create config
            config = create_default_config()
            config.save_directory = output_dir
            config.experiment_name = f"{parameter_name}_{param_value}_run{run_idx}"
            config.enable_visualization = False
            config.world.random_seed = 42 + run_idx
            
            # Set parameter
            param_parts = parameter_name.split('.')
            obj = config
            for part in param_parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, param_parts[-1], param_value)
            
            # Run simulation
            sim = BeaverEcosystemSimulation(config)
            sim.run(num_steps)
            
            # Extract metrics
            final_metrics = sim.history[-1]['metrics']
            
            results.append({
                'parameter': parameter_name,
                'value': param_value,
                'run': run_idx,
                'final_reward': final_metrics['reward_total'],
                'final_num_alive': final_metrics['num_alive'],
                'mean_reward': np.mean([h['metrics']['reward_total'] for h in sim.history]),
                'mean_wisdom': np.mean([h['metrics']['wisdom_signal'] for h in sim.history])
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save
    output_path = Path(output_dir) / f"sensitivity_{parameter_name.replace('.', '_')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    
    # Plot
    _plot_sensitivity(df, parameter_name, output_dir)
    
    return df


def _plot_sensitivity(df: pd.DataFrame, parameter_name: str, output_dir: str) -> None:
    """Plot sensitivity analysis results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by parameter value
    grouped = df.groupby('value')
    
    values = []
    mean_rewards = []
    std_rewards = []
    mean_alive = []
    std_alive = []
    
    for value, group in grouped:
        values.append(value)
        mean_rewards.append(group['mean_reward'].mean())
        std_rewards.append(group['mean_reward'].std())
        mean_alive.append(group['final_num_alive'].mean())
        std_alive.append(group['final_num_alive'].std())
    
    # Plot 1: Mean reward
    ax = axes[0]
    ax.errorbar(values, mean_rewards, yerr=std_rewards, marker='o', capsize=5)
    ax.set_title(f'Sensitivity: {parameter_name} vs Mean Reward')
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Mean Reward')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final population
    ax = axes[1]
    ax.errorbar(values, mean_alive, yerr=std_alive, marker='s', capsize=5, color='green')
    ax.set_title(f'Sensitivity: {parameter_name} vs Population')
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Final Num Alive')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / f"sensitivity_{parameter_name.replace('.', '_')}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Sensitivity plot saved to {plot_path}")
    
    plt.close(fig)

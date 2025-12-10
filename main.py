"""
Main Entry Point
Beaver Ecosystem Simulation with Contemplative Overmind and Architect Cognitive Prior

Usage:
    python main.py --mode single --steps 1000 --visualize
    python main.py --mode ablation --steps 500 --runs 3
    python main.py --mode sensitivity --parameter agent.num_agents --values 10,20,30,40
"""

import argparse
import sys
from pathlib import Path

from config import (
    create_default_config,
    create_greedy_config,
    create_ablation_config
)
from simulation import BeaverEcosystemSimulation
from visualization import EcosystemVisualizer, create_summary_plots
from analysis import AblationAnalyzer, run_parameter_sensitivity_analysis


def run_single_simulation(args):
    """Run single simulation"""
    print("\n" + "=" * 80)
    print("BEAVER ECOSYSTEM SINGLE SIMULATION")
    print("=" * 80)
    
    # Create configuration
    if args.policy == 'greedy':
        config = create_greedy_config()
    else:
        config = create_default_config()
    
    config.world.max_steps = args.steps
    config.enable_visualization = args.visualize
    config.experiment_name = args.name
    
    # Create and run simulation
    sim = BeaverEcosystemSimulation(config)
    sim.run()
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        
        # Current state visualization
        visualizer = EcosystemVisualizer(sim)
        state = sim.get_current_state()
        visualizer.visualize_state(state)
        visualizer.save_current_state('final_state.png')
        
        # Summary plots
        create_summary_plots(sim)
        
        print(f"Visualizations saved to {sim.output_dir}")
    
    print("\nSimulation complete!")
    print(f"Results saved to: {sim.output_dir}")
    
    # Print final stats
    if sim.history:
        final_metrics = sim.history[-1]['metrics']
        print("\nFinal Statistics:")
        print(f"  Reward: {final_metrics['reward_total']:.2f}")
        print(f"  Wisdom: {final_metrics['wisdom_signal']:.2f}")
        print(f"  Agents Alive: {final_metrics['num_alive']}")
        print(f"  Structural Entropy: {final_metrics['structural_entropy']:.3f}")


def run_ablation_study(args):
    """Run ablation study"""
    analyzer = AblationAnalyzer(args.output_dir)
    analyzer.run_ablation_study(
        num_steps=args.steps,
        num_runs_per_condition=args.runs
    )


def run_sensitivity_analysis(args):
    """Run parameter sensitivity analysis"""
    # Parse parameter values
    try:
        values = [float(v) for v in args.values.split(',')]
    except:
        print(f"Error: Could not parse values '{args.values}'")
        print("Expected format: --values 10,20,30,40")
        return
    
    run_parameter_sensitivity_analysis(
        parameter_name=args.parameter,
        parameter_values=values,
        num_steps=args.steps,
        num_runs=args.runs,
        output_dir=args.output_dir
    )


def run_comparison(args):
    """Run comparison between greedy and contemplative"""
    print("\n" + "=" * 80)
    print("GREEDY vs CONTEMPLATIVE COMPARISON")
    print("=" * 80)
    
    results = {}
    
    for policy_name in ['greedy', 'contemplative']:
        print(f"\nRunning {policy_name} policy...")
        
        if policy_name == 'greedy':
            config = create_greedy_config()
        else:
            config = create_default_config()
        
        config.world.max_steps = args.steps
        config.enable_visualization = False
        config.experiment_name = f"comparison_{policy_name}"
        config.save_directory = args.output_dir
        
        # Run multiple times
        policy_results = []
        
        for run_idx in range(args.runs):
            config.world.random_seed = 42 + run_idx
            
            sim = BeaverEcosystemSimulation(config)
            sim.run()
            
            # Extract final metrics
            final_metrics = sim.history[-1]['metrics']
            policy_results.append({
                'final_reward': final_metrics['reward_total'],
                'final_alive': final_metrics['num_alive'],
                'mean_reward': sum(h['metrics']['reward_total'] for h in sim.history) / len(sim.history)
            })
        
        # Aggregate
        import numpy as np
        results[policy_name] = {
            'mean_final_reward': np.mean([r['final_reward'] for r in policy_results]),
            'std_final_reward': np.std([r['final_reward'] for r in policy_results]),
            'mean_alive': np.mean([r['final_alive'] for r in policy_results]),
            'mean_reward': np.mean([r['mean_reward'] for r in policy_results])
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for policy_name, metrics in results.items():
        print(f"\n{policy_name.upper()}:")
        print(f"  Final Reward: {metrics['mean_final_reward']:.2f} ± {metrics['std_final_reward']:.2f}")
        print(f"  Final Alive: {metrics['mean_alive']:.1f}")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
    
    # Compute improvement
    improvement = (
        results['contemplative']['mean_final_reward'] /
        results['greedy']['mean_final_reward'] - 1
    ) * 100
    
    print(f"\nImprovement: {improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Beaver Ecosystem Simulation with Contemplative Overmind',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single simulation
  python main.py --mode single --steps 1000 --visualize
  
  # Run with greedy baseline
  python main.py --mode single --policy greedy --steps 1000
  
  # Run ablation study
  python main.py --mode ablation --steps 500 --runs 3
  
  # Run sensitivity analysis
  python main.py --mode sensitivity --parameter agent.num_agents --values 10,20,30,40
  
  # Compare greedy vs contemplative
  python main.py --mode comparison --steps 1000 --runs 5
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'ablation', 'sensitivity', 'comparison'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of simulation steps'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization (single mode only)'
    )
    
    parser.add_argument(
        '--policy',
        type=str,
        default='contemplative',
        choices=['greedy', 'contemplative'],
        help='Policy type (single mode only)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='beaver_ecosystem',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of runs per condition (ablation/sensitivity/comparison modes)'
    )
    
    parser.add_argument(
        '--parameter',
        type=str,
        help='Parameter name for sensitivity analysis (e.g., agent.num_agents)'
    )
    
    parser.add_argument(
        '--values',
        type=str,
        help='Comma-separated parameter values for sensitivity analysis'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function
    try:
        if args.mode == 'single':
            run_single_simulation(args)
        
        elif args.mode == 'ablation':
            run_ablation_study(args)
        
        elif args.mode == 'sensitivity':
            if not args.parameter or not args.values:
                print("Error: --parameter and --values required for sensitivity mode")
                parser.print_help()
                return
            run_sensitivity_analysis(args)
        
        elif args.mode == 'comparison':
            run_comparison(args)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

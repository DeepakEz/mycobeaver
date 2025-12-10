"""
Demo Script
Shows how to use the beaver ecosystem simulation programmatically
"""

import numpy as np
import matplotlib.pyplot as plt

from config import create_default_config, create_greedy_config
from simulation import BeaverEcosystemSimulation
from visualization import EcosystemVisualizer, create_summary_plots


def demo_basic_simulation():
    """Demo 1: Basic simulation with default settings"""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Simulation")
    print("=" * 80)
    
    # Create default configuration
    config = create_default_config()
    config.world.max_steps = 200  # Short run for demo
    config.experiment_name = "demo_basic"
    
    # Create and run simulation
    sim = BeaverEcosystemSimulation(config)
    print("\nRunning simulation for 200 steps...")
    sim.run(200)
    
    # Get final state
    state = sim.get_current_state()
    
    print("\nFinal State:")
    print(f"  Step: {state.step}")
    print(f"  Time: {state.time:.2f}")
    print(f"  Agents Alive: {state.metrics['num_alive']}")
    print(f"  Total Reward: {state.metrics['reward_total']:.2f}")
    print(f"  Wisdom Signal: {state.wisdom_signal:.2f}")
    print(f"  Structural Entropy: {state.metrics['structural_entropy']:.3f}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualizer = EcosystemVisualizer(sim)
    visualizer.visualize_state(state)
    visualizer.save_current_state('demo_basic_final.png')
    
    # Create summary plots
    create_summary_plots(sim)
    
    print(f"\nResults saved to: {sim.output_dir}")


def demo_custom_configuration():
    """Demo 2: Simulation with custom configuration"""
    print("\n" + "=" * 80)
    print("DEMO 2: Custom Configuration")
    print("=" * 80)
    
    # Create custom configuration
    config = create_default_config()
    
    # Modify parameters
    config.world.grid_height = 30
    config.world.grid_width = 30
    config.agent.num_agents = 20
    config.world.max_steps = 300
    config.experiment_name = "demo_custom"
    
    # Adjust Overmind to be more aggressive about diversity
    config.overmind.lambda_Hs = 2.0  # Double structural entropy reward
    config.overmind.lambda_B_brittle = 3.0  # Triple brittleness penalty
    
    print("\nCustom Parameters:")
    print(f"  Grid Size: {config.world.grid_height}×{config.world.grid_width}")
    print(f"  Num Agents: {config.agent.num_agents}")
    print(f"  Structural Entropy Reward: {config.overmind.lambda_Hs}")
    print(f"  Brittleness Penalty: {config.overmind.lambda_B_brittle}")
    
    # Run simulation
    sim = BeaverEcosystemSimulation(config)
    sim.run(300)
    
    # Extract metrics
    final_metrics = sim.history[-1]['metrics']
    print("\nFinal Metrics:")
    print(f"  Reward: {final_metrics['reward_total']:.2f}")
    print(f"  Structural Entropy: {final_metrics['structural_entropy']:.3f}")
    print(f"  Brittleness: {final_metrics['brittleness']:.3f}")


def demo_compare_policies():
    """Demo 3: Compare greedy vs contemplative policies"""
    print("\n" + "=" * 80)
    print("DEMO 3: Policy Comparison")
    print("=" * 80)
    
    results = {}
    
    for policy_name in ['greedy', 'contemplative']:
        print(f"\nRunning {policy_name} policy...")
        
        if policy_name == 'greedy':
            config = create_greedy_config()
        else:
            config = create_default_config()
        
        config.world.max_steps = 500
        config.experiment_name = f"demo_{policy_name}"
        config.enable_visualization = False
        
        # Run simulation
        sim = BeaverEcosystemSimulation(config)
        sim.run(500)
        
        # Extract results
        rewards = [h['metrics']['reward_total'] for h in sim.history]
        alive = [h['metrics']['num_alive'] for h in sim.history]
        
        results[policy_name] = {
            'rewards': rewards,
            'alive': alive,
            'final_reward': rewards[-1],
            'mean_reward': np.mean(rewards),
            'final_alive': alive[-1]
        }
        
        print(f"  Final Reward: {results[policy_name]['final_reward']:.2f}")
        print(f"  Mean Reward: {results[policy_name]['mean_reward']:.2f}")
        print(f"  Final Alive: {results[policy_name]['final_alive']}")
    
    # Compare
    print("\n" + "-" * 80)
    print("COMPARISON:")
    print("-" * 80)
    
    improvement = (
        results['contemplative']['mean_reward'] / 
        results['greedy']['mean_reward'] - 1
    ) * 100
    
    print(f"Contemplative vs Greedy:")
    print(f"  Reward Improvement: {improvement:.1f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = range(len(results['greedy']['rewards']))
    
    # Reward comparison
    ax = axes[0]
    ax.plot(steps, results['greedy']['rewards'], 'r-', linewidth=2, label='Greedy')
    ax.plot(steps, results['contemplative']['rewards'], 'b-', linewidth=2, label='Contemplative')
    ax.set_title('Reward Comparison')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Population comparison
    ax = axes[1]
    ax.plot(steps, results['greedy']['alive'], 'r-', linewidth=2, label='Greedy')
    ax.plot(steps, results['contemplative']['alive'], 'b-', linewidth=2, label='Contemplative')
    ax.set_title('Population Comparison')
    ax.set_xlabel('Step')
    ax.set_ylabel('Agents Alive')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/demo_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: ./output/demo_comparison.png")
    
    plt.close(fig)


def demo_step_by_step():
    """Demo 4: Step-by-step simulation with custom intervention"""
    print("\n" + "=" * 80)
    print("DEMO 4: Step-by-Step Execution")
    print("=" * 80)
    
    # Create simulation
    config = create_default_config()
    config.world.max_steps = 100
    config.experiment_name = "demo_step_by_step"
    
    sim = BeaverEcosystemSimulation(config)
    
    print("\nExecuting steps manually...")
    
    # Run first 50 steps
    for i in range(50):
        metrics = sim.step()
        
        if i % 10 == 0:
            print(f"  Step {i}: Reward={metrics['reward_total']:.2f}, "
                  f"Alive={metrics['num_alive']}, "
                  f"Wisdom={metrics['wisdom_signal']:.2f}")
    
    # Inject a disturbance at step 50
    print("\n  >>> Injecting disturbance: Removing water from core habitat")
    core_cells = list(sim.environment.core_habitat)
    for cell in core_cells[:10]:  # Drain 10 cells
        sim.environment.state.h[cell] *= 0.3
    
    # Continue for another 50 steps
    print("\nContinuing after disturbance...")
    for i in range(50, 100):
        metrics = sim.step()
        
        if i % 10 == 0:
            print(f"  Step {i}: Reward={metrics['reward_total']:.2f}, "
                  f"Alive={metrics['num_alive']}, "
                  f"Wisdom={metrics['wisdom_signal']:.2f}")
    
    print("\nStep-by-step execution complete!")
    
    # Check if Overmind intervened
    interventions = sum(h['metrics'].get('intervention', 0) for h in sim.history)
    print(f"Overmind interventions: {int(interventions)}")


def demo_access_components():
    """Demo 5: Direct access to simulation components"""
    print("\n" + "=" * 80)
    print("DEMO 5: Accessing Components")
    print("=" * 80)
    
    # Create simulation
    config = create_default_config()
    config.world.max_steps = 100
    config.experiment_name = "demo_components"
    
    sim = BeaverEcosystemSimulation(config)
    sim.run(100)
    
    # Access environment
    print("\nEnvironment State:")
    print(f"  Mean water depth: {np.mean(sim.environment.state.h):.3f}")
    print(f"  Mean vegetation: {np.mean(sim.environment.state.v):.3f}")
    print(f"  Mean soil moisture: {np.mean(sim.environment.state.m):.3f}")
    
    # Access Physarum network
    print("\nPhysarum Network:")
    from config import CommodityType
    if CommodityType.WATER in sim.physarum.commodities:
        entropy = sim.physarum.get_structural_entropy()
        print(f"  Structural entropy: {entropy:.3f}")
        is_degenerate = sim.physarum.detect_degenerate_network()
        print(f"  Network degenerate: {is_degenerate}")
    
    # Access agents
    print("\nAgent Population:")
    alive_agents = sim.population.get_alive_agents()
    print(f"  Num alive: {len(alive_agents)}")
    
    if alive_agents:
        energies = [a.state.energy for a in alive_agents]
        print(f"  Mean energy: {np.mean(energies):.2f}")
        print(f"  Energy range: [{np.min(energies):.2f}, {np.max(energies):.2f}]")
    
    # Access Overmind
    print("\nOvermind State:")
    overmind_state = sim.overmind.get_overmind_state()
    print(f"  ρ (evaporation): {overmind_state['rho']:.3f}")
    print(f"  β_R (sharpness): {overmind_state['beta_R']:.3f}")
    print(f"  Wisdom trend: {overmind_state['wisdom_trend']:.3f}")
    
    # Access projects
    print("\nProjects:")
    for project in sim.projects.state.projects[:3]:  # Show first 3
        print(f"  Project {project.id}: Q={project.quality:.2f}, R={project.recruitment:.2f}")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("BEAVER ECOSYSTEM SIMULATION - DEMOS")
    print("=" * 80)
    
    try:
        # Run demos
        demo_basic_simulation()
        
        demo_custom_configuration()
        
        demo_compare_policies()
        
        demo_step_by_step()
        
        demo_access_components()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETE!")
        print("=" * 80)
        print("\nCheck the ./output/ directory for results and visualizations.")
    
    except KeyboardInterrupt:
        print("\n\nDemos interrupted by user")
    
    except Exception as e:
        print(f"\nError running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

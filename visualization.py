"""
Visualization Module
Comprehensive visualization for beaver ecosystem
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Optional
from pathlib import Path

from simulation import BeaverEcosystemSimulation, SimulationState


class EcosystemVisualizer:
    """
    Comprehensive visualization for beaver ecosystem
    
    Features:
    - Environment state (water, vegetation, dams)
    - Agent positions
    - Pheromone fields
    - Physarum network
    - Project locations
    - Metrics over time
    """
    
    def __init__(self, simulation: BeaverEcosystemSimulation):
        self.sim = simulation
        
        # Figure setup
        self.fig = None
        self.axes = {}
    
    def create_figure(self) -> plt.Figure:
        """Create comprehensive visualization figure"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main spatial visualizations (top 2 rows)
        self.axes['water'] = fig.add_subplot(gs[0, 0])
        self.axes['vegetation'] = fig.add_subplot(gs[0, 1])
        self.axes['dams'] = fig.add_subplot(gs[0, 2])
        self.axes['agents'] = fig.add_subplot(gs[0, 3])
        
        self.axes['pheromones'] = fig.add_subplot(gs[1, 0])
        self.axes['physarum'] = fig.add_subplot(gs[1, 1])
        self.axes['moisture'] = fig.add_subplot(gs[1, 2])
        self.axes['projects'] = fig.add_subplot(gs[1, 3])
        
        # Metrics (bottom row)
        self.axes['reward'] = fig.add_subplot(gs[2, 0])
        self.axes['population'] = fig.add_subplot(gs[2, 1])
        self.axes['wisdom'] = fig.add_subplot(gs[2, 2])
        self.axes['overmind'] = fig.add_subplot(gs[2, 3])
        
        self.fig = fig
        return fig
    
    def visualize_state(self, state: SimulationState) -> None:
        """Visualize current simulation state"""
        if self.fig is None:
            self.create_figure()
        
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # Get 2D views
        h_2d, v_2d, m_2d, d_2d, L_2d, z_2d = state.environment.get_state_2d()
        
        # 1. Water depth
        ax = self.axes['water']
        im = ax.imshow(h_2d, cmap='Blues', aspect='auto')
        ax.set_title(f'Water Depth (Step {state.step})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Depth')
        
        # 2. Vegetation
        ax = self.axes['vegetation']
        im = ax.imshow(v_2d, cmap='Greens', aspect='auto')
        ax.set_title('Vegetation Biomass')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Biomass')
        
        # 3. Dams
        ax = self.axes['dams']
        dam_strength = 1 - d_2d  # Show dam strength (1 - permeability)
        im = ax.imshow(dam_strength, cmap='Reds', aspect='auto')
        ax.set_title('Dam Strength')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Strength')
        
        # 4. Agents
        ax = self.axes['agents']
        # Background: elevation
        ax.imshow(z_2d, cmap='terrain', aspect='auto', alpha=0.3)
        # Agent positions
        agent_positions = state.population.get_agent_positions()
        if agent_positions:
            rows, cols = zip(*agent_positions)
            ax.scatter(cols, rows, c='red', s=50, marker='o', edgecolors='black')
        ax.set_title(f'Agents (n={len(agent_positions)})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 5. Pheromones
        ax = self.axes['pheromones']
        pheromone_2d = state.pheromones.get_pheromone_2d(
            state.environment.H,
            state.environment.W
        )
        im = ax.imshow(pheromone_2d, cmap='hot', aspect='auto')
        ax.set_title('Pheromone Trails')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Concentration')
        
        # 6. Physarum network
        ax = self.axes['physarum']
        from config import CommodityType
        if CommodityType.WATER in state.physarum.commodities:
            conductivity_2d = state.physarum.get_conductivity_2d(CommodityType.WATER)
            im = ax.imshow(conductivity_2d, cmap='viridis', aspect='auto')
            ax.set_title('Physarum Network (Water)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Conductivity')
        
        # 7. Soil moisture
        ax = self.axes['moisture']
        im = ax.imshow(m_2d, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Soil Moisture')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Moisture')
        
        # 8. Projects
        ax = self.axes['projects']
        # Background: elevation
        ax.imshow(z_2d, cmap='terrain', aspect='auto', alpha=0.3)
        # Project centers and regions
        for project in state.projects.state.projects:
            center_row, center_col = project.center
            # Draw region
            for cell_idx in list(project.region)[:20]:  # Limit for visibility
                row, col = state.environment._index_to_coords(cell_idx)
                ax.plot(col, row, 'o', color='orange', alpha=0.3, markersize=3)
            # Draw center
            ax.plot(center_col, center_row, 's', color='red', markersize=10,
                   label=f'P{project.id}: Q={project.quality:.2f}, R={project.recruitment:.2f}')
        ax.set_title('Projects')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if state.projects.state.projects:
            ax.legend(loc='upper right', fontsize=6)
        
        # 9. Reward history
        if len(self.sim.history) > 1:
            ax = self.axes['reward']
            steps = [h['step'] for h in self.sim.history]
            rewards = [h['metrics']['reward_total'] for h in self.sim.history]
            ax.plot(steps, rewards, 'b-', linewidth=2)
            ax.set_title('Global Reward')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # 10. Population
        if len(self.sim.history) > 1:
            ax = self.axes['population']
            steps = [h['step'] for h in self.sim.history]
            num_alive = [h['metrics']['num_alive'] for h in self.sim.history]
            mean_energy = [h['metrics']['mean_energy'] for h in self.sim.history]
            
            ax2 = ax.twinx()
            ax.plot(steps, num_alive, 'g-', linewidth=2, label='Num Alive')
            ax2.plot(steps, mean_energy, 'r--', linewidth=2, label='Mean Energy')
            
            ax.set_title('Population Dynamics')
            ax.set_xlabel('Step')
            ax.set_ylabel('Num Alive', color='g')
            ax2.set_ylabel('Mean Energy', color='r')
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # 11. Wisdom signal
        if len(self.sim.history) > 1:
            ax = self.axes['wisdom']
            steps = [h['step'] for h in self.sim.history]
            wisdom = [h['metrics']['wisdom_signal'] for h in self.sim.history]
            ax.plot(steps, wisdom, 'm-', linewidth=2)
            ax.set_title('Wisdom Signal')
            ax.set_xlabel('Step')
            ax.set_ylabel('Wisdom')
            ax.grid(True, alpha=0.3)
        
        # 12. Overmind meta-parameters
        if len(self.sim.history) > 1:
            ax = self.axes['overmind']
            steps = [h['step'] for h in self.sim.history]
            
            # Extract overmind parameters (if available)
            try:
                rho = [h['metrics'].get('overmind_rho', 0) for h in self.sim.history]
                beta_R = [h['metrics'].get('overmind_beta_R', 0) for h in self.sim.history]
                
                ax.plot(steps, rho, 'b-', linewidth=2, label='ρ (evaporation)')
                ax.plot(steps, beta_R, 'r-', linewidth=2, label='β_R (sharpness)')
                
                ax.set_title('Overmind Meta-Parameters')
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'No Overmind data', ha='center', va='center',
                       transform=ax.transAxes)
        
        plt.suptitle(
            f'Beaver Ecosystem | Step {state.step} | Time {state.time:.1f} | '
            f'Alive: {state.metrics["num_alive"]} | '
            f'Reward: {state.metrics["reward_total"]:.2f} | '
            f'Wisdom: {state.wisdom_signal:.2f}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
    
    def save_current_state(self, filename: str) -> None:
        """Save current visualization to file"""
        if self.fig is not None:
            filepath = self.sim.output_dir / filename
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.sim.logger.info(f"Saved visualization to {filepath}")
    
    def create_animation(
        self,
        states: List[SimulationState],
        output_path: Path,
        fps: int = 10
    ) -> None:
        """Create animation from list of states"""
        self.sim.logger.info(f"Creating animation with {len(states)} frames...")
        
        # Create figure
        self.create_figure()
        
        def update(frame_idx):
            state = states[frame_idx]
            self.visualize_state(state)
            return self.fig,
        
        anim = FuncAnimation(
            self.fig,
            update,
            frames=len(states),
            interval=1000//fps,
            blit=False
        )
        
        # Save
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        
        self.sim.logger.info(f"Animation saved to {output_path}")
        
        plt.close(self.fig)


def create_summary_plots(simulation: BeaverEcosystemSimulation) -> None:
    """Create summary plots from simulation history"""
    if len(simulation.history) == 0:
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    steps = [h['step'] for h in simulation.history]
    
    # Row 1: Rewards
    ax = axes[0, 0]
    reward_total = [h['metrics']['reward_total'] for h in simulation.history]
    ax.plot(steps, reward_total, 'b-', linewidth=2)
    ax.set_title('Total Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    reward_survival = [h['metrics']['reward_survival'] for h in simulation.history]
    reward_habitat = [h['metrics']['reward_habitat'] for h in simulation.history]
    ax.plot(steps, reward_survival, 'g-', linewidth=2, label='Survival')
    ax.plot(steps, reward_habitat, 'm-', linewidth=2, label='Habitat')
    ax.set_title('Reward Components')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    wisdom = [h['metrics']['wisdom_signal'] for h in simulation.history]
    ax.plot(steps, wisdom, 'm-', linewidth=2)
    ax.set_title('Wisdom Signal')
    ax.set_xlabel('Step')
    ax.set_ylabel('Wisdom')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Environment
    ax = axes[1, 0]
    h_mean = [h['metrics']['h_mean_core'] for h in simulation.history]
    h_std = [h['metrics']['h_std_core'] for h in simulation.history]
    ax2 = ax.twinx()
    ax.plot(steps, h_mean, 'b-', linewidth=2, label='Mean')
    ax2.plot(steps, h_std, 'r--', linewidth=2, label='Std')
    ax.set_title('Water Depth (Core)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Depth', color='b')
    ax2.set_ylabel('Std Depth', color='r')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    num_flood = [h['metrics']['num_flood_cells'] for h in simulation.history]
    ax.plot(steps, num_flood, 'r-', linewidth=2)
    ax.set_title('Flood Cells')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    structural_entropy = [h['metrics']['structural_entropy'] for h in simulation.history]
    ax.plot(steps, structural_entropy, 'g-', linewidth=2)
    ax.set_title('Structural Entropy')
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.grid(True, alpha=0.3)
    
    # Row 3: Population and Overmind
    ax = axes[2, 0]
    num_alive = [h['metrics']['num_alive'] for h in simulation.history]
    ax.plot(steps, num_alive, 'g-', linewidth=2)
    ax.set_title('Population Size')
    ax.set_xlabel('Step')
    ax.set_ylabel('Num Alive')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    mean_energy = [h['metrics']['mean_energy'] for h in simulation.history]
    mean_satiety = [h['metrics']['mean_satiety'] for h in simulation.history]
    ax.plot(steps, mean_energy, 'r-', linewidth=2, label='Energy')
    ax.plot(steps, mean_satiety, 'b-', linewidth=2, label='Satiety')
    ax.set_title('Agent State')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 2]
    rho = [h['metrics'].get('overmind_rho', 0) for h in simulation.history]
    beta_R = [h['metrics'].get('overmind_beta_R', 0) for h in simulation.history]
    ax.plot(steps, rho, 'b-', linewidth=2, label='ρ')
    ax.plot(steps, beta_R, 'r-', linewidth=2, label='β_R')
    ax.set_title('Overmind Parameters')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = simulation.output_dir / 'summary_plots.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    simulation.logger.info(f"Summary plots saved to {output_path}")
    
    plt.close(fig)

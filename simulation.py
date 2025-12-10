"""
Simulation Module
Main simulation orchestration bringing together all components (§16)
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from config import (
    SimulationConfig, CommodityType, TaskType,
    create_default_config
)
from environment import Environment
from physarum import PhysarumNetwork
from pheromones import PheromoneField
from projects import ProjectManager
from agents import AgentPopulation
from metrics import MetricsCalculator
from overmind import ContemplativeOvermind
from policies import PolicyManager


@dataclass
class SimulationState:
    """Complete simulation state snapshot"""
    step: int
    time: float
    environment: Environment
    physarum: PhysarumNetwork
    pheromones: PheromoneField
    projects: ProjectManager
    population: AgentPopulation
    overmind: ContemplativeOvermind
    metrics: Dict[str, float]
    wisdom_signal: float


class BeaverEcosystemSimulation:
    """
    Complete beaver ecosystem simulation (§16)
    
    Multi-agent Markov game with:
    - Environment dynamics (hydrology, vegetation, dams)
    - Physarum adaptive network
    - Ant-style pheromone communication
    - Bee-style recruitment
    - Beaver agents with task division
    - Contemplative Overmind with ACP
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        if config is None:
            config = create_default_config()
        
        self.cfg = config
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.logger.info("Initializing beaver ecosystem simulation...")
        
        # Environment
        self.logger.info("Creating environment...")
        self.environment = Environment(
            self.cfg.world,
            self.cfg.hydrology,
            self.cfg.vegetation,
            self.cfg.dam
        )
        
        # Physarum network
        self.logger.info("Creating Physarum network...")
        self.physarum = PhysarumNetwork(
            self.environment,
            self.cfg.physarum
        )
        
        # Pheromone field
        self.logger.info("Creating pheromone field...")
        self.pheromones = PheromoneField(
            self.environment.N,
            self.environment.state.neighbors,
            self.cfg.pheromone
        )
        
        # Project manager
        self.logger.info("Creating project manager...")
        self.projects = ProjectManager(
            self.environment,
            self.cfg.project
        )
        
        # Agent population
        self.logger.info("Creating agent population...")
        self.population = AgentPopulation(
            self.environment,
            self.cfg.agent
        )
        
        # Metrics calculator
        self.logger.info("Creating metrics calculator...")
        self.metrics_calc = MetricsCalculator(
            self.cfg.reward,
            self.cfg.overmind,
            self.cfg.dam
        )
        
        # Overmind
        self.logger.info("Creating contemplative overmind...")
        self.overmind = ContemplativeOvermind(
            self.cfg.overmind
        )
        
        # Policy manager
        self.logger.info("Creating policy manager...")
        self.policy_manager = PolicyManager(
            self.cfg.policy,
            use_contemplative=self.cfg.use_contemplative_policy
        )
        
        # Simulation state
        self.current_step = 0
        self.current_time = 0.0
        
        # Data collection
        self.history = []
        
        # Output directory
        self.output_dir = Path(self.cfg.save_directory) / self.cfg.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialization complete!")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self) -> None:
        """Setup logging"""
        if self.cfg.enable_logging:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            logging.basicConfig(
                level=getattr(logging, self.cfg.log_level),
                format=log_format
            )
        
        self.logger = logging.getLogger('BeaverEcosystem')
    
    def step(self) -> Dict[str, float]:
        """
        Execute one simulation time step
        
        Order of operations:
        1. Update Physarum sources/sinks
        2. Step Physarum network
        3. Update project qualities
        4. Update project recruitment
        5. Execute agent actions
        6. Step pheromones
        7. Step environment
        8. Compute metrics
        9. Update Overmind
        10. Adapt meta-parameters
        """
        dt = self.cfg.world.dt
        
        # 1. Update Physarum sources/sinks based on environment
        self._update_physarum_sources_sinks()
        
        # 2. Step Physarum network
        self.physarum.step()
        
        # 3. Update project qualities
        self.projects.update_project_qualities()
        
        # 4. Update project recruitment
        # (simplified: assume some scouts return)
        scout_returns = self._simulate_scout_returns()
        self.projects.update_recruitment(dt, scout_returns)
        
        # 5. Execute agent actions with current meta-parameters
        task_stimuli = self.overmind.meta_params.task_stimuli
        
        # Assign projects to workers if needed
        self._assign_projects()
        
        # Execute agent step
        action_metrics = self.policy_manager.execute_population_step(
            self.population,
            self.environment,
            self.pheromones,
            self.physarum,
            self.projects,
            task_stimuli,
            current_wisdom=0.0,  # Will be computed below
            dt=dt
        )
        
        # 6. Step pheromones
        self.pheromones.step(dt)
        
        # 7. Step environment
        self.environment.step()
        
        # 8. Compute metrics
        reward_components = self.metrics_calc.compute_reward(
            self.environment,
            self.population
        )
        
        wisdom_signal = self.metrics_calc.compute_wisdom_signal(
            self.environment,
            self.population,
            self.physarum,
            reward_components
        )
        
        all_metrics = self.metrics_calc.compute_all_metrics(
            self.environment,
            self.population,
            self.physarum
        )
        
        # Add action metrics
        all_metrics.update(action_metrics)
        
        # 9. Update Overmind and check for intervention
        should_intervene, reason = self.overmind.should_intervene(all_metrics)
        
        if should_intervene:
            self.logger.warning(f"Overmind intervention: {reason}")
            self.overmind.execute_intervention(reason)
            all_metrics['intervention'] = 1.0
            all_metrics['intervention_reason'] = reason
        else:
            all_metrics['intervention'] = 0.0
        
        # 10. Adapt meta-parameters
        self.overmind.update_meta_parameters(wisdom_signal, all_metrics)
        
        # Update pheromone config with new evaporation rate
        self.pheromones.cfg.rho = self.overmind.meta_params.rho
        
        # Store overmind state in metrics
        overmind_state = self.overmind.get_overmind_state()
        all_metrics.update({
            f'overmind_{k}': v for k, v in overmind_state.items()
            if isinstance(v, (int, float))
        })
        
        # Update step counter
        self.current_step += 1
        self.current_time += dt
        
        # Store in history
        self.history.append({
            'step': self.current_step,
            'time': self.current_time,
            'metrics': all_metrics.copy()
        })
        
        return all_metrics
    
    def _update_physarum_sources_sinks(self) -> None:
        """Update Physarum sources and sinks based on environment state"""
        sources = {c: [] for c in self.physarum.commodities}
        sinks = {c: [] for c in self.physarum.commodities}
        
        # Water commodity: sources = high water areas, sinks = lodges/dams
        if CommodityType.WATER in self.physarum.commodities:
            # Sources: cells with high water
            high_water = np.where(self.environment.state.h > 1.5)[0]
            sources[CommodityType.WATER] = high_water[:10].tolist()  # Limit to 10
            
            # Sinks: cells with dams (low permeability)
            dam_cells = np.where(self.environment.state.d < 0.5)[0]
            sinks[CommodityType.WATER] = dam_cells[:10].tolist()
        
        # Log transport: sources = vegetation, sinks = project centers
        if CommodityType.LOG_TRANSPORT in self.physarum.commodities:
            # Sources: high vegetation cells
            high_veg = np.where(self.environment.state.v > 5.0)[0]
            sources[CommodityType.LOG_TRANSPORT] = high_veg[:10].tolist()
            
            # Sinks: project centers
            for project in self.projects.state.projects:
                center_idx = self.environment._coords_to_index(*project.center)
                sinks[CommodityType.LOG_TRANSPORT].append(center_idx)
        
        # Food transport: sources = vegetation, sinks = agents/lodges
        if CommodityType.FOOD_TRANSPORT in self.physarum.commodities:
            high_veg = np.where(self.environment.state.v > 5.0)[0]
            sources[CommodityType.FOOD_TRANSPORT] = high_veg[:10].tolist()
            
            # Sinks: agent locations
            agent_positions = [a.state.position for a in self.population.get_alive_agents()]
            sinks[CommodityType.FOOD_TRANSPORT] = agent_positions[:10]
        
        self.physarum.update_sources_sinks(sources, sinks)
    
    def _simulate_scout_returns(self) -> Dict[int, int]:
        """Simulate scouts returning to advertise projects"""
        alive_agents = self.population.get_alive_agents()
        
        # Count scouts near each project
        scout_returns = {p.id: 0 for p in self.projects.state.projects}
        
        for agent in alive_agents:
            if agent.state.role.name == 'SCOUT':
                # Check if near any project
                for project in self.projects.state.projects:
                    if agent.state.position in project.region:
                        scout_returns[project.id] += 1
        
        return scout_returns
    
    def _assign_projects(self) -> None:
        """Assign projects to worker agents"""
        alive_agents = self.population.get_alive_agents()
        beta_R = self.overmind.meta_params.beta_R
        
        for agent in alive_agents:
            # Only assign to workers without a project
            if agent.state.role.name == 'WORKER' and agent.state.assigned_project is None:
                agent.assign_project(self.projects, beta_R)
    
    def run(self, num_steps: Optional[int] = None) -> None:
        """
        Run simulation for specified number of steps
        
        Args:
            num_steps: Number of steps to run (defaults to config max_steps)
        """
        if num_steps is None:
            num_steps = self.cfg.world.max_steps
        
        self.logger.info(f"Running simulation for {num_steps} steps...")
        
        for step in range(num_steps):
            # Execute step
            metrics = self.step()
            
            # Log progress
            if step % 100 == 0:
                self.logger.info(
                    f"Step {step}/{num_steps} | "
                    f"Alive: {metrics['num_alive']} | "
                    f"Reward: {metrics['reward_total']:.2f} | "
                    f"Wisdom: {metrics['wisdom_signal']:.2f}"
                )
            
            # Check termination
            if metrics['num_alive'] == 0:
                self.logger.warning(f"Population extinct at step {step}")
                break
        
        self.logger.info("Simulation complete!")
        
        # Save results
        self.save_results()
    
    def save_results(self) -> None:
        """Save simulation results"""
        self.logger.info("Saving results...")
        
        # Save configuration
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            # Convert config to dict (simplified)
            config_dict = {
                'experiment_name': self.cfg.experiment_name,
                'num_agents': self.cfg.agent.num_agents,
                'grid_size': f"{self.cfg.world.grid_height}x{self.cfg.world.grid_width}",
                'max_steps': self.cfg.world.max_steps,
                'use_contemplative_policy': self.cfg.use_contemplative_policy
            }
            json.dump(config_dict, f, indent=2)
        
        # Save metrics history
        self._save_metrics_history()
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _save_metrics_history(self) -> None:
        """Save metrics history to CSV"""
        import pandas as pd
        
        # Extract metrics
        steps = [h['step'] for h in self.history]
        times = [h['time'] for h in self.history]
        
        # Get all metric keys
        all_keys = set()
        for h in self.history:
            all_keys.update(h['metrics'].keys())
        
        # Build dataframe
        data = {'step': steps, 'time': times}
        
        for key in all_keys:
            if isinstance(self.history[0]['metrics'].get(key), (int, float)):
                data[key] = [h['metrics'].get(key, np.nan) for h in self.history]
        
        df = pd.DataFrame(data)
        
        # Save
        metrics_path = self.output_dir / 'metrics.csv'
        df.to_csv(metrics_path, index=False)
        
        self.logger.info(f"Metrics saved to {metrics_path}")
    
    def get_current_state(self) -> SimulationState:
        """Get current simulation state"""
        metrics = self.metrics_calc.compute_all_metrics(
            self.environment,
            self.population,
            self.physarum
        )
        
        reward = self.metrics_calc.compute_reward(
            self.environment,
            self.population
        )
        
        wisdom = self.metrics_calc.compute_wisdom_signal(
            self.environment,
            self.population,
            self.physarum,
            reward
        )
        
        return SimulationState(
            step=self.current_step,
            time=self.current_time,
            environment=self.environment,
            physarum=self.physarum,
            pheromones=self.pheromones,
            projects=self.projects,
            population=self.population,
            overmind=self.overmind,
            metrics=metrics,
            wisdom_signal=wisdom
        )

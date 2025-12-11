"""
Comprehensive Diagnostics Module (Enhancement #8)
Tracks system-wide metrics for analysis, visualization, and ablation studies

Features:
- Multi-layer entropy tracking (pheromone, Physarum, agent actions)
- Spatial statistics (autocorrelation, clustering, gradients)
- Coordination metrics (agent synchrony, division of labor)
- Network analysis (Physarum sparsity, connectivity, efficiency)
- Wisdom/reward decomposition
- Real-time ablation toggles
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class SystemDiagnostics:
    """Comprehensive diagnostic state at a single timestep"""
    step: int
    
    # Pheromone entropy (per channel)
    pheromone_entropy: Dict[str, float] = field(default_factory=dict)
    pheromone_total_entropy: float = 0.0
    
    # Physarum network metrics
    physarum_sparsity: float = 0.0  # Gini coefficient of edge conductivities
    physarum_connectivity: float = 0.0  # Fraction of non-zero edges
    physarum_efficiency: float = 0.0  # Average shortest path length
    
    # Agent coordination
    action_synchrony: float = 0.0  # Spatial autocorrelation of actions
    role_diversity: float = 0.0  # Shannon entropy of role distribution
    task_coordination: float = 0.0  # Fraction working on same task
    
    # Spatial statistics
    agent_clustering: float = 0.0  # Moran's I for agent positions
    water_gradient: float = 0.0  # Max gradient of water depth
    vegetation_gradient: float = 0.0  # Max gradient of vegetation
    
    # Wisdom decomposition
    wisdom_flood_component: float = 0.0
    wisdom_drought_component: float = 0.0
    wisdom_habitat_component: float = 0.0
    wisdom_failure_component: float = 0.0
    
    # Reward decomposition
    reward_survival: float = 0.0
    reward_habitat: float = 0.0
    reward_stability: float = 0.0
    reward_cooperation: float = 0.0
    
    # Population dynamics
    birth_count: int = 0
    death_count: int = 0
    generation_avg: float = 0.0
    
    # Performance metrics
    dam_count: int = 0
    lodge_count: int = 0
    flood_cells: int = 0
    drought_cells: int = 0


class DiagnosticTracker:
    """
    Tracks comprehensive system diagnostics over time
    
    Use for:
    - Publication-quality analysis
    - Ablation studies
    - Real-time monitoring
    - System debugging
    """
    
    def __init__(self):
        self.history: List[SystemDiagnostics] = []
        self.current_step = 0
        
        # Ablation toggles (can be modified at runtime)
        self.ablations = {
            'pheromone_enabled': True,
            'physarum_enabled': True,
            'overmind_enabled': True,
            'multi_channel_pheromone': True,
            'agent_memory': True,
            'role_specialization': True,
            'population_dynamics': True
        }
        
        # Running statistics (for efficiency)
        self._running_means = defaultdict(float)
        self._running_vars = defaultdict(float)
        
    def compute_step_diagnostics(
        self,
        environment,
        agents: List,
        pheromone_field,
        physarum_network,
        overmind,
        wisdom_signal: float,
        reward_components: Dict[str, float]
    ) -> SystemDiagnostics:
        """
        Compute all diagnostic metrics for current timestep
        
        Args:
            environment: Environment instance
            agents: List of agent objects
            pheromone_field: PheromoneField instance
            physarum_network: PhysarumNetwork instance
            overmind: Overmind instance
            wisdom_signal: Current wisdom value
            reward_components: Dictionary of reward component values
        
        Returns:
            SystemDiagnostics object with all computed metrics
        """
        diag = SystemDiagnostics(step=self.current_step)
        
        # Pheromone metrics
        if self.ablations['pheromone_enabled'] and pheromone_field is not None:
            diag.pheromone_total_entropy = pheromone_field.compute_entropy()
            
            if self.ablations['multi_channel_pheromone']:
                # Per-channel entropy
                from pheromones import PheromoneChannel
                for channel in PheromoneChannel:
                    ch_entropy = pheromone_field.compute_entropy(channel)
                    diag.pheromone_entropy[channel.name] = ch_entropy
        
        # Physarum network metrics
        if self.ablations['physarum_enabled'] and physarum_network is not None:
            diag.physarum_sparsity = self._compute_physarum_sparsity(physarum_network)
            diag.physarum_connectivity = self._compute_physarum_connectivity(physarum_network)
            diag.physarum_efficiency = self._compute_network_efficiency(physarum_network)
        
        # Agent coordination metrics
        if agents:
            diag.action_synchrony = self._compute_action_synchrony(agents, environment)
            diag.role_diversity = self._compute_role_diversity(agents)
            diag.task_coordination = self._compute_task_coordination(agents)
            diag.agent_clustering = self._compute_spatial_clustering(agents, environment)
        
        # Spatial gradients
        if environment is not None:
            diag.water_gradient = self._compute_max_gradient(environment.state.h, environment.H, environment.W)
            diag.vegetation_gradient = self._compute_max_gradient(environment.state.v, environment.H, environment.W)
        
        # Wisdom decomposition (if overmind active)
        if self.ablations['overmind_enabled'] and overmind is not None:
            wisdom_comp = self._decompose_wisdom(environment, overmind)
            diag.wisdom_flood_component = wisdom_comp['flood']
            diag.wisdom_drought_component = wisdom_comp['drought']
            diag.wisdom_habitat_component = wisdom_comp['habitat']
            diag.wisdom_failure_component = wisdom_comp['failure']
        
        # Reward decomposition
        if reward_components:
            diag.reward_survival = reward_components.get('survival', 0.0)
            diag.reward_habitat = reward_components.get('habitat', 0.0)
            diag.reward_stability = reward_components.get('stability', 0.0)
            diag.reward_cooperation = reward_components.get('cooperation', 0.0)
        
        # Population metrics
        if self.ablations['population_dynamics']:
            diag.birth_count = getattr(environment, 'birth_count_this_step', 0)
            diag.death_count = getattr(environment, 'death_count_this_step', 0)
            if agents:
                generations = [getattr(a, 'generation', 0) for a in agents]
                diag.generation_avg = np.mean(generations)
        
        # Infrastructure metrics
        if environment is not None:
            diag.dam_count = int(np.sum(environment.state.d < 0.5))  # Strong dams
            diag.flood_cells = int(np.sum(environment.state.h > environment.hydro_cfg.h_flood))
            diag.drought_cells = int(np.sum(environment.state.h < environment.hydro_cfg.h_drought))
        
        # Store in history
        self.history.append(diag)
        self.current_step += 1
        
        return diag
    
    def _compute_physarum_sparsity(self, physarum_network) -> float:
        """
        Compute Gini coefficient of Physarum edge conductivities
        
        Returns 0 (equal) to 1 (all flow on one edge)
        """
        try:
            from config import CommodityType
            conductivities = []
            
            for edge in physarum_network.edges:
                D_value = physarum_network.state.D[edge][CommodityType.WATER]
                conductivities.append(D_value)
            
            if not conductivities:
                return 0.0
            
            # Gini coefficient
            conductivities = np.sort(conductivities)
            n = len(conductivities)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * conductivities)) / (n * np.sum(conductivities)) - (n + 1) / n
            return max(0.0, min(1.0, gini))
        except:
            return 0.0
    
    def _compute_physarum_connectivity(self, physarum_network) -> float:
        """Fraction of edges with non-zero conductivity"""
        try:
            from config import CommodityType
            total_edges = len(physarum_network.edges)
            if total_edges == 0:
                return 0.0
            
            non_zero = 0
            for edge in physarum_network.edges:
                D_value = physarum_network.state.D[edge][CommodityType.WATER]
                if D_value > 1e-6:
                    non_zero += 1
            
            return non_zero / total_edges
        except:
            return 0.0
    
    def _compute_network_efficiency(self, physarum_network) -> float:
        """
        Compute average path efficiency (inverse of shortest path lengths)
        
        Higher = more efficient network topology
        """
        # Simplified: use average conductivity as proxy
        try:
            from config import CommodityType
            conductivities = []
            for edge in physarum_network.edges:
                D_value = physarum_network.state.D[edge][CommodityType.WATER]
                conductivities.append(D_value)
            
            if conductivities:
                return float(np.mean(conductivities))
            return 0.0
        except:
            return 0.0
    
    def _compute_action_synchrony(self, agents, environment) -> float:
        """
        Compute spatial autocorrelation of agent actions
        
        High = agents near each other do similar things (coordinated)
        Low = independent actions
        """
        if len(agents) < 2:
            return 0.0
        
        try:
            # Get agent positions and last actions
            positions = []
            actions = []
            
            for agent in agents:
                if hasattr(agent, 'position') and hasattr(agent, 'last_action'):
                    positions.append(agent.position)
                    # Convert action to numeric (hack: use action enum value)
                    action_val = int(agent.last_action) if hasattr(agent.last_action, 'value') else 0
                    actions.append(action_val)
            
            if len(positions) < 2:
                return 0.0
            
            # Compute pairwise distances and action similarities
            n = len(positions)
            total_weight = 0.0
            weighted_sum = 0.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Spatial distance
                    pos_i = positions[i]
                    pos_j = positions[j]
                    dist = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])  # Manhattan
                    
                    # Weight by inverse distance
                    if dist > 0:
                        weight = 1.0 / dist
                    else:
                        weight = 10.0  # Same cell
                    
                    # Action similarity (1 if same, 0 if different)
                    action_sim = 1.0 if actions[i] == actions[j] else 0.0
                    
                    weighted_sum += weight * action_sim
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            return 0.0
        except:
            return 0.0
    
    def _compute_role_diversity(self, agents) -> float:
        """
        Shannon entropy of role distribution
        
        High = diverse roles, Low = all agents same role
        """
        if not agents:
            return 0.0
        
        try:
            role_counts = defaultdict(int)
            for agent in agents:
                if hasattr(agent, 'role'):
                    role_counts[agent.role] += 1
            
            if not role_counts:
                return 0.0
            
            total = sum(role_counts.values())
            probs = [count / total for count in role_counts.values()]
            
            # Shannon entropy
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            return entropy
        except:
            return 0.0
    
    def _compute_task_coordination(self, agents) -> float:
        """
        Fraction of agents working on same task (highest task)
        
        High = focused effort, Low = dispersed effort
        """
        if not agents:
            return 0.0
        
        try:
            task_counts = defaultdict(int)
            for agent in agents:
                if hasattr(agent, 'current_task'):
                    task_counts[agent.current_task] += 1
            
            if not task_counts:
                return 0.0
            
            max_count = max(task_counts.values())
            return max_count / len(agents)
        except:
            return 0.0
    
    def _compute_spatial_clustering(self, agents, environment) -> float:
        """
        Moran's I for agent spatial distribution
        
        > 0: clustered, = 0: random, < 0: dispersed
        """
        if len(agents) < 2:
            return 0.0
        
        try:
            # Get agent positions as grid coordinates
            positions = []
            for agent in agents:
                if hasattr(agent, 'position'):
                    positions.append(agent.position)
            
            if len(positions) < 2:
                return 0.0
            
            # Simplified Moran's I: check if agents cluster
            n = len(positions)
            mean_x = np.mean([p[0] for p in positions])
            mean_y = np.mean([p[1] for p in positions])
            
            # Variance from center
            variance = np.mean([
                (p[0] - mean_x)**2 + (p[1] - mean_y)**2
                for p in positions
            ])
            
            # Expected variance if uniformly distributed
            H, W = environment.H, environment.W
            expected_var = (H**2 + W**2) / 12.0
            
            # Normalized clustering metric
            if expected_var > 0:
                clustering = 1.0 - (variance / expected_var)
                return np.clip(clustering, -1.0, 1.0)
            return 0.0
        except:
            return 0.0
    
    def _compute_max_gradient(self, field: np.ndarray, H: int, W: int) -> float:
        """Compute maximum spatial gradient of a field"""
        try:
            # Reshape to 2D if flat
            if field.ndim == 1:
                field_2d = field.reshape(H, W)
            else:
                field_2d = field
            
            # Compute gradients
            grad_y, grad_x = np.gradient(field_2d)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return float(np.max(gradient_magnitude))
        except:
            return 0.0
    
    def _decompose_wisdom(self, environment, overmind) -> Dict[str, float]:
        """Decompose wisdom signal into components"""
        try:
            # Get components from overmind observation
            obs = overmind.get_observation(environment)
            
            return {
                'flood': -obs.get('C_flood', 0.0),
                'drought': -obs.get('C_drought', 0.0),
                'habitat': obs.get('R_habitat', 0.0),
                'failure': -obs.get('C_failure', 0.0)
            }
        except:
            return {'flood': 0.0, 'drought': 0.0, 'habitat': 0.0, 'failure': 0.0}
    
    def get_time_series(self, metric_name: str) -> List[float]:
        """Extract time series for a specific metric"""
        return [getattr(diag, metric_name, 0.0) for diag in self.history]
    
    def get_summary_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        series = self.get_time_series(metric_name)
        if not series:
            return {}
        
        return {
            'mean': float(np.mean(series)),
            'std': float(np.std(series)),
            'min': float(np.min(series)),
            'max': float(np.max(series)),
            'final': float(series[-1])
        }
    
    def export_to_dict(self) -> List[Dict[str, Any]]:
        """Export full diagnostic history as list of dicts (for JSON/CSV)"""
        return [
            {
                'step': d.step,
                **d.pheromone_entropy,
                'pheromone_total_entropy': d.pheromone_total_entropy,
                'physarum_sparsity': d.physarum_sparsity,
                'physarum_connectivity': d.physarum_connectivity,
                'physarum_efficiency': d.physarum_efficiency,
                'action_synchrony': d.action_synchrony,
                'role_diversity': d.role_diversity,
                'task_coordination': d.task_coordination,
                'agent_clustering': d.agent_clustering,
                'water_gradient': d.water_gradient,
                'vegetation_gradient': d.vegetation_gradient,
                'wisdom_flood': d.wisdom_flood_component,
                'wisdom_drought': d.wisdom_drought_component,
                'wisdom_habitat': d.wisdom_habitat_component,
                'wisdom_failure': d.wisdom_failure_component,
                'reward_survival': d.reward_survival,
                'reward_habitat': d.reward_habitat,
                'reward_stability': d.reward_stability,
                'reward_cooperation': d.reward_cooperation,
                'birth_count': d.birth_count,
                'death_count': d.death_count,
                'generation_avg': d.generation_avg,
                'dam_count': d.dam_count,
                'flood_cells': d.flood_cells,
                'drought_cells': d.drought_cells
            }
            for d in self.history
        ]
    
    def save_to_json(self, filepath: str):
        """Save diagnostic history to JSON file"""
        data = self.export_to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_current_summary(self):
        """Print summary of most recent diagnostics"""
        if not self.history:
            print("No diagnostics collected yet")
            return
        
        diag = self.history[-1]
        print(f"\n=== Diagnostics at Step {diag.step} ===")
        print(f"Pheromone entropy: {diag.pheromone_total_entropy:.3f}")
        print(f"Physarum sparsity: {diag.physarum_sparsity:.3f}")
        print(f"Agent coordination: {diag.action_synchrony:.3f}")
        print(f"Role diversity: {diag.role_diversity:.3f}")
        print(f"Dams: {diag.dam_count}, Floods: {diag.flood_cells}, Droughts: {diag.drought_cells}")
        print(f"Births: {diag.birth_count}, Deaths: {diag.death_count}")
        print("=" * 50)

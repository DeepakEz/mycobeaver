"""
Pheromone Module
Implements ant-style pheromone fields for stigmergic communication (§6)
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

from config import PheromoneConfig


@dataclass
class PheromoneState:
    """State of pheromone fields"""
    # Pheromone concentrations on directed edges
    tau: Dict[Tuple[int, int], float]


class PheromoneField:
    """
    Ant-style pheromone field for stigmergic communication (§6)
    
    Features:
    - Directed edge pheromones
    - Evaporation dynamics
    - Agent deposition
    """
    
    def __init__(
        self,
        num_cells: int,
        neighbors: Dict[int, List[int]],
        config: PheromoneConfig
    ):
        self.N = num_cells
        self.neighbors = neighbors
        self.cfg = config
        
        # Build directed edge list
        self.directed_edges = self._build_directed_edges()
        
        # Initialize state
        self.state = self._initialize_state()
    
    def _build_directed_edges(self) -> List[Tuple[int, int]]:
        """Build list of all directed edges"""
        edges = []
        for i in range(self.N):
            for j in self.neighbors[i]:
                edges.append((i, j))
        return edges
    
    def _initialize_state(self) -> PheromoneState:
        """Initialize pheromone state (all zeros)"""
        tau = {edge: 0.0 for edge in self.directed_edges}
        return PheromoneState(tau=tau)
    
    def step(self, dt: float) -> None:
        """
        Execute one pheromone time step (§6)
        
        τ_ij^{t+1} = (1 - ρ) * τ_ij^t + Δτ_ij^t
        """
        rho = self.cfg.rho
        
        for edge in self.directed_edges:
            # Evaporation
            self.state.tau[edge] *= (1 - rho * dt)
            
            # Clip to zero if very small
            if self.state.tau[edge] < 1e-8:
                self.state.tau[edge] = 0.0
    
    def deposit(self, i: int, j: int, amount: float) -> None:
        """
        Deposit pheromone on edge (i, j)
        
        Δτ_ij = δ_k (agent k deposition)
        """
        edge = (i, j)
        if edge in self.state.tau:
            self.state.tau[edge] += amount
    
    def get_pheromone(self, i: int, j: int) -> float:
        """Get pheromone concentration on edge (i, j)"""
        edge = (i, j)
        return self.state.tau.get(edge, 0.0)
    
    def get_movement_probability(
        self,
        i: int,
        heuristics: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Compute movement probabilities from cell i (§11)
        
        P_k(j | i) = (τ_ij^α * η_ij^β) / Σ_l (τ_il^α * η_il^β)
        """
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        
        # Get neighbors
        neighbor_list = self.neighbors[i]
        
        if len(neighbor_list) == 0:
            return {}
        
        # Compute numerators
        numerators = {}
        total = 0.0
        
        for j in neighbor_list:
            tau_ij = self.get_pheromone(i, j)
            eta_ij = heuristics.get(j, 1.0)
            
            # Avoid division by zero
            if tau_ij < 1e-8:
                tau_ij = 1e-8
            if eta_ij < 1e-8:
                eta_ij = 1e-8
            
            numerator = (tau_ij ** alpha) * (eta_ij ** beta)
            numerators[j] = numerator
            total += numerator
        
        # Normalize
        if total < 1e-8:
            # Uniform distribution if no information
            prob = 1.0 / len(neighbor_list)
            return {j: prob for j in neighbor_list}
        
        probabilities = {j: numerators[j] / total for j in neighbor_list}
        return probabilities
    
    def reset(self) -> None:
        """Reset all pheromones to zero"""
        for edge in self.directed_edges:
            self.state.tau[edge] = 0.0
    
    def get_pheromone_2d(self, grid_height: int, grid_width: int) -> np.ndarray:
        """Get 2D view of average pheromone concentrations for visualization"""
        pheromone_2d = np.zeros((grid_height, grid_width))
        
        for i in range(self.N):
            # Average pheromone on outgoing edges
            row = i // grid_width
            col = i % grid_width
            
            total_tau = 0.0
            count = 0
            
            for j in self.neighbors[i]:
                tau_ij = self.get_pheromone(i, j)
                total_tau += tau_ij
                count += 1
            
            if count > 0:
                pheromone_2d[row, col] = total_tau / count
        
        return pheromone_2d

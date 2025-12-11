"""
Pheromone Module - MULTI-CHANNEL IMPLEMENTATION
Implements sophisticated ant-style pheromone fields with specialized channels
for different behaviors (§6 + Enhancement #3)

Key features:
- Multiple pheromone channels (FOOD, CONSTRUCTION, DANGER, RETURN_HOME, EXPLORATION)
- Channel-specific evaporation rates
- Success-based deposition bonuses
- Directional bias (boost trails aligned with targets)
- Comprehensive diagnostics (entropy, spatial distribution)
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

from config import PheromoneConfig


class PheromoneChannel(Enum):
    """Specialized pheromone channels for different behaviors"""
    FOOD = 0           # Trail to food sources
    CONSTRUCTION = 1   # Trail to dam/lodge building sites
    DANGER = 2         # Warning about flooded/hazardous areas
    RETURN_HOME = 3    # Trail back to lodge (like bee navigation)
    EXPLORATION = 4    # General exploration trails


@dataclass
class PheromoneState:
    """State of multi-channel pheromone fields"""
    # Multi-channel pheromones: (i, j) -> {channel -> concentration}
    tau: Dict[Tuple[int, int], Dict[PheromoneChannel, float]]
    
    # Diagnostics
    entropy_history: List[float]
    channel_statistics: Dict[PheromoneChannel, Dict[str, float]]


class MultiChannelPheromoneField:
    """
    MULTI-CHANNEL pheromone field for specialized communication.
    
    Each edge (i, j) can have multiple pheromone types:
    - FOOD: leads to food sources
    - CONSTRUCTION: leads to building sites
    - DANGER: warns about hazards
    - RETURN_HOME: guides back to lodge
    - EXPLORATION: general exploration
    
    This enables sophisticated coordination beyond single-channel systems.
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
        """Initialize multi-channel pheromone state (all zeros)"""
        tau = {}
        for edge in self.directed_edges:
            tau[edge] = {channel: 0.0 for channel in PheromoneChannel}
        
        return PheromoneState(
            tau=tau,
            entropy_history=[],
            channel_statistics={ch: {} for ch in PheromoneChannel}
        )
    
    def step(self, dt: float) -> None:
        """
        Execute one pheromone time step with CHANNEL-SPECIFIC evaporation
        
        τ_ij^{ch,t+1} = (1 - ρ_ch) * τ_ij^{ch,t}
        
        Different channels evaporate at different rates!
        """
        for edge in self.directed_edges:
            for channel in PheromoneChannel:
                # Get channel-specific evaporation rate
                rho = self.cfg.channel_evaporation.get(channel, self.cfg.rho)
                
                # Evaporate
                self.state.tau[edge][channel] *= (1 - rho * dt)
                
                # Clip to zero if very small
                if self.state.tau[edge][channel] < 1e-8:
                    self.state.tau[edge][channel] = 0.0
        
        # Update diagnostics
        self._update_diagnostics()
    
    def deposit(
        self,
        i: int,
        j: int,
        channel: PheromoneChannel,
        amount: float = None,
        success_multiplier: float = 1.0,
        directional_alignment: float = 0.0
    ) -> None:
        """
        Deposit pheromone on edge (i, j) for specific channel
        
        Args:
            i: Source cell
            j: Destination cell
            channel: Which pheromone channel
            amount: Base deposit amount (uses config default if None)
            success_multiplier: Multiply by this (e.g., 2.0 if carrying food)
            directional_alignment: Cosine similarity with target [0,1]
        """
        if amount is None:
            amount = self.cfg.delta
        
        # Apply success multiplier
        amount *= success_multiplier
        
        # Apply directional bias if enabled and aligned
        if self.cfg.use_directional_bias and directional_alignment > 0.5:
            amount *= self.cfg.directional_boost * directional_alignment
        
        # Deposit on specific channel
        edge = (i, j)
        if edge in self.state.tau:
            self.state.tau[edge][channel] += amount
    
    def get_pheromone(
        self,
        i: int,
        j: int,
        channel: PheromoneChannel
    ) -> float:
        """Get pheromone concentration on edge (i, j) for specific channel"""
        edge = (i, j)
        if edge not in self.state.tau:
            return 0.0
        return self.state.tau[edge][channel]
    
    def get_all_channels(
        self,
        i: int,
        j: int
    ) -> Dict[PheromoneChannel, float]:
        """Get all channel concentrations for edge (i, j)"""
        edge = (i, j)
        if edge not in self.state.tau:
            return {ch: 0.0 for ch in PheromoneChannel}
        return self.state.tau[edge].copy()
    
    def get_neighbors_concentration(
        self,
        i: int,
        channel: PheromoneChannel
    ) -> Dict[int, float]:
        """
        Get pheromone concentrations from cell i to all neighbors for a channel
        
        Returns:
            Dictionary mapping neighbor_cell -> concentration
        """
        neighbor_list = self.neighbors[i]
        return {
            j: self.get_pheromone(i, j, channel)
            for j in neighbor_list
        }
    
    def get_neighbors_all_channels(
        self,
        i: int
    ) -> Dict[int, Dict[PheromoneChannel, float]]:
        """
        Get all channel concentrations for all neighbors of cell i
        
        Returns:
            Dictionary mapping neighbor_cell -> {channel -> concentration}
        """
        neighbor_list = self.neighbors[i]
        return {
            j: self.get_all_channels(i, j)
            for j in neighbor_list
        }
    
    def get_movement_probability(
        self,
        i: int,
        heuristics: Dict[int, float],
        channel: PheromoneChannel = None,
        channel_weights: Dict[PheromoneChannel, float] = None
    ) -> Dict[int, float]:
        """
        Compute movement probabilities from cell i using pheromones + heuristics
        
        Single channel:
            P_k(j | i) = (τ_ij^α * η_ij^β) / Σ_l (τ_il^α * η_il^β)
        
        Multi-channel weighted:
            τ_ij = Σ_ch w_ch * τ_ij^ch
            P_k(j | i) = (τ_ij^α * η_ij^β) / Σ_l (τ_il^α * η_il^β)
        
        Args:
            i: Current cell
            heuristics: Heuristic desirability for each neighbor
            channel: Use single channel (if specified)
            channel_weights: Weights for combining multiple channels
        """
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        
        neighbor_list = self.neighbors[i]
        
        if len(neighbor_list) == 0:
            return {}
        
        # Compute pheromone values
        pheromone_values = {}
        
        if channel is not None:
            # Single channel mode
            pheromone_values = self.get_neighbors_concentration(i, channel)
        elif channel_weights is not None:
            # Multi-channel weighted combination
            for j in neighbor_list:
                all_ch = self.get_all_channels(i, j)
                combined = sum(
                    channel_weights.get(ch, 0.0) * all_ch[ch]
                    for ch in PheromoneChannel
                )
                pheromone_values[j] = combined
        else:
            # Default: equal weight on all channels
            for j in neighbor_list:
                all_ch = self.get_all_channels(i, j)
                combined = sum(all_ch.values()) / len(PheromoneChannel)
                pheromone_values[j] = combined
        
        # Compute probabilities
        numerators = {}
        total = 0.0
        
        for j in neighbor_list:
            tau_ij = pheromone_values.get(j, 0.0)
            eta_ij = heuristics.get(j, 1.0)
            
            # Avoid zero
            if tau_ij < 1e-8:
                tau_ij = 1e-8
            if eta_ij < 1e-8:
                eta_ij = 1e-8
            
            numerator = (tau_ij ** alpha) * (eta_ij ** beta)
            numerators[j] = numerator
            total += numerator
        
        # Normalize
        if total < 1e-8:
            # Uniform if no information
            prob = 1.0 / len(neighbor_list)
            return {j: prob for j in neighbor_list}
        
        return {j: numerators[j] / total for j in neighbor_list}
    
    def compute_entropy(self, channel: PheromoneChannel = None) -> float:
        """
        Compute Shannon entropy of pheromone distribution
        
        High entropy = diffuse trails (exploration)
        Low entropy = concentrated trails (exploitation)
        """
        concentrations = []
        
        for edge in self.directed_edges:
            if channel is None:
                # All channels combined
                concentrations.extend(self.state.tau[edge].values())
            else:
                # Specific channel only
                concentrations.append(self.state.tau[edge][channel])
        
        # Filter positive concentrations
        concentrations = [c for c in concentrations if c > 0]
        
        if not concentrations or sum(concentrations) < 1e-9:
            return 0.0
        
        # Normalize to probability distribution
        total = sum(concentrations)
        probs = [c / total for c in concentrations]
        
        # Shannon entropy: H = -Σ p log p
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        return entropy
    
    def _update_diagnostics(self) -> None:
        """Update diagnostic statistics (called after each step)"""
        # Overall entropy
        entropy = self.compute_entropy()
        self.state.entropy_history.append(entropy)
        
        # Per-channel statistics
        for channel in PheromoneChannel:
            concentrations = [
                self.state.tau[edge][channel]
                for edge in self.directed_edges
                if self.state.tau[edge][channel] > 0
            ]
            
            if concentrations:
                self.state.channel_statistics[channel] = {
                    'mean': np.mean(concentrations),
                    'max': np.max(concentrations),
                    'count': len(concentrations),
                    'entropy': self.compute_entropy(channel)
                }
            else:
                self.state.channel_statistics[channel] = {
                    'mean': 0.0,
                    'max': 0.0,
                    'count': 0,
                    'entropy': 0.0
                }
    
    def reset(self) -> None:
        """Reset all pheromones to zero"""
        for edge in self.directed_edges:
            for channel in PheromoneChannel:
                self.state.tau[edge][channel] = 0.0
        
        self.state.entropy_history.clear()
        self.state.channel_statistics = {ch: {} for ch in PheromoneChannel}
    
    def get_pheromone_2d(
        self,
        grid_height: int,
        grid_width: int,
        channel: PheromoneChannel = None
    ) -> np.ndarray:
        """
        Get 2D view of pheromone concentrations for visualization
        
        Args:
            grid_height: Height of grid
            grid_width: Width of grid
            channel: Specific channel to visualize (None for all channels combined)
        """
        pheromone_2d = np.zeros((grid_height, grid_width))
        
        for i in range(self.N):
            row = i // grid_width
            col = i % grid_width
            
            total_tau = 0.0
            count = 0
            
            for j in self.neighbors[i]:
                if channel is None:
                    # Average across all channels
                    all_ch = self.get_all_channels(i, j)
                    tau_ij = sum(all_ch.values())
                else:
                    # Specific channel
                    tau_ij = self.get_pheromone(i, j, channel)
                
                total_tau += tau_ij
                count += 1
            
            if count > 0:
                pheromone_2d[row, col] = total_tau / count
        
        return pheromone_2d


# Backward compatibility: alias for existing code
PheromoneField = MultiChannelPheromoneField

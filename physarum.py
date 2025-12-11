"""
Physarum Network Module
Implements adaptive flow network (§17) with multi-commodity support (§17.5)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg
from dataclasses import dataclass

from config import PhysarumConfig, CommodityType
from environment import Environment


@dataclass
class PhysarumState:
    """State of Physarum network"""
    # Conductivities for each commodity [num_edges, num_commodities]
    D: Dict[Tuple[int, int], Dict[CommodityType, float]]
    
    # Potentials for each commodity [N, num_commodities]
    Phi: Dict[CommodityType, np.ndarray]
    
    # Fluxes for each commodity [num_edges, num_commodities]
    Q: Dict[Tuple[int, int], Dict[CommodityType, float]]
    
    # Edge lengths (costs)
    L: Dict[Tuple[int, int], float]


class PhysarumNetwork:
    """
    Adaptive flow network inspired by Physarum polycephalum (§17)
    
    Features:
    - Multi-source, multi-sink flow (§17.2)
    - Adaptive conductivity dynamics (§17.3)
    - Coupling with hydrology and terrain (§17.4)
    - Multi-commodity flows (§17.5)
    """
    
    def __init__(
        self,
        environment: Environment,
        config: PhysarumConfig
    ):
        self.env = environment
        self.cfg = config
        
        # Grid dimensions
        self.N = environment.N
        self.H = environment.H
        self.W = environment.W
        
        # Build edge list
        self.edges = self._build_edge_list()
        self.num_edges = len(self.edges)
        
        # Commodity types
        if config.enable_multicommodity:
            self.commodities = [
                CommodityType.WATER,
                CommodityType.LOG_TRANSPORT,
                CommodityType.FOOD_TRANSPORT
            ]
        else:
            self.commodities = [CommodityType.WATER]
        
        # Initialize state
        self.state = self._initialize_state()
        
        # Sources and sinks (will be updated dynamically)
        self.sources: Dict[CommodityType, List[int]] = {c: [] for c in self.commodities}
        self.sinks: Dict[CommodityType, List[int]] = {c: [] for c in self.commodities}
    
    def _build_edge_list(self) -> List[Tuple[int, int]]:
        """Build list of all edges"""
        edges = []
        for i in range(self.N):
            for j in self.env.state.neighbors[i]:
                if i < j:  # Add each edge once
                    edges.append((i, j))
        return edges
    
    def _initialize_state(self) -> PhysarumState:
        """Initialize Physarum state"""
        # Initialize conductivities (small positive values)
        D = {}
        for edge in self.edges:
            D[edge] = {c: 0.1 for c in self.commodities}
        
        # Initialize potentials (zeros)
        Phi = {c: np.zeros(self.N) for c in self.commodities}
        
        # Initialize fluxes (zeros)
        Q = {}
        for edge in self.edges:
            Q[edge] = {c: 0.0 for c in self.commodities}
        
        # Compute edge lengths
        L = self._compute_edge_lengths()
        
        return PhysarumState(D=D, Phi=Phi, Q=Q, L=L)
    
    def _compute_edge_lengths(self) -> Dict[Tuple[int, int], float]:
        """
        Compute edge lengths with terrain and hydrology coupling (§17.4)
        
        L_ij = L_ij^(0) + λ_h * h̄_ij + λ_z * |z_i - z_j|
        """
        L = {}
        
        for i, j in self.edges:
            # Base geometric distance
            row_i, col_i = self.env._index_to_coords(i)
            row_j, col_j = self.env._index_to_coords(j)
            L_base = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
            
            # Average water depth
            h_avg = 0.5 * (self.env.state.h[i] + self.env.state.h[j])
            
            # Elevation difference
            z_diff = abs(self.env.state.z[i] - self.env.state.z[j])
            
            # Total cost
            L[i, j] = L_base + self.cfg.lambda_h * h_avg + self.cfg.lambda_z * z_diff
        
        return L
    
    def update_sources_sinks(
        self,
        sources: Dict[CommodityType, List[int]],
        sinks: Dict[CommodityType, List[int]]
    ) -> None:
        """Update source and sink locations"""
        self.sources = sources
        self.sinks = sinks
    
    def step(self) -> None:
        """
        Execute one Physarum time step:
        1. Update edge lengths (coupling with environment)
        2. Solve for potentials and fluxes (§17.2)
        3. Update conductivities (§17.3)
        """
        # Update edge lengths
        self.state.L = self._compute_edge_lengths()
        
        # Solve for each commodity
        for commodity in self.commodities:
            self._solve_commodity(commodity)
        
        # Update conductivities
        self._update_conductivities()
    
    def _solve_commodity(self, commodity: CommodityType) -> None:
        """
        Solve Poisson equation for one commodity (§17.2)
        
        Σ_j (D_ij / L_ij) * (Φ_i - Φ_j) = b_i
        """
        # Build injection vector
        b = self._build_injection_vector(commodity)
        
        # Check if there's any flow to solve
        if np.abs(b).sum() < 1e-8:
            return
        
        # Build system matrix
        A = self._build_laplacian_matrix(commodity)
        
        # Add small regularization to diagonal for numerical stability
        epsilon = 1e-10
        A_reg = A.tolil()
        for i in range(self.N):
            A_reg[i, i] += epsilon
        A_reg = A_reg.tocsr()
        
        # Fix one potential to break gauge freedom (set first source to 0)
        sources_c = self.sources[commodity]
        if len(sources_c) > 0:
            anchor = sources_c[0]
            # Fix potential at anchor
            A_reg = A_reg.tolil()
            A_reg[anchor, :] = 0
            A_reg[anchor, anchor] = 1
            A_reg = A_reg.tocsr()
            b[anchor] = 0
        
        # Solve linear system
        try:
            # Use direct solver for small systems or as fallback
            if self.N < 500:
                Phi = spsolve(A_reg, b)
            else:
                # Try iterative solver with better parameters
                # Use previous solution as initial guess
                x0 = self.state.Phi[commodity].copy()
                
                Phi, info = cg(
                    A_reg,
                    b,
                    x0=x0,
                    rtol=1e-3,  # Relaxed tolerance
                    atol=1e-6,
                    maxiter=min(self.N, 500)
                )
                
                if info != 0:
                    # Fall back to direct solver
                    try:
                        Phi = spsolve(A_reg, b)
                    except:
                        # If everything fails, keep previous solution
                        return
            
            self.state.Phi[commodity] = Phi
            
        except Exception as e:
            # Keep previous solution if solver fails
            return
        
        # Compute fluxes
        self._compute_fluxes(commodity)
    
    def _build_injection_vector(self, commodity: CommodityType) -> np.ndarray:
        """Build injection vector b for commodity"""
        b = np.zeros(self.N)
        
        # Add sources
        for i in self.sources[commodity]:
            b[i] = self.cfg.I_source
        
        # Add sinks
        for i in self.sinks[commodity]:
            b[i] = -self.cfg.I_sink
        
        return b
    
    def _build_laplacian_matrix(self, commodity: CommodityType) -> lil_matrix:
        """Build weighted Laplacian matrix"""
        A = lil_matrix((self.N, self.N))
        
        for i, j in self.edges:
            D_ij = self.state.D[(i, j)][commodity]
            L_ij = self.state.L[(i, j)]
            
            # Ensure minimum conductivity and length for numerical stability
            D_ij = max(D_ij, 1e-6)
            L_ij = max(L_ij, 1e-6)
            
            weight = D_ij / L_ij
            
            # Laplacian entries
            A[i, i] += weight
            A[j, j] += weight
            A[i, j] -= weight
            A[j, i] -= weight
        
        return A
    
    def _compute_fluxes(self, commodity: CommodityType) -> None:
        """
        Compute edge fluxes from potentials (§17.2)
        
        Q_ij = (D_ij / L_ij) * (Φ_i - Φ_j)
        """
        Phi = self.state.Phi[commodity]
        
        for i, j in self.edges:
            D_ij = self.state.D[(i, j)][commodity]
            L_ij = self.state.L[(i, j)]
            
            if L_ij < 1e-8:
                Q_ij = 0.0
            else:
                Q_ij = (D_ij / L_ij) * (Phi[i] - Phi[j])
            
            self.state.Q[(i, j)][commodity] = Q_ij
    
    def _update_conductivities(self) -> None:
        """
        Update conductivities based on flux (§17.3)
        
        D_ij^{t+1} = D_ij^t + Δt * (α_D * g(|Q_ij|) - β_D * D_ij)
        g(|Q|) = |Q|^γ
        """
        dt = self.env.dt
        alpha_D = self.cfg.alpha_D
        beta_D = self.cfg.beta_D
        gamma = self.cfg.gamma_flux
        
        for edge in self.edges:
            for commodity in self.commodities:
                Q_ij = self.state.Q[edge][commodity]
                D_ij = self.state.D[edge][commodity]
                
                # Growth term
                g_flux = np.abs(Q_ij) ** gamma
                
                # Update
                dD = alpha_D * g_flux - beta_D * D_ij
                D_new = D_ij + dt * dD
                
                # Keep in reasonable range [0.01, 10.0]
                D_new = np.clip(D_new, 0.01, 10.0)
                
                self.state.D[edge][commodity] = D_new
    
    def get_edge_desirability(self, i: int, j: int) -> float:
        """
        Compute edge desirability for agent movement (§17.4)
        
        η_ij = exp(Σ_c ω_c * D_ij^(c) / L_ij)
        """
        edge = (min(i, j), max(i, j))
        
        if edge not in self.edges:
            return 0.0
        
        L_ij = self.state.L[edge]
        
        if L_ij < 1e-8:
            return 0.0
        
        # Equal weights for all commodities
        weights = {c: 1.0 / len(self.commodities) for c in self.commodities}
        
        total = 0.0
        for commodity in self.commodities:
            D_ij = self.state.D[edge][commodity]
            total += weights[commodity] * D_ij / L_ij
        
        return np.exp(total)
    
    def get_structural_entropy(self) -> float:
        """
        Compute structural entropy of network (§18.1)
        
        H_struct = -Σ_{ij} p_ij * log(p_ij + ε)
        
        Where p_ij = D_ij / Σ_{kl} D_kl
        """
        epsilon = self.cfg.alpha_D * 1e-8
        
        # Sum conductivities across all commodities and edges
        total_D = 0.0
        for edge in self.edges:
            for commodity in self.commodities:
                total_D += self.state.D[edge][commodity]
        
        if total_D < epsilon:
            return 0.0
        
        # Compute entropy
        H = 0.0
        for edge in self.edges:
            for commodity in self.commodities:
                D_ij = self.state.D[edge][commodity]
                p_ij = D_ij / (total_D + epsilon)
                
                if p_ij > epsilon:
                    H -= p_ij * np.log(p_ij + epsilon)
        
        return H
    
    def get_conductivity_2d(self, commodity: CommodityType) -> np.ndarray:
        """Get 2D view of average edge conductivities for visualization"""
        conductivity_2d = np.zeros((self.H, self.W))
        
        for i in range(self.N):
            # Average conductivity of edges connected to this node
            total_D = 0.0
            count = 0
            
            for j in self.env.state.neighbors[i]:
                edge = (min(i, j), max(i, j))
                if edge in self.state.D:
                    total_D += self.state.D[edge][commodity]
                    count += 1
            
            if count > 0:
                avg_D = total_D / count
                row, col = self.env._index_to_coords(i)
                conductivity_2d[row, col] = avg_D
        
        return conductivity_2d
    
    def detect_degenerate_network(self, threshold: float = 0.9) -> bool:
        """
        Detect if network has degenerated to a single dominant path (§18.1)
        
        Returns True if top edge captures >90% of total conductivity
        """
        # Get all conductivities
        all_D = []
        for edge in self.edges:
            for commodity in self.commodities:
                all_D.append(self.state.D[edge][commodity])
        
        if len(all_D) == 0:
            return False
        
        all_D = np.array(all_D)
        total = np.sum(all_D)
        
        if total < 1e-8:
            return False
        
        max_D = np.max(all_D)
        fraction = max_D / total
        
        return fraction > threshold
    
    def get_edge_conductivities(
        self, 
        commodity: CommodityType,
        normalize: bool = True
    ) -> Dict[Tuple[int, int], float]:
        """
        Export edge-wise conductivities for a specific commodity
        
        Returns dictionary mapping (i, j) tuples to conductivity values.
        This is used for Physarum-hydrology coupling (§1.1, §3.5).
        
        CRITICAL ORDER:
        1. Clip raw values to [0.1, 5.0] (prevent extremes)
        2. THEN normalize to mean=1.0
        This preserves relative variation while preventing runaway
        
        Args:
            commodity: Which commodity to export conductivities for
            normalize: If True, normalize conductivities to have mean=1.0
                      This prevents runaway positive feedback loops
            
        Returns:
            Dictionary of {(i, j): D_ij} for all edges
        """
        conductivities = {}
        
        for edge in self.edges:
            D_value = self.state.D[edge][commodity]
            # CLIP FIRST (before normalization)
            # This ensures weak edges stay weak and strong edges stay strong
            # relative to each other, even after normalization
            D_clipped = np.clip(D_value, 0.1, 5.0)
            conductivities[edge] = D_clipped
        
        # THEN normalize to mean=1.0 (after clipping)
        if normalize and len(conductivities) > 0:
            values = list(conductivities.values())
            mean_D = np.mean(values)
            
            # Normalize so mean = 1.0
            if mean_D > 1e-8:
                conductivities = {
                    edge: D / mean_D 
                    for edge, D in conductivities.items()
                }
        
        return conductivities
    
    # ========================================================================
    # AGENT COUPLING INTERFACE (Enhancement #2)
    # ========================================================================
    
    def get_neighbor_conductivities(
        self,
        cell_index: int,
        neighbor_indices: List[int],
        commodity: CommodityType = CommodityType.WATER
    ) -> Dict[int, float]:
        """
        Get Physarum conductivities for agent's neighbors (Enhancement #2)
        
        This allows agents to use Physarum "highways" for pathfinding.
        High conductivity = recommended transport corridor.
        
        Args:
            cell_index: Agent's current cell
            neighbor_indices: List of neighbor cells
            commodity: Which commodity type to query
        
        Returns:
            Dictionary mapping neighbor_index -> conductivity score [0, 1]
        """
        conductivities = {}
        
        for neighbor in neighbor_indices:
            # Get edge key (canonical form)
            edge_key = (min(cell_index, neighbor), max(cell_index, neighbor))
            
            if edge_key in self.state.D:
                D_value = self.state.D[edge_key][commodity]
                
                # Normalize to [0, 1] range for agent scoring
                # Physarum D is bounded [0.1, 5.0] and normalized to mean=1.0
                # Map [0.3, 3.0] (after env clipping) to [0, 1]
                D_normalized = np.clip((D_value - 0.3) / (3.0 - 0.3), 0.0, 1.0)
                conductivities[neighbor] = D_normalized
            else:
                # No edge or zero conductivity
                conductivities[neighbor] = 0.5  # Neutral score
        
        return conductivities
    
    def get_edge_desirability(
        self,
        from_cell: int,
        to_cell: int,
        commodity: CommodityType = CommodityType.WATER
    ) -> float:
        """
        Get desirability score for a specific edge (Enhancement #2)
        
        Returns:
            Score in [0, 1] where 1 = highly recommended path
        """
        edge_key = (min(from_cell, to_cell), max(from_cell, to_cell))
        
        if edge_key in self.state.D:
            D_value = self.state.D[edge_key][commodity]
            # Normalize to [0, 1]
            return float(np.clip((D_value - 0.3) / (3.0 - 0.3), 0.0, 1.0))
        else:
            return 0.5  # Neutral
    
    def compute_shortest_path_via_physarum(
        self,
        start_cell: int,
        end_cell: int,
        max_steps: int = 100
    ) -> Optional[List[int]]:
        """
        Compute path from start to end following Physarum conductivities
        
        Uses greedy best-first search guided by high-conductivity edges.
        This is a simple heuristic - agents can follow this or deviate.
        
        Returns:
            List of cell indices forming path, or None if no path found
        """
        if start_cell == end_cell:
            return [start_cell]
        
        visited = {start_cell}
        path = [start_cell]
        current = start_cell
        
        for _ in range(max_steps):
            # Get neighbors
            if current not in self.neighbors:
                break
            
            neighbors = self.neighbors[current]
            
            # Filter unvisited
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited:
                break
            
            # Score by conductivity (prefer high D) and distance to goal
            best_neighbor = None
            best_score = -float('inf')
            
            for neighbor in unvisited:
                # Conductivity score
                D_score = self.get_edge_desirability(current, neighbor)
                
                # Distance to goal (simple Manhattan heuristic)
                # Would need grid_width to compute properly - simplified here
                distance_score = 0.0  # Placeholder
                
                # Combined score (bias toward high conductivity)
                score = 0.7 * D_score + 0.3 * distance_score
                
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                break
            
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
            
            # Check if reached goal
            if current == end_cell:
                return path
        
        return None  # No path found
        
        return conductivities

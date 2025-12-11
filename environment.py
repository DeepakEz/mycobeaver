"""
Environment Module
Implements hydrology dynamics (§3), vegetation dynamics (§4), and dam/lodge structures (§5)
"""

import numpy as np
from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
from config import WorldConfig, HydrologyConfig, VegetationConfig, DamConfig


@dataclass
class EnvironmentState:
    """Complete environment state (§2)"""
    # Spatial fields (N cells)
    h: np.ndarray  # Water depth [N]
    v: np.ndarray  # Vegetation biomass [N]
    m: np.ndarray  # Soil moisture [N]
    d: np.ndarray  # Dam permeability [N]
    L: np.ndarray  # Lodge indicator [N]
    
    # Static fields
    z: np.ndarray  # Elevation [N]
    
    # Graph structure
    neighbors: Dict[int, List[int]]  # Neighbor mapping
    
    # Exogenous inputs
    r: np.ndarray  # Rainfall [N]
    b: np.ndarray  # Boundary inflow [N]
    
    # Work tracking
    u_build: np.ndarray  # Dam building effort [N]
    c_consume: np.ndarray  # Vegetation consumption [N]


class Environment:
    """
    Complete environment simulation with:
    - Hydrology dynamics (§3)
    - Vegetation and soil moisture (§4)
    - Structural memory (dams and lodges) (§5)
    """
    
    def __init__(
        self,
        world_config: WorldConfig,
        hydro_config: HydrologyConfig,
        veg_config: VegetationConfig,
        dam_config: DamConfig
    ):
        self.world_cfg = world_config
        self.hydro_cfg = hydro_config
        self.veg_cfg = veg_config
        self.dam_cfg = dam_config
        
        # Grid dimensions
        self.H = world_config.grid_height
        self.W = world_config.grid_width
        self.N = self.H * self.W
        
        # Time step
        self.dt = world_config.dt
        
        # Initialize state
        self.state = self._initialize_state()
        
        # Core habitat region (for metrics)
        self.core_habitat = self._define_core_habitat()
        
        # Downstream region (for flood metrics)
        self.downstream_region = self._define_downstream_region()
        
        # River source cells
        self.source_cells = self._define_source_cells()
        
        # Physarum network conductivity (for hydrology coupling)
        # D_water[(i,j)] = conductivity of edge between cells i and j
        self.physarum_D_water: Dict[Tuple[int, int], float] = {}
        
        # POPULATION DYNAMICS TRACKING (Enhancement #4 + #9)
        self.birth_count_this_step = 0
        self.death_count_this_step = 0
        self.total_births = 0
        self.total_deaths = 0
        
        # Terrain state tracking (for reward computation)
        self.previous_terrain_state: Optional[Dict] = None
        self.current_terrain_state: Optional[Dict] = None
    
    def _initialize_state(self) -> EnvironmentState:
        """Initialize environment state"""
        np.random.seed(self.world_cfg.random_seed)
        
        # Generate elevation field with some structure
        z = self._generate_elevation()
        
        # Initialize water depth (slightly higher in low-elevation areas)
        h = np.maximum(0, 0.5 - 0.1 * (z - z.min()) / (z.max() - z.min() + 1e-8))
        
        # Initialize vegetation
        v_mean = self.veg_cfg.initial_vegetation_mean
        v_std = self.veg_cfg.initial_vegetation_std
        v = np.maximum(0, np.random.normal(v_mean, v_std, self.N))
        
        # Initialize soil moisture
        m = np.full(self.N, self.veg_cfg.initial_moisture)
        
        # Initialize dam permeability (1 = no dam)
        d = np.ones(self.N)
        
        # Initialize lodge indicator (no lodges initially)
        L = np.zeros(self.N, dtype=np.int32)
        
        # Initialize rainfall (will be updated)
        r = np.zeros(self.N)
        
        # Initialize boundary inflow
        b = np.zeros(self.N)
        
        # Initialize work tracking
        u_build = np.zeros(self.N)
        c_consume = np.zeros(self.N)
        
        # Build neighbor structure
        neighbors = self._build_neighbor_structure()
        
        return EnvironmentState(
            h=h, v=v, m=m, d=d, L=L, z=z,
            neighbors=neighbors,
            r=r, b=b,
            u_build=u_build,
            c_consume=c_consume
        )
    
    def _generate_elevation(self) -> np.ndarray:
        """Generate elevation field with some structure"""
        # Create elevation gradient from top to bottom
        y_coords = np.linspace(0, 1, self.H)
        x_coords = np.linspace(0, 1, self.W)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Base gradient (higher at top)
        base = self.world_cfg.elevation_scale * (1 - Y)
        
        # Add roughness with Perlin-like noise
        freq = 5
        noise = np.sin(freq * 2 * np.pi * X) * np.cos(freq * 2 * np.pi * Y)
        roughness = self.world_cfg.elevation_roughness * noise
        
        elevation = base + roughness
        return elevation.flatten()
    
    def _build_neighbor_structure(self) -> Dict[int, List[int]]:
        """Build neighbor mapping (4-connected grid)"""
        neighbors = {}
        for i in range(self.N):
            row, col = self._index_to_coords(i)
            neighs = []
            
            # Up
            if row > 0:
                neighs.append(self._coords_to_index(row - 1, col))
            # Down
            if row < self.H - 1:
                neighs.append(self._coords_to_index(row + 1, col))
            # Left
            if col > 0:
                neighs.append(self._coords_to_index(row, col - 1))
            # Right
            if col < self.W - 1:
                neighs.append(self._coords_to_index(row, col + 1))
            
            neighbors[i] = neighs
        
        return neighbors
    
    def _index_to_coords(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (row, col)"""
        return (idx // self.W, idx % self.W)
    
    def _coords_to_index(self, row: int, col: int) -> int:
        """Convert (row, col) to linear index"""
        return row * self.W + col
    
    def _define_core_habitat(self) -> Set[int]:
        """Define core habitat region (center 40% of grid)"""
        core = set()
        r_start = int(0.3 * self.H)
        r_end = int(0.7 * self.H)
        c_start = int(0.3 * self.W)
        c_end = int(0.7 * self.W)
        
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                core.add(self._coords_to_index(r, c))
        
        return core
    
    def _define_downstream_region(self) -> Set[int]:
        """Define downstream region (bottom 30% of grid)"""
        downstream = set()
        r_start = int(0.7 * self.H)
        
        for r in range(r_start, self.H):
            for c in range(self.W):
                downstream.add(self._coords_to_index(r, c))
        
        return downstream
    
    def _define_source_cells(self) -> Set[int]:
        """Define river source cells (top 10% of grid, center columns)"""
        sources = set()
        r_end = int(0.1 * self.H)
        c_start = int(0.4 * self.W)
        c_end = int(0.6 * self.W)
        
        for r in range(r_end):
            for c in range(c_start, c_end):
                sources.add(self._coords_to_index(r, c))
        
        return sources
    
    def step(self) -> None:
        """
        Execute one environment time step:
        1. Update rainfall
        2. Update hydrology (§3)
        3. Update vegetation (§4)
        4. Update soil moisture (§4)
        5. Update dam permeability (§5)
        6. Reset work trackers
        """
        # Update exogenous inputs
        self._update_rainfall()
        self._update_boundary_inflow()
        
        # Update environment dynamics
        self._update_hydrology()
        self._update_vegetation()
        self._update_soil_moisture()
        self._update_dam_permeability()
        
        # Reset work trackers for next step
        self.state.u_build = np.zeros(self.N)
        self.state.c_consume = np.zeros(self.N)
    
    def set_physarum_conductivity(self, D_water: Dict[Tuple[int, int], float]):
        """
        Set Physarum network conductivity for hydrology coupling (§1.1, §3.5)
        
        This allows the Physarum slime network to directly affect
        water flow paths in the hydrology simulation.
        
        Args:
            D_water: Dictionary mapping (i, j) edges to conductivity values
        """
        self.physarum_D_water = D_water
    
    def _update_rainfall(self) -> None:
        """Generate spatially varying rainfall"""
        mean_r = self.hydro_cfg.mean_rainfall
        std_r = self.hydro_cfg.rainfall_std
        self.state.r = np.maximum(0, np.random.normal(mean_r, std_r, self.N))
    
    def _update_boundary_inflow(self) -> None:
        """Update boundary inflow at source cells"""
        self.state.b = np.zeros(self.N)
        for cell in self.source_cells:
            self.state.b[cell] = self.hydro_cfg.boundary_inflow
    
    def _update_hydrology(self) -> None:
        """
        Update water depth using head-based flow (§3)
        
        h_{i}^{t+1} = h_{i}^{t} + Δt * (r_i + b_i + Σ_j q_{ji} - Σ_j q_{ij} - ℓ(h_i))
        """
        # Compute water surface heights (head)
        H_head = self.state.z + self.state.h
        
        # Compute flows
        flow_in = np.zeros(self.N)
        flow_out = np.zeros(self.N)
        
        for i in range(self.N):
            H_i = H_head[i]
            
            for j in self.state.neighbors[i]:
                H_j = H_head[j]
                
                if H_i > H_j:
                    # Flow from i to j
                    g_ij = self._compute_conductance(i, j)
                    q_ij = g_ij * (H_i - H_j)
                    flow_out[i] += q_ij
                    flow_in[j] += q_ij
        
        # Add boundary drainage (water escapes at edges)
        boundary_drainage = self._compute_boundary_drainage()
        
        # Compute losses
        losses = self._compute_losses()
        
        # Update water depth
        dh = self.state.r + self.state.b + flow_in - flow_out - losses - boundary_drainage
        self.state.h = np.maximum(0, self.state.h + self.dt * dh)
        
        # SAFETY: Progressive overflow protection
        max_h = np.max(self.state.h)
        
        # Warning at h > 3.0
        if max_h > 3.0:
            print(f"WARNING: Water depth high (max={max_h:.2f}), may need adjustment")
        
        # Emergency clamp at h > 5.0
        if max_h > 5.0:
            print(f"EMERGENCY: Clamping water depth from {max_h:.2f} to 5.0")
            self.state.h = np.clip(self.state.h, 0, 5.0)
        
        # Catastrophic overflow protection
        if np.any(np.isnan(self.state.h)) or np.any(np.isinf(self.state.h)):
            print("CRITICAL ERROR: Numerical overflow in water depth!")
            print(f"  max(h) = {np.max(self.state.h)}")
            print(f"  max(dh) = {np.max(np.abs(dh))}")
            print(f"  max(flow_in) = {np.max(flow_in)}")
            print(f"  max(flow_out) = {np.max(flow_out)}")
            print(f"  Applying emergency reset...")
            self.state.h = np.clip(self.state.h, 0, 2.0)  # Reset to safe
    
    def _compute_boundary_drainage(self) -> np.ndarray:
        """
        Compute boundary drainage (water escaping at edges)
        This prevents water accumulation at domain boundaries
        
        CRITICAL: Reduced from 10% to 2% per step to prevent complete drainage
        """
        drainage = np.zeros(self.N)
        drainage_rate = 0.02  # 2% of water drains at boundaries per step (was 0.1)
        
        for i in range(self.N):
            row, col = self._index_to_coords(i)
            # Cells at edges lose extra water
            if row == 0 or row == self.H - 1 or col == 0 or col == self.W - 1:
                drainage[i] = drainage_rate * self.state.h[i]
        
        return drainage
    
    def _compute_conductance(self, i: int, j: int) -> float:
        """
        Compute edge conductance with Physarum coupling (§3, §1.1, §3.5)
        
        κ_{ij} = g_0 * φ(d_i, d_j) * ψ(z_i, z_j) * D_{ij}^{water}
        
        Where:
        - φ(d_i, d_j) = (d_i + d_j) / 2  (dam component)
        - ψ(z_i, z_j) = exp(-λ_z * |z_i - z_j|)  (terrain component)
        - D_{ij}^{water} = Physarum conductivity (slime network component)
        
        CRITICAL: D_water comes pre-clipped and normalized from Physarum
        We apply a final safety clip to [0.3, 3.0] (10x range) to preserve
        variation while preventing extreme values
        """
        # Dam component
        d_i = self.state.d[i]
        d_j = self.state.d[j]
        phi = 0.5 * (d_i + d_j)
        
        # Terrain component (higher elevation difference → lower conductivity)
        z_i = self.state.z[i]
        z_j = self.state.z[j]
        lambda_z = 0.1  # Terrain resistance coefficient
        psi = np.exp(-lambda_z * abs(z_i - z_j))
        
        # Physarum component (slime network conductivity)
        # Comes pre-clipped to [0.1, 5.0] and normalized to mean=1.0
        # Apply final safety clip to [0.3, 3.0] (10x range, wider than before)
        edge_key = (min(i, j), max(i, j))  # Canonical edge representation
        D_water_raw = self.physarum_D_water.get(edge_key, 1.0)
        D_water = np.clip(D_water_raw, 0.3, 3.0)  # Safety bounds (wider than [0.5, 2.0])
        
        # Total conductance
        kappa = phi * psi * D_water
        
        return self.hydro_cfg.g0 * kappa
    
    def _compute_losses(self) -> np.ndarray:
        """
        Compute evaporation and seepage losses (§3)
        
        ℓ(h_i) = α_evap * h_i + α_seep * h_i
        """
        alpha_evap = self.hydro_cfg.alpha_evap
        alpha_seep = self.hydro_cfg.alpha_seep
        return (alpha_evap + alpha_seep) * self.state.h
    
    def _update_vegetation(self) -> None:
        """
        Update vegetation biomass (§4)
        
        v_{i}^{t+1} = v_{i}^{t} + Δt * (g_v(m_i, v_i) - c_i)
        g_v(m_i, v_i) = γ_v * m_i * (1 - v_i / V_max)
        """
        # Growth term (logistic, moisture-limited)
        growth = self.veg_cfg.gamma_v * self.state.m * (
            1 - self.state.v / self.veg_cfg.V_max
        )
        
        # Update with consumption
        dv = growth - self.state.c_consume
        self.state.v = np.maximum(0, self.state.v + self.dt * dv)
    
    def _update_soil_moisture(self) -> None:
        """
        Update soil moisture (§4)
        
        m_{i}^{t+1} = (1 - β_m) * m_i + β_infil * h_i - β_drain * m_i
        """
        beta_m = self.veg_cfg.beta_m
        beta_infil = self.veg_cfg.beta_infil
        beta_drain = self.veg_cfg.beta_drain
        
        dm = -beta_m * self.state.m + beta_infil * self.state.h - beta_drain * self.state.m
        self.state.m = np.clip(self.state.m + self.dt * dm, 0, 1)
    
    def _update_dam_permeability(self) -> None:
        """
        Update dam permeability (§5)
        
        d_{i}^{t+1} = clip[0,1](d_i - η_build * u_{i,build} + η_erode * u_{i,erode})
        """
        # Compute erosion pressure from flow
        u_erode = self._compute_erosion_pressure()
        
        # Update permeability
        dd = -self.dam_cfg.eta_build * self.state.u_build + self.dam_cfg.eta_erode * u_erode
        self.state.d = np.clip(self.state.d + dd, 0, 1)
    
    def _compute_erosion_pressure(self) -> np.ndarray:
        """
        Compute erosion pressure from flow magnitude (§5)
        
        u_{i,erode} = φ(Σ_j |q_{ij}|)
        φ(x) = min(x, φ_max)
        """
        H_head = self.state.z + self.state.h
        flow_magnitude = np.zeros(self.N)
        
        for i in range(self.N):
            H_i = H_head[i]
            
            for j in self.state.neighbors[i]:
                H_j = H_head[j]
                
                if H_i > H_j:
                    g_ij = self._compute_conductance(i, j)
                    q_ij = g_ij * (H_i - H_j)
                    flow_magnitude[i] += abs(q_ij)
        
        return np.minimum(flow_magnitude, self.dam_cfg.phi_max)
    
    def add_building_effort(self, cell: int, effort: float) -> None:
        """Add dam building effort at cell"""
        if 0 <= cell < self.N:
            self.state.u_build[cell] += effort
    
    def add_vegetation_consumption(self, cell: int, amount: float) -> None:
        """Add vegetation consumption at cell"""
        if 0 <= cell < self.N:
            self.state.c_consume[cell] += amount
    
    def set_lodge(self, cell: int, is_lodge: bool = True) -> None:
        """Mark cell as lodge"""
        if 0 <= cell < self.N:
            self.state.L[cell] = 1 if is_lodge else 0
    
    def get_state_2d(self) -> Tuple[np.ndarray, ...]:
        """Get 2D views of state for visualization"""
        h_2d = self.state.h.reshape((self.H, self.W))
        v_2d = self.state.v.reshape((self.H, self.W))
        m_2d = self.state.m.reshape((self.H, self.W))
        d_2d = self.state.d.reshape((self.H, self.W))
        L_2d = self.state.L.reshape((self.H, self.W))
        z_2d = self.state.z.reshape((self.H, self.W))
        
        return h_2d, v_2d, m_2d, d_2d, L_2d, z_2d
    
    def get_core_habitat_metrics(self) -> Dict[str, float]:
        """Get metrics for core habitat region"""
        core_indices = list(self.core_habitat)
        
        h_core = self.state.h[core_indices]
        h_mean = np.mean(h_core)
        h_std = np.std(h_core)
        
        return {
            'h_mean_core': h_mean,
            'h_std_core': h_std
        }
    
    def get_downstream_metrics(self) -> Dict[str, float]:
        """Get metrics for downstream region"""
        downstream_indices = list(self.downstream_region)
        
        h_down = self.state.h[downstream_indices]
        num_flood = np.sum(h_down > self.hydro_cfg.h_flood)
        
        return {
            'num_flood_cells': int(num_flood)
        }
    
    # ========================================================================
    # POPULATION DYNAMICS & TERRAIN REWARDS (Enhancement #5 + #9)
    # ========================================================================
    
    def capture_terrain_state(self) -> Dict:
        """
        Capture current terrain state for reward computation
        
        Returns snapshot of key metrics for before/after comparison
        """
        flood_cells = np.sum(self.state.h > self.hydro_cfg.h_flood)
        drought_cells = np.sum(self.state.h < self.hydro_cfg.h_drought)
        
        terrain_state = {
            'flood_cells': int(flood_cells),
            'drought_cells': int(drought_cells),
            'total_vegetation': float(np.sum(self.state.v)),
            'water_variance': float(np.var(self.state.h)),
            'mean_water_depth': float(np.mean(self.state.h)),
            'strong_dams': int(np.sum(self.state.d < 0.5)),
        }
        
        return terrain_state
    
    def compute_terrain_rewards(
        self,
        agent_positions: Dict[int, int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute terrain-based rewards for all agents (Enhancement #5)
        
        Compares before/after terrain state to reward agents for
        improving the environment
        
        Args:
            agent_positions: Mapping of agent_id -> cell_index
        
        Returns:
            Dict mapping agent_id -> reward_components
        """
        if self.previous_terrain_state is None:
            # First step, no comparison possible
            return {agent_id: {} for agent_id in agent_positions.keys()}
        
        # Capture current state
        current = self.capture_terrain_state()
        previous = self.previous_terrain_state
        
        # Compute global changes
        Δflooding = current['flood_cells'] - previous['flood_cells']
        Δdrought = current['drought_cells'] - previous['drought_cells']
        Δvegetation = current['total_vegetation'] - previous['total_vegetation']
        Δwater_var = current['water_variance'] - previous['water_variance']
        Δdams = current['strong_dams'] - previous['strong_dams']
        
        # Shared rewards (all agents benefit from improvements)
        shared_rewards = {
            'flood_prevention': -10.0 * Δflooding,
            'drought_prevention': -5.0 * Δdrought,
            'vegetation_growth': 2.0 * Δvegetation,
            'water_stability': -3.0 * Δwater_var,
            'infrastructure': 15.0 * max(0, Δdams)  # Reward for building dams
        }
        
        # Compute per-agent rewards
        agent_rewards = {}
        
        for agent_id, position in agent_positions.items():
            rewards = shared_rewards.copy()
            
            # Local rewards (based on agent's location)
            local_h = self.state.h[position]
            
            # Reward for being near improved areas
            if previous['flood_cells'] > current['flood_cells']:
                # Check if agent near previously flooded area
                was_flooded = self.state.h[position] > self.hydro_cfg.h_flood
                if was_flooded and local_h <= self.hydro_cfg.h_flood:
                    rewards['local_flood_fix'] = 20.0
            
            if previous['drought_cells'] > current['drought_cells']:
                # Check if agent near previously dry area
                was_dry = self.state.h[position] < self.hydro_cfg.h_drought
                if was_dry and local_h >= self.hydro_cfg.h_drought:
                    rewards['local_drought_fix'] = 15.0
            
            agent_rewards[agent_id] = rewards
        
        return agent_rewards
    
    def begin_step(self):
        """Call at start of environment step to capture state"""
        self.previous_terrain_state = self.current_terrain_state
        self.current_terrain_state = self.capture_terrain_state()
        
        # Reset per-step counters
        self.birth_count_this_step = 0
        self.death_count_this_step = 0
    
    def record_birth(self):
        """Record that an agent was born this step"""
        self.birth_count_this_step += 1
        self.total_births += 1
    
    def record_death(self):
        """Record that an agent died this step"""
        self.death_count_this_step += 1
        self.total_deaths += 1
    
    def get_population_statistics(self) -> Dict[str, int]:
        """Get population dynamics statistics"""
        return {
            'births_this_step': self.birth_count_this_step,
            'deaths_this_step': self.death_count_this_step,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'net_population_change': self.total_births - self.total_deaths
        }


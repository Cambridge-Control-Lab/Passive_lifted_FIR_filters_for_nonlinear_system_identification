from __future__ import annotations

import numpy as np


def build_feature_map(active_dims: tuple[int, ...],
                      delay_steps: dict[int,int]) -> np.ndarray:
    """
      Build ordered feature map for scheduling signal pt.

      Python type note:
      - active_dims: tuple[int, ...] means active_dims is a tuple of
        integers with variable length.

      Conceptual Example:
      - if scheduling signal pt at any time t = pt = [a(t) b(t) c(t) ... ]
      Then active_dims = (0,2,3) with {0:3,2:4} 
      means the extended scheduling signal at time t is
      [a(t) c(t) a(t-1) a(t-2) a(t-3) c(t-1) c(t-2) c(t-3) c(t-4)]
      with F = feature dimension = 9.
      Here a(t), b(t), c(t), etc. are scalar values.

      Real example: 
      - active_dims = (0, 1)
      - delay_steps = {0:2, 1:3}
      - Then output of the function is 
      feature_map = np.array([
                    [0, 0],  # current dim 0
                    [1, 0],  # current dim 1
                    [0, 1],  # dim 0 delayed by 1
                    [0, 2],  # dim 0 delayed by 2
                    [1, 1],  # dim 1 delayed by 1
                    [1, 2],  # dim 1 delayed by 2
                    [1, 3],  # dim 1 delayed by 3
                ], dtype=int)
                
      Input:
      - active_dims: tuple of active base dimensions.
        For this experiment, active_dims=(0, 1) uses the first two dimensions of pt.
      - delay_steps: mapping base_dim -> number_of_delays.
        For this experiment, delay_steps={0:5, 1:5} uses five delayed copies
        for each of the first two dimensions.

      Output:
      - feature_map: numpy.ndarray, shape (F, 2), dtype int
        Each row is [base_dim, lag].
        F = number of extended features, or feature dimension.
        For active_dims=(0,1), delay_steps={0:5,1:5}, F=12.
      
      


      Ordering rule:
      1. Current-time features first: (d,0) for each d in ascending active_dims.
      2. Delayed features next: (d,1), (d,2), ..., (d,L_d), by dim then lag.
    """

    # Check input types.
    # Validate active_dims input type.
    if not isinstance(active_dims, tuple):
        raise ValueError("active_dims must be a tuple of integers.")
    # Validate delay_steps input type.
    if not isinstance(delay_steps, dict):
        raise ValueError("delay_steps must be a dict of {dim: lag_count}.")

    # Map to clean: e.g., map active_dims=(0,5,2) to [0,2,5]
    dims = []
    for dim_value in active_dims:
        dim_int = int(dim_value) # e.g., active_dims=(0,1), so dim_value is 0 then 1.
        if dim_int < 0 or dim_int > 6:
            raise ValueError("active_dims values must be in range [0, 6].")
        dims.append(dim_int)
    dims.sort()
    if len(dims) < 1:
        raise ValueError("active_dims must contain at least one dimension.")
    if len(np.unique(np.asarray(dims, dtype=int))) != len(dims):
        raise ValueError("active_dims must not contain duplicate dimensions.")

    rows = []
    for dim_int in dims: # now dims = [0,2,5]
        rows.append([dim_int, 0]) # rows = [ [0,0], [2,0], [5,0] ]
    """Build the undelayed part of the feature map.
    """

    """Append delayed copies for each active dimension.

      With:
      dims = [0, 1]
      delay_steps = {0:2, 1:3}
      it appends:

      for dim 0: [0,1], [0,2]
      for dim 1: [1,1], [1,2], [1,3]
    """
    for dim_int in dims: # If active_dims = (0,1), then dims = [0,1], dim_int = 0 or 1
        lag_count = 0
        if dim_int in delay_steps: # e.g., if delay_steps={0:2,1:3}, then delay_steps[0] = 2,  delay_steps[1] = 3
            lag_count = int(delay_steps[dim_int]) # so lag_count = 2 or 3 when dim_int = 0 or 1
            if lag_count < 0:
                raise ValueError("delay_steps values must be >= 0.")
        for lag in range(1, lag_count + 1):
            rows.append([dim_int, lag])

    # Convert list to numpy array 
    feature_map = np.asarray(rows, dtype=int)

    # Validate shape has exactly 2 columns.
    if feature_map.ndim != 2 or feature_map.shape[1] != 2:
        raise ValueError("feature_map must have shape (F, 2).")

    return feature_map


def build_p_ext_from_p7(p_7tb: np.ndarray, feature_map: np.ndarray,
                        scale_io_by_20: bool) -> np.ndarray:
    """
      Build extended scheduling tensor from base 7-channel scheduling tensor.

      Dimensions:
      - 7: scheduling signal dimension at each time step.
      - T: number of samples for each batch of training data.
      - B: number of batches of training data.

      Input:
      - p_7tb: numpy.ndarray, shape (7, T, B)
        base scheduling channels over time and trajectories.
      - feature_map: numpy.ndarray, shape (F, 2)
        each row is [base_dim, lag]. 
        Real example: active_dims = (0, 1) delay_steps = {0:2, 1:3}. 
        Then output of the function is 
        feature_map = np.array([
                    [0, 0],  # current dim 0
                    [1, 0],  # current dim 1
                    [0, 1],  # dim 0 delayed by 1
                    [0, 2],  # dim 0 delayed by 2
                    [1, 1],  # dim 1 delayed by 1
                    [1, 2],  # dim 1 delayed by 2
                    [1, 3],  # dim 1 delayed by 3
                ], dtype=int)
      - scale_io_by_20: bool
        if True, any feature whose base_dim is 0 or 1 is divided by 20.
        This applies to lag-0 and delayed copies.
        This option provides fixed scaling for the input/output channels.

      Output:
      - p_ext_ftb: numpy.ndarray, shape (F, T, B), dtype float64
        Extended scheduling-signal feature data obtained by applying the
        feature_map from build_feature_map to raw training data p_7tb.
      
      Example:
      - If T = 4, B = 2, feature_map = as listed above of shape (F=7,2)
      Then p_ext_ftb has the shape (F=7,T=4,B=2)
      - if feature is [a,b,D1a,D2a,D1b,D2b,D3b]
      where D2 means apply two steps of delay. 
      Then assume that a and b have two batches of data. 
      a1 = [10;20;30;40]
      a2 = [11;21;31;41]
      b1 = [0;200;300;400]
      b2 = [0;201;301;401]
      - Then p_ext_ftb[0,:,:] stores all batches of data (a1,a2) for first feature a
      That is: 
      p_ext_ftb[0,:,:] = [10 11; 20 21; 30 31; 40 41] of shape(4,2)
      And 
      p_ext_ftb[1,:,:] = [100 101; 200 201; 300 301; 400 401] of shape(4,2)
      And 
      p_ext_ftb[2,:,:] = data for D1a1 and D1a2 ( 1 step delay )
      [10 11; 10 11; 20 21; 30 31] of shape(4,2)
      And 
      p_ext_ftb[3,:,:] = data for D2a1 and D2a2 ( 2 step delay )
      [10 11; 10 11; 10 11; 20 21] of shape(4,2)
      p_ext_ftb[6,:,:] = data for D3b1 and D2b2 ( 3 step delay )
      [0 0; 0 0; 0 0; 0 0] of shape(4,2)

      Delay edge rule:
      - For a requested lag where t-lag < 0, use source time index 0.
        This is the "edge hold" rule.
    """
    # Convert input to a numeric array.
    p_7tb = np.asarray(p_7tb, dtype=float)

    # Validate input dimensions.
    if p_7tb.ndim != 3:
        raise ValueError(f"p_7tb must be 3D (7,T,B), got shape {p_7tb.shape}")
    if p_7tb.shape[0] != 7:
        raise ValueError(f"p_7tb first axis must be 7, got {p_7tb.shape[0]}")

    # Validate feature map dimensions.
    feature_map = np.asarray(feature_map, dtype=int)
    if feature_map.ndim != 2 or feature_map.shape[1] != 2:
        raise ValueError("feature_map must have shape (F,2).")

    # Determine the dimension of the output of this function: shape (F, T, B),
    feature_count = int(feature_map.shape[0]) # Get F 
    n_time = int( p_7tb.shape[1] ) # Get T 
    n_batch = int( p_7tb.shape[2] ) # Get B

    # Allocate zero tensor for output of this function of shape (F,T,B)
    p_ext_ftb = np.zeros((feature_count, n_time, n_batch), dtype=float)

    # Fill each extended feature channel with explicit loops.
    # Fill delayed features using the edge-hold rule.
    for f_index in range(feature_count):
        # Read base dimension and lag for this feature.
        base_dim = int(feature_map[f_index, 0])
        lag = int(feature_map[f_index, 1])
        if base_dim < 0 or base_dim >= p_7tb.shape[0]:
            raise ValueError(
                f"feature_map[{f_index},0]={base_dim} is out of valid range [0,{p_7tb.shape[0]-1}]"
            )
        if lag < 0:
            raise ValueError(f"feature_map[{f_index},1]={lag} must be >= 0.")

        # Fill over time and batch.
        for t_index in range(n_time):
            # Compute source time with edge hold.
            src_t = t_index - lag
            if src_t < 0:
                src_t = 0

            # Copy entire batch row at once for this time.
            p_ext_ftb[f_index, t_index, :] = p_7tb[base_dim, src_t, :]

        # Optional scaling for channels based on base dims 0 and 1.
        if scale_io_by_20:
            if base_dim == 0 or base_dim == 1:
                p_ext_ftb[f_index, :, :] = p_ext_ftb[f_index, :, :] / 20.0


    return p_ext_ftb


def split_batch_indices(
        n_batch: int,
        split_counts: tuple[int, int, int],
        split_seed: int,
        shuffle: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build deterministic train/val/test indices on batch axis.

    Usage:
        cfg["train_val_test_split"] = (16, 2, 2)
        cfg["split_seed"] = 42
        cfg["shuffle_split"] = True
        cfg["deterministic_epoch_shuffle"] = True

        tr_idx, va_idx, te_idx = features.split_batch_indices(
        n_batch=n_batch,
        split_counts=cfg["train_val_test_split"],
        split_seed=int(cfg["split_seed"]),
        shuffle=bool(cfg["shuffle_split"]))

    Input:
    - n_batch: int
      total number of trajectories B.
    - split_counts: tuple (n_tr, n_va, n_te)
      must sum to n_batch.
    - split_seed: int
      random seed for deterministic shuffling.
    - shuffle: bool
      whether to shuffle before split.

    Output:
    - tr_idx: numpy.ndarray, shape (n_tr,), dtype int
    - va_idx: numpy.ndarray, shape (n_va,), dtype int
    - te_idx: numpy.ndarray, shape (n_te,), dtype int

    Rule:
    - Match the old code behavior:
      1) create index array [0,1,...,B-1],
      2) shuffle using default_rng(seed) if shuffle=True,
      3) take first n_tr for train, next n_va for val, remaining for test,
      4) sort each split array.
    """

    # Get split rules and total batch count.
    n_batch = int(n_batch) # B 
    n_tr = int(split_counts[0])
    n_va = int(split_counts[1])
    n_te = int(split_counts[2])

    if n_tr + n_va + n_te != n_batch:
        raise ValueError("split_counts must sum to n_batch.")

    # Build full index list. [0,1,...,B-1],
    all_idx = np.arange(n_batch, dtype=int)

    # Deterministic shuffle if requested.
    if shuffle:
        rng = np.random.default_rng(int(split_seed))
        rng.shuffle(all_idx)

    # Slice split segments.
    tr_idx = all_idx[0:n_tr]
    va_idx = all_idx[n_tr:n_tr + n_va]
    te_idx = all_idx[n_tr + n_va:n_tr + n_va + n_te]

    # Sort each split so instead of 5, 10, 2 for training, we get 2 5 10.
    tr_idx = np.sort(tr_idx)
    va_idx = np.sort(va_idx)
    te_idx = np.sort(te_idx)

    # Return split arrays.
    return tr_idx, va_idx, te_idx


def make_fixed_exponential_fir_bank(
        n_branch: int,
        m_fir: int,
        dt: float,
        seed: int,
        tau_min: float,
        tau_max: float,
        gain_min: float,
        gain_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
    Build deterministic positive decaying FIR bank.

    Input:
    - n_branch: int, J = number of branches
    - m_fir: int =  M = FIR length
    - dt: float, sample time
    - seed: int, random seed used for small jitter
    - tau_min, tau_max: float, exponential time-constant range
    - gain_min, gain_max: float, gain range

    Output:
    - g_bank_jm: numpy.ndarray, shape (J, M), dtype float64
      FIR taps for each branch.
    - taus_j: numpy.ndarray, shape (J,), dtype float64
      time constant per branch.
    - gains_j: numpy.ndarray, shape (J,), dtype float64
      gain per branch.

    Formula for each branch j and tap k:
    - g_j[k] = gain_j * exp(-(k*dt)/tau_j)
  """

  # Get branch number, FIR order and dt
  n_branch = int(n_branch)
  m_fir = int(m_fir)
  dt = float(dt)

  # Basic validation.
  if n_branch < 1:
      raise ValueError("n_branch must be >= 1")
  if m_fir < 1:
      raise ValueError("m_fir must be >= 1")
  if dt <= 0.0:
      raise ValueError("dt must be > 0")
  if tau_min <= 0.0 or tau_max <= 0.0:
      raise ValueError("tau_min and tau_max must be > 0")
  if tau_min > tau_max:
      raise ValueError("tau_min must be <= tau_max")
  if gain_min <= 0.0 or gain_max <= 0.0:
      raise ValueError("gain_min and gain_max must be > 0")
  if gain_min > gain_max:
      raise ValueError("gain_min must be <= gain_max")
  
  # Create a random generator with the specified seed.
  rng = np.random.default_rng(int(seed))

  # Build initial tau grid.
  taus_j = np.linspace(tau_min, tau_max, n_branch, dtype=float)

  # Add small deterministic jitter if branch count is more than 1.
  # Apply optional deterministic jitter.
  if n_branch > 1:
      tau_span = tau_max - tau_min
      tau_jitter = 0.02 * tau_span * (rng.random(n_branch) - 0.5)
      taus_j = taus_j + tau_jitter

  # Clip and sort
  taus_j = np.clip(taus_j, tau_min, tau_max)
  taus_j = np.sort(taus_j)

  # Build gain grid
  gains_j = np.linspace(gain_min, gain_max, n_branch, dtype=float)

  # Add small deterministic jitter to gains.
  if n_branch > 1:
      gain_span = gain_max - gain_min
      gain_jitter = 0.03 * gain_span * (rng.random(n_branch) - 0.5)
      gains_j = gains_j + gain_jitter
    
  # Clip gains.
  gains_j = np.clip(gains_j, gain_min, gain_max)

  # Build FIR taps: g_j[k] = gain_j * exp(-(k*dt)/tau_j)
  g_bank_jm = np.zeros((n_branch,m_fir), dtype=float)
  for j in range(n_branch):
      tau_j = float(taus_j[j])
      gain_j = float(gains_j[j])
      for k in range(m_fir):
          exponent_value = - ( (k * dt) / tau_j )
          g_value = gain_j * np.exp(exponent_value)
          g_bank_jm[j, k] = g_value

  return g_bank_jm, taus_j.astype(float), gains_j.astype(float)
  

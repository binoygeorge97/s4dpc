"""Data generation for the microgrid identification pipeline.

fast_vectorized_aprbs is ported byte-verbatim from the user's data.py.
generate_microgrid_trajectory and create_microgrid_dataset are ported with
one deliberate change from verbatim: the APRBS amplitude range, originally
hardcoded to (-10.0, 10.0) inside generate_microgrid_trajectory, is now the
aprbs_low/aprbs_high keyword arguments (same defaults, so behavior is
unchanged unless a caller overrides them) - so a controller's max_action
can be set relative to it instead of a buried constant. Nothing else about
either function changed.

Cases 8 and 9 (from s4dpc.systems.get_discrete_matrices) are 1x1 and 2x2
systems - different d_input/d_output than cases 1-7's fixed 9/6. These
functions handle them fine individually (fully dimension-agnostic), but
that breaks the "stack cases into one batch axis" assumption a vmapped
sweep needs. Sweeps use cases 1-7 only; 8/9 stay supported for one-off
dimension-agnostic testing.
"""
import numpy as np

from s4dpc.systems import get_discrete_matrices


def fast_vectorized_aprbs(batch_size, length, low, high, hold_prob, rng, Nu):
    """
    Generates Amplitude Pseudo-Random Binary Signals (APRBS) across a dynamic 
    number of input control channels (Nu) without hardcoded loop limits.
    """
    signals = np.zeros((batch_size, Nu, length))
    current_vals = rng.uniform(low, high, size=(batch_size, Nu))
    
    for t in range(length):
        # Determine which channels in which batches switch values at this step
        switch_mask = rng.rand(batch_size, Nu) > hold_prob
        new_vals = rng.uniform(low, high, size=(batch_size, Nu))
        
        # Update values based on the dynamic mask
        current_vals = np.where(switch_mask, new_vals, current_vals)
        signals[:, :, t] = current_vals
        
    return signals

def generate_microgrid_trajectory(batch_size, length=100, seed=42, system_case=3, dt=0.01, aprbs_low=-10.0, aprbs_high=10.0):
    """
    Rolls out state trajectories using discrete physics matrices. 
    Constructs the exact input matrix format [X_k, U_k] expected by the S4 model.
    """
    rng = np.random.RandomState(seed)
    Ad, Bd = get_discrete_matrices(dt, case=system_case)
    
    # --- DYNAMIC MATRIX BOUNDING ---
    Nx = Ad.shape[0]   # Number of system states
    Nu = Bd.shape[1]   # Number of control inputs
    D_in = Nx + Nu     # Total neural network input channels
    
    # Generate random control steps for all channels simultaneously
    U_signals = fast_vectorized_aprbs(
        batch_size=batch_size, 
        length=length, 
        low=aprbs_low, 
        high=aprbs_high, 
        hold_prob=0.8, 
        rng=rng, 
        Nu=Nu
    )
    
    # Initialize arrays based on dynamic dimensions
    batch_inputs = np.zeros((batch_size, length, D_in))
    batch_targets = np.zeros((batch_size, length, Nx))
    
    # Initialize initial physical state X_0
    X_current = rng.uniform(-2.0, 2.0, size=(batch_size, Nx))
    
    for k in range(length):
        U_k = U_signals[:, :, k]  # Control input at step k: Shape (Batch, Nu)
        
        # --- DIMENSION-AGNOSTIC SLICING ---
        batch_inputs[:, k, 0:Nx] = X_current
        batch_inputs[:, k, Nx:D_in] = U_k
        
        # State-space transition equation: X_{k+1} = X_k * Ad^T + U_k * Bd^T
        X_next = X_current @ Ad.T + U_k @ Bd.T
        batch_targets[:, k, :] = X_next
        X_current = X_next

    return batch_inputs, batch_targets

def create_microgrid_dataset(bsz=32, L=100, system_case=3, dt=0.01, aprbs_low=-10.0, aprbs_high=10.0):
    """
    Unified entry point wrapper. Mimics standard PyTorch/JAX dataset structures 
    by outputting accessible sequence batches, along with extracted dimension bounds.
    """
    # Generate full standalone training and validation sets
    train_x, train_y = generate_microgrid_trajectory(
        batch_size=bsz * 10, length=L, seed=42, system_case=system_case, dt=dt,
        aprbs_low=aprbs_low, aprbs_high=aprbs_high,
    )
    test_x, test_y = generate_microgrid_trajectory(
        batch_size=bsz, length=L, seed=101, system_case=system_case, dt=dt,
        aprbs_low=aprbs_low, aprbs_high=aprbs_high,
    )
    
    # Package into iterable list-of-tuples structure matching your evaluation loop
    train_loader = [(train_x[i:i+bsz], train_y[i:i+bsz]) for i in range(0, len(train_x), bsz)]
    test_loader = [(test_x, test_y)]
    
    # Extract dimensions explicitly from the generated data arrays
    d_input = train_x.shape[-1]
    d_output = train_y.shape[-1]
    
    return train_loader, test_loader, d_input, d_output

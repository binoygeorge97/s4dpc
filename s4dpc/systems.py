"""Discrete-time LTI plants (CLAUDE.md §1, §2): 7 systems, all A:(6,6), B:(6,3)
for cases 1-7 (cases 8/9 exist for dimension-agnostic testing but have
different shapes - 1x1 and 2x2 - and are not used in vmapped sweeps).

get_discrete_matrices is the CANONICAL discretization: bilinear (Tustin)
at dt=0.01, ported verbatim from the user's data.py (byte-exact, including
its original trailing whitespace/CRLF, extracted programmatically rather
than retyped) - WITH ONE DELIBERATE EXCEPTION: case 4's off-diagonals were
rescaled (100/80/120 -> 5/4/6) after np.linalg.eig produced a nonsensical
cond(eigenvectors)=0 (impossible; condition numbers are >= 1) on the
original, too-extreme defective Jordan structure. See docs/DECISIONS.md.

get_discrete_matrices_zoh is a comparison-only variant, added per instruction
(NOT verbatim, new code): the same continuous-time A/B construction,
discretized via scipy.signal.cont2discrete(method="zoh") at dt=0.02 instead
of the manual bilinear transform, to settle which discretization is
canonical (bilinear@0.01 vs an independent ZOH@0.02 codepath the user has
elsewhere). See tests/test_systems.py for the empirical comparison against
known rho(A_d) values that made this determination, and its printed table.
"""
import numpy as np
from scipy.signal import cont2discrete


def get_discrete_matrices(dt=0.01, case=3):
    """
    Generates continuous-time state-space systems and performs bilinear 
    (Tustin) discretization. Completely agnostic to state/control dimensions.
    """
    Z = np.zeros((2, 2))
    
    # ==========================================
    # SELECT CONTINUOUS 'A' MATRIX
    # ==========================================
    if case == 1:
        # Case 1: Base Original (Stable with some coupling)
        A = np.block([
            [np.array([[-3.5, -2.4], [0.0, 0.0]]), np.array([[0.0, 0.03], [0.0, 0.0]]), np.array([[0.0, 0.06], [0.0, 0.0]])],
            [Z, np.array([[-3.5, -2.3], [0.0, 0.0]]), Z], 
            [Z, Z, np.array([[-5.2, -5.3], [0.0, 0.0]])]
        ])

    elif case == 2:
        # Case 2: Unstable Uncoupled
        A = np.block([
            [np.array([[1.0, 0.5], [0.0, 2.0]]), Z, Z], 
            [Z, np.array([[1.5, 0.5], [0.0, 3.0]]), Z], 
            [Z, Z, np.array([[0.5, 0.5], [0.0, 4.0]])]
        ])

    elif case == 3:
        # Case 3: Oscillatory Unstable (Complex Eigenvalues)
        A4 = np.array([[ 1.0,  2.0], [-2.0,  1.0]])
        A5 = np.array([[ 2.0,  5.0], [-5.0,  2.0]])
        A6 = np.array([[ 1.5, 10.0], [-10.0, 1.5]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])

    elif case == 4:
        # Case 4: Non-normal Jordan Block
        A4 = np.array([[1.0, 5.0], [0.0,   1.0]])
        A5 = np.array([[1.5,  4.0], [0.0,   1.5]])
        A6 = np.array([[2.0, 6.0], [0.0,   2.0]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])

    elif case == 5:
        # Case 5: Weakly Coupled Unstable Transition
        A4 = np.array([[1.0, 1.0], [0.0, 2.0]])
        A5 = np.array([[1.5, 0.7], [0.0, 3.0]])
        A6 = np.array([[0.5, 1.2], [0.0, 4.0]])
        
        H45 = np.array([[ 0.3, -0.2], [ 0.0,  0.4]])
        H46 = np.array([[ 0.1,  0.0], [ 0.0, -0.3]])
        H56 = np.array([[ 0.25, -0.15], [ 0.0,  0.2]])
        
        A = np.block([
            [A4, H45, H46],
            [Z,  A5,  H56],
            [Z,  Z,   A6 ]
        ])

    elif case == 6:
        # Case 6: Ill-conditioned Eigenvectors
        A4 = np.array([[1.0, 1000.0], [0.001,   1.0]])
        A5 = np.array([[1.5,  800.0], [0.002,   1.5]])
        A6 = np.array([[2.0, 1200.0], [0.0015,  2.0]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])

    elif case == 7:
        # Case 7: Mixed Stable / Unstable Modes
        A4 = np.array([[-2.0, 0.5], [ 0.0, 0.5]])
        A5 = np.array([[-1.0, 0.5], [ 0.0, 1.5]])
        A6 = np.array([[-3.0, 0.5], [ 0.0, 2.0]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])
        
    elif case == 8:
        # Case 8: 1x1 System 
        A = np.array([[3.0]])
        B = np.array([[1.0]]) 
        
    elif case == 9:
        # Case 9: 2x2 System
        A = np.array([
            [2.5, 1.0],
            [0.0, 1.5]
        ])
        # 1 control input affecting the second state
        B = np.array([
            [0.0],
            [1.0]
        ])
        
    else:
        raise ValueError(f"Case {case} is not supported. Please select a case between 1 and 9.")

    # ==========================================
    # SELECT 'B' MATRIX FOR CASES 1-7
    # ==========================================
    if 1 <= case <= 7:
        B_i = np.array([[0.0], [1.0]])
        B = np.block([
            [B_i, np.zeros((2,1)), np.zeros((2,1))],
            [np.zeros((2,1)), B_i, np.zeros((2,1))],
            [np.zeros((2,1)), np.zeros((2,1)), B_i]
        ])
        
    # --- DYNAMIC DIMENSION EXTRACTION ---
    Nx = A.shape[0]
    Nu = B.shape[1]

    # Bilinear (Tustin) Transformation: 
    I = np.eye(Nx)
    inv_term = np.linalg.inv(I - (dt / 2.0) * A)
    
    Ad = inv_term @ (I + (dt / 2.0) * A)
    Bd = inv_term @ B * dt
    
    return Ad, Bd

def get_discrete_matrices_zoh(dt=0.02, case=3):
    """Comparison-only. Same continuous-time systems as get_discrete_matrices
    (construction duplicated below, not shared, so that function stays
    byte-verbatim), discretized via scipy.signal.cont2discrete(method="zoh")
    instead of the bilinear transform. Not the canonical path."""
    Z = np.zeros((2, 2))
    
    # ==========================================
    # SELECT CONTINUOUS 'A' MATRIX
    # ==========================================
    if case == 1:
        # Case 1: Base Original (Stable with some coupling)
        A = np.block([
            [np.array([[-3.5, -2.4], [0.0, 0.0]]), np.array([[0.0, 0.03], [0.0, 0.0]]), np.array([[0.0, 0.06], [0.0, 0.0]])],
            [Z, np.array([[-3.5, -2.3], [0.0, 0.0]]), Z], 
            [Z, Z, np.array([[-5.2, -5.3], [0.0, 0.0]])]
        ])

    elif case == 2:
        # Case 2: Unstable Uncoupled
        A = np.block([
            [np.array([[1.0, 0.5], [0.0, 2.0]]), Z, Z], 
            [Z, np.array([[1.5, 0.5], [0.0, 3.0]]), Z], 
            [Z, Z, np.array([[0.5, 0.5], [0.0, 4.0]])]
        ])

    elif case == 3:
        # Case 3: Oscillatory Unstable (Complex Eigenvalues)
        A4 = np.array([[ 1.0,  2.0], [-2.0,  1.0]])
        A5 = np.array([[ 2.0,  5.0], [-5.0,  2.0]])
        A6 = np.array([[ 1.5, 10.0], [-10.0, 1.5]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])

    elif case == 4:
        # Case 4: Non-normal Jordan Block
        A4 = np.array([[1.0, 5.0], [0.0,   1.0]])
        A5 = np.array([[1.5,  4.0], [0.0,   1.5]])
        A6 = np.array([[2.0, 6.0], [0.0,   2.0]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])

    elif case == 5:
        # Case 5: Weakly Coupled Unstable Transition
        A4 = np.array([[1.0, 1.0], [0.0, 2.0]])
        A5 = np.array([[1.5, 0.7], [0.0, 3.0]])
        A6 = np.array([[0.5, 1.2], [0.0, 4.0]])
        
        H45 = np.array([[ 0.3, -0.2], [ 0.0,  0.4]])
        H46 = np.array([[ 0.1,  0.0], [ 0.0, -0.3]])
        H56 = np.array([[ 0.25, -0.15], [ 0.0,  0.2]])
        
        A = np.block([
            [A4, H45, H46],
            [Z,  A5,  H56],
            [Z,  Z,   A6 ]
        ])

    elif case == 6:
        # Case 6: Ill-conditioned Eigenvectors
        A4 = np.array([[1.0, 1000.0], [0.001,   1.0]])
        A5 = np.array([[1.5,  800.0], [0.002,   1.5]])
        A6 = np.array([[2.0, 1200.0], [0.0015,  2.0]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])

    elif case == 7:
        # Case 7: Mixed Stable / Unstable Modes
        A4 = np.array([[-2.0, 0.5], [ 0.0, 0.5]])
        A5 = np.array([[-1.0, 0.5], [ 0.0, 1.5]])
        A6 = np.array([[-3.0, 0.5], [ 0.0, 2.0]])
        A = np.block([
            [A4, Z, Z],
            [Z, A5, Z],
            [Z, Z, A6]
        ])
        
    elif case == 8:
        # Case 8: 1x1 System 
        A = np.array([[3.0]])
        B = np.array([[1.0]]) 
        
    elif case == 9:
        # Case 9: 2x2 System
        A = np.array([
            [2.5, 1.0],
            [0.0, 1.5]
        ])
        # 1 control input affecting the second state
        B = np.array([
            [0.0],
            [1.0]
        ])
        
    else:
        raise ValueError(f"Case {case} is not supported. Please select a case between 1 and 9.")

    # ==========================================
    # SELECT 'B' MATRIX FOR CASES 1-7
    # ==========================================
    if 1 <= case <= 7:
        B_i = np.array([[0.0], [1.0]])
        B = np.block([
            [B_i, np.zeros((2,1)), np.zeros((2,1))],
            [np.zeros((2,1)), B_i, np.zeros((2,1))],
            [np.zeros((2,1)), np.zeros((2,1)), B_i]
        ])
        
    # --- DYNAMIC DIMENSION EXTRACTION ---
    Nx = A.shape[0]
    Nu = B.shape[1]

    # Zero-order hold, for comparison against the canonical bilinear path only
    C = np.eye(Nx)
    D = np.zeros((Nx, Nu))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt, method="zoh")

    return Ad, Bd

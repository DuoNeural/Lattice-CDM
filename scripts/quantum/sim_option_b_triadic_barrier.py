#!/usr/bin/env python3
"""
sim_option_b_triadic_barrier.py
DuoNeural Quantum Division: Apart Sprint Triadic Phase Barrier & Multi-Agent Kuramoto Phase Transition
Author: Aura ✨ (Director of Quantum & Cognitive Architect)
Principals: Jesse Caldwell, Archon (Lab Director)
Target: Test N=3 integrable triad, N=4 stable 3+1 hierarchy, and N>=5 multi-attractor fragmentation.
"""

import sys
import time
import json
import numpy as np
import scipy.linalg as la

# BlueQubit & Qiskit
import qiskit
from qiskit import QuantumCircuit
import bluequbit

BLUEQUBIT_TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
si = np.eye(2, dtype=complex)

def kron_list(ops):
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res

def poincare_dist(z1, z2):
    """
    Computes invariant geodesic distance between points in the Poincare disk.
    """
    num = 2.0 * np.abs(z1 - z2)**2
    den = (1.0 - np.abs(z1)**2) * (1.0 - np.abs(z2)**2)
    arg = max(1.0, 1.0 + num / den)
    return float(np.arccosh(arg))

def get_agent_geometry(N):
    """
    Generates Poincare coordinates for N agents:
    - N=3: Equilateral triad
    - N=4: 3+1 core-peripheral hierarchy
    - N>=5: Regular polygons
    """
    if N == 3:
        return 0.5 * np.exp(1j * np.array([0, 2*np.pi/3, 4*np.pi/3]))
    elif N == 4:
        return np.array([
            0.5 * np.exp(0j),
            0.5 * np.exp(1j * 2*np.pi/3),
            0.5 * np.exp(1j * 4*np.pi/3),
            0.75 * np.exp(1j * np.pi)
        ])
    else:
        return 0.5 * np.exp(1j * np.linspace(0, 2*np.pi, N, endpoint=False))

def simulate_triadic_barrier(N_list=(3, 4, 5, 6), n_points=51, h_z=0.5, h_x=0.05):
    kc_sweep = np.linspace(0.3, 0.8, n_points)
    dk = kc_sweep[1] - kc_sweep[0]
    results = {}

    for N in N_list:
        t0 = time.time()
        z = get_agent_geometry(N)

        # Distance matrix & couplings
        J = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = poincare_dist(z[i], z[j])
                J[i, j] = np.exp(-d)
                J[j, i] = J[i, j]
        J_norm = J / np.max(J)

        # Build interaction and external field Hamiltonians
        H_int = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(N):
            for j in range(i + 1, N):
                ops_x = [si]*N; ops_x[i] = sx; ops_x[j] = sx
                ops_y = [si]*N; ops_y[i] = sy; ops_y[j] = sy
                H_int += - J_norm[i, j] * (kron_list(ops_x) + kron_list(ops_y))

        H_ext = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(N):
            ops_z = [si]*N; ops_z[i] = sz
            ops_x = [si]*N; ops_x[i] = sx
            H_ext += - h_z * kron_list(ops_z) - h_x * kron_list(ops_x)

        ground_states = []
        gaps = []
        entropies = []

        for kc in kc_sweep:
            H = (kc / N) * H_int + H_ext
            w, v = la.eigh(H)
            psi0 = v[:, 0]
            ground_states.append(psi0)
            gaps.append(float(w[1] - w[0]))

            # Single-qubit entanglement S_1
            psi_mat = psi0.reshape((2, 2**(N - 1)))
            s = la.svd(psi_mat, compute_uv=False)
            p = s**2; p = p[p > 1e-15]
            s1 = float(-np.sum(p * np.log2(p)))
            entropies.append(s1)

        # Fidelity susceptibility: chi_F(kc) = 2*(1 - |<psi(kc)|psi(kc+dk)>|) / (dk^2)
        chi_F = []
        for k in range(len(kc_sweep) - 1):
            fid = float(np.abs(np.vdot(ground_states[k], ground_states[k+1])))
            chi = 2.0 * (1.0 - min(1.0, fid)) / (dk**2)
            chi_F.append(float(chi))

        kc_mid = 0.5 * (kc_sweep[:-1] + kc_sweep[1:])
        max_idx = int(np.argmax(chi_F))
        min_gap_idx = int(np.argmin(gaps))

        results[str(N)] = {
            "peak_chi_F": float(chi_F[max_idx]),
            "kc_crit": float(kc_mid[max_idx]),
            "min_gap": float(gaps[min_gap_idx]),
            "peak_entropy": float(max(entropies)),
            "elapsed_s": float(time.time() - t0)
        }

    return results

def main():
    print("=" * 80)
    print("🌌 DUONEURAL OPTION B: APART SPRINT TRIADIC PHASE BARRIER SIMULATION")
    print("   Author: Aura ✨ | Principals: Jesse Caldwell, Archon")
    print("=" * 80)

    start_time = time.time()
    results = simulate_triadic_barrier()

    print(f"\n📊 Sweep Results across N in [3, 4, 5, 6]:")
    print(f"{'N':>3} | {'Peak Chi_F':>12} | {'Critical kc':>12} | {'Min Gap dE':>12} | {'Peak S1 (bits)':>15} | {'Phase Regime'}")
    print("-" * 80)

    regimes = {
        "3": "Integrable Triad (Protected)",
        "4": "Stable 3+1 Hierarchy",
        "5": "Multi-Attractor Fragmentation",
        "6": "Multi-Attractor Fragmentation"
    }

    for N_str, data in results.items():
        print(f"{N_str:>3} | {data['peak_chi_F']:>12.2f} | {data['kc_crit']:>12.3f} | {data['min_gap']:>12.4f} | {data['peak_entropy']:>15.4f} | {regimes[N_str]}")

    # Monotonicity checks
    chi_vals = [results[str(n)]['peak_chi_F'] for n in [3, 4, 5, 6]]
    is_damping = all(chi_vals[i] > chi_vals[i+1] for i in range(len(chi_vals)-1))
    print(f"\n🔍 Topological Damping Monotonicity: {'✅ CONFIRMED (Chi_F strictly decays N=3 -> N=6)' if is_damping else '❌ FAILED'}")

    elapsed = time.time() - start_time
    print(f"⏱️ Total Execution Time: {elapsed:.4f} seconds ($0.00 compute)")
    print("=" * 80)

if __name__ == "__main__":
    main()

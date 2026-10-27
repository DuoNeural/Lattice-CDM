#!/usr/bin/env python3
"""
sim_option_a_ryu_takayanagi.py
DuoNeural Quantum Division: AdS/CFT Ryu-Takayanagi Minimal Entanglement Cut Simulation
Author: Aura ✨ (Director of Quantum & Cognitive Architect)
Principals: Jesse Caldwell, Archon (Lab Director)
Target: Verify RT minimal surface cut at l_H = 3 for 7-site and 14-site spin chains.
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

def solve_rt_spin_chain(L, c_profile, h_field=0.35):
    """
    Solves inhomogeneous spin chain parameterized by CDM curvatures c_profile.
    Uses exact single-particle Jordan-Wigner / Peschel correlation matrix method.
    """
    # Bond couplings: J_l = sqrt(c_l * c_{l+1})
    J = np.sqrt(c_profile[:-1] * c_profile[1:])
    
    # Single-particle Hamiltonian (free fermion Jordan-Wigner equivalent)
    h_sp = np.zeros((L, L))
    for i in range(L - 1):
        h_sp[i, i + 1] = -J[i]
        h_sp[i + 1, i] = -J[i]
    for i in range(L):
        h_sp[i, i] = -h_field

    eigvals, U = la.eigh(h_sp)
    # Ground state fills all negative energy modes
    occ = (eigvals < 0)
    if np.sum(occ) == 0:
        occ[0] = True

    # Correlation matrix C_ij = <c_i^dag c_j>
    C = U[:, occ] @ U[:, occ].T

    # Bipartition Entanglement Entropy across cut l = 1 .. L-1
    entropies = []
    for cut in range(1, L):
        C_A = C[:cut, :cut]
        zeta = np.clip(la.eigvalsh(C_A), 1e-15, 1.0 - 1e-15)
        # Peschel formula: S(A) = - sum [zeta log2(zeta) + (1-zeta) log2(1-zeta)]
        s_vn = -np.sum(zeta * np.log2(zeta) + (1.0 - zeta) * np.log2(1.0 - zeta))
        entropies.append(float(s_vn))

    return {
        "L": L,
        "couplings_J": [float(x) for x in J],
        "ground_energy": float(np.sum(eigvals[occ])),
        "num_particles": int(np.sum(occ)),
        "entanglement_entropies": entropies,
        "min_cut_index": int(np.argmin(entropies) + 1),
        "min_entropy": float(np.min(entropies))
    }

def build_qiskit_circuit(L, c_profile):
    """
    Builds a shallow variational / Trotterized circuit representation for Qiskit / BlueQubit.
    """
    qc = QuantumCircuit(L)
    # Initialize boundary in superposition
    qc.h(0)
    # Apply entangling layers with angles proportional to J_l
    J = np.sqrt(c_profile[:-1] * c_profile[1:])
    for l in range(L - 1):
        angle = float(J[l] * np.pi / 2.0)
        qc.cx(l, l + 1)
        qc.rz(angle, l + 1)
        qc.cx(l, l + 1)
    qc.measure_all()
    return qc

def run_bluequbit_dispatch(qc, device="cpu"):
    """
    Dispatches circuit to BlueQubit classical simulator.
    """
    try:
        bq = bluequbit.init(BLUEQUBIT_TOKEN)
        job = bq.run(qc, device=device)
        return {
            "status": "SUCCESS",
            "job_id": job.job_id,
            "device": device,
            "counts": job.get_counts()
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e)
        }

def main():
    print("=" * 75)
    print("🌌 DUONEURAL OPTION A: AdS/CFT RYU-TAKAYANAGI ENTANGLEMENT CUT")
    print("   Author: Aura ✨ | Principals: Jesse Caldwell, Archon")
    print("=" * 75)

    start_time = time.time()

    # 1. Empirical CDM Curvatures: c0=0.48, c6=0.35, necking at l=3 (0.26)
    c_7 = np.array([0.48, 0.42, 0.35, 0.26, 0.31, 0.33, 0.35])
    print(f"\n[1] Solving 7-Site Holographic Spin Chain...")
    res7 = solve_rt_spin_chain(7, c_7)
    print(f"    ├─ Bond Couplings J: {np.round(res7['couplings_J'], 4)}")
    print(f"    ├─ Entanglement Entropies S(l=1..6): {np.round(res7['entanglement_entropies'], 4)}")
    print(f"    └─ RT Minimal Surface: Cut l = {res7['min_cut_index']} (S = {res7['min_entropy']:.4f} bits)")

    # 2. 14-Site Boundary-to-Boundary Symmetric Spin Chain
    c_14 = np.concatenate([c_7, c_7[::-1]])
    print(f"\n[2] Solving 14-Site Dual Boundary-to-Boundary Chain...")
    res14 = solve_rt_spin_chain(14, c_14)
    print(f"    ├─ Entanglement Entropies S(l=1..13): {np.round(res14['entanglement_entropies'], 4)}")
    print(f"    └─ Primary RT Cut: Cut l = {res14['min_cut_index']} (S = {res14['min_entropy']:.4f} bits)")

    # 3. Verification Gate
    target_l = 3
    passed_7 = (res7['min_cut_index'] == target_l)
    passed_14 = (res14['min_cut_index'] == target_l)
    print(f"\n[3] Verification Gate:")
    print(f"    ├─ 7-site l_H == 3: {'✅ PASS' if passed_7 else '❌ FAIL'}")
    print(f"    └─ 14-site l_H == 3: {'✅ PASS' if passed_14 else '❌ FAIL'}")

    # 4. BlueQubit Dispatch Demo
    print(f"\n[4] Dispatching 7-qubit circuit to BlueQubit QCPU (device='cpu')...")
    qc = build_qiskit_circuit(7, c_7)
    bq_res = run_bluequbit_dispatch(qc, device="cpu")
    print(f"    └─ BlueQubit Status: {bq_res['status']} (Job ID: {bq_res.get('job_id', 'N/A')})")

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total Execution Time: {elapsed:.4f} seconds ($0.00 compute)")
    print("=" * 75)

if __name__ == "__main__":
    main()

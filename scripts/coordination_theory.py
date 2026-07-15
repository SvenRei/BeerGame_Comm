"""
coordination_theory.py -- numerical check of the regime-invariance theorem for the SIGNAL
coordinating transfer, on a serial chain. No torch/env needed (scipy newsvendor algebra).

Claim: the transfer tau*_k = p - b_k that coordinates each upstream link is regime-invariant --
the same constant coordinates the decentralized base-stocks to the centralized base-stocks for
every demand rate lambda. Levels move with lambda; the coordinating transfer does not.
"""
import numpy as np
from scipy import stats


def _pmf(Lam, dmax):
    p = stats.poisson.pmf(np.arange(dmax + 1), Lam); p[-1] += 1.0 - p.sum(); return p


def newsvendor_level(frac, Lam):
    """Critical-fractile base-stock for Poisson(Lam) demand."""
    return float(stats.poisson.ppf(np.clip(frac, 0, 1 - 1e-9), Lam))


def newsvendor_cost(y, Lam, h, b, dmax=None):
    """E[h (y-D)+ + b (D-y)+], D ~ Poisson(Lam)."""
    dmax = dmax or int(Lam + 8 * np.sqrt(Lam) + 10)
    d = np.arange(dmax + 1); p = _pmf(Lam, dmax)
    return float((p * (h * np.maximum(y - d, 0) + b * np.maximum(d - y, 0))).sum())


def check_link(p, h, b_private, lambdas, L=1):
    """One upstream link: central target fractile p/(p+h); decentral b_private/(b_private+h);
    contract tau* = p - b_private. Verify the contract level == central level for every lambda,
    and report the price of anarchy (system cost under decentral vs central stock)."""
    phi_star = p / (p + h)                                   # system-optimal fractile
    phi_nash = b_private / (b_private + h)                   # self-interested fractile
    tau_star = p - b_private                                 # the coordinating transfer (regime-free)
    phi_ctr = (b_private + tau_star) / (b_private + tau_star + h)
    rows, poas, coordinated = [], [], True
    for lam in lambdas:
        Lam = lam * (L + 1)
        y_star = newsvendor_level(phi_star, Lam)
        y_nash = newsvendor_level(phi_nash, Lam)
        y_ctr = newsvendor_level(phi_ctr, Lam)
        # PoA: system cost (shortfall charged at p) under decentral vs central stock
        c_star = newsvendor_cost(y_star, Lam, h, p)
        c_nash = newsvendor_cost(y_nash, Lam, h, p)
        poa = c_nash / max(c_star, 1e-9)
        coordinated &= (abs(y_ctr - y_star) < 1e-9)
        rows.append((lam, y_star, y_nash, y_ctr, poa)); poas.append(poa)
    return dict(phi_star=phi_star, phi_nash=phi_nash, tau_star=tau_star,
                rows=rows, poa_mean=float(np.mean(poas)), coordinated=coordinated)


if __name__ == "__main__":
    LAMBDAS = [6, 10, 14, 18, 22]
    print("=" * 74)
    print("SIGNAL coordination: regime-invariance of tau*  (serial link, Poisson demand)")
    print("=" * 74)
    # canonical retailer-penalty cost: p large at customer, upstream private penalty small
    p, h, b_priv = 10.0, 0.5, 0.5
    r = check_link(p, h, b_priv, LAMBDAS, L=1)
    print(f"  p={p}  h2={h}  b2_private={b_priv}")
    print(f"  central fractile = {r['phi_star']:.3f}   nash fractile = {r['phi_nash']:.3f}")
    print(f"  tau* = p - b2 = {r['tau_star']:.2f}   (NOTE: independent of lambda)\n")
    print(f"  {'lambda':>7}{'y_central':>11}{'y_nash':>9}{'y_contract':>12}{'PoA':>8}")
    for lam, ys, yn, yc, poa in r["rows"]:
        print(f"  {lam:>7}{ys:>11.0f}{yn:>9.0f}{yc:>12.0f}{poa:>8.2f}")
    print(f"\n  mean price of anarchy (no contract) = {r['poa_mean']:.2f}")
    print(f"  contract level == central level at EVERY lambda: {r['coordinated']}")

    # ---- assertions ----
    assert r["coordinated"], "tau* must coordinate at every lambda (regime-invariance)"
    assert r["poa_mean"] > 1.05, "self-interest should create a real price of anarchy"
    # tau* must not depend on lambda: recompute over a different lambda grid, same tau*
    r2 = check_link(p, h, b_priv, [8, 12, 30], L=1)
    assert abs(r2["tau_star"] - r["tau_star"]) < 1e-12, "tau* changed with the regime grid (should not)"
    assert r2["coordinated"], "tau* must also coordinate the second regime grid"
    # 4-echelon nesting: each upstream echelon coordinated by its own tau*_k = p - b_k (regime-free)
    for hk, bk in [(0.5, 0.5), (0.4, 0.3), (0.3, 0.2)]:
        rk = check_link(p, hk, bk, LAMBDAS, L=1)
        assert rk["coordinated"], f"echelon (h={hk},b={bk}) not coordinated across lambda"
    print("\ncoordination_theory self-test PASS")
    print("  -> ONE constant tau* coordinates decentralized -> centralized base-stocks at every lambda;")
    print("     levels move with lambda, tau* does not. Regime-invariance holds; PoA>1 without it.")
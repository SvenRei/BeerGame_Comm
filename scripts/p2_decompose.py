"""
p2_decompose.py -- identify the mechanism behind the preregistered P2 reversal.

REGISTERED FACT (FINAL_RESULTS v2.1, n=25): Gamma = V_raw(clip12) - V_raw(inf) = -462.2
[-577.2, -331.0], every seed negative; V under clip12 ~ 0. P2 predicted the opposite sign.
The review's verdict: the reversal is publishable only with an identification of WHY. Three
candidate mechanisms, separable with data that already exists plus scripts/clipped_refs.py:

  (M1) INFORMATION REDUNDANCY: under clipping the broadcast no longer carries recoverable
       value even for a perfect user.            -> testable: WEDGE(12) collapses.
  (M2) LEARNING FAILURE: the value is still there (WEDGE(12) large) but MAPPO cannot learn to
       use the (unclipped) message when its complementary own-observations are garbled.
                                                 -> wedge-capture(12) ~ 0 while WEDGE(12) >> 0.
  (M3) BASELINE MOVEMENT: clipping degrades the nocomm arm rather than the raw arm (Gamma
       driven by DeltaC_noc, not DeltaC_raw).    -> testable: the Gamma split below.

The exact seed-paired algebra (per seed s, per clip c):
  Gamma_s(c) = V_s(c) - V_s(inf)
             = [C_noc,s(c) - C_noc,s(inf)] - [C_raw,s(c) - C_raw,s(inf)]
             = DeltaC_noc,s(c) - DeltaC_raw,s(c)
so the reversal decomposes into "what clipping did to the baseline" minus "what it did to the
comm arm", both CRN/seed-paired. All cells load fail-closed on the registered seed vector via
confirmatory_v2.load_cell -- the n=30 contamination class is structurally impossible here.

Optional blocks (each degrades to a SKIP note if its inputs are absent):
  * references: results/baselines_ar_clip_v2.json (scripts/clipped_refs.py) -> per-clip
    privileged/ownobs frontier, WEDGE, and wedge-capture = V(c)/WEDGE(c).
  * QMIX concordance: v13 qmix_{ar1,clip12} cells -> Gamma_qmix and the registered
    sign-concordance read for algorithm-conditionality (review item 8, P2 slice).
  * transfer probes: sweep_out/p2_transfer/{raw,noc}_t{inf,clip12}_e{inf,clip12} dumps
    (produced tonight via scripts/make_env_override.py + eval_signal --env-json) -> the
    train-time vs eval-time garbling split: if an inf-trained raw policy KEEPS its value when
    evaluated under clip12 observations, the message-using policy exists and works under
    garbled complements -- clipping breaks LEARNING, not execution (sharpens M2).

Run (repo root, after the analyze rerun and clipped_refs):
  python scripts/p2_decompose.py --root sweep_out/v13 --seeds 25-49 \
      --refs results/baselines_ar_clip_v2.json --transfer-root sweep_out/p2_transfer \
      --out reports/P2_DECOMPOSITION.md
"""
import os
import sys
import json
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.confirmatory_v2 import load_cell                        # noqa: E402  fail-closed loader
from scripts.c1_stats import bootstrap_ci, paired                    # noqa: E402  BCa CI + Wilcoxon/sign
from scripts.comm_stats import parse_seed_spec                       # noqa: E402  registered spec grammar

RHO_KEY = "0.9"                                   # the P2 world is the rho=0.9 AR(1) regime
CLIPS = ["inf", "20", "12"]                       # registered dose ladder (Blackwell-nested)
CELLS = {"inf": ("ar1r9_nocomm", "ar1r9_raw"),
         "20":  ("clip20_nocomm", "clip20_raw"),
         "12":  ("clip12_nocomm", "clip12_raw")}
QMIX_CELLS = {"inf": ("qmix_ar1_nocomm", "qmix_ar1_raw"),
              "12":  ("qmix_clip12_nocomm", "qmix_clip12_raw")}
TRANSFER_DIRS = ["raw_tinf_eclip12", "raw_tclip12_einf",             # off-diagonal raw probes
                 "noc_tinf_eclip12", "noc_tclip12_einf"]             # nocomm controls
# M1-vs-M2 classification thresholds (descriptive labels, not hypothesis tests -- stated up
# front so the verdict is not free to move after seeing the numbers):
WEDGE_COLLAPSE_FRAC = 0.25    # WEDGE(12) < 25% of WEDGE(inf)  -> "collapsed" (M1 branch)
CAPTURE_NULL = 0.10           # V(12)/WEDGE(12) < 10%          -> "captured ~none" (M2 branch)


def _dist(v):
    v = np.asarray(v, float)
    q = np.percentile(v, [10, 50, 90])
    return {"P_gt0": float(np.mean(v > 0)), "p10": float(q[0]), "p50": float(q[1]),
            "p90": float(q[2]), "min": float(np.min(v)), "max": float(np.max(v))}


def _fmt_ci(ci):
    return f"[{ci[0]:+.1f},{ci[1]:+.1f}]"


def _cells_present(root, cells, seeds):
    missing = [os.path.join(c, f"seed{s}.json") for c in cells for s in seeds
               if not os.path.exists(os.path.join(root, c, f"seed{s}.json"))]
    return (len(missing) == 0), missing


def load_pair(root, noc_cell, raw_cell, seeds):
    """(C_noc, C_raw, V) seed-aligned vectors for one clip level; fail-closed."""
    c_noc = load_cell(root, noc_cell, seeds)
    c_raw = load_cell(root, raw_cell, seeds)
    return c_noc, c_raw, c_noc - c_raw


def main_tables(root, seeds, lines):
    data = {}
    lines.append("## 1. Per-clip value of the raw broadcast (registered seeds, CRN-paired)\n")
    lines.append("| clip | C_nocomm | C_raw | V = C_noc - C_raw | 95% CI (BCa) | Wilcoxon p | "
                 "P(V>0) | p10 / p50 / p90 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ck in CLIPS:
        noc, raw, v = load_pair(root, *CELLS[ck], seeds=seeds)
        st = paired(noc, raw)                          # diff = noc - raw = V
        ci = bootstrap_ci(v)
        d = _dist(v)
        data[ck] = {"noc": noc, "raw": raw, "V": v, "ci": ci, "stats": st, "dist": d}
        lines.append(f"| {ck} | {np.mean(noc):.1f} | {np.mean(raw):.1f} | {np.mean(v):+.1f} | "
                     f"{_fmt_ci(ci)} | {st['wilcoxon_p']:.3g} | {d['P_gt0']:.2f} | "
                     f"{d['p10']:+.1f} / {d['p50']:+.1f} / {d['p90']:+.1f} |")
    lines.append("")
    lines.append("## 2. Gamma and its decomposition: Gamma(c) = DeltaC_noc(c) - DeltaC_raw(c)\n")
    lines.append("DeltaC_x(c) = C_x(c) - C_x(inf), seed-paired. Positive DeltaC = clipping made "
                 "that arm MORE expensive. Gamma < 0 with DeltaC_noc ~ 0 and DeltaC_raw ~ |Gamma| "
                 "means the entire reversal is the comm arm losing its edge (rules out M3).\n")
    lines.append("| clip | Gamma = V(c)-V(inf) | 95% CI | DeltaC_noc | 95% CI | DeltaC_raw | "
                 "95% CI | seeds Gamma<0 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ck in [c for c in CLIPS if c != "inf"]:
        g = data[ck]["V"] - data["inf"]["V"]
        dn = data[ck]["noc"] - data["inf"]["noc"]
        dr = data[ck]["raw"] - data["inf"]["raw"]
        data[ck]["gamma"], data[ck]["d_noc"], data[ck]["d_raw"] = g, dn, dr
        lines.append(f"| {ck} | {np.mean(g):+.1f} | {_fmt_ci(bootstrap_ci(g))} | "
                     f"{np.mean(dn):+.1f} | {_fmt_ci(bootstrap_ci(dn))} | "
                     f"{np.mean(dr):+.1f} | {_fmt_ci(bootstrap_ci(dr))} | "
                     f"{int(np.sum(g < 0))}/{len(g)} |")
    lines.append("")
    return data


def refs_block(refs_path, data, lines):
    if not (refs_path and os.path.exists(refs_path)):
        lines.append("## 3. Clipped-frontier references -- SKIPPED\n")
        lines.append(f"`{refs_path}` not found. Run `python scripts/clipped_refs.py "
                     "--episodes 200` first; sections 3-4 are the M1-vs-M2 separator.\n")
        return None
    with open(refs_path) as f:
        R = json.load(f)
    if RHO_KEY not in R.get("rhos", {}):
        lines.append(f"## 3. Clipped-frontier references -- SKIPPED (no rho={RHO_KEY} block)\n")
        return None
    r = R["rhos"][RHO_KEY]
    inv_ok = all(b["pass"] for b in r["invariance"].values())
    lines.append("## 3. Clipped-frontier references (scripts/clipped_refs.py)\n")
    lines.append(f"Privileged-rung clip-invariance verified: **{'PASS' if inv_ok else 'FAIL'}** "
                 f"(exact under CRN, tol {R['meta']['invariant_tol']:g}). The registered "
                 f"frontier did not move; only observations changed.\n")
    lines.append("| clip | AR_CondBS (privileged) | AR_CondBS_ownobs | WEDGE = ownobs - priv |")
    lines.append("|---|---|---|---|")
    for ck in CLIPS:
        if ck in r["wedge_ownobs_minus_priv"]:
            lines.append(f"| {ck} | {r['costs']['AR_CondBS'][ck]:.1f} | "
                         f"{r['costs']['AR_CondBS_ownobs'][ck]:.1f} | "
                         f"{r['wedge_ownobs_minus_priv'][ck]:+.1f} |")
    lines.append("")
    lines.append("## 4. Wedge capture and the M1/M2/M3 verdict\n")
    lines.append("capture(c) = V(c) / WEDGE(c): the fraction of the conditional-base-stock-"
                 "recoverable broadcast value the learned arm actually realized.\n")
    lines.append("| clip | V (learned) | WEDGE (recoverable) | capture |")
    lines.append("|---|---|---|---|")
    verdict_bits = {}
    for ck in CLIPS:
        if ck not in r["wedge_ownobs_minus_priv"]:
            continue
        w = float(r["wedge_ownobs_minus_priv"][ck])
        vm = float(np.mean(data[ck]["V"]))
        cap = vm / w if abs(w) > 1e-9 else float("nan")
        verdict_bits[ck] = (vm, w, cap)
        lines.append(f"| {ck} | {vm:+.1f} | {w:+.1f} | {cap:+.2%} |")
    lines.append("")
    if "12" in verdict_bits and "inf" in verdict_bits:
        v12, w12, cap12 = verdict_bits["12"]
        _, winf, _ = verdict_bits["inf"]
        collapsed = (w12 < WEDGE_COLLAPSE_FRAC * winf) if winf > 0 else False
        null_cap = (not np.isnan(cap12)) and (cap12 < CAPTURE_NULL)
        if collapsed:
            v = ("M1 -- INFORMATION REDUNDANCY: WEDGE(12) is below "
                 f"{WEDGE_COLLAPSE_FRAC:.0%} of WEDGE(inf); under clip-12 even a perfect "
                 "conditional-base-stock user of the broadcast recovers little. The reversal "
                 "is a property of the decision problem.")
        elif null_cap:
            v = ("M2 -- LEARNING FAILURE: WEDGE(12) remains large "
                 f"({w12:+.1f}, {w12 / winf:.0%} of the unclipped wedge) but the learned arm "
                 f"captured {cap12:+.1%} of it. The value survives garbling; MAPPO's ability "
                 "to learn to use the (unclipped) message does not, once the complementary "
                 "own-observations are censored.")
        else:
            v = (f"MIXED: WEDGE(12)={w12:+.1f} with capture {cap12:+.1%} -- neither the "
                 "collapse (M1) nor the null-capture (M2) threshold is met; report the "
                 "decomposition without a single-mechanism label.")
        lines.append(f"**Verdict (thresholds fixed a priori in this script's header): {v}**\n")
        lines.append("(M3 is adjudicated by Section 2: DeltaC_noc ~ 0 rules out baseline "
                     "movement regardless of the M1/M2 branch.)\n")
    return r


def qmix_block(root, seeds, lines):
    lines.append("## 5. Algorithm concordance (QMIX, registered sign rule)\n")
    for ck, cells in QMIX_CELLS.items():
        ok, missing = _cells_present(root, cells, seeds)
        if not ok:
            lines.append(f"SKIPPED at clip={ck}: {len(missing)} missing files "
                         f"(first: `{missing[0]}`).\n")
            return
    rows = {}
    for ck, (nc, rc) in QMIX_CELLS.items():
        noc, raw, v = load_pair(root, nc, rc, seeds)
        rows[ck] = v
        lines.append(f"- clip={ck}: V_qmix = {np.mean(v):+.1f} {_fmt_ci(bootstrap_ci(v))} "
                     f"(P(V>0)={np.mean(v > 0):.2f}, n={len(v)})")
    g = rows["12"] - rows["inf"]
    lines.append(f"- Gamma_qmix = {np.mean(g):+.1f} {_fmt_ci(bootstrap_ci(g))} "
                 f"({int(np.sum(g < 0))}/{len(g)} seeds negative)")
    lines.append("")
    lines.append("Registered V1 decision rule is SIGN CONCORDANCE: the P2 reversal is "
                 "algorithm-general iff Gamma_qmix and Gamma_mappo share sign with CIs "
                 "excluding 0 on the same side.\n")


def transfer_block(troot, root, seeds, lines):
    lines.append("## 6. Train-time vs eval-time garbling (transfer probes)\n")
    if not (troot and os.path.isdir(troot)):
        lines.append(f"SKIPPED: `{troot}` absent. Produce via scripts/make_env_override.py + "
                     "`eval_signal.py --env-json ... --dump-comm` (see RUNBOOK step 5); this "
                     "block then separates 'clipping breaks learning' from 'clipping breaks "
                     "execution' within M2.\n")
        return
    anchors = {"raw_inf": np.asarray(load_cell(root, "ar1r9_raw", seeds), float),
               "raw_clip12": np.asarray(load_cell(root, "clip12_raw", seeds), float),
               "noc_inf": np.asarray(load_cell(root, "ar1r9_nocomm", seeds), float),
               "noc_clip12": np.asarray(load_cell(root, "clip12_nocomm", seeds), float)}
    lines.append("Native (train=eval) anchors from Section 1; off-diagonal cells below evaluate "
                 "a checkpoint trained in one observation world inside the other (message "
                 "channel identical; env dynamics identical; only o[3] garbling differs).\n")
    lines.append("| probe dir | mean cost | vs train-native | vs eval-native |")
    lines.append("|---|---|---|---|")
    name_to_anchor = {"raw_tinf_eclip12": ("raw_inf", "raw_clip12"),
                      "raw_tclip12_einf": ("raw_clip12", "raw_inf"),
                      "noc_tinf_eclip12": ("noc_inf", "noc_clip12"),
                      "noc_tclip12_einf": ("noc_clip12", "noc_inf")}
    got_any = False
    for d in TRANSFER_DIRS:
        full = os.path.join(troot, d)
        ok, missing = _cells_present(troot, [d], seeds)
        if not os.path.isdir(full) or not ok:
            why = "dir absent" if not os.path.isdir(full) else f"{len(missing)} seed files missing"
            lines.append(f"| {d} | SKIP ({why}) | | |")
            continue
        got_any = True
        c = np.asarray(load_cell(troot, d, seeds), float)
        tn, en = name_to_anchor[d]
        dt, de = c - anchors[tn], c - anchors[en]
        lines.append(f"| {d} | {np.mean(c):.1f} | {np.mean(dt):+.1f} {_fmt_ci(bootstrap_ci(dt))} | "
                     f"{np.mean(de):+.1f} {_fmt_ci(bootstrap_ci(de))} |")
    lines.append("")
    if got_any:
        lines.append("Read: `raw_tinf_eclip12` ~ its train-native cost (small 'vs train-native') "
                     "=> the inf-trained message-using policy still functions when its own "
                     "observations are garbled at execution -- the deficit is in LEARNING under "
                     "garbling, not in using the message alongside garbled inputs. "
                     "`raw_tclip12_einf` ~ its train-native cost => the clip12-trained policy "
                     "learned to ignore the message and stays message-blind even with clean "
                     "observations restored.\n")


def main():
    ap = argparse.ArgumentParser(description="P2 reversal: Gamma decomposition + mechanism id.")
    ap.add_argument("--root", default="sweep_out/v13")
    ap.add_argument("--seeds", default="25-49", help="registered seed spec (fail-closed)")
    ap.add_argument("--refs", default="results/baselines_ar_clip_v2.json")
    ap.add_argument("--transfer-root", default="sweep_out/p2_transfer")
    ap.add_argument("--out", default="reports/P2_DECOMPOSITION.md")
    a = ap.parse_args()
    seeds = parse_seed_spec(a.seeds)
    lines = ["# P2 decomposition -- mechanism identification for the registered reversal",
             f"\nroot=`{a.root}`  seeds={a.seeds} (n={len(seeds)}, fail-closed via "
             f"confirmatory_v2.load_cell)  refs=`{a.refs}`\n"]
    data = main_tables(a.root, seeds, lines)
    refs_block(a.refs, data, lines)
    qmix_block(a.root, seeds, lines)
    transfer_block(a.transfer_root, a.root, seeds, lines)
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt)
        print(f"-> wrote {a.out}")


def _selftest():
    """Synthetic end-to-end: plant a known Gamma driven purely by DeltaC_raw and check every
    block reproduces it (including graceful skips). Run: python scripts/p2_decompose.py --selftest"""
    import tempfile
    rng = np.random.default_rng(0)
    seeds = list(range(25, 50))
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "v13")
        def put(cell, mu):
            os.makedirs(os.path.join(root, cell), exist_ok=True)
            for s in seeds:
                with open(os.path.join(root, cell, f"seed{s}.json"), "w") as f:
                    json.dump({"0.9": float(mu + rng.normal(0, 5))}, f)
        put("ar1r9_nocomm", 4600); put("ar1r9_raw", 4150)       # V(inf) ~ +450
        put("clip20_nocomm", 4600); put("clip20_raw", 4450)     # V(20)  ~ +150
        put("clip12_nocomm", 4600); put("clip12_raw", 4600)     # V(12)  ~ 0 -> Gamma ~ -450, all via DeltaC_raw
        refs = {"meta": {"invariant_tol": 1e-6},
                "rhos": {"0.9": {"costs": {"AR_CondBS": {c: 3200.0 for c in CLIPS},
                                           "AR_CondBS_ownobs": {"inf": 7000.0, "20": 5000.0, "12": 4700.0}},
                                 "invariance": {"AR_CondBS": {"pass": True}, "AR_StaticBS": {"pass": True}},
                                 "wedge_ownobs_minus_priv": {"inf": 3800.0, "20": 1800.0, "12": 1500.0}}}}
        rp = os.path.join(td, "refs.json")
        with open(rp, "w") as f:
            json.dump(refs, f)
        lines = []
        data = main_tables(root, seeds, lines)
        g12 = float(np.mean(data["12"]["V"] - data["inf"]["V"]))
        dr12 = float(np.mean(data["12"]["raw"] - data["inf"]["raw"]))
        dn12 = float(np.mean(data["12"]["noc"] - data["inf"]["noc"]))
        assert -520 < g12 < -380, g12
        assert abs(dn12) < 6 and 380 < dr12 < 520, (dn12, dr12)         # reversal all in DeltaC_raw
        r = refs_block(rp, data, lines)
        assert r is not None
        joined = "\n".join(lines)
        assert "M2 -- LEARNING FAILURE" in joined, "expected M2 verdict on planted numbers"
        qmix_block(root, seeds, lines)                                   # qmix cells absent -> SKIP
        transfer_block(os.path.join(td, "nope"), root, seeds, lines)     # absent -> SKIP
        joined = "\n".join(lines)
        assert "SKIPPED" in joined
        # M1 branch: collapse the wedge and re-run the verdict
        refs["rhos"]["0.9"]["wedge_ownobs_minus_priv"]["12"] = 200.0
        with open(rp, "w") as f:
            json.dump(refs, f)
        lines2 = []
        refs_block(rp, main_tables(root, seeds, lines2), lines2)
        assert "M1 -- INFORMATION REDUNDANCY" in "\n".join(lines2)
    print("p2_decompose selftest: PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

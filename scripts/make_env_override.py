"""
make_env_override.py -- materialize a FULL env config JSON = (checkpoint's trained env) + patches,
for agents/eval_signal.py --env-json. This is what makes the P2 transfer probes runnable with the
existing eval binary: --env-json REPLACES the saved env wholesale (resolve_env_base precedence:
CLI > ckpt > ENV_BASE), so a bare patch file like {"obs_order_clip": 12} would silently drop the
horizon/costs/AR keys the checkpoint was trained with. This script starts from the ckpt's own env
dict, applies explicit KEY=VAL patches (or KEY=del to REMOVE a key, e.g. lifting the clip), prints
the exact diff, and writes the merged dict.

Examples (repo root):
  # evaluate an UNCLIPPED-trained raw ckpt under clip-12 observations:
  python scripts/make_env_override.py --ckpt <inf-trained ckpt> \
      --set obs_order_clip=12 --out /tmp/env_tinf_eclip12.json
  # evaluate a CLIP12-trained ckpt with the clip lifted:
  python scripts/make_env_override.py --ckpt <clip12-trained ckpt> \
      --set obs_order_clip=del --out /tmp/env_tclip12_einf.json
then:
  python agents/eval_signal.py --ckpt <same ckpt> --env-json <out> \
      --dump-comm sweep_out/p2_transfer/<cell> --dump-ar1 0.9 --dump-episodes 15

Value parsing: int/float/bool("true"/"false")/null("none") inferred; anything else stays a string;
KEY=del removes the key. Refuses to write if the ckpt has no saved env (pre-fix checkpoints):
patching on top of the module fallback would eval a DIFFERENT world than trained -- exactly the
inconsistency this script exists to prevent.
"""
import os
import sys
import json
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_val(v):
    s = str(v)
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="repeatable; VAL=del removes KEY")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch                                     # after argparse so --help stays torch-free
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    env = ckpt.get("env") or ckpt.get("config", {}).get("env")
    if not isinstance(env, dict) or not env:
        sys.exit("FAIL: checkpoint carries no saved env dict; refusing to synthesize one "
                 "(patching the module fallback would eval a world the policy never trained in).")
    env = dict(env)
    before = dict(env)
    for kv in a.set:
        if "=" not in kv:
            sys.exit(f"FAIL: --set expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        if v.lower() == "del":
            env.pop(k, None)
        else:
            env[k] = parse_val(v)
    print(f"ckpt env <- {a.ckpt}")
    keys = sorted(set(before) | set(env))
    for k in keys:
        b, e = before.get(k, "<absent>"), env.get(k, "<absent>")
        if b != e:
            print(f"  {k}: {b!r} -> {e!r}")
    if before == env:
        print("  (no changes -- writing the ckpt env verbatim)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(env, f, indent=2, sort_keys=True)
    print(f"-> wrote {a.out}")


if __name__ == "__main__":
    main()
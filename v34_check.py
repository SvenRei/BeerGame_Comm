#!/usr/bin/env python3
"""v34_check.py -- verify the v3.4 conditioning-parity patch is fully installed. Run: python v34_check.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ok = True
y = open("conf/agent/signal.yaml").read()
for k in ("obs_norm", "split_optimizer", "lr_critic", "state_scale"):
    good = k + ":" in y
    ok &= good
    print(f"  yaml {k:16s}: {'OK' if good else 'MISSING'}")
try:
    from agents.signal_agent import SIGNALTrainer
    from agents.topologies import get_adj
    tr = SIGNALTrainer({"hidden": 16, "msg_content": "raw", "use_comm": False,
                        "obs_norm": True, "split_optimizer": True}, 4, 4, 20,
                       get_adj("retailer_broadcast"))
    a = tr.actors[0].in_norm is True
    b = tr.critic_opt is not None and abs(tr.critic_opt.param_groups[0]["lr"] - 1e-3) < 1e-9
    ok &= a and b
    print(f"  actor in_norm wired      : {'OK' if a else 'FAIL'}")
    print(f"  split critic Adam @1e-3  : {'OK' if b else 'FAIL'}")
except Exception as e:
    ok = False
    print(f"  trainer construction     : FAIL ({type(e).__name__}: {e})")
print("\n" + ("V3.4 PATCH VERIFIED -- proceed with: python v34_run_nocomm.py"
              if ok else "*** PATCH NOT (FULLY) APPLIED -- unzip v34_conditioning_parity.zip over the project, then rerun this check ***"))
sys.exit(0 if ok else 1)

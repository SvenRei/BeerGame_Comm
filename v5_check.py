#!/usr/bin/env python3
"""v5_check.py -- verify the v5 categorical action head is fully installed. Run: python v5_check.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ok = True
y = open("conf/agent/signal.yaml").read()
for k in ("head_type", "act_bins", "entropy_start", "obs_norm", "split_optimizer"):
    good = k + ":" in y
    ok &= good
    print(f"  yaml {k:16s}: {'OK' if good else 'MISSING'}")
try:
    from agents.signal_agent import SIGNALTrainer
    from agents.topologies import get_adj
    tr = SIGNALTrainer({"hidden": 16, "msg_content": "raw", "use_comm": False,
                        "head_type": "categorical", "obs_norm": True,
                        "split_optimizer": True}, 4, 4, 20, get_adj("retailer_broadcast"))
    ac = tr.actors[0]
    a = ac.head_type == "categorical" and len(ac.s_grid) == 41 and float(ac.s_grid[-1]) == 160.0
    import torch as T
    p = T.softmax(ac.s_logits(T.randn(1, 4) * 20, T.randn(1, 16) * .3, T.randn(1, 1) * 10), -1)
    b = float(p.max()) < 0.05                      # near-uniform cold start
    cvo = tr.critic_opt is not None
    ok &= a and b and cvo
    print(f"  categorical S-grid 41x160: {'OK' if a else 'FAIL'}")
    print(f"  cold start (max p<0.05)  : {'OK' if b else 'FAIL'} (max p={float(p.max()):.4f})")
    print(f"  split critic optimizer   : {'OK' if cvo else 'FAIL'}")
except Exception as e:
    ok = False
    print(f"  trainer construction     : FAIL ({type(e).__name__}: {e})")
print("\n" + ("V5 CATEGORICAL HEAD VERIFIED -- proceed with: python v5_run_nocomm.py"
              if ok else "*** V5 NOT (FULLY) APPLIED -- unzip v5_categorical_head.zip over the project, then rerun this check ***"))
sys.exit(0 if ok else 1)

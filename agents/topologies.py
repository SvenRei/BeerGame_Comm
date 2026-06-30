"""
topologies.py -- communication topologies (ADJ matrices) for the Study-2 sweep.

Row-stochastic convention (same as the original ADJ): incoming[i] = sum_j ADJ[i,j]*msg[j],
so each agent's received message is a convex combination of the messages it can hear.
Wider topology = each agent hears further along the serial chain.

  neighbor : chain adjacency (the ORIGINAL setup). r<->w<->d<->m, range-1.
  skip     : range-2. each agent also hears the agent two hops away.
  full     : all-to-all. every agent hears every other.

Select at train time with `agent.comm_topology=neighbor|skip|full` (requires use_comm=true).
The trainer builds the live ADJ via get_adj(cfg.agent.comm_topology); the eval script reads
the topology back out of the checkpoint config so messages route exactly as trained.
"""
import torch

# Listens-to (who each agent hears):
#   neighbor: 0<->1<->2<->3
#   no_neighbor: 0->{2,3}  1->{3}  2->{0}  3->{0,1}
#   skip:     0->{1,2}  1->{0,2,3}  2->{0,1,3}  3->{1,2}
#   full:     everyone -> everyone else
ADJ_TOPOLOGIES = {
    "neighbor": [[0.0, 1.0, 0.0, 0.0],
                 [0.5, 0.0, 0.5, 0.0],
                 [0.0, 0.5, 0.0, 0.5],
                 [0.0, 0.0, 1.0, 0.0]],
    # no_neighbor: a CONTROL/placebo -- each agent hears only NON-adjacent agents (the "wrong"
    # partners). If comm helps via the serial demand-signal channel (Lee see-through-bullwhip),
    # this should NOT help; a gain here would mean the benefit is not that mechanism.
   "no_neighbor":[[0.0, 0.0, 0.5, 0.5],
                 [0.0, 0.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.0, 0.0]],                
    "skip":     [[0.0,     0.5,     0.5,     0.0],
                 [1 / 3.0, 0.0,     1 / 3.0, 1 / 3.0],
                 [1 / 3.0, 1 / 3.0, 0.0,     1 / 3.0],
                 [0.0,     0.5,     0.5,     0.0]],
    "full":     [[0.0,     1 / 3.0, 1 / 3.0, 1 / 3.0],
                 [1 / 3.0, 0.0,     1 / 3.0, 1 / 3.0],
                 [1 / 3.0, 1 / 3.0, 0.0,     1 / 3.0],
                 [1 / 3.0, 1 / 3.0, 1 / 3.0, 0.0]],
    # retailer_broadcast: every UPSTREAM stage hears the retailer (agent 0) UNDILUTED.
    # The maximally favorable case for Lee-Padmanabhan-Whang see-through-bullwhip: if sharing
    # the cleanest demand signal with everyone still buys nothing, the serial null is decisive.
    # Retailer hears nothing (row 0 all-zero -> incoming[0]=0; it observes demand directly).
    "retailer_broadcast": [[0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0]],

    # upstream_only: each stage hears ONLY its immediate DOWNSTREAM neighbor (the one closer to
    # the customer). Demand belief propagates UP the chain, hop by hop -- the Lee-correct direction.
    # This is the realistic VMI-style local sharing and the theory-predicted beneficial geometry.
    "upstream_only":   [[0.0, 0.0, 0.0, 0.0],   # retailer hears no one (observes demand directly)
                        [1.0, 0.0, 0.0, 0.0],   # wholesaler  <- retailer
                        [0.0, 1.0, 0.0, 0.0],   # distributor <- wholesaler
                        [0.0, 0.0, 1.0, 0.0]],  # manufacturer<- distributor

    # downstream_only: each stage hears ONLY its immediate UPSTREAM neighbor (its supplier). Info
    # flows DOWN -- the WRONG direction for demand sharing (you'd be hearing a MORE-distorted upstream
    # belief). A sharper placebo than no_neighbor: ADJACENT but wrong-direction, isolating direction alone.
    "downstream_only": [[0.0, 1.0, 0.0, 0.0],   # retailer    <- wholesaler  (useless: already sees demand)
                        [0.0, 0.0, 1.0, 0.0],   # wholesaler  <- distributor
                        [0.0, 0.0, 0.0, 1.0],   # distributor <- manufacturer
                    [0.0, 0.0, 0.0, 0.0]],  # manufacturer hears no one
    # manufacturer_broadcast: directional MIRROR of retailer_broadcast -- the LEAST-informed stage
    # broadcasts to everyone. Its belief is the most bullwhip-distorted, so this should NOT help.
    # A decisive control: retailer_broadcast helping while this does not pins the value to the clean
    # demand signal, not to "broadcasting" per se.
    "manufacturer_broadcast": [[0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]],

    # single-link probes: share at ONE link only, to ask whether the value concentrates at the
    # most-upstream link (worst bullwhip) vs the cleanest downstream link.
    "link_top_only":    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]],  # only manufacturer <- distributor
    "link_bottom_only": [[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]],  # only wholesaler 
}


def get_adj(name="neighbor"):
    """Return the row-stochastic ADJ tensor for the named topology."""
    if name not in ADJ_TOPOLOGIES:
        raise ValueError(f"unknown comm_topology '{name}'; choose from {list(ADJ_TOPOLOGIES)}")
    A = torch.tensor(ADJ_TOPOLOGIES[name], dtype=torch.float32)
    rs = A.sum(dim=1, keepdim=True)
    rs = torch.where(rs == 0, torch.ones_like(rs), rs)
    return A / rs            # defensive row-normalization (hand-edited matrices stay convex)


# Back-compat: `from agents.topologies import ADJ` gives the original neighbor chain.
ADJ = get_adj("neighbor")
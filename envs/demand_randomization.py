"""
demand_randomization.py -- DemandRandomizedBeerGame, a self-contained subclass of the env
(no agent dependency).

Per episode: lambda ~ U[lo,hi] and, with prob p_shift, one mid-episode level shift. Perturbs ONLY
demand_type 'poisson' (training); black_swan/extreme_chaos pass straight through so any OOD
benchmark is never altered. This is the rate-randomized regime-inference training env; the held-out
gate reuses it with lo=hi=lambda, p_shift=0 for stationary per-lambda evaluation.
"""
import numpy as np
from envs.beer_game_env import BeerGameParallelEnv


class DemandRandomizedBeerGame(BeerGameParallelEnv):
    def __init__(self, config, lam_lo=4.0, lam_hi=16.0, p_shift=0.5, shift_scale=2.0):
        super().__init__(config)
        self._dr = dict(lo=lam_lo, hi=lam_hi, p_shift=p_shift, scale=shift_scale)
        self._dr_rng = np.random.default_rng()
        self._dr_lambda = 8.0
        self._dr_shift_t = None
        self._dr_shift_lambda = 8.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._dr_rng = np.random.default_rng(seed + 99991)
        self._dr_lambda = float(self._dr_rng.uniform(self._dr["lo"], self._dr["hi"]))
        if self._dr_rng.random() < self._dr["p_shift"]:
            lo_t = max(1, self.horizon // 4)
            hi_t = max(lo_t + 1, 3 * self.horizon // 4)
            self._dr_shift_t = int(self._dr_rng.integers(lo_t, hi_t))
            factor = self._dr["scale"] if self._dr_rng.random() < 0.5 else 1.0 / self._dr["scale"]
            self._dr_shift_lambda = max(0.0, self._dr_lambda * factor)
        else:
            self._dr_shift_t = None
        return super().reset(seed=seed, options=options)

    def _roll_stochastic_demand(self, step):
        if self._config.get("demand_type") == "poisson":
            lam = self._dr_lambda
            if self._dr_shift_t is not None and step >= self._dr_shift_t:
                lam = self._dr_shift_lambda
            return self.np_random.poisson(lam)
        return super()._roll_stochastic_demand(step)
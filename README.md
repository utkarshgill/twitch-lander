# twitch-lander

REINFORCE for LunarLander. 150 lines.

## run

```bash
python tiny_reinforce.py
```

~5 mins on CPU. Starts at -400, hits positive around iter 200-300.

```bash
PLOT=1 python tiny_reinforce.py  # live plot
```

## algorithm

1. run policy, collect episode
2. compute returns G_t (discounted rewards)
3. gradient ascent on E[log π(a|s) * G]
4. repeat

High G → increase log π. Low G → decrease log π.

## tanh squashing

Lander needs actions in [-1, 1]. Sample u ~ N(μ, σ), then a = tanh(u). Always bounded, better than clipping.

We also need a correction term:
```
log P(a) = log P(u) - log(1 - a²)
```

## results

Gets to 100-150 in 500 iters. "Solved" is 200 but this lands fine.

Sometimes stuck on bad init. Just ctrl-c and restart.

## install

```bash
pip install gymnasium[box2d] torch numpy tqdm matplotlib
```

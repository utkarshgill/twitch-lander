# twitch-lander

REINFORCE for LunarLander in ~200 lines.

## run

```bash
python tiny_reinforce.py
```

~5 mins on CPU. Starts at -400, hits positive around iter 200-300.

```bash
PLOT=1 python tiny_reinforce.py  # live plot
```

## tanh squashing + logp correction

Lander needs actions in [-1, 1]. Sample u ~ N(μ, σ), then a = tanh(u). Always bounded, better than clipping.

We also need a correction term:
```
log P(a) = log P(u) - log(1 - a²)
```

## install dependencies

```bash
pip install gymnasium[box2d] torch numpy tqdm matplotlib
```

#!/usr/bin/env python3
"""Four-tile generative models, standard-library only.

Run: python3 reference.py
The website's complete interactive engine is model.mjs.
This independent reference shows the core training and sampling mathematics.
"""
import math
import random
from pathlib import Path

STATES = range(16)
CLEAN = (0, 15)
WEIGHT = 0.65


def bit(x, i):
    return (x >> (3 - i)) & 1


def conditional(x, z, signal):
    """Uniform-to-clean product path, conditioned on clean endpoint z."""
    return math.prod((1 - signal) / 2 + signal * (bit(x, i) == bit(z, i)) for i in range(4))


def distribution(signal):
    return [WEIGHT * conditional(x, CLEAN[0], signal) + (1 - WEIGHT) * conditional(x, CLEAN[1], signal) for x in STATES]


def posterior(x, signal):
    u = WEIGHT * conditional(x, CLEAN[0], signal)
    v = (1 - WEIGHT) * conditional(x, CLEAN[1], signal)
    return u / (u + v) if u + v else WEIGHT


def categorical(probabilities, rng):
    u, total = rng.random(), 0
    for x, p in enumerate(probabilities):
        total += p
        if u < total:
            return x
    return len(probabilities) - 1


class Denoiser:
    """Logistic lookup table, not a neural net. Predicts first clean endpoint."""
    def __init__(self):
        self.logits = [[0.0] * 16 for _ in range(25)]

    @staticmethod
    def sigmoid(logit):
        return 1 / (1 + math.exp(-logit))

    def predict(self, x, signal):
        u = signal * 24
        lo, hi = int(u), min(24, int(u) + 1)
        f = u - lo
        return self.sigmoid((1 - f) * self.logits[lo][x] + f * self.logits[hi][x])

    def loss(self):
        loss = 0
        for b in range(25):
            for x in STATES:
                p = min(1 - 1e-12, max(1e-12, self.sigmoid(self.logits[b][x])))
                loss -= WEIGHT * conditional(x, CLEAN[0], b / 24) * math.log(p)
                loss -= (1 - WEIGHT) * conditional(x, CLEAN[1], b / 24) * math.log(1 - p)
        return loss / 25

    def train(self, epochs=160):
        for _ in range(epochs):
            for b in range(25):
                for x in STATES:
                    u = WEIGHT * conditional(x, CLEAN[0], b / 24)
                    v = (1 - WEIGHT) * conditional(x, CLEAN[1], b / 24)
                    # Sum over clean labeled conditional examples; no posterior oracle.
                    gradient = (u + v) * self.sigmoid(self.logits[b][x]) - u
                    self.logits[b][x] -= 4 * gradient


def flow_rates(x, t, denoiser=None):
    """Linear schedule kappa(t)=t; valid for t<1."""
    w = denoiser.predict(x, t) if denoiser else posterior(x, t)
    return [(w * (bit(CLEAN[0], i) != bit(x, i)) + (1 - w) * (bit(CLEAN[1], i) != bit(x, i))) / (1 - t) for i in range(4)]


def sample_flow(n=4000, seed=7, denoiser=None):
    """Adaptive one-jump Euler approximation; no forced endpoint correction."""
    rng = random.Random(seed)
    samples = [rng.randrange(16) for _ in range(n)]
    t = 0.0
    while t < 0.995 - 1e-12:
        rates = [flow_rates(x, t, denoiser) for x in STATES]
        max_rate = max(map(sum, rates))
        dt = min(0.995 - t, 1 / 300, 0.075 / max(max_rate, 1e-12))
        for j, x in enumerate(samples):
            u, cumulative = rng.random(), 0
            for i, rate in enumerate(rates[x]):
                cumulative += dt * rate
                if u < cumulative:
                    samples[j] ^= 1 << (3 - i)
                    break
        t += dt
    return samples


def reverse_kernel(x, k, steps=20, denoiser=None):
    signal, previous_signal = 1 - k / steps, 1 - (k - 1) / steps
    ratio = signal / previous_signal
    w = denoiser.predict(x, signal) if denoiser else posterior(x, signal)
    result = []
    for y in STATES:
        # q(y | x,z) = q(x | y) q(y | z) / q(x | z).
        bridges = w * conditional(y, CLEAN[0], previous_signal) / conditional(x, CLEAN[0], signal)
        bridges += (1 - w) * conditional(y, CLEAN[1], previous_signal) / conditional(x, CLEAN[1], signal)
        result.append(conditional(x, y, ratio) * bridges)
    total = sum(result)
    return [v / total for v in result]


def sample_diffusion(n=4000, seed=7, denoiser=None):
    rng = random.Random(seed)
    samples = [rng.randrange(16) for _ in range(n)]
    for k in range(20, 0, -1):
        kernels = [reverse_kernel(x, k, denoiser=denoiser) for x in STATES]
        samples = [categorical(kernels[x], rng) for x in samples]
    return samples


def verify():
    for a in (0, .3, .8, 1):
        assert abs(sum(distribution(a)) - 1) < 1e-12
    # Exact transport of probability mass, independent of Monte Carlo noise.
    p = [1 / 16] * 16
    for k in range(20, 0, -1):
        kernels = [reverse_kernel(x, k) for x in STATES]
        p = [sum(p[x] * kernels[x][y] for x in STATES) for y in STATES]
    assert max(abs(p[x] - distribution(1)[x]) for x in STATES) < 1e-12
    # Generator continuity equation checked against a numerical derivative.
    t, h = .4, 1e-6
    p, rhs = distribution(t), [0.] * 16
    for x in STATES:
        for i, rate in enumerate(flow_rates(x, t)):
            rhs[x] -= p[x] * rate
            rhs[x ^ (1 << (3 - i))] += p[x] * rate
    assert max(abs(rhs[x] - (distribution(t + h)[x] - distribution(t - h)[x]) / (2 * h)) for x in STATES) < 1e-8


def main():
    verify()
    kind = 'flow' if 'flow' in Path(__file__).parent.name else 'diffusion'
    learner = Denoiser()
    before = learner.loss()
    learner.train()
    after = learner.loss()
    assert after < before
    sampler = sample_flow if kind == 'flow' else sample_diffusion
    print(f'{kind.upper()}: exact finite-state checks passed')
    print(f'Training loss: {before:.4f} -> {after:.4f} nats (160 epochs)')
    for label, model in [('Exact posterior', None), ('Learned posterior', learner)]:
        samples = sampler(denoiser=model)
        print(f'{label}:')
        for x in CLEAN:
            print(f'  {x:04b}: {samples.count(x) / len(samples):.3f}')
        print(f'  other states: {sum(x not in CLEAN for x in samples) / len(samples):.3f}')
    print('Target: 0000 = 0.650, 1111 = 0.350')
    if kind == 'flow':
        print('CTMC simulation stops at t=0.995; residual off-target mass is expected.')


if __name__ == '__main__':
    main()

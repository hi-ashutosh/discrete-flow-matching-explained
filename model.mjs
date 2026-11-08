/**
 * Exact four-tile mathematics, shared by the two teaching sites.
 * State x is an integer 0..15; bits are read row-major, most significant first.
 * Matrices use R(source, destination); probability vectors are columns.
 * No DOM, third-party packages, or network calls.
 */
export const STATES = Array.from({ length: 16 }, (_, i) => i);
export const BINS = 25;
export const END = 0.995; // The CTMC endpoint is a limit; never divide by zero.
export const bit = (x, i) => (x >> (3 - i)) & 1;
export const patterns = { solid: [0, 15], checker: [5, 10], rows: [3, 12] };

export function data(config) {
  const p = Array(16).fill(0);
  const pair = patterns[config.pattern];
  p[pair[0]] = config.weight;
  p[pair[1]] = 1 - config.weight;
  return p;
}

/** Return [signal strength kappa(t), derivative kappa'(t)]. */
export function schedule(t, kind = 'linear') {
  if (kind === 'slow') return [t * t, 2 * t];
  if (kind === 'fast') return [2 * t - t * t, 2 - 2 * t];
  return [t, 1];
}

/** q(x | clean endpoint z, signal a): product of four categorical probabilities. */
export function likelihood(x, z, a) {
  let p = 1;
  for (let i = 0; i < 4; i++) {
    p *= (1 - a) / 2 + a * Number(bit(x, i) === bit(z, i));
  }
  return p;
}

export function marginal(a, config) {
  const d = data(config);
  return STATES.map(x => d.reduce((sum, weight, z) => sum + weight * likelihood(x, z, a), 0));
}

/** Exact posterior probability of the first of our two clean patterns. */
export function posterior(x, a, config) {
  const pair = patterns[config.pattern];
  const u = config.weight * likelihood(x, pair[0], a);
  const v = (1 - config.weight) * likelihood(x, pair[1], a);
  // An impossible observation at a=1 has no defined posterior; not used to sample.
  return u + v > 0 ? u / (u + v) : config.weight;
}

/** Deterministic Mulberry32 random source; reproduces a run when scrubbing time. */
export function rng(seed = 7) {
  let a = seed >>> 0;
  return () => {
    a += 0x6D2B79F5;
    let t = a;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function categorical(p, random) {
  const u = random();
  let cumulative = 0;
  for (let i = 0; i < p.length; i++) {
    cumulative += p[i];
    if (u < cumulative) return i;
  }
  return p.length - 1; // Floating-point roundoff fallback.
}

const sigmoid = x => 1 / (1 + Math.exp(-x));

/**
 * Trainable binary endpoint classifier: 25 signal bins × 16 current states.
 * This is a logistic lookup table, not a neural network.
 * Signal interpolation makes predictions available between training bins.
 */
export class Denoiser {
  constructor() {
    this.logits = Array.from({ length: BINS }, () => Array(16).fill(0));
    this.history = [];
  }

  predict(x, a) {
    const u = Math.min(1, Math.max(0, a)) * (BINS - 1);
    const lo = Math.floor(u), hi = Math.min(BINS - 1, lo + 1), f = u - lo;
    return sigmoid(this.logits[lo][x] * (1 - f) + this.logits[hi][x] * f);
  }

  loss(config) {
    const pair = patterns[config.pattern];
    let loss = 0;
    for (let b = 0; b < BINS; b++) {
      const a = b / (BINS - 1);
      for (const x of STATES) {
        const p = Math.min(1 - 1e-12, Math.max(1e-12, sigmoid(this.logits[b][x])));
        loss -= config.weight * likelihood(x, pair[0], a) * Math.log(p)
          + (1 - config.weight) * likelihood(x, pair[1], a) * Math.log(1 - p);
      }
    }
    return loss / BINS;
  }

  train(config, epochs = 40) {
    if (!this.history.length) this.history.push(this.loss(config));
    const pair = patterns[config.pattern];
    for (let e = 0; e < epochs; e++) {
      for (let b = 0; b < BINS; b++) {
        const a = b / (BINS - 1);
        for (const x of STATES) {
          const u = config.weight * likelihood(x, pair[0], a);
          const v = (1 - config.weight) * likelihood(x, pair[1], a);
          // Sum over labeled conditional examples, without using posterior().
          // Gradient of the SUM over signal bins: (u+v)*prediction - u.
          // Equivalent learning rate for the reported MEAN loss is 4*BINS.
          this.logits[b][x] -= 4 * ((u + v) * sigmoid(this.logits[b][x]) - u);
        }
      }
      if ((e + 1) % 5 === 0) this.history.push(this.loss(config));
    }
  }
}

export function endpointProbability(x, a, config, model = null) {
  return model ? model.predict(x, a) : posterior(x, a, config);
}

/** One off-diagonal flip rate per coordinate; all other non-diagonal rates are zero. */
export function rates(x, t, config, model = null) {
  const [a, derivative] = schedule(t, config.schedule);
  const w = endpointProbability(x, a, config, model);
  const pair = patterns[config.pattern];
  return Array.from({ length: 4 }, (_, i) => derivative / (1 - a) * (
    w * Number(bit(pair[0], i) !== bit(x, i))
    + (1 - w) * Number(bit(pair[1], i) !== bit(x, i))
  ));
}

/** Probability of x after corrupting y with cumulative retention ratio. */
export function transition(y, x, ratio) {
  return likelihood(x, y, ratio);
}

/**
 * Full joint finite-step reverse kernel over all 16 previous states y.
 * Mix exact conditional bridges q(y | x,z) under the clean-endpoint predictor.
 * Exact posterior => exact Bayes reversal; learned posterior => approximate model.
 */
export function reverseKernel(x, k, config, model = null) {
  const T = config.steps, a = 1 - k / T, previous = 1 - (k - 1) / T;
  const ratio = a / previous, pair = patterns[config.pattern];
  const w = endpointProbability(x, a, config, model);
  const p = STATES.map(y => transition(y, x, ratio) * (
    w * likelihood(y, pair[0], previous) / Math.max(1e-300, likelihood(x, pair[0], a))
    + (1 - w) * likelihood(y, pair[1], previous) / Math.max(1e-300, likelihood(x, pair[1], a))
  ));
  const sum = p.reduce((a, b) => a + b, 0);
  return p.map(v => v / sum);
}

export function histogram(samples) {
  const h = Array(16).fill(0);
  samples.forEach(x => h[x]++);
  return h.map(v => v / samples.length);
}

export function tv(p, q) {
  return p.reduce((sum, v, i) => sum + Math.abs(v - q[i]), 0) / 2;
}

/**
 * Approximate the CTMC using adaptive one-jump Euler steps.
 * Bound dt * total outgoing rate by 0.075 for every state.
 * Samples jump using the marginal rate; no clean endpoint is assigned to them.
 */
export function simulateFlow(config, model = null) {
  const random = rng(config.seed), n = config.count;
  const samples = Array.from({ length: n }, () => Math.floor(random() * 16));
  const frames = [samples.slice()];
  let t = 0;
  for (let f = 1; f <= 100; f++) {
    const end = f / 100 * END;
    while (t < end - 1e-12) {
      const allRates = STATES.map(x => rates(x, t, config, model));
      const maxRate = Math.max(...allRates.map(r => r.reduce((a, b) => a + b, 0)));
      const dt = Math.min(end - t, 1 / 300, 0.075 / Math.max(maxRate, 1e-12));
      for (let j = 0; j < n; j++) {
        const r = allRates[samples[j]], u = random();
        let cumulative = 0;
        for (let i = 0; i < 4; i++) {
          cumulative += r[i] * dt;
          if (u < cumulative) {
            samples[j] ^= 1 << (3 - i);
            break;
          }
        }
      }
      t += dt;
    }
    frames.push(samples.slice());
  }
  return frames;
}

/** Sample the chosen finite-step corruption chain, or its modeled reverse. */
export function simulateDiffusion(config, model = null, forward = false) {
  const random = rng(config.seed), d = data(config);
  const samples = Array.from({ length: config.count }, () => (
    forward ? categorical(d, random) : Math.floor(random() * 16)
  ));
  const frames = [samples.slice()];
  for (let f = 1; f <= config.steps; f++) {
    const k = forward ? f : config.steps - f + 1;
    const kernels = STATES.map(x => forward
      ? STATES.map(y => transition(x, y, (1 - k / config.steps) / (1 - (k - 1) / config.steps)))
      : reverseKernel(x, k, config, model));
    for (let j = 0; j < samples.length; j++) {
      samples[j] = categorical(kernels[samples[j]], random);
    }
    frames.push(samples.slice());
  }
  return frames;
}

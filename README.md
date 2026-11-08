# Discrete Flow Matching — a visual field guide

**[Open the interactive website](https://hi-ashutosh.github.io/discrete-flow-matching-explained/)** · [GitHub repository](https://github.com/hi-ashutosh/discrete-flow-matching-explained)

An interactive, dependency-free teaching website. Four binary tiles explain generative modeling before any downstream application. Paired with [`discrete-diffusion-explained`](https://hi-ashutosh.github.io/discrete-diffusion-explained/).

![Presentation view of the interactive teaching site](preview.png)

## Open locally

From this repository:

```sh
python3 -m http.server 8000
```

Visit http://localhost:8000. Use a local HTTP server: browsers restrict JavaScript modules when opening `index.html` as a `file://` URL. No installation or build is needed.

To serve both companion repositories together, run `python3 -m http.server 8000` from their parent Code folder and visit `http://localhost:8000/discrete-flow-matching-explained/`. This also enables the companion navigation links. Stop the server with Ctrl+C.

## What to try

1. Press Play or scrub the time slider. Track the outlined sample and compare population bars with exact marginals.
2. Change the two target patterns, their mixture weight, and the schedule or number of steps.
3. Choose **Trainable**, train for 40 epochs, and compare again. Repeat training or reset it.
4. Click a histogram state to inspect its jump rates or reverse probabilities.
5. Open the mathematical explanations. Use **Presentation view** for live teaching; Escape restores the lesson.

Changing the target dataset resets training. The two colors are categorical values. Time sliders replay one seeded stochastic run; “New random run” changes the seed.

## Run the code and checks

```sh
python3 reference.py
node --test tests/core.test.mjs
```

Python 3.9+ and a modern Node.js version with the built-in test runner are enough. There are no third-party dependencies. `npm test` runs the same Node checks; `npm start` serves the website on port 8000.

## Deployment with GitHub Pages

This repository publishes automatically from **main → /(root)**. Its live URL is [https://hi-ashutosh.github.io/discrete-flow-matching-explained/](https://hi-ashutosh.github.io/discrete-flow-matching-explained/).

After making changes locally:

```sh
git add .
git commit -m "Update the teaching site"
git push
```

GitHub rebuilds Pages after a push. Check the repository’s Actions tab for deployment status. `.nojekyll` serves the static assets directly; there is no application build step. The companion site links work because both projects are hosted under the same account.

For a fork, configure **Settings → Pages → Build and deployment → Deploy from a branch → main → /(root)**. See [GitHub’s publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Files

- `index.html`: lesson, equations, references, and accessible controls.
- `styles.css`: responsive layout; system fonts; no external requests.
- `model.mjs`: conditional paths, posterior, trainable denoiser, generators, reverse kernels, seeded sampling.
- `app.mjs`: interaction and rendering.
- `reference.py`: independent standard-library implementation, checks, training, and sampling.
- `tests/core.test.mjs`: normalization, endpoint limits, master equation, Bayes reversal, exact mass transport, expected gradient identity, training improvement, and seeded Monte Carlo checks.

## Mathematical scope and limitations

The data contains two complete four-bit patterns, with a configurable mixture. Conditional noisy samples factor across coordinates; the marginal distribution need not factor. Both demos use the same uniform-to-clean product path. The browser’s classifier predicts the two complete endpoints, unlike a large token-factorized neural denoiser. Its 25 × 16 logits are trained by exact full-batch cross-entropy; this is real optimization but no neural generalization claim is made.

The flow sampler uses adaptive one-jump Euler steps and ends at t=0.995, with finite-step error and residual corruption. It does not forcibly assign final targets. The diffusion sampler uses the exact finite-state reverse kernel in oracle mode and an endpoint-posterior mixture in learned mode. Monte Carlo plots use 192 samples, displaying the first 72; their empirical frequencies fluctuate. Outlined bars always show the ideal analytical path, even when sampling with a learned model.

Conditional/marginal squared-loss gradient equivalence is explained with its assumptions, separately from the cross-entropy posterior-learning identity. The demo does not claim all discrete objectives have identical gradients. Matching intermediate marginals does not make two sample processes identical.

## Primary references

- Campbell et al., *Generative Flows on Discrete State-Spaces*, first posted February 2024: https://arxiv.org/abs/2402.04997
- Gat et al., *Discrete Flow Matching*, 2024: https://arxiv.org/abs/2407.15595
- Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces*, 2021: https://arxiv.org/abs/2107.03006
- Lipman et al., *Flow Matching for Generative Modeling*, 2023: https://arxiv.org/abs/2210.02747

This is an original educational implementation, not an official paper reproduction. Sources are cited for the concepts, not copied code or figures.

The two possible clean endpoints are known to the toy classifier. In the diffusion example, even an untrained classifier has the correct final support because of the conditional bridge; training improves posterior weights and intermediate transition probabilities. This dataset is deliberately small enough to enumerate, so direct mixture sampling would be easier than either generative construction.

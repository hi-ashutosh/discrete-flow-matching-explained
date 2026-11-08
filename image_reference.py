#!/usr/bin/env python3
"""Exact finite-mixture discrete image diffusion, standard-library only.
Run python3 image_reference.py. For both flow and diffusion use image-model.mjs.
This is an exact teaching posterior over synthetic digits, not a trained network.
"""
import json
import math
import random
from pathlib import Path

CATALOG = json.loads((Path(__file__).parent / 'digits.json').read_text())


def posterior(image, signal):
    logs = []
    for item in CATALOG:
        distance = sum(x != z for x, z in zip(image, item['pixels']))
        logs.append((64 - distance) * math.log((1 + signal) / 2)
                    + distance * math.log(max(1e-300, (1 - signal) / 2)))
    maximum = max(logs)
    weights = [math.exp(v - maximum) for v in logs]
    total = sum(weights)
    return [v / total for v in weights]


def reverse_step(image, k, steps, rng):
    signal, previous = 1 - k / steps, 1 - (k - 1) / steps
    weights = posterior(image, signal)
    # A NEW latent endpoint is sampled at every step, not fixed for the run.
    clean = rng.choices(CATALOG, weights=weights)[0]['pixels']
    ratio = signal / previous
    output = []
    for x, z in zip(image, clean):
        numerator = ((1 - ratio) / 2 + ratio * (x == 1)) * ((1 - previous) / 2 + previous * z)
        denominator = (1 - signal) / 2 + signal * (x == z)
        output.append(int(rng.random() < numerator / denominator))
    return output


def draw(image):
    return '\n'.join(''.join('██' if v else '  ' for v in image[y:y+8]) for y in range(0,64,8))


def main():
    rng = random.Random(41)
    images = [[rng.randrange(2) for _ in range(64)] for _ in range(24)]
    print('Source: independent random pixels\n' + draw(images[0]))
    for k in range(40, 0, -1):
        images = [reverse_step(x, k, 40, rng) for x in images]
        if k in (30, 15, 1):
            print(f'\nSelected sample after reverse step to k={k-1}:\n' + draw(images[0]))
    support = {tuple(item['pixels']) for item in CATALOG}
    assert all(tuple(x) in support for x in images)
    print('\nAll 24 final samples are in the target dataset. No nearest-image correction was used.')
    print('Each trajectory uses fresh posterior inference at every reverse step.')


if __name__ == '__main__':
    main()

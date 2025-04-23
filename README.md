# Compute the minimal enclosing circle

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/min_circle/badge)](https://www.codefactor.io/repository/github/tschm/min_circle)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)

We use different solvers and compare readability and speed.

## The problem

Given $N$ random points in an $m$ dimensional space we compute
the center $x$ and the radius $r$ of a ball such that all $N$
points are contained in this ball.

## Makefile

Create the virtual environment defined in requirements.txt using

```bash
make install
```

## Marimo

We use Marimo (instead of Jupyter) to perform our experiments. Start with

```bash
make marimo
```

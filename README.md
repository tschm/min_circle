# Compute the minimal enclosing circle

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

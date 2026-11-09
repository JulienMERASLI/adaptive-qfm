# Adapted from https://github.com/SimpleArt/spsa

from math import isinf, isnan, sqrt
from typing import Callable, Optional, Sequence, SupportsFloat, SupportsIndex, Union
import numpy as np
import operator

ArrayLike = Union[
    np.ndarray,
    float,
    Sequence[float],
    Sequence[Sequence[float]],
    Sequence[Sequence[Sequence[float]]],
    Sequence[Sequence[Sequence[Sequence[float]]]],
    Sequence[Sequence[Sequence[Sequence[Sequence[float]]]]],
]

def type_check(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    adam: bool,
    iterations: int,
    lr: Optional[float],
    lr_decay: float,
    lr_power: float,
    px: Optional[float],
    px_decay: float,
    px_power: float,
    momentum: float,
    beta: float,
    epsilon: float,
    /,
) -> np.ndarray:
    """Type check the parameters and casts `x` to a numpy array."""
    # Type-check.
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(x, (SupportsFloat, np.ndarray, Sequence)):
        raise TypeError(f"x must be either a real number, numpy array, or sequence, got {x!r}")
    elif not isinstance(iterations, SupportsIndex):
        raise TypeError(f"iterations cannot be interpreted as an integer, got {iterations!r}")
    elif not isinstance(adam, SupportsIndex):
        raise TypeError(f"adam cannot be interpreted as an integer, got {adam!r}")
    adam = bool(operator.index(adam))
    names = ("lr_decay", "lr_power")
    values = (lr_decay, lr_power)
    if lr is not None:
        names = ("lr", *names)
        values = (lr, *values)
    if px is not None:
        names = (*names, "px")
        values = (*values, px)
    names = (*names, "px_decay", "px_power", "momentum", "beta", "epsilon")
    values = (*values, px_decay, px_power, momentum, beta, epsilon)
    for name, value in zip(names, values):
        if not isinstance(value, SupportsFloat):
            raise TypeError(f"{name} must be real, got {value!r}")
        elif isnan(float(value)):
            raise ValueError(f"{name} must not be nan, got {value!r}")
        elif isinf(float(value)):
            raise ValueError(f"{name} must not be infinite, got {value!r}")
        elif float(value) <= 0:
            raise ValueError(f"{name} must not be negative, got {value!r}")
    names = ("lr_power", "px_power", "momentum", "beta")
    values = (lr_power, px_power, momentum, beta)
    for name, value in zip(names, values):
        if float(value) >= 1:
            raise ValueError(f"{name} must not be greater than 1, got {value!r}")
    # Cast to numpy array.
    x = np.array(x, dtype=float)
    # Type-check.
    if x.size == 0:
        raise ValueError("cannot optimize with array of size 0")
    elif np.isnan(x).any():
        raise ValueError(f"x must not contain nan")
    elif np.isinf(x).any():
        raise ValueError(f"x must not contain infinity")
    return x

def minimize(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    /,
    *,
    adam: bool = True,
	iterations: int = 10_000,
	lr: Optional[float] = None,
	lr_decay: float = 1e-3,
	lr_power: float = 0.5,
	px: Optional[float] = None,
	px_decay: float = 1e-2,
	px_power: float = 0.161,
	momentum: float = 0.97,
	beta: float = 0.999,
	epsilon: float = 1e-7,
) -> np.ndarray:

    try:
        x = type_check(
            f, x, adam, iterations, lr, lr_decay, lr_power,
            px, px_decay, px_power, momentum, beta, epsilon
        )
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)

    adam = bool(operator.index(adam))
    iterations = operator.index(iterations)

    if lr is not None:
        lr = float(lr)

    lr_decay = float(lr_decay)
    lr_power = float(lr_power)

    if px is not None:
        px = float(px)

    px_decay = float(px_decay)
    px_power = float(px_power)
    momentum = float(momentum)
    beta = float(beta)
    epsilon = float(epsilon)

    rng = np.random.default_rng()

    m1 = 1.0 - momentum
    m2 = 1.0 - beta

    # Initial objective evaluation.
    y = f(x)

    bn = 1.0
    noise = 0.0

    # Initial perturbation size.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))

        for _ in range(8):
            dx = rng.choice((-1.0, 1.0), x.shape) * px

            y1 = f(x + dx)
            y2 = f(x - dx)

            if y1 != y2:
                break

            px *= 1.2

    # Initial gradient estimation.
    b1 = 0.0
    b2 = 0.0

    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)

    for _ in range(8):
        dx = rng.choice((-1.0, 1.0), x.shape) * px

        df_dx = (f(x + dx) - f(x - dx)) * 0.5 / dx

        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)

        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)

        square_gx += m2 * (
            (slow_gx / b2) ** 2 - square_gx
        )

    # Estimate learning rate.
    dx = 3 / b1 * gx

    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)

    if lr is None:
        lr = 1e-5

        for _ in range(3):
            for _ in range(32):
                if f(x - lr * dx) >= y:
                    break
                lr *= 1.4

    # Track average solution.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x

    # Track best solution.
    y_best = y
    x_best = x.copy()

    consecutive_fails = 0
    improvement_fails = 0

    # Initial step.
    dx = gx / b1

    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)

    # Optimization.
    for i in range(iterations):

        # Estimate next point.
        x_next = x - lr * dx

        # Estimate gradient.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        dx /= np.sqrt(square_gx / b2 + epsilon)

        y1 = f(x_next + dx)
        y2 = f(x_next - dx)

        df = (y1 - y2) * 0.5
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)

        # Update gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)

        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)

        square_gx += m2 * (
            (slow_gx / b2) ** 2 - square_gx
        )

        # Compute step size.
        dx = gx / (b1 * (1 + lr_decay * i) ** lr_power)

        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)

        # Reuse cached objective value at x.
        y3 = y

        y4 = f(x - lr * m1 * dx)
        y5 = f(x - lr / m1 * dx)

        # Deterministic line search.
        if y3 < min(y4, y5):
            lr /= 1.3

        if y4 < min(y3, y5):
            lr *= 1.3 / 1.4

        if y5 < min(y3, y4):
            lr *= 1.4

        # Minimum learning rate.
        lr = max(
            lr,
            epsilon / (1 + 0.01 * i) ** 0.5
            * (1 + 0.25 * np.linalg.norm(x))
        )

        # Update solution.
        x -= lr * dx

        # Evaluate updated solution once.
        y = f(x)

        # Update average.
        bx += mx / (1 + 0.01 * i) ** 0.303 * (1 - bx)
        x_avg += mx / (1 + 0.01 * i) ** 0.303 * (x - x_avg)

        consecutive_fails += 1

        # Track best solution.
        if y < y_best:
            y_best = y
            x_best = x.copy()
            consecutive_fails = 0

        # Restart if progress stalls.
        if consecutive_fails < 128 * (improvement_fails + 8):
            continue

        consecutive_fails = 0
        improvement_fails += 1

        x = x_best.copy()

        bx = mx * (1 - mx)
        x_avg = bx * x

        b1 = m1 * (1 - m1)
        gx = b1 / b2 * slow_gx

        slow_gx *= m2 * (1 - m2) / b2
        square_gx *= m2 * (1 - m2) / b2

        b2 = m2 * (1 - m2)

        lr /= 64 * improvement_fails

        # Cached objective value is y_best.
        y = y_best

    return x_best if y_best <= y else x

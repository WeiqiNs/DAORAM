import math


def max_bucket_load(num_bins: int, security_bits: int = 128) -> int:
    x = (math.log2(num_bins) + security_bits - 1) / math.e
    return math.ceil(math.e ** (lambert_w(x) + 1))


def lambert_w(x: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    if x == 0:
        return 0.0
    if x < -math.exp(-1):
        raise ValueError("lambert_w(x) is not defined for x < -1/e.")

    w = 0 if x <= 1 else math.log(x) - math.log(math.log(x))

    for _ in range(max_iter):
        ew = math.exp(w)
        wew = w * ew
        w_next = w - (wew - x) / (ew * (w + 1) - (w + 2) * (wew - x) / (2 * w + 2))

        if abs(w_next - w) < tol:
            return w_next

        w = w_next

    raise RuntimeError("Lambert W function did not converge")

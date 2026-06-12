import torch
import math

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def LayerNorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-5)


class CheckError(RuntimeError):
    pass


def _normalize_for_check(x):
    # Treat torch.Size, tuple, and list shapes as equivalent.
    if isinstance(x, (list, tuple)):
        return tuple(x)

    # Avoid requiring torch import just for this helper.
    if type(x).__name__ == "Size" and type(x).__module__.startswith("torch"):
        return tuple(x)

    return x


def check_eq(left, right, left_name="left", right_name="right"):
    left_cmp = _normalize_for_check(left)
    right_cmp = _normalize_for_check(right)

    if left_cmp != right_cmp:
        raise CheckError(
            f"CHECK_EQ failed: {left_name} == {right_name}\n"
            f"  {left_name}:  {left!r}\n"
            f"  {right_name}: {right!r}"
        )
    
def XavierFactor(param):
    return 0.02
    # return math.sqrt(2 / (param.shape[0] + param.shape[1]))
    
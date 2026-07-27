import torch

def make_problem_spec(problem, *, tkwargs, noise_se=None, ref_point=None):
    """
    Returns: bounds (2,d), ref_point (m,) or None, noise_se (m,) or None
    - If you pass ref_point explicitly, it wins.
    - If problem has problem.ref_point, we use it.
    - noise_se is optional; if you don't pass it, we try problem.noise_std if present.
    """
    # bounds
    if not hasattr(problem, "bounds"):
        raise ValueError("Problem must have a .bounds tensor (2, d) or you must provide bounds another way.")
    bounds = problem.bounds.to(**tkwargs)

    # ref_point (needed for qNEHVI / hypervolume methods)
    rp = ref_point
    if rp is None and hasattr(problem, "ref_point"):
        rp = problem.ref_point
    if rp is not None:
        rp = torch.as_tensor(rp, **tkwargs).view(-1)

    # noise
    ns = noise_se
    if ns is None and hasattr(problem, "noise_std") and problem.noise_std is not None:
        ns = problem.noise_std
    if ns is not None:
        ns = torch.as_tensor(ns, **tkwargs).view(-1)

    return bounds, rp, ns

import torch


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Pure PyTorch FPS fallback.

    Args:
        xyz: (B, N, 3)
        npoint: number of samples to pick

    Returns:
        (B, npoint) long indices
    """
    if xyz.ndim != 3 or xyz.size(-1) != 3:
        raise ValueError(f"Expected xyz shape (B, N, 3), got {tuple(xyz.shape)}")

    bsz, num_points, _ = xyz.shape
    if npoint <= 0:
        raise ValueError(f"npoint must be > 0, got {npoint}")
    npoint = min(npoint, num_points)

    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((bsz, num_points), 1e10, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.randint(0, num_points, (bsz,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1)[1]

    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features by index.

    Args:
        features: (B, C, N)
        idx: (B, S)

    Returns:
        (B, C, S)
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features shape (B, C, N), got {tuple(features.shape)}")
    if idx.ndim != 2:
        raise ValueError(f"Expected idx shape (B, S), got {tuple(idx.shape)}")

    idx = idx.long()
    expanded_idx = idx.unsqueeze(1).expand(-1, features.size(1), -1)
    return torch.gather(features, dim=2, index=expanded_idx)


def three_nn(unknown: torch.Tensor, known: torch.Tensor):
    """Find 3 nearest neighbors in known for every unknown point.

    Args:
        unknown: (B, N, 3)
        known: (B, M, 3)

    Returns:
        dist: (B, N, 3)
        idx: (B, N, 3)
    """
    if unknown.ndim != 3 or known.ndim != 3:
        raise ValueError(
            f"Expected unknown/known to be rank-3 tensors, got {tuple(unknown.shape)} and {tuple(known.shape)}"
        )

    dist_matrix = torch.cdist(unknown, known, p=2)
    dist, idx = torch.topk(dist_matrix, k=3, dim=-1, largest=False, sorted=False)
    return dist, idx.long()


def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Interpolate features by weighted 3-neighbor lookup.

    Args:
        features: (B, C, M)
        idx: (B, N, 3)
        weight: (B, N, 3)

    Returns:
        (B, C, N)
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features shape (B, C, M), got {tuple(features.shape)}")
    if idx.ndim != 3 or weight.ndim != 3:
        raise ValueError(f"Expected idx/weight shape (B, N, 3), got {tuple(idx.shape)} and {tuple(weight.shape)}")

    idx = idx.long()
    bsz, channels, _ = features.shape
    num_unknown = idx.size(1)

    features_expanded = features.unsqueeze(2).expand(-1, -1, num_unknown, -1)
    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1, -1)
    gathered = torch.gather(features_expanded, dim=3, index=idx_expanded)
    interpolated = torch.sum(gathered * weight.unsqueeze(1), dim=-1)
    return interpolated

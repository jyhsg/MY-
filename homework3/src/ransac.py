"""RANSAC with minimal eight-point samples and all-inlier refinement."""
import numpy as np
from .fundamental import eight_point, sampson_squared


def estimate_ransac(left, right, threshold_px=1.5, confidence=.999,
                    max_iterations=10000, seed=42):
    if len(left) < 8 or len(left) != len(right):
        raise ValueError("Need at least eight paired correspondences")
    if threshold_px <= 0 or not 0 < confidence < 1 or max_iterations < 1:
        raise ValueError("Invalid RANSAC settings")
    left, right = np.asarray(left), np.asarray(right)
    rng = np.random.default_rng(seed)
    limit, iteration, degenerate = max_iterations, 0, 0
    best_F, best_mask, best_score = None, None, (-1, -np.inf)
    threshold2 = threshold_px**2

    def classify(F):
        d = sampson_squared(F, left, right)
        mask = d <= threshold2
        return mask, (int(mask.sum()), -float(np.minimum(d, threshold2).sum()))

    while iteration < limit:
        iteration += 1
        sample = rng.choice(len(left), 8, replace=False)
        try:
            F = eight_point(left[sample], right[sample])
        except (ValueError, np.linalg.LinAlgError):
            degenerate += 1
            continue
        mask, score = classify(F)
        if score > best_score and mask.sum() >= 8:
            # Local optimization: fit all current inliers, accept only an improvement.
            for _ in range(5):
                try:
                    refined = eight_point(left[mask], right[mask])
                except (ValueError, np.linalg.LinAlgError):
                    break
                new_mask, new_score = classify(refined)
                if new_score < score:
                    break
                F, mask, score = refined, new_mask, new_score
            best_F, best_mask, best_score = F, mask, score
            probability = (mask.mean())**8
            if probability >= 1:
                limit = iteration
            elif probability > 0:
                required = int(np.ceil(np.log1p(-confidence) / np.log1p(-probability)))
                limit = min(limit, max(iteration, required))
    if best_F is None:
        raise RuntimeError("No non-degenerate fundamental matrix found")
    # Deliver the all-consensus least-squares fit, then report its actual inliers.
    final_F = eight_point(left[best_mask], right[best_mask])
    final_mask, _ = classify(final_F)
    if final_mask.sum() < 8:
        raise RuntimeError("Consensus refit produced fewer than eight inliers")
    return final_F, final_mask, dict(iterations=iteration, degenerate_samples=degenerate,
        threshold_px=threshold_px, confidence=confidence, max_iterations=max_iterations,
        seed=seed, fit_count=int(best_mask.sum()), inliers=int(final_mask.sum()),
        matches=len(left), inlier_ratio=float(final_mask.mean()))

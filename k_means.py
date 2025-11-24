"""
kmeans_from_scratch_full.py

Comprehensive, well-documented implementation of K-Means clustering from scratch
(using only NumPy for the core algorithm). This file is intentionally long (500+ lines)
and includes:

- Utilities to generate synthetic datasets (blobs, elongated clusters)
- A KMeans class with random and k-means++ initialization
- Methods for fit / predict / fit_predict
- Inertia and silhouette score computed from scratch
- Multiple experiment helpers: elbow analysis, silhouette analysis, sensitivity checks
- Lots of inline comments and docstrings for teaching and learning

Requirements:
- numpy
- matplotlib (for visualizations)
- pandas (optional, for tabular summaries)

Usage:
- Run as a script: python kmeans_from_scratch_full.py
- Or import KMeans and utilities into a notebook for interactive experiments.

Author: ChatGPT (educational implementation)
"""

import numpy as np
import math
import random
from typing import Optional, Tuple, List, Dict

# Optional imports for demo visualizations and tabular display
try:
    import matplotlib.pyplot as plt
    import pandas as pd
except Exception:
    plt = None
    pd = None

# Set global random seed for reproducible examples when running as script
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)


# ----------------------------- Utilities ---------------------------------

def make_blobs(n_samples=300, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), random_state=None):
    """
    Create synthetic isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    n_samples : int or sequence
        If int, total number of samples. If sequence, must match number of centers
        and gives per-center sample counts.
    centers : int or array-like
        Number of centers to generate or explicit center coordinates (shape (k, n_features)).
    cluster_std : float or sequence
        Standard deviation of clusters (scalar or per-center).
    center_box : tuple (low, high)
        Bounding box used when generating random centers.
    random_state : int or None
        Use a RandomState for reproducibility.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,) integer labels
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    # If centers is an integer, create random centers in the box
    if isinstance(centers, int):
        n_centers = centers
        centers = rng.uniform(center_box[0], center_box[1], size=(n_centers, 2))
    else:
        centers = np.asarray(centers)
        n_centers = centers.shape[0]

    # Determine how many samples go to each center
    if isinstance(n_samples, int):
        base = n_samples // n_centers
        counts = [base] * n_centers
        for i in range(n_samples - base * n_centers):
            counts[i % n_centers] += 1
    else:
        counts = list(n_samples)
        if len(counts) != n_centers:
            raise ValueError("n_samples sequence length must equal number of centers")

    # Spread cluster_std to per-center list
    if np.isscalar(cluster_std):
        stds = [cluster_std] * n_centers
    else:
        stds = list(cluster_std)
        if len(stds) != n_centers:
            raise ValueError("cluster_std sequence length must equal number of centers")

    X_parts = []
    y_parts = []
    for idx, (center, cnt, std) in enumerate(zip(centers, counts, stds)):
        points = rng.normal(loc=center, scale=std, size=(cnt, centers.shape[1]))
        X_parts.append(points)
        y_parts.append(np.full(cnt, idx, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle rows so that labels are not ordered by cluster
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of X and rows of Y.
    Returns D where D[i, j] = ||X[i] - Y[j]||^2
    """
    # Efficient broadcasting computation using (a-b)^2 = a^2 + b^2 - 2ab
    X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]  # shape (n_x, 1)
    Y_sq = np.sum(Y ** 2, axis=1)[np.newaxis, :]  # shape (1, n_y)
    cross = X.dot(Y.T)  # shape (n_x, n_y)
    D = X_sq + Y_sq - 2 * cross
    # Numerical safety: clip tiny negatives to zero
    D = np.maximum(D, 0.0)
    return D


# ----------------------------- KMeans Class -------------------------------

class KMeans:
    """
    K-Means clustering implemented using NumPy for numerical operations.

    Features implemented:
    - random and k-means++ initialization
    - multiple restarts (n_init) keeping best inertia
    - inertia computation
    - silhouette score (from scratch using pairwise distances)
    - fit, predict, fit_predict
    - plotting helpers (if matplotlib available)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        if init not in ("random", "kmeans++"):
            raise ValueError("init must be 'random' or 'kmeans++'")

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        # Attributes set after fit
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        # RNG for reproducible initializations per instance
        self._rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()

    def _init_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """Pick k distinct random samples from X as initial centroids."""
        n_samples = X.shape[0]
        indices = self._rng.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _init_centroids_kmeanspp(self, X: np.ndarray) -> np.ndarray:
        """
        k-means++ initialization algorithm.
        Picks first centroid uniformly at random from the data, then iteratively picks next
        centroids with probability proportional to squared distance to the closest chosen centroid.
        """
        n_samples = X.shape[0]
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        # Choose first centroid randomly
        first_idx = self._rng.randint(0, n_samples)
        centroids[0] = X[first_idx]

        # distances to nearest centroid so far (squared)
        closest_dist_sq = pairwise_distances(X, centroids[0:1]).ravel()

        for c in range(1, self.n_clusters):
            probs = closest_dist_sq / np.sum(closest_dist_sq)
            # numerical safety
            if np.isnan(probs).any():
                probs = np.nan_to_num(probs, copy=True)
                probs = probs / probs.sum()

            cumulative = np.cumsum(probs)
            r = self._rng.rand()
            next_idx = np.searchsorted(cumulative, r)
            centroids[c] = X[next_idx]

            # update optimal distances
            new_dist_sq = pairwise_distances(X, centroids[c:c+1]).ravel()
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centroids

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each row in X to the nearest centroid index."""
        D = pairwise_distances(X, centroids)  # squared distances
        return np.argmin(D, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as mean of assigned members. Reinitialize empty centroids randomly."""
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features), dtype=X.dtype)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                # empty cluster: reinitialize to a random data point
                centroids[k] = X[self._rng.randint(0, X.shape[0])]
            else:
                centroids[k] = np.mean(members, axis=0)
        return centroids

    def _compute_inertia(self, X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
        """Compute sum of squared distances (inertia) for given labeling and centroids."""
        D = pairwise_distances(X, centroids)
        assigned_dist_sq = D[np.arange(X.shape[0]), labels]
        return float(np.sum(assigned_dist_sq))

    def _has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Convergence when maximum centroid movement <= tol."""
        shifts = np.linalg.norm(new_centroids - old_centroids, axis=1)
        return np.max(shifts) <= self.tol

    def _single_run(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Run K-Means once starting from an initialization. Returns centroids, labels, inertia, n_iter used.
        """
        # Initialize
        if self.init == "random":
            centroids = self._init_centroids_random(X)
        else:
            centroids = self._init_centroids_kmeanspp(X)

        for i in range(1, self.max_iter + 1):
            labels = self._assign_labels(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            if self._has_converged(centroids, new_centroids):
                centroids = new_centroids
                inertia = self._compute_inertia(X, centroids, labels)
                return centroids, labels, inertia, i
            centroids = new_centroids

        # Max iterations reached
        labels = self._assign_labels(X, centroids)
        inertia = self._compute_inertia(X, centroids, labels)
        return centroids, labels, inertia, self.max_iter

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means to data X. Runs self.n_init restarts and keeps the best result (lowest inertia).
        """
        X = np.asarray(X)
        best_inertia = float("inf")
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        # To ensure reproducibility across runs, create a local RNG seeded by the instance RNG
        # and draw seeds for each restart. This preserves determinism when random_state is set.
        run_seeds = [self._rng.randint(0, 2 ** 31 - 1) for _ in range(self.n_init)]

        for run_idx, seed in enumerate(run_seeds):
            # For each restart, create a temporary KMeans object using that seed for initialization
            temp = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=1,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=seed,
            )
            centroids, labels, inertia, n_iter = temp._single_run(X)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample in X to the nearest fitted centroid."""
        X = np.asarray(X)
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit to X then return labels."""
        self.fit(X)
        return self.labels_

    # ------------------ Evaluation metrics from scratch ------------------

    def silhouette_score(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        """
        Compute mean silhouette score for dataset X and labels.
        Silhouette for sample i: s(i) = (b(i) - a(i)) / max(a(i), b(i))
        where a(i) = mean distance to other points in same cluster
              b(i) = min mean distance to points in other clusters
        """
        X = np.asarray(X)
        if labels is None:
            if self.labels_ is None:
                raise ValueError("Labels are required to compute silhouette score")
            labels = self.labels_

        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        if unique_labels.shape[0] == 1:
            # silhouette undefined for single cluster; return 0
            return 0.0

        # Compute full pairwise Euclidean distances (not squared) because silhouette uses distances
        sqD = pairwise_distances(X, X)
        D = np.sqrt(sqD)

        a = np.zeros(n_samples, dtype=float)
        b = np.full(n_samples, np.inf, dtype=float)

        # Compute intra-cluster mean distances a(i)
        for lab in unique_labels:
            idxs = np.where(labels == lab)[0]
            if idxs.size == 1:
                a[idxs] = 0.0
            else:
                # submatrix of distances among points in the same cluster
                subD = D[np.ix_(idxs, idxs)].copy()
                # exclude diagonal by filling it with nan
                np.fill_diagonal(subD, np.nan)
                a_vals = np.nanmean(subD, axis=1)
                a[idxs] = a_vals

        # Compute nearest-cluster mean distances b(i)
        for lab in unique_labels:
            idxs_lab = np.where(labels == lab)[0]
            for other_lab in unique_labels:
                if other_lab == lab:
                    continue
                idxs_other = np.where(labels == other_lab)[0]
                if idxs_other.size == 0:
                    continue
                # mean distance from each point in idxs_lab to points in idxs_other
                mean_to_other = np.mean(D[np.ix_(idxs_lab, idxs_other)], axis=1)
                b[idxs_lab] = np.minimum(b[idxs_lab], mean_to_other)

        sil_vals = (b - a) / np.maximum(a, b)
        # Replace any NaNs (possible when both a and b are zero) with zero
        sil_vals = np.nan_to_num(sil_vals)
        return float(np.mean(sil_vals))

    # ------------------ Visualization helpers (if matplotlib available) ------------------

    def plot_clusters(self, X: np.ndarray, labels: Optional[np.ndarray] = None, show_centroids: bool = True, title: str = "") -> None:
        """Plot 2D clusters. Requires matplotlib. Only supports 2D features."""
        if plt is None:
            raise RuntimeError("matplotlib not available; cannot plot")
        X = np.asarray(X)
        if X.shape[1] != 2:
            raise ValueError("plot_clusters only supports 2D data")
        if labels is None:
            if self.labels_ is None:
                raise ValueError("Labels required for plotting")
            labels = self.labels_
        fig, ax = plt.subplots(figsize=(7, 5))
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(X[mask, 0], X[mask, 1], label=f"Cluster {lab}", alpha=0.6, s=35)
        if show_centroids and self.cluster_centers_ is not None:
            ax.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], marker="x", s=120, linewidths=3, label="Centroids")
        ax.set_title(title if title else f"KMeans k={self.n_clusters} init={self.init}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        plt.show()


# ---------------------- Experiment helpers ---------------------------------

def evaluate_k_range(X: np.ndarray, k_values: List[int], init_methods: List[str], n_init=5, random_state=None) -> Optional["pd.DataFrame"]:
    """
    Run KMeans across a range of k values and initialization methods.
    Returns a pandas DataFrame if pandas is available, otherwise returns a list of dicts.
    """
    results = []
    for init in init_methods:
        for k in k_values:
            km = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=300, tol=1e-4, random_state=random_state)
            km.fit(X)
            sil = km.silhouette_score(X)
            results.append({
                "k": k,
                "init": init,
                "inertia": km.inertia_,
                "silhouette": sil,
                "n_iter": km.n_iter_,
            })
            print(f"Completed k={k}, init={init}: inertia={km.inertia_:.3f}, silhouette={sil:.4f}, n_iter={km.n_iter_}")
    if pd is not None:
        df = pd.DataFrame(results).sort_values(["init", "k"]).reset_index(drop=True)
        return df
    return results


def elbow_plot_from_results(df, init_method: str):
    if plt is None or df is None:
        return
    subset = df[df["init"] == init_method]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(subset["k"], subset["inertia"], marker="o")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia (sum of squared distances)")
    ax.set_title(f"Elbow plot (init={init_method})")
    plt.show()


def silhouette_plot_from_results(df, init_method: str):
    if plt is None or df is None:
        return
    subset = df[df["init"] == init_method]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(subset["k"], subset["silhouette"], marker="o")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Mean Silhouette Score")
    ax.set_title(f"Silhouette vs k (init={init_method})")
    plt.show()


# ---------------------- Long demonstration (script) -------------------------

def long_demo():
    """
    Full demonstration that:
    - constructs several synthetic datasets of varying difficulty
    - runs KMeans with different inits and k values
    - shows elbow and silhouette diagnostics
    - visualizes centroid sensitivity across different runs
    - demonstrates limitations

    This method displays plots (requires matplotlib).
    """
    if plt is None:
        print("Matplotlib not installed — demo will only run computations and print summaries.")

    print("Generating datasets...")
    X_easy, y_easy = make_blobs(n_samples=450, centers=3, cluster_std=0.8, center_box=(-8, 8), random_state=1)
    centers_med = np.array([[-4, -2], [0, 0], [3, 4]])
    X_med, y_med = make_blobs(n_samples=450, centers=centers_med, cluster_std=[1.4, 1.2, 1.8], random_state=2)
    centers_hard = np.array([[0, 0], [5, 5], [-6, 4]])
    X_hard, y_hard = make_blobs(n_samples=450, centers=centers_hard, cluster_std=[0.5, 2.2, 0.9], random_state=3)

    datasets = [
        ("Easy - well-separated", X_easy, y_easy),
        ("Medium - overlapping", X_med, y_med),
        ("Hard - different densities", X_hard, y_hard),
    ]

    for title, X, y in datasets:
        print(f"\n\nDataset: {title}")
        print(f"Data shape: {X.shape}, true clusters: {len(np.unique(y))}")

        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            for lab in np.unique(y):
                mask = y == lab
                ax.scatter(X[mask, 0], X[mask, 1], label=f"True {lab}", alpha=0.6, s=30)
            ax.set_title(f"{title} (true labels)")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.legend()
            plt.show()

        k_values = list(range(2, 7))
        df_results = evaluate_k_range(X, k_values, init_methods=["random", "kmeans++"], n_init=8, random_state=42)
        if pd is not None:
            print(df_results)
        elbow_plot_from_results(df_results, "random")
        elbow_plot_from_results(df_results, "kmeans++")
        silhouette_plot_from_results(df_results, "random")
        silhouette_plot_from_results(df_results, "kmeans++")

        # Pick best k by silhouette for kmeans++
        if pd is not None:
            best_row = df_results[df_results["init"] == "kmeans++"].sort_values("silhouette", ascending=False).iloc[0]
            best_k = int(best_row["k"])
            print(f"Best k by silhouette for {title} using kmeans++: {best_k} (silhouette={best_row['silhouette']:.4f})")
            km_best = KMeans(n_clusters=best_k, init="kmeans++", n_init=12, random_state=123)
            km_best.fit(X)
            print(f"Best inertia: {km_best.inertia_:.3f}, n_iter: {km_best.n_iter_}")
            if plt is not None:
                km_best.plot_clusters(X, title=f"{title} - KMeans (k={best_k}, init=kmeans++)")
            sil = km_best.silhouette_score(X)
            print(f"Silhouette score (best fit): {sil:.4f}")

        # Sensitivity test with true k
        true_k = len(np.unique(y))
        print(f"\nSensitivity test for true k={true_k} (multiple runs with different seeds):")
        centroids_collection = []
        inertia_list = []
        for seed in range(5):
            km_sample = KMeans(n_clusters=true_k, init="random", n_init=1, random_state=seed)
            km_sample.fit(X)
            inertia_list.append(km_sample.inertia_)
            centroids_collection.append(km_sample.cluster_centers_)
            print(f" seed={seed}: inertia={km_sample.inertia_:.3f}, n_iter={km_sample.n_iter_}")
        print("Inertia across seeds:", inertia_list)

        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20)
            for i, centroids in enumerate(centroids_collection):
                ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, label=f"Run {i}")
            ax.set_title(f"Centroid positions across different random initializations (k={true_k})")
            ax.legend()
            plt.show()

        print("\nEmpirical Observations and Limitations:")
        print("- K-Means assumes spherical clusters and equal variance; it struggles with clusters of different shapes/densities.")
        print("- Initialization matters: random init can converge to poor local minima; k-means++ often helps but does not guarantee global optimum.")
        print("- K must be specified (we used elbow and silhouette as heuristics).")
        print("- K-Means is sensitive to outliers because it uses means (centroids).")
        print("- Euclidean distance causes scale sensitivity; features should be normalized in real-world datasets.")

    # Final deeper experiment: elongated cluster
    print("\n\nFinal deep-dive: dataset with elongated cluster where KMeans struggles")
    rng = np.random.RandomState(99)
    # Create an elongated diagonal line of points
    coords = np.linspace(-5, 5, 120)
    long_line = np.column_stack([coords, coords]) + rng.normal(scale=0.3, size=(len(coords), 2))
    sph1 = rng.normal(loc=[7, -6], scale=0.6, size=(120, 2))
    sph2 = rng.normal(loc=[-7, 6], scale=0.6, size=(120, 2))
    X_weird = np.vstack([long_line, sph1, sph2])

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_weird[:, 0], X_weird[:, 1], s=20, alpha=0.6)
        ax.set_title("Weird dataset: elongated cluster + two spherical clusters")
        plt.show()

    km3 = KMeans(n_clusters=3, init="kmeans++", n_init=12, random_state=11)
    km3.fit(X_weird)
    print("Inertia:", km3.inertia_)
    if plt is not None:
        km3.plot_clusters(X_weird, title="KMeans on elongated dataset (k=3)")
    print("Silhouette score:", km3.silhouette_score(X_weird))
    print("\nObservation: KMeans tends to split elongated clusters incorrectly because centroid-based partitions assume convex spherical clusters.")


# ---------------------------- Module entrypoint ---------------------------

if __name__ == "__main__":
    # Run the long demo when this file is executed directly
    long_demo()

# End of file

import pandas as pd
from glob import glob
from ast import literal_eval
import numpy as np
from sklearn.manifold import TSNE
from scipy.sparse.csgraph import laplacian
import torch
from numpy.typing import NDArray
from typing import Union, Any
from src.criteria.image_comparison import CFrobeniusDistance, SSIMD2


class ExperimentMetrics:
    """A class to extract metrics from df."""

    image_distance: list[float]  # Frobenius Distance between images.
    boundary_distance: list[float]  # Euclidean Distance towards boundary.
    lap_variance: list[float]  # Laplacian variance on image differences.

    escape_ratios: list[float]  # Escape ratio of boundary search.
    coverage: list[float]  # Boundary coverage metric.

    runtime: list[float]
    runtime_norm: list[float]

    def __init__(self, df: pd.DataFrame, xs: list[str], ys: list[str]) -> None:
        """
        Initialize the object and extract metrics from df.

        :param df: The dataframe to extract metrics from.
        :param xs: The names of the x columns to extract.
        :param ys: The names of the y columns to extract.
        """
        """Extract image distances."""
        im_comp = CFrobeniusDistance()._frob
        diffs = [(df["X"] - df[x]) for x in xs]
        f_dist = [d.apply(im_comp) for d in diffs]
        f_dist = self._apply_strategy(pd.DataFrame(f_dist), "min")
        self.image_distance = f_dist.tolist()

        """Extract boundary distance."""
        b_dists = [df[y].apply(distance_to_boundary) for y in ys]
        b_dists = self._apply_strategy(pd.DataFrame(b_dists), "min")
        self.boundary_distance = b_dists.tolist()

        """Extract laplacian variance."""
        lap_var = [d.apply(laplacian_variance) for d in diffs]
        lap_var = self._apply_strategy(pd.DataFrame(lap_var), "max")
        self.lap_variance = lap_var.tolist()

        """Get boundary stats."""
        cov_esc = [get_boundary_stats(df["y"], df[y]) for y in ys]
        cov, esc = tuple(zip(*cov_esc))

        esc = self._apply_strategy(pd.DataFrame(esc), "min")
        self.escape_ratios = esc.tolist()

        cov = self._apply_strategy(pd.DataFrame(cov), "max")
        self.coverage = cov.tolist()

        runt = df["runtime"]
        runt_norm = runt.copy()
        if isinstance(runt[0], str):
            runt = runt.apply(lambda x: _convert_to_seconds(x))
            runt_norm = runt.copy()
        if "iter" in df.columns:
            runt_norm = (runt / df["iter"]).apply(lambda x: x * 15_000)  # Normalize toward budget

        self.runtime_norm = runt_norm.tolist()
        self.runtime = runt.tolist()

    def _apply_strategy(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply strategy for merging df columns."""
        if strategy == "min":
            return df.min(axis=0)
        elif strategy == "max":
            return df.min(axis=0)
        elif strategy == "mean":
            return df.mean(axis=0)
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented.")


def softmax(x: Union[list, NDArray]) -> NDArray:
    """Apply a softmax operation on a list or numpy array."""
    if isinstance(x, list):
        x = np.array(x)
    return (np.e**x) / np.sum(np.e**x)


def cohens_d(x, y) -> float:
    """Calculate CohensD for effect size analysis."""
    return (np.mean(x) - np.mean(y)) / (np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2))


def transform_image(x: Union[str, NDArray]) -> NDArray:
    """Transform an image into standard config."""
    if isinstance(x, str):
        x = literal_eval(x)
    x = np.array(x).squeeze()
    x = x.transpose(1, 2, 0)
    return x


def to_tensor(x: str) -> torch.Tensor:
    """Convert a string to tensor."""
    x = np.array(literal_eval(x))
    return torch.Tensor(x)


def load_and_combine_dfs(path: str, filters: list[str]) -> pd.DataFrame:
    """
    Load and combined experiment dfs from a path.

    :param path: Path to experiment dfs.
    :param filters: Filers to select dfs.
    :returns: Combined experiment dfs.
    """
    maybe_slash = "" if path[-1] == "/" else "/"
    all_dfs = glob(f"{path}{maybe_slash}*.csv")
    if len(all_dfs) == 0:
        raise FileNotFoundError(f"No dfs in directory {path}")
    combined_df = pd.DataFrame()
    for elem in all_dfs:
        split_file = elem.split(".")[0].split("_")
        res_size = 4 if "wrn" in split_file else 3
        if all((kw in split_file for kw in filters)) and (
            len(split_file) - len(filters) == res_size
        ):
            tmp_df = pd.read_csv(elem)
            combined_df = pd.concat([tmp_df, combined_df], ignore_index=True)
            cols = tmp_df.columns

    combined_df.columns = cols
    return combined_df


def get_tsne_from_values(
    *values: list[pd.Series], components: int = 2, random_state: int = 0
) -> list[Any]:
    """Combine multiple data points and compute TSNE."""
    tsne = TSNE(n_components=components, random_state=random_state)

    values = [v.to_list() for v in values]  # Concert pd.Series to list
    cl = [0] + [len(v) for v in values]  # Component lengths for extraction
    total = np.vstack([np.array(v) for v in values])

    emb_total = tsne.fit_transform(total)
    return [emb_total[cl[i] : cl[i] + cl[i + 1]] for i in range(len(cl) - 1)]


def filter_for_classes(
    *elements: list[pd.Series],
    class_information: pd.Series,
    classes: list[int],
    filter_class_information: bool = True,
) -> list[pd.Series]:
    """
    Filter elements based on classes, last element is used to filter.

    :param elements: Elements to filter.
    :param class_information: Classes information for elements.
    :param classes: Classes to filter for.
    :param filter_class_information: Whether to filter class information elements aswell.
    :returns: The filtered elements.
    """
    assert all(
        (len(e) == len(class_information) for e in elements)
    ), "Error, series provided have different lengths."
    mask = class_information.isin(classes)

    elements = [e[mask] for e in elements]
    elements = elements + [class_information[mask]] if filter_class_information else elements
    return elements


def _convert_to_seconds(elem: str) -> float:
    *_, time = elem.split(" ")
    h, m, s = time.split(":")
    s = float(s) + 60 * int(m) + 3600 * int(h)
    return s


def get_boundary_stats(y1: pd.Series, y2: pd.Series) -> tuple[NDArray, float]:
    """
    Get boundary coverage measures based on Kolmogorov-Smirnov distance.

    Get boundary vicinity.

    :param y1: First series of confidence values.
    :param y2: Second series of confidence values.
    :returns: Mean boundary coverage across classes and std.
    """
    boundary_distribution = {i: [] for i in range(10)}
    escaped_boundaries = {i: [] for i in range(10)}
    esc = 0
    for y, yp in zip(y1, y2):
        yp = np.array(yp)
        label = np.argmax(y)
        boundary_location = yp.argsort()[::-1][:2]
        if label not in boundary_location:
            escaped_boundaries[label].append(boundary_location)
            esc += 1
        else:
            bl = list(boundary_location)
            bl.remove(label)
            boundary_distribution[label] += bl

    unif = np.full(9, 1 / 9)
    distances = []
    for label, dist in boundary_distribution.items():
        hist, _ = np.histogram(dist, bins=10, range=(0, 10))
        hist = np.delete(hist, label)
        hist = hist / 10

        dist = 0 if hist.sum() == 0 else (2 * np.sum(np.minimum(unif, hist)) / 2) * hist.sum()

        distances.append(dist)
    distances = np.array(distances)
    return distances, esc / len(y1)


def laplacian_variance(arr: NDArray) -> float:
    """
    Calculate the laplacian variance.

    :param arr: Array to calculate laplacian variance for.
    :returns: The variance value.
    """
    arr = arr.squeeze()
    if len(arr.shape) == 3:
        arr = arr.sum(axis=np.argmin(arr.shape))

    filtered = laplacian(arr)
    return filtered.var()


def reduce_dim(arr: NDArray, reduce_channels: bool) -> NDArray:
    arr = arr.squeeze()
    shape = arr.shape
    if len(shape) == 3 and reduce_channels:
        arr = arr.sum(axis=np.argmin(shape)) / min(shape)
    return arr


def format_cols(df: pd.DataFrame, reduce_channels: bool = False) -> pd.DataFrame:
    for c in df.columns:
        if "X" in c:
            df[c] = df[c].apply(lambda x: reduce_dim(np.array(literal_eval(x)), reduce_channels))
        elif "y" in c:
            df[c] = df[c].apply(lambda x: np.array(literal_eval(x)))
        elif "genome" in c:
            df[c] = df[c].apply(lambda x: np.array([float(v) for v in x[1:-1].split(" ") if v]))
    return df


def distance_to_boundary(arr: NDArray) -> float:
    arr = arr.squeeze()
    boundary_indices = np.argsort(arr)[::-1][:2]
    ideal = np.zeros_like(arr)
    ideal[boundary_indices] = 0.5
    return np.linalg.norm(ideal - arr)

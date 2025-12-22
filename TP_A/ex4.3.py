# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# This is the folder on the Desktop where the data lives.
# The folder name should be changed if your setup is different.
DATA_FOLDER = Path.home() / "Desktop" / "HWDS1" / "data for python"

# All results, including plots and CSV summaries, will be written in the below file.
RESULTS_FOLDER = DATA_FOLDER / "analysis_results"
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

# These are the dataset files that was expected to be analyzed here below written as a list.
EXPECTED_DATASETS = [
    "tp1_artificialdata1.npz",
    "tp1_artificialdata2.npz",
    "tp1_artificialdata3.npz",
    "tp1_artificialdata4.npz",
    "tp1_freyfaces.npz",
    "tp1_digit2.npz",
]

# The eigen spectrum plots planned to display up to this many eigenvalues.
MAX_EIGENVALUES_TO_SHOW = 100


# HELPER FUNCTIONS used:
def extract_matrix_from_npz(npz_path: Path) -> np.ndarray:
    """
    Opens a .npz file and pulls out the main data matrix.

    Some .npz archives use 'X' as the variable name, others use 'data' or
    something similar. If none of those exist, the function simply picks
    the first array inside the file that has two dimensions.

    Everything is converted to float64 so that later linear algebra routines
    run smoothly without type issues.
    """
    with np.load(npz_path) as archive:
        chosen_key = None
        for candidate in ["X", "data", "D", "A", "arr_0"]:
            if candidate in archive.files and np.asarray(archive[candidate]).ndim == 2:
                chosen_key = candidate
                break

        if chosen_key is None:
            for key in archive.files:
                if np.asarray(archive[key]).ndim == 2:
                    chosen_key = key
                    break

        if chosen_key is None:
            raise ValueError(
                f"Could not find a 2D matrix in {npz_path.name}. Keys found: {archive.files}"
            )

        matrix = np.asarray(archive[chosen_key])

        if matrix.dtype.kind in ("i", "u", "b"):
            matrix = matrix.astype(np.float64)
        return matrix


def compute_covariance_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the sample covariance matrix.

    The rows of the data matrix are assumed to be examples, and the columns
    are features. Each column is mean-centered, and then the standard formula
    for covariance is applied: (X_centered^T X_centered) / (n - 1).
    """
    centered = data_matrix - data_matrix.mean(axis=0, keepdims=True)
    n_samples = data_matrix.shape[0]
    return (centered.T @ centered) / (n_samples - 1)


def analyze_eigenvalues(cov_matrix: np.ndarray):
    """
    Works out some facts about the eigenvalues of the covariance matrix.

    It sorts them from largest to smallest computes the determinant in a
    numerically safe way and also computes the product of all eigenvalues
    In exact arithmetic the determinant and the product of eigenvalues
    are identical but in floating point arithmetic they can differ slightly.
    """
    eigenvals = np.linalg.eigvalsh(cov_matrix)[::-1]

    sign, log_abs_det = np.linalg.slogdet(cov_matrix)
    determinant_value = float(sign * np.exp(log_abs_det))

    if np.any(eigenvals <= 0):
        eigen_product = 0.0
    else:
        eigen_product = float(np.exp(np.sum(np.log(eigenvals))))

    consistent = np.isclose(eigen_product, determinant_value, rtol=1e-6, atol=1e-12)
    return eigenvals, determinant_value, eigen_product, bool(consistent)


def save_eigen_spectrum_plot(
    eigenvals: np.ndarray, title: str, output_folder: Path, top_k: int = 100
) -> str:
    """
    Draws a simple line plot of the top eigenvalues and saves it as a PNG
    to use in the question answering

    Looking at the spectrum is often the easiest way to see whether the data
    has a few dominant directions of variance (a sharp elbow in the plot) or
    whether the variance is spread more evenly across many directions.
    """
    k = min(top_k, len(eigenvals))
    x = np.arange(1, k + 1)

    plt.figure()
    plt.plot(x, eigenvals[:k])
    plt.xlabel("Eigenvalue rank (largest first)")
    plt.ylabel("Eigenvalue magnitude")
    plt.title(f"{title} — eigen spectrum (top {k})")

    fname = re.sub(r"[^a-zA-Z0-9_]+", "_", title) + "_spectrum.png"
    out_path = output_folder / fname

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


# MAIN SCRIPT 
summary_records = []

for dataset_name in EXPECTED_DATASETS:
    dataset_path = DATA_FOLDER / dataset_name

    if not dataset_path.exists():
        print(f"[WARNING] File not found: {dataset_path}")
        continue

    X = extract_matrix_from_npz(dataset_path)
    cov = compute_covariance_matrix(X)
    eigvals, det_val, prod_val, consistent = analyze_eigenvalues(cov)
    spectrum_file = save_eigen_spectrum_plot(
        eigvals, dataset_name, RESULTS_FOLDER, top_k=MAX_EIGENVALUES_TO_SHOW
    )

    summary_records.append(
        {
            "dataset_name": dataset_name,
            "num_samples": int(X.shape[0]),
            "num_features": int(X.shape[1]),
            "determinant": det_val,
            "product_of_eigenvalues": prod_val,
            "product_matches_det": consistent,
            "smallest_eigenvalue": float(eigvals.min()),
            "largest_eigenvalue": float(eigvals.max()),
            "eigenvalues_plotted": int(min(MAX_EIGENVALUES_TO_SHOW, len(eigvals))),
            "spectrum_plot_path": spectrum_file,
        }
    )

summary_df = pd.DataFrame(
    summary_records,
    columns=[
        "dataset_name",
        "num_samples",
        "num_features",
        "determinant",
        "product_of_eigenvalues",
        "product_matches_det",
        "smallest_eigenvalue",
        "largest_eigenvalue",
        "eigenvalues_plotted",
        "spectrum_plot_path",
    ],
)

csv_output = RESULTS_FOLDER / "covariance_eigenspectra_summary.csv"
summary_df.to_csv(csv_output, index=False)

print(summary_df.to_string(index=False))
print(f"\nCSV saved to: {csv_output}")
print(f"Plots saved in: {RESULTS_FOLDER}")

"""Post-processing figures for the 1D time-dependent Schrödinger PINN.

Expected objects from the training script
-----------------------------------------
model              : trained DeepXDE model
losshistory         : DeepXDE LossHistory returned by model.train(...)
schrodinger_pde     : PDE operator returning [f_u, f_v]
initial_u, initial_v: initial-condition functions used by DeepXDE

The script creates five publication-ready figures:
1. Training and grouped loss convergence
2. FDM reference, PINN prediction, and absolute density error
3. FDM/PINN density comparison at selected times
4. PDE residual heatmaps
5. Conservation of total probability
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

Array = np.ndarray


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, 1.0
NX = 200
NT = 101

# DeepXDE stores the losses in this order when the constraints are supplied as:
# [bc_u, bc_v, ic_u, ic_v, observe_u, observe_v]
# The first two columns are the two PDE residual losses.
DEFAULT_LOSS_GROUPS: Mapping[str, Tuple[int, ...]] = {
    "PDE loss": (0, 1),
    "Boundary loss": (2, 3),
    "Initial loss": (4, 5),
    "Data loss": (6, 7),
}


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _integrate(values: Array, x: Array, axis: int = -1) -> Array:
    """Use numpy.trapezoid when available, otherwise fall back to numpy.trapz."""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return trapezoid(values, x=x, axis=axis)


# -----------------------------------------------------------------------------
# Reference solution and common evaluation grid
# -----------------------------------------------------------------------------
def evaluate_initial_wavefunction(
    x: Array,
    initial_u: Callable[[Array], Array],
    initial_v: Optional[Callable[[Array], Array]],
) -> Array:
    """Evaluate the same initial condition used by the PINN at t=0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    points_t0 = np.column_stack((x, np.zeros_like(x)))

    u0 = np.asarray(initial_u(points_t0), dtype=float).reshape(-1)
    if initial_v is None:
        v0 = np.zeros_like(u0)
    else:
        v0 = np.asarray(initial_v(points_t0), dtype=float).reshape(-1)

    if u0.size != x.size or v0.size != x.size:
        raise ValueError(
            "initial_u and initial_v must return one value for every input point."
        )

    return u0 + 1j * v0


def compute_fdm_reference(
    x: Array,
    t: Array,
    initial_u: Callable[[Array], Array],
    initial_v: Optional[Callable[[Array], Array]],
    enforce_dirichlet: bool = True,
) -> Array:
    """Compute the finite-difference reference solution.

    The Hamiltonian is
        H = -1/2 D_xx + 1/2 x^2.

    When enforce_dirichlet=True, only the interior grid points are evolved and
    psi(-5,t)=psi(5,t)=0 is imposed exactly.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    if x.size < 3:
        raise ValueError("At least three spatial grid points are required.")
    if t.size < 2:
        raise ValueError("At least two time points are required.")
    if not np.allclose(np.diff(x), x[1] - x[0]):
        raise ValueError("The FDM reference solver requires a uniform x-grid.")
    if not np.allclose(np.diff(t), t[1] - t[0]):
        raise ValueError("The expm_multiply implementation requires a uniform t-grid.")

    dx = x[1] - x[0]
    psi0_full = evaluate_initial_wavefunction(x, initial_u, initial_v)

    if enforce_dirichlet:
        x_operator = x[1:-1]
        psi0 = psi0_full[1:-1]
    else:
        x_operator = x
        psi0 = psi0_full

    n = x_operator.size
    main_diagonal = -2.0 * np.ones(n)
    off_diagonal = np.ones(n - 1)
    laplacian = diags(
        [off_diagonal, main_diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        format="csr",
    ) / dx**2

    potential = diags(0.5 * x_operator**2, offsets=0, format="csr")
    hamiltonian = -0.5 * laplacian + potential

    # exp(-i H t) psi(0) for every time in the uniform time grid.
    evolved = expm_multiply(
        -1j * hamiltonian,
        psi0,
        start=float(t[0]),
        stop=float(t[-1]),
        num=t.size,
        endpoint=True,
    )

    if enforce_dirichlet:
        psi_reference = np.zeros((t.size, x.size), dtype=complex)
        psi_reference[:, 1:-1] = evolved
        return psi_reference

    return np.asarray(evolved)


def evaluate_solutions(
    model,
    initial_u: Callable[[Array], Array],
    initial_v: Optional[Callable[[Array], Array]],
    x_min: float = X_MIN,
    x_max: float = X_MAX,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    nx: int = NX,
    nt: int = NT,
    enforce_dirichlet: bool = True,
) -> Dict[str, Array]:
    """Evaluate PINN and FDM solutions on the same rectangular grid."""
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)
    x_mesh, t_mesh = np.meshgrid(x, t)
    evaluation_points = np.column_stack((x_mesh.ravel(), t_mesh.ravel()))

    prediction = np.asarray(model.predict(evaluation_points))
    if prediction.ndim != 2 or prediction.shape[1] < 2:
        raise ValueError("model.predict must return at least two columns: u and v.")

    u_pinn = prediction[:, 0].reshape(nt, nx)
    v_pinn = prediction[:, 1].reshape(nt, nx)
    psi_pinn = u_pinn + 1j * v_pinn

    psi_fdm = compute_fdm_reference(
        x=x,
        t=t,
        initial_u=initial_u,
        initial_v=initial_v,
        enforce_dirichlet=enforce_dirichlet,
    )

    rho_pinn = np.abs(psi_pinn) ** 2
    rho_fdm = np.abs(psi_fdm) ** 2
    rho_absolute_error = np.abs(rho_pinn - rho_fdm)

    return {
        "x": x,
        "t": t,
        "x_mesh": x_mesh,
        "t_mesh": t_mesh,
        "points": evaluation_points,
        "u_pinn": u_pinn,
        "v_pinn": v_pinn,
        "psi_pinn": psi_pinn,
        "psi_fdm": psi_fdm,
        "rho_pinn": rho_pinn,
        "rho_fdm": rho_fdm,
        "rho_absolute_error": rho_absolute_error,
    }


# -----------------------------------------------------------------------------
# Figure 1: total and component loss histories
# -----------------------------------------------------------------------------
def plot_loss_convergence(
    losshistory,
    output_path: Path,
    adam_end_step: Optional[int] = None,
    loss_groups: Mapping[str, Sequence[int]] = DEFAULT_LOSS_GROUPS,
) -> None:
    """Plot total train/test loss and grouped PINN loss components."""
    steps = np.asarray(losshistory.steps, dtype=float)
    train_losses = np.asarray(losshistory.loss_train, dtype=float)
    test_losses = np.asarray(losshistory.loss_test, dtype=float)

    if train_losses.ndim != 2:
        raise ValueError("losshistory.loss_train must be a two-dimensional array.")

    total_train = np.sum(train_losses, axis=1)
    total_test = np.sum(test_losses, axis=1)

    tiny = np.finfo(float).tiny
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.semilogy(steps, np.maximum(total_train, tiny), label="Total training loss")
    ax.semilogy(steps, np.maximum(total_test, tiny), linestyle="--", label="Total test loss")

    for group_name, indices in loss_groups.items():
        indices = tuple(indices)
        if max(indices) >= train_losses.shape[1]:
            raise ValueError(
                f"{group_name} uses column {max(indices)}, but only "
                f"{train_losses.shape[1]} loss columns are available."
            )
        grouped_loss = np.sum(train_losses[:, indices], axis=1)
        ax.semilogy(steps, np.maximum(grouped_loss, tiny), label=group_name)

    if adam_end_step is not None:
        ax.axvline(adam_end_step, linestyle=":", label="Adam to L-BFGS")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean-squared loss")
    ax.set_title("Training convergence and loss components")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)


# -----------------------------------------------------------------------------
# Figure 2: FDM, PINN, and density-error heatmaps
# -----------------------------------------------------------------------------
def plot_density_heatmaps(data: Mapping[str, Array], output_path: Path) -> None:
    x = data["x"]
    t = data["t"]
    rho_fdm = data["rho_fdm"]
    rho_pinn = data["rho_pinn"]
    error = data["rho_absolute_error"]

    density_min = min(float(rho_fdm.min()), float(rho_pinn.min()))
    density_max = max(float(rho_fdm.max()), float(rho_pinn.max()))
    extent = [x.min(), x.max(), t.min(), t.max()]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)

    image_0 = axes[0].imshow(
        rho_fdm,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=density_min,
        vmax=density_max,
    )
    axes[0].set_title(r"FDM reference $|\psi|^2$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")
    fig.colorbar(image_0, ax=axes[0])

    image_1 = axes[1].imshow(
        rho_pinn,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=density_min,
        vmax=density_max,
    )
    axes[1].set_title(r"PINN prediction $|\psi|^2$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$t$")
    fig.colorbar(image_1, ax=axes[1])

    image_2 = axes[2].imshow(
        error,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axes[2].set_title("Absolute density error")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$t$")
    fig.colorbar(image_2, ax=axes[2])

    _save_figure(fig, output_path)


# -----------------------------------------------------------------------------
# Figure 3: PINN versus FDM at selected times
# -----------------------------------------------------------------------------
def plot_selected_time_comparisons(
    data: Mapping[str, Array],
    output_path: Path,
    selected_times: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
) -> None:
    x = data["x"]
    t = data["t"]
    rho_fdm = data["rho_fdm"]
    rho_pinn = data["rho_pinn"]

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for axis, requested_time in zip(axes, selected_times):
        time_index = int(np.argmin(np.abs(t - requested_time)))
        actual_time = t[time_index]

        axis.plot(x, rho_fdm[time_index], label="FDM reference")
        axis.plot(x, rho_pinn[time_index], linestyle="--", label="PINN")
        axis.set_title(fr"$t={actual_time:.2f}$")
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$|\psi(x,t)|^2$")
        axis.grid(True, alpha=0.3)

    for unused_axis in axes[len(selected_times):]:
        unused_axis.axis("off")

    axes[0].legend()
    fig.suptitle("PINN and FDM probability-density comparison")
    fig.tight_layout()
    _save_figure(fig, output_path)


# -----------------------------------------------------------------------------
# Figure 4: PDE-residual heatmaps
# -----------------------------------------------------------------------------
def _split_two_residuals(residual_output, number_of_points: int) -> Tuple[Array, Array]:
    """Handle the common DeepXDE output formats for a two-equation operator."""
    if isinstance(residual_output, (list, tuple)):
        if len(residual_output) != 2:
            raise ValueError("The PDE operator must return exactly [f_u, f_v].")
        residual_u = np.asarray(residual_output[0]).reshape(-1)
        residual_v = np.asarray(residual_output[1]).reshape(-1)
    else:
        residual_array = np.asarray(residual_output)
        residual_array = np.squeeze(residual_array)

        if residual_array.ndim != 2:
            raise ValueError(
                "Could not interpret the residual output. Expected an (N, 2) array "
                "or a list [f_u, f_v]."
            )

        if residual_array.shape == (number_of_points, 2):
            residual_u = residual_array[:, 0]
            residual_v = residual_array[:, 1]
        elif residual_array.shape == (2, number_of_points):
            residual_u = residual_array[0]
            residual_v = residual_array[1]
        else:
            raise ValueError(
                f"Unexpected residual shape {residual_array.shape}; expected "
                f"({number_of_points}, 2)."
            )

    if residual_u.size != number_of_points or residual_v.size != number_of_points:
        raise ValueError("Residual output size does not match the evaluation grid.")

    return residual_u, residual_v


def _log_norm_for(values: Array) -> LogNorm:
    positive_values = np.asarray(values)[np.asarray(values) > 0]
    if positive_values.size == 0:
        return LogNorm(vmin=1e-16, vmax=1e-15)

    vmin = max(float(np.percentile(positive_values, 1.0)), 1e-16)
    vmax = max(float(np.percentile(positive_values, 99.5)), 10.0 * vmin)
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_pde_residual_heatmaps(
    model,
    schrodinger_pde: Callable,
    data: Mapping[str, Array],
    output_path: Path,
) -> None:
    points = data["points"]
    x = data["x"]
    t = data["t"]
    nt = t.size
    nx = x.size

    residual_output = model.predict(points, operator=schrodinger_pde)
    residual_u, residual_v = _split_two_residuals(residual_output, points.shape[0])

    absolute_residual_u = np.abs(residual_u.reshape(nt, nx))
    absolute_residual_v = np.abs(residual_v.reshape(nt, nx))
    extent = [x.min(), x.max(), t.min(), t.max()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    image_u = axes[0].imshow(
        np.maximum(absolute_residual_u, 1e-16),
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=_log_norm_for(absolute_residual_u),
    )
    axes[0].set_title(r"Real residual $|f_u(x,t)|$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")
    fig.colorbar(image_u, ax=axes[0])

    image_v = axes[1].imshow(
        np.maximum(absolute_residual_v, 1e-16),
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=_log_norm_for(absolute_residual_v),
    )
    axes[1].set_title(r"Imaginary residual $|f_v(x,t)|$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$t$")
    fig.colorbar(image_v, ax=axes[1])

    _save_figure(fig, output_path)


# -----------------------------------------------------------------------------
# Figure 5: conservation of total probability
# -----------------------------------------------------------------------------
def plot_probability_conservation(
    data: Mapping[str, Array],
    output_path: Path,
) -> Dict[str, float]:
    x = data["x"]
    t = data["t"]
    rho_pinn = data["rho_pinn"]
    rho_fdm = data["rho_fdm"]

    probability_pinn = _integrate(rho_pinn, x=x, axis=1)
    probability_fdm = _integrate(rho_fdm, x=x, axis=1)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(t, probability_fdm, label="FDM reference")
    ax.plot(t, probability_pinn, linestyle="--", label="PINN")
    ax.axhline(probability_fdm[0], linestyle=":", label=r"Initial probability $P(0)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P(t)=\int |\psi(x,t)|^2\,dx$")
    ax.set_title("Conservation of total probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)

    metrics = {
        "maximum_PINN_probability_drift": float(
            np.max(np.abs(probability_pinn - probability_pinn[0]))
        ),
        "maximum_FDM_probability_drift": float(
            np.max(np.abs(probability_fdm - probability_fdm[0]))
        ),
        "maximum_PINN_FDM_probability_difference": float(
            np.max(np.abs(probability_pinn - probability_fdm))
        ),
    }

    print("Probability-conservation metrics")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6e}")

    return metrics


# -----------------------------------------------------------------------------
# Quantitative errors and one-call wrapper
# -----------------------------------------------------------------------------
def print_global_errors(data: Mapping[str, Array]) -> Dict[str, float]:
    psi_pinn = data["psi_pinn"]
    psi_fdm = data["psi_fdm"]
    rho_pinn = data["rho_pinn"]
    rho_fdm = data["rho_fdm"]

    wavefunction_denominator = np.linalg.norm(psi_fdm.ravel())
    density_denominator = np.linalg.norm(rho_fdm.ravel())

    relative_wavefunction_l2 = (
        np.linalg.norm((psi_pinn - psi_fdm).ravel()) / wavefunction_denominator
    )
    relative_density_l2 = (
        np.linalg.norm((rho_pinn - rho_fdm).ravel()) / density_denominator
    )
    maximum_density_error = np.max(np.abs(rho_pinn - rho_fdm))

    metrics = {
        "relative_wavefunction_L2_error": float(relative_wavefunction_l2),
        "relative_density_L2_error": float(relative_density_l2),
        "maximum_absolute_density_error": float(maximum_density_error),
    }

    print("Global validation metrics")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6e}")

    return metrics


def make_all_five_figures(
    model,
    losshistory,
    schrodinger_pde: Callable,
    initial_u: Callable[[Array], Array],
    initial_v: Optional[Callable[[Array], Array]],
    output_directory: str = "figures",
    adam_end_step: Optional[int] = 1000,
    loss_groups: Mapping[str, Sequence[int]] = DEFAULT_LOSS_GROUPS,
    selected_times: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    nx: int = NX,
    nt: int = NT,
    enforce_dirichlet: bool = True,
) -> Dict[str, Array]:
    """Create all five figures after the PINN has finished training."""
    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    data = evaluate_solutions(
        model=model,
        initial_u=initial_u,
        initial_v=initial_v,
        nx=nx,
        nt=nt,
        enforce_dirichlet=enforce_dirichlet,
    )

    plot_loss_convergence(
        losshistory=losshistory,
        output_path=output_directory_path / "figure_1_loss_convergence.png",
        adam_end_step=adam_end_step,
        loss_groups=loss_groups,
    )

    plot_density_heatmaps(
        data=data,
        output_path=output_directory_path / "figure_2_density_comparison.png",
    )

    plot_selected_time_comparisons(
        data=data,
        output_path=output_directory_path / "figure_3_selected_times.png",
        selected_times=selected_times,
    )

    plot_pde_residual_heatmaps(
        model=model,
        schrodinger_pde=schrodinger_pde,
        data=data,
        output_path=output_directory_path / "figure_4_pde_residuals.png",
    )

    plot_probability_conservation(
        data=data,
        output_path=output_directory_path / "figure_5_probability_conservation.png",
    )

    print_global_errors(data)
    return data


# -----------------------------------------------------------------------------
# Add this call at the end of your training script:
# -----------------------------------------------------------------------------
# results = make_all_five_figures(
#     model=model,
#     losshistory=losshistory,
#     schrodinger_pde=schrodinger_pde,
#     initial_u=initial_u,
#     initial_v=initial_v,
#     output_directory="figures",
#     adam_end_step=1000,
# )

import numpy as np
import sys
from deimos.utils.plotting import plt, get_intermediate_points

def plot_colormap(ax, x, y, z, zlabel=None, **kw):
    x = get_intermediate_points(x, bounding_points=True)
    y = get_intermediate_points(y, bounding_points=True)
    cmesh = ax.pcolormesh(x, y, z.T, **kw)
    if not any([x in kw for x in ["edgecolor", "edgecolors"]]):
        cmesh.set_edgecolor("face")
    return cmesh

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_sidereal_data.py file1.npz file2.npz")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    data1 = np.load(file1)
    data2 = np.load(file2)

    # Use the same order as your production script
    field_directions = ["x", "y", "z"]
    detectors = ["ICECUBE", "ARCA", "P_ONE", "TRIDENT", "HUNT", "GVD"]

    # Check keys and shapes
    for fd in field_directions:
        for det in detectors:
            key = f"('{fd}', '{det}')"
            if key not in data1 or key not in data2:
                print(f"Missing key: {key}")
                sys.exit(1)
            if data1[key].shape != data2[key].shape:
                print(f"Shape mismatch for {key}: {data1[key].shape} vs {data2[key].shape}")
                sys.exit(1)

    # Get grid from one array
    arr_shape = data1[f"('{field_directions[0]}', '{detectors[0]}')"].shape
    n_ra = arr_shape[0]
    n_dec = arr_shape[1]
    n_flav = arr_shape[2] if len(arr_shape) > 2 else 1
    ra_values_deg = np.linspace(0.0, 360.0, num=n_ra)
    dec_values_deg = np.linspace(-90.0, 90.0, num=n_dec)

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(len(field_directions), len(detectors), figsize=(7*len(detectors), 5*len(field_directions)), sharex=True, sharey=True)
    fig.suptitle(f"Residuals: {file2} - {file1}")

    vmin, vmax = None, None
    # Compute global vmin/vmax for colorbar
    all_residuals = []
    for i, fd in enumerate(field_directions):
        for j, det in enumerate(detectors):
            key = f"('{fd}', '{det}')"
            arr1 = data1[key]
            arr2 = data2[key]
            residual = arr2 - arr1
            if residual.ndim == 3:
                residual = residual[..., 1]
            all_residuals.append(residual)
    all_residuals = np.array(all_residuals)
    vmax = np.nanmax(np.abs(all_residuals))
    vmin = -vmax

    for i, fd in enumerate(field_directions):
        for j, det in enumerate(detectors):
            key = f"('{fd}', '{det}')"
            arr1 = data1[key]
            arr2 = data2[key]
            residual = arr2 - arr1
            if residual.ndim == 3:
                residual = residual[..., 1]
            zlabel = f"Residual $P_{{{det}}}$ ({fd})"
            cmesh = plot_colormap(ax=ax[i, j], x=ra_values_deg, y=dec_values_deg, z=residual, zlabel=zlabel, cmap="bwr", vmin=vmin, vmax=vmax)
            ax[i, j].set_title(f"{det} ({fd})")
            ax[i, j].set_xlabel("RA [deg]")
            ax[i, j].set_ylabel("DEC [deg]")

    fig.subplots_adjust(right=0.88, wspace=0.07, hspace=0.07)
    cbar_ax = fig.add_axes([0.90, 0.107, 0.025, 0.775])
    cbar = fig.colorbar(ax[0,0].collections[0], cax=cbar_ax, orientation="vertical")
    cbar.set_label("Residual $P$")

    plt.savefig("sidereal_residuals.png")
    print("Residual plot saved as sidereal_residuals.png")
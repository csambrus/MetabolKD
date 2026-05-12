from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import PLOT_DPI, ensure_dir

try:
    from IPython.display import Image as IPyImage, display
except Exception:
    IPyImage = None
    display = None


def display_png_if_available(path: str | Path) -> None:
    """Notebookban megjeleniti a kepet, scriptben csendben tovabblep."""
    if display is None or IPyImage is None:
        return
    path = Path(path)
    if path.exists():
        display(IPyImage(filename=str(path)))


def save_show_close_figure(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Egységes figura kezeles:
    - opcionális mentes
    - opcionális megjelenites
    - eroforras-felszabaditas (close)
    """
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

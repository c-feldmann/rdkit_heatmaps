import io
import numpy as np
from PIL import Image
from rdkit import Chem
from typing import *


def get_mol_lims(mol: Chem.Mol) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Returns the extend of the molecule.

    x- and y-coordinates of all atoms in the molecule are accessed, returning min- and max-values for both axes.

    Parameters
    ----------
    mol

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
    """
    coords = []
    conf = mol.GetConformer(0)
    for i, _ in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append((pos.x, pos.y))
    coords = np.array(coords)
    min_p = np.min(coords, axis=0)
    max_p = np.max(coords, axis=0)
    x_lim = min_p[0], max_p[0]
    y_lim = min_p[1], max_p[1]
    return x_lim, y_lim


def pad(lim: Union[Sequence[float], np.ndarray], ratio: float):
    """Takes a 2 dimensional vector and adds len(vector) * ratio / 2 to each side and returns obtained vector.



    Parameters
    ----------
    lim: Sequence[float, float]

    ratio

    Returns
    -------

    """
    diff = max(lim) - min(lim)
    diff *= ratio / 2
    return [lim[0] - diff, lim[1] + diff]


def transform2png(data) -> Image:
    """Transforms bytes from RDKit MolDraw2DCairo to a png-image"""
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img

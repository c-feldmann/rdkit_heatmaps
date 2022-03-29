from matplotlib.colors import Colormap
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit_heatmaps import utils
from rdkit_heatmaps.heatmaps import ValueGrid
from rdkit_heatmaps.heatmaps import color_canvas
from rdkit_heatmaps.functions import GaussFunction2D
from typing import *


def mapvalues2mol(mol: Chem.Mol,
                  atom_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                  bond_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                  atom_width: float = 0.3,
                  bond_width: float = 0.25,
                  bond_length: float = 0.5,
                  canvas: Optional[rdMolDraw2D.MolDraw2D] = None,
                  grid_resolution=None,
                  value_lims: Optional[Sequence[float]] = None,
                  color: Union[str, Colormap] = "bwr",
                  padding: Optional[Sequence[float]] = None) -> rdMolDraw2D:

    if grid_resolution is None:
        grid_resolution = [1000, 500]
    if atom_weights is not None:
        if not len(atom_weights) == len(mol.GetAtoms()):
            raise ValueError("len(atom_weights) is not equal to number of bonds in mol")
    if bond_weights is not None:
        if not len(bond_weights) == len(mol.GetBonds()):
            raise ValueError("len(bond_weights) is not equal to number of bonds in mol")

    if padding is None:
        padding = [1, 1]

    if not canvas:
        canvas = rdMolDraw2D.MolDraw2DCairo(800, 450)
        draw_opt = canvas.drawOptions()
        draw_opt.padding = 0.2
        draw_opt.bondLineWidth = 3
        canvas.SetDrawOptions(draw_opt)

    xl, yl = utils.get_mol_lims(mol)
    xl = utils.pad(xl, padding[0])
    yl = utils.pad(yl, padding[1])
    v_map = ValueGrid(xl, yl, grid_resolution[0], grid_resolution[1])
    conf = mol.GetConformer(0)

    for i, _ in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords = pos.x, pos.y
        f = GaussFunction2D(center=coords, std1=atom_width, std2=atom_width, scale=atom_weights[i], rotation=0)
        v_map.add_function(f)

    for i, b in enumerate(mol.GetBonds()):  # type: Chem.Bond
        a1 = b.GetBeginAtom().GetIdx()
        a2 = b.GetEndAtom().GetIdx()
        a1_pos = conf.GetAtomPosition(a1)
        a1_coords = np.array([a1_pos.x, a1_pos.y])
        a2_pos = conf.GetAtomPosition(a2)
        a2_coords = np.array([a2_pos.x, a2_pos.y])

        diff = a2_coords - a1_coords
        angle = np.arctan2(diff[0], diff[1])
        bond_center = (a1_coords + a2_coords) / 2

        f = GaussFunction2D(center=bond_center, std1=bond_width, std2=bond_length, scale=bond_weights[i],
                            rotation=angle)
        v_map.add_function(f)

    v_map.recalculate()
    if not value_lims:
        abs_max = np.max(np.abs(v_map.values))
        value_lims = [-abs_max, abs_max]


    c_grid = v_map.map2color(color, v_lim=value_lims)
    canvas.DrawMolecule(mol)
    color_canvas(canvas, c_grid)
    canvas.DrawMolecule(mol)
    return canvas

from typing import *
import numpy as np
from rdkit.Chem import Draw
from rdkit.Geometry.rdGeometry import Point2D
import abc
import matplotlib.colors as colors
from matplotlib import cm
from rdkit_heatmaps.functions import Function2D


class Grid2D(abc.ABC):
    def __init__(self, x_lim: Tuple[float, float], y_lim: Tuple[float, float], x_res: int, y_res: int):
        self.function_list = []
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_res = x_res
        self.y_res = y_res
        self.values = np.zeros((self.x_res, self.y_res))

    @property
    def dx(self):
        return (max(self.x_lim) - min(self.x_lim)) / self.x_res

    @property
    def dy(self):
        return (max(self.y_lim) - min(self.y_lim)) / self.y_res

    def grid_field_center(self, x_idx: int, y_idx: int) -> Tuple[float, float]:
        x_coord = min(self.x_lim) + self.dx * (x_idx + 0.5)
        y_coord = min(self.y_lim) + self.dy * (y_idx + 0.5)
        return x_coord, y_coord

    def grid_field_lim(self, x_idx: int, y_idx: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        upper_left = (min(self.x_lim) + self.dx * x_idx, min(self.y_lim) + self.dy * y_idx)
        lower_right = (min(self.x_lim) + self.dx * (x_idx + 1), min(self.y_lim) + self.dy * (y_idx + 1))
        return upper_left, lower_right


class ValueGrid(Grid2D):
    def __init__(self, x_lim: Tuple[float, float], y_lim: Tuple[float, float], x_res: int, y_res: int, ):
        super().__init__(x_lim, y_lim, x_res, y_res)
        self.function_list: List[Function2D] = []
        self.values = np.zeros((self.x_res, self.y_res))

    def add_function(self, function: Function2D):
        self.function_list.append(function)

    def recalculate(self):
        self.values = np.zeros((self.x_res, self.y_res))
        x_y0_list = np.array([self.grid_field_center(x, 0)[0] for x in range(self.x_res)])
        x0_y_list = np.array([self.grid_field_center(0, y)[1] for y in range(self.y_res)])
        xv, yv = np.meshgrid(x_y0_list, x0_y_list)
        xv = xv.ravel()
        yv = yv.ravel()
        coordinate_pairs = np.vstack([xv, yv]).T
        for f in self.function_list:
            values = f(coordinate_pairs)
            values = values.reshape(self.y_res, self.x_res).T
            assert values.shape == self.values.shape, (values.shape, self.values.shape)
            self.values += values

    def map2color(self, c_map: Union[colors.Colormap, str], v_lim=None):
        color_grid = ColorGrid(self.x_lim, self.y_lim, self.x_res, self.y_res)
        if not v_lim:
            v_lim = np.min(self.values), np.max(self.values)
        normalizer = colors.Normalize(vmin=v_lim[0], vmax=v_lim[1])
        if isinstance(c_map, str):
            c_map = cm.get_cmap(c_map)
        norm = normalizer(self.values)
        color_grid.color_grid = np.array(c_map(norm))
        return color_grid


class ColorGrid(Grid2D):
    def __init__(self, x_lim: Tuple[float, float], y_lim: Tuple[float, float], x_res: int, y_res:int):
        super().__init__(x_lim, y_lim, x_res, y_res)
        self.color_grid = np.ones((self.x_res, self.y_res, 4))


def color_canvas(canvas: Draw.MolDraw2D, colorgrid: ColorGrid):
    for x in range(colorgrid.x_res):
        for y in range(colorgrid.y_res):
            upper_left, lower_right = colorgrid.grid_field_lim(x, y)
            upper_left, lower_right = Point2D(*upper_left), Point2D(*lower_right)
            canvas.SetColour(tuple(colorgrid.color_grid[x, y]))
            canvas.DrawRect(upper_left, lower_right)

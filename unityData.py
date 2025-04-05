from __future__ import annotations


import xml.etree.ElementTree as ET
import pyvista as pv
import numpy as np
from enum import Enum
from dataclasses import dataclass
import copy
import os
import eyeDataViz as eyeViz
from scipy.spatial import cKDTree
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .classifiers import FixationClassifier

@dataclass
class FixationGroup:
    idx: int
    centroid: np.ndarray[np.float64]
    spread: np.ndarray[np.float64]
    time_in: float
    time_out: float
    num_points: int

@dataclass
class SaccadeGroup:
    idx: int
    start_coord: np.ndarray[np.float64]
    end_coord: np.ndarray[np.float64]
    avg_velocity: float
    time_in: float
    time_out: float
    smooth_pursuit: bool

@dataclass
class picture:
    idx: int
    time: float
    origin: np.ndarray[np.float64]
    direction: np.ndarray[np.float64]
    path: os.PathLike


def global_coords_to_face_coords(global_X, face_vertices):
    """
    @param global_X: np.array[np.float64[3,1]] a coordinate in the global inertial frame
    @param face_vertices: np.array[np.float64[3,3]] the global coordinates of the face to project onto.
            face_vertices[n-1] = global coordinate for vertex n.

    the face coordinates are defined in the following way: 
        - The origin is defined at vertex 1. 
        - The x axis is defined from the origin and pointing to vertex 2.
        - The z axis is defined from the origin and pointing normal to the face plane.
        - The y axis is defineed from the origin and normal to the xz plane.
    """
    O = face_vertices[0]
    v2_v1 = face_vertices[1] - face_vertices[0]
    v3_v1 = face_vertices[2] - face_vertices[0]
    x_axis = (v2_v1)/np.linalg.norm(v2_v1)
    z_axis = np.cross(v2_v1, v3_v1)/np.linalg.norm(np.cross(v2_v1, v3_v1))
    y_axis = np.cross(z_axis, x_axis)/np.linalg.norm(np.cross(x_axis, z_axis))
    transform_matrix = np.array([[*x_axis, O[0]],
                                 [*y_axis, O[1]],
                                 [*z_axis, O[2]]], np.float64)
    return transform_matrix @ global_X


class UnityData:
    def __init__(self, xml_path, obj_path = None, obj_transform = None, texture_path = None) -> None:
        self.tree = ET.parse(xml_path)
        self.X_3d = [] #gaze position in 3D environment
        self.X_2d = [] #gaze position on display area
        self.cam_D = [] #camera viewing direction
        self.cam_X = [] #camera position in 3D evironment
        self.triangle_idx = [] #Triangle on obj being viewed by gaze
        self.pupil = [] #pupil dimameter in mm
        self.t = [] #time in seconds
        self.gaze_origin = [] #gaze ray origin in 3D Unity env
        self.gaze_direction = [] #gaze ray direction vector in 3D Unity env
        self.display_dimensions = [int(self.tree.find("./GazeData/DisplayDimensions").get(val)) 
                            for val in ["Width", "Height"]]
        self.start_time = int(self.tree.find("./GazeData/Timestamp").text)
        
        #Parse data from XML file
        self.X_3d = np.array(list(map(self._get_x_y_z, self.tree.findall("GazeData/IntersectionPoint"))), np.float64)
        self.X_2d = np.array(list(map(self._get_x_y, self.tree.findall("GazeData/PositionOnDisplayArea"))), np.float64)
        self.pupil = np.array(list(map(lambda _e: self._convert_to_float(_e.get("average_pupildiameter")), self.tree.findall("GazeData/pupil"))), np.float64)
        self.gaze_origin = np.array(list(map(lambda _e: _e.get("Origin")[1:-1].split(", "), self.tree.findall("GazeData/GazeOrigin/CombinedGazeRayScreen"))), np.float64)
        self.gaze_direction = np.array(list(map(lambda _e: _e.get("Direction")[1:-1].split(", "), self.tree.findall("GazeData/GazeOrigin/CombinedGazeRayScreen"))), np.float64)
        self.cam_D = np.array(list(map(self._get_x_y_z, self.tree.findall("CameraData/CameraDirection"))), np.float64)
        self.cam_X = np.array(list(map(self._get_x_y_z, self.tree.findall("CameraData/CameraOrigin"))), np.float64)
        self.t = np.array(list(map(lambda _t: (int(_t.text) - self.start_time)/1e6, self.tree.findall("GazeData/Timestamp"))), np.float64)
        
        #Initialize data attributes
        self.gaze_type = None
        self.pictures = None
        self.mesh = None
        self.surface_heatmap = None
        self.volumetric_heatmap = None
        self.tex = None
        
        #Load obj and texture if provided
        if obj_path:
            obj_transform = np.eye(4) if obj_transform is None else obj_transform
            self.mesh = pv.read(obj_path)
            self.mesh.transform(obj_transform)
        if texture_path:
            self.tex = pv.read_texture(texture_path)
            


    def _merge_gaze_groups(self):
        """
        self.gaze_types must be defined for this function to run
        it will combine the gazes into groups by collapsing
        consecutive gazes of the same type into gaze groups
        such as fixation groups or saccade groups.
        """
        assert self.gaze_type is not None, "classify points into gaze type first"
        self.groups = []
        self.saccade_groups = []
        self.fixation_groups = []
        self.smooth_pursuit_groups = []
        w_start = 0
        w_end = 1
        while w_end < len(self.gaze_type):
            group_type = self.gaze_type[w_start]
            while self.gaze_type[w_end] == group_type and w_end < len(self.gaze_type)-1:
                w_end += 1
            if (group_type == eyeViz.GazeTypes.SACCADE 
                or group_type == eyeViz.GazeTypes.SMOOTH_PURSUIT):
                group_idx = slice(max(0,w_start-1), w_end+1)
                t = self.t[group_idx]
                points = self.X_3d[group_idx]
                dt = (t - np.roll(t, 1, axis=0))[:, np.newaxis]
                vel = ((points - np.roll(points, 1, axis=0))/dt)[1:]
                avg_vel = np.average(np.linalg.norm(vel, axis = 1))
                group = SaccadeGroup(idx=len(self.groups),
                                     start_coord=points[0],
                                     end_coord=points[-1],
                                     avg_velocity=avg_vel,
                                     time_in=t[0],
                                     time_out=t[-1],
                                     smooth_pursuit=group_type==eyeViz.GazeTypes.SMOOTH_PURSUIT)
                if group_type == eyeViz.GazeTypes.SACCADE:
                    self.saccade_groups.append(group)
                else:
                    self.smooth_pursuit_groups.append(group)
            elif group_type == eyeViz.GazeTypes.FIXATION:
                group_idx = slice(w_start, w_end)
                t = self.t[group_idx]
                points = self.X_3d[group_idx]
                group = FixationGroup(idx=len(self.groups),
                                      centroid=np.average(points, axis=0),
                                      spread=np.var(points, axis=0),
                                      time_in=t[0],
                                      time_out=t[-1],
                                      num_points=len(points))
                self.fixation_groups.append(group)
            elif group_type == eyeViz.GazeTypes.nan:
                pass
            else:
                raise ValueError(f"Unrecognized gaze type {group_type}")
            self.groups.append(group)
            w_start = w_end
            w_end = w_start+1

    def plot_movement_chart_2d(self):
        """
        plots 2 charts for gaze movement in x and y directions over time
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, sharex=True)
        dt = (self.t - np.roll(self.t, 1, axis=0))[:, np.newaxis]
        vel_2d = ((self.X_2d - np.roll(self.X_2d, 1, axis=0))/dt)[1:] #discard the first point
        vel_2d = np.linalg.norm(vel_2d, axis=1)
        ax[0].scatter(self.t, self.X_2d[:, 0], s=2)
        ax[1].scatter(self.t, self.X_2d[:, 1], s=2)
        ax[2].scatter(self.t[1:], vel_2d, s=2)
        ax[0].set_ylabel("x position (pixels)")
        ax[1].set_ylabel("y position(pixels)")
        ax[2].set_ylabel("velocity (pixel/s)")
        ax[2].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()

    def plot_movement_chart_3d(self):
        """
        plots 3 charts for gaze movement in x, y and z directions over time
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, sharex=True)
        dt = (self.t - np.roll(self.t, 1, axis=0))[:, np.newaxis]
        vel_3d = ((self.X_3d - np.roll(self.X_3d, 1, axis=0))/dt)[1:] #discard the first point
        vel_3d = np.linalg.norm(vel_3d, axis=1)
        ax[0].scatter(self.t, self.X_3d[:, 0], s=2)
        ax[1].scatter(self.t, self.X_3d[:, 1], s=2)
        ax[2].scatter(self.t, self.X_3d[:, 2], s=2)
        ax[3].scatter(self.t[1:], vel_3d, s=2)
        ax[0].set_ylabel("x position (meter)")
        ax[1].set_ylabel("y position(meter)")
        ax[2].set_ylabel("z position(meter)")
        ax[3].set_ylabel("velocity (meter/s)")
        ax[3].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()

    def show_volumetric_heatmap(self, volume_spacing=0.5, texture=False):
        pcd = pv.PointSet(self.X_3d)
        pcd_min = pcd.points.min(axis=0)
        pcd_max = pcd.points.max(axis=0)
        grid_dims = ((pcd_max-pcd_min)/volume_spacing).round().astype(np.int32) + 2
        grid = pv.ImageData(
            dimensions=grid_dims, spacing=(volume_spacing, volume_spacing, volume_spacing)
            )
        grid = grid.translate(pcd_min)
        #calculate number of points in each cell
        num_points = []
        for cell in grid.cell:
            x_min, x_max, y_min, y_max, z_min, z_max = cell.bounds
            points_in_cell = np.all(
                np.logical_and(np.greater_equal(self.X_3d, [x_min, y_min, z_min]),
                        np.less_equal(self.X_3d, [x_max, y_max, z_max]))
                        , axis=1).sum()
            num_points.append(int(points_in_cell))
        grid['data'] = num_points
        pl = pv.Plotter()
        _ = pl.add_mesh(self.mesh,
                        texture=self.tex if texture else None)
        _ = pl.add_volume(grid, cmap="jet")
        pl.show()
    
    def calc_surface_heatmap(self, subdivide=0, radius=1, use_fixations=False, smooth=1):
        self.surface_heatmap = self.mesh.subdivide(subdivide, "linear")
        density = np.zeros(self.surface_heatmap.n_points)
        if use_fixations:
            assert self.fixation_groups, "If using fixations for heatmap you must first run a classfier on points"
            points = np.array([group.centroid for group in self.fixation_groups])
        else:
            points = self.X_3d
        min_points = np.min(points, axis=0) - radius
        max_points = np.max(points, axis=0) + radius
        tree = cKDTree(points)
        _tqdm = tqdm(enumerate(self.surface_heatmap.points), 
                     desc="calculating surface heatmap vertex colors...",
                     total = self.surface_heatmap.n_points)
        for i, vertex in _tqdm:
            if not np.all((vertex >= min_points) & (vertex <= max_points)):
                #vertex is out of bounds of points, ignore to speed up computation
                density[i] = 0
                continue
            indices = tree.query_ball_point(vertex, radius)
            if len(indices) == 0:
                density[i] = 0
                continue
            if smooth == 0:
                density[i] = len(indices)
                continue
            points_inside = points[indices]
            density[i] = (1 - (np.linalg.norm(points_inside-vertex, axis=1)/radius)**smooth).sum()
        self.surface_heatmap["density"] = density

    @property
    def has_surface_heatmap(self) -> bool:
        if self.surface_heatmap:
            return (self.surface_heatmap.active_scalars is not None)
        return False

    def classify_fixations(self, classifier: FixationClassifier) -> None:
        self.gaze_type = classifier(self)
        self._merge_gaze_groups()
        
    def _convert_to_float(self, string):
        """Converts a string to a float
        handles the case where the string is not a number"""
        try:
            return float(string)
        except ValueError:
            return None
        
    def _get_x_y_z(self, element):
        """Returns the x, y, z coordinates of the element 
        <given that it is an XML element with X, Y, Z attributes>"""
        return [self._convert_to_float(element.get("X")), 
                self._convert_to_float(element.get("Y")), 
                self._convert_to_float(element.get("Z"))]
    
    def _get_x_y(self, element):
        """Returns the x, y coordinates of the element 
        <given that it is an XML element with X, Y attributes>"""
        return [self._convert_to_float(element.get("X")), 
                self._convert_to_float(element.get("Y"))]

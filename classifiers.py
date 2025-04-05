"""
Classes that implement different 3D fixation classifiers
"""
from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import eyeDataViz as eyeViz
from typing import TYPE_CHECKING
import copy
if TYPE_CHECKING:
    from .unityData import UnityData

class FixationClassifier(ABC):
    def __init__(self, **args):
        pass
    @abstractmethod
    def __call__(self, data:UnityData) -> np.ndarray:
        pass

@dataclass
class Duchowski_I_VT(FixationClassifier):
    """
    I-VT in 3D implementation by Duchowski et. al. 2002.

    """
    convolution_filter: np.NDArray = field(default_factory=lambda:np.array([1,2,3,2,1]))
    velocity_threshold: float = 130 #velocity in deg/sec

    def __call__(self, data: eyeViz.UnityData) -> np.NDArray:
        sample_period = len(self.convolution_filter)
        avg_gaze_origin = np.average([data.gaze_origin[i:i+sample_period] 
                                        for i in range(len(data.gaze_origin) - sample_period)], 
                                        axis=1)
        v = data.X_3d[:-sample_period] - avg_gaze_origin
        t = data.t[:-sample_period]
        v_1 = np.roll(v, -1, axis=0)
        dt = np.roll(t, -1) - t
        theta = np.arccos(np.einsum('ij,ij->i', v, v_1)
                        /(np.linalg.norm(v,axis=1)*np.linalg.norm(v_1, axis=1)))
        theta = theta / dt
        theta = theta[:-1]
        dtheta = np.convolve(theta, self.convolution_filter, "valid")
        classes = []
        for i in range(len(data.X_3d)):
            try:
                if dtheta[i] >= self.velocity_threshold:
                    classes.append(eyeViz.GazeTypes.SACCADE)
                else:
                    classes.append(eyeViz.GazeTypes.FIXATION)
            except IndexError:
                classes.append(eyeViz.GazeTypes.nan)
        return classes

@dataclass
class I_VT_2D(FixationClassifier):
    velocity_threshold: float = 30 #deg/s
    def __call__(self, data: eyeViz.UnityData) -> np.NDArray:
        """
        threshold: max eye movement velocity value in pixel/second to classify 
            a gaze as fixation
        
        ========= I-VT algorithm ========
        (Salvucci and Goldberg, 2000)
            Calculate point-to-point velocities for each point in protocol
            label each point below velocity threshold as fixation point, otherwise saccade points
            Collapse consecutive fixation points into fixation groups, removing saccade points
            Map each fixation group to a fixation at the centroid of its points
            return fixations
        """
        last_point = np.zeros((2,))
        self.gaze_type = []
        self.fixations_2d = []
        fixation_group = []
        dt = (self.t - np.roll(self.t, 1, axis=0))[:, np.newaxis]
        for point in self.X_2d:
            vel = np.linalg.norm((point - last_point)/dt)
            last_point = point
            if vel < self.velocity_threshold:
                self.gaze_type.append(eyeViz.GazeTypes.FIXATION)
                fixation_group.append(point)
            else:
                self.gaze_type.append(eyeViz.GazeTypes.SACCADE)
                if fixation_group:
                    self.fixations_2d.append(
                        np.average(fixation_group, axis=0)
                        )
                    fixation_group = []
        return self.fixations_2d

@dataclass
class I_VT_3D(FixationClassifier):
    """Alternative to Duchowski"""
    velocity_threshold: float
    def __call__(self, data: eyeViz.UnityData) -> np.NDArray:
        """
        threshold: max eye movement velocity value in meter/second to classify
            a gaze as fixation
        
        see self.ivt_2d.__doc__ for an explanation of the ivt algorithm
        """
        last_point = np.zeros((3,))
        self.gaze_type = []
        self.fixations_3d = []
        fixation_group = []
        for i, point in enumerate(self.X_3d):
            dt = self.t[i] - self.t[i-1]
            vel = np.linalg.norm((point - last_point)/dt)
            last_point = point
            if vel < self.velocity_threshold:
                self.gaze_type.append(eyeViz.GazeTypes.FIXATION)
                fixation_group.append(point)
            else:
                self.gaze_type.append(eyeViz.GazeTypes.SACCADE)
                if fixation_group:
                    self.fixations_3d.append(
                        np.average(fixation_group, axis=0)
                    )
                    fixation_group = []
        return self.fixations_3d
    
@dataclass
class I_VDT_3D(FixationClassifier):
    """
        Parameters
        ----------
        velocity_threshold: float
            gazes with 3D velocities above this threshold will be classified as saccades
        dispersion_threshold: float
            disperssion threshold to use for IDT algorithm, groups of points with a
            disperssion less than this will be classified as fixation groups
        window_size: 
        
        ====== I-VDT algorithm ======
        (Komogortsev and Karpov, 2012)
            >> Calculate point-to-point velocities for each point
            >> Mark all points above velocity threshold as saccades
            >> filter saccades (this part is discarted in this implementation 
            ... filtering can be done by a separate function)
            >> initialize temporal window over first points in the remaining 
            ... eye movement trace
            >> While the temporal window does not reach the end of array
            ... >> Calculate dispersion of points in window
            ... >> if dispersion < dispersion_threshold
            ... ... >> while dispersion < dispersion_threshold
            ... ... ... >> add one more point to temporal window
            ... ... ... >> calculate dispersion of points in window
            ... ... >> end while
            ... ... >> Mark points inside the window as fixations
            ... ... >> Clear window
            ... >> Else
            ... ... >> Remove first point from window
            ... ... >> Mark first point as smooth pursuit
            ... >> End if
            >> End While
            >> Merge smooth pursuits, fixations, and saccades into groups
            >> return saccades, fixations, and smooth pursuits
    """
    velocity_threshold: float
    dispersion_threshold: float
    window_size: int = 10
    def __call__(self, data: eyeViz.UnityData) -> np.NDArray:
        points = copy.deepcopy(data.X_3d)
        dt = (data.t - np.roll(data.t, 1, axis=0))[:, np.newaxis]
        vel_3d = ((points - np.roll(points, 1, axis=0))/dt)[1:] #discard the first point
        vel_3d = np.linalg.norm(vel_3d, axis=1)
        classes = np.zeros(len(points), np.object_)
        classes[np.where(vel_3d>self.velocity_threshold)] = eyeViz.GazeTypes.SACCADE
        points = points[np.where(vel_3d<=self.velocity_threshold)]
        new_idx = np.where(vel_3d<=self.velocity_threshold)[0]
        w_start = 0
        w_end = self.window_size
        dispersion = lambda window: (window.max(axis=0) - window.min(axis=0)).sum()
        while w_start<=len(points):
            window = points[w_start:w_end]
            if dispersion(window) < self.dispersion_threshold:
                while dispersion(window) < self.dispersion_threshold and w_end <= len(points):
                    w_end += 1
                    window = points[w_start:w_end]
                classes[new_idx[w_start:w_end]] = eyeViz.GazeTypes.FIXATION
                w_start = w_end
                w_end = w_start + self.window_size
            else:
                classes[new_idx[w_start]] = eyeViz.GazeTypes.SMOOTH_PURSUIT
                w_start += 1
                w_end += 1        
        return classes

@dataclass
class I_VDT_2D(FixationClassifier):
    velocity_threshold: float
    dispersion_threshold: float
    def __call__(self, data: eyeViz.UnityData) -> np.NDArray:
        pass
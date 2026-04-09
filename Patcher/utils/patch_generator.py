""" 
Filename: patch_generator.py
Author: Romain Defferrard
Date: 04-06-2025

Description:
    This module defines the PatchGenerator class, which generates rectangular patch polygons
    within overlapping LiDAR flight zones. It uses the raster mesh as the Swiss coordinate grid, 
    apply PCA to find the dominant direction of the estimated overlap, and sample patches along that centerline.

    The generated patches are stored as Patch objects with geometric metadata (center, direction, etc.).
"""
import numpy as np
from typing import List, Tuple, Optional
from skimage.measure import find_contours, approximate_polygon, label as sk_label, regionprops
from shapely.geometry import Polygon, LineString
import warnings

from .patch_model import PatchParams, Patch


class PatchGenerator:
    def __init__(
        self, superpos_zones: List[np.ndarray], raster_mesh: Tuple[np.ndarray, np.ndarray],
        patch_params: Tuple[float, float, float],
        n_lines: int = 1, line_offset: float = 0.0,
    ):
        """
        Initializes the PatchGenerator class.

        Inputs:
            superpos_zones (List[np.ndarray]):           List of overlapping area masks.
            raster_mesh (Tuple[np.ndarray, np.ndarray]): (x_mesh, y_mesh) for spatial mapping.
            patch_params (Tuple[float, float, float]):   (length, width, sample_distance) of patches.
            n_lines (int):   Number of parallel centerlines. 1 = only the main centerline (default).
                             Even values produce symmetric lines about the main axis; odd values
                             include the main centerline itself.
            line_offset (float): Perpendicular distance [m] between adjacent parallel centerlines.

        Output:
            None
        """

        self.superpos_zones_all = superpos_zones
        self.x_mesh, self.y_mesh = raster_mesh
        self.band_length, self.band_width, self.sample_distance = patch_params
        self.n_lines = max(1, int(n_lines))
        self.line_offset = float(line_offset)

        self.tol = 0.05  # tolerance parameter for the contour generation
        self.contour = None
        self.patch_id = 1
        self.res_x = float(abs(self.x_mesh[0, 1] - self.x_mesh[0, 0]))
        self.res_y = float(abs(self.y_mesh[1, 0] - self.y_mesh[0, 0]))

        warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely.predicates")

        # Output
        self.patches_list = []
        self.centerlines_list = []
        self.contours_list = []
        self.max_patch_len = []
        self.patches_poly_list = []
        # Parallel shifted centerlines for each zone: List[List[np.ndarray]]
        self.parallel_centerlines_list = []

        # Process all superposition zones
        self.process_zones()

    def process_zones(self) -> None:
        """
        Iterate through each overlapping zone to generate the contour polygon, PCA centerline and 
        series of rectangular patches.

         Stores outputs in:
            - self.contours_list (List[np.ndarray]):    polygon in Swiss coordinates
            - self.centerlines_list (List[np.ndarray]): PCA of overlaps 
            - self.patches_list (List[List[Patch]]):    list of patch objects per zone

        """
        for superpos_zone in self.superpos_zones_all:
            self.get_contour(superpos_zone)
            self.get_centerline(superpos_zone)
            patches, par_lines = self.patches_along_centerline()
            self.patches_list.append(patches)
            self.parallel_centerlines_list.append(par_lines)

    def get_contour(self, superpos_zone: np.ndarray) -> None:
        """
        Extract a polygonal contour from the *largest connected component* of a
        superposition zone. Isolating the largest component prevents small isolated
        pixels from being picked as the reference region, which would produce an
        incorrect centerline and patches.

        Input:
            superpos_zone (np.ndarray): Single boolean mask of overlapping area

        Output:
            None
        """
        if not np.any(superpos_zone):
            self.contours_list.append(np.zeros((0, 2)))
            return

        # UPDATE: keep largest comnponent 
        labeled = sk_label(superpos_zone.astype(int))
        props = regionprops(labeled)
        largest = max(props, key=lambda r: r.area)
        superpos_largest = (labeled == largest.label)

        contour_bulk = find_contours(superpos_largest.astype(int))[0]
        coords = approximate_polygon(contour_bulk, tolerance=self.tol)

        contour_x = coords[:, 1].astype(int)  # Column indices
        contour_y = coords[:, 0].astype(int)  # Row indices

        # Convert indices to Swiss coordinates frame
        contour = np.array(
                        [self.x_mesh[contour_y, contour_x] + self.res_x / 2, self.y_mesh[contour_y, contour_x] - self.res_y / 2]
                        ).T  # Add half pixel to get the center of the cell
        self.contours_list.append(contour)

    def get_centerline(self, superpos_zone: np.ndarray) -> None:
        """
        Estimate the main orientation of a superposition zone using PCA on its pixel coordinates.

        Input:
            superpos_zone (np.ndarray): Single boolean mask of overlapping area

        Output:
            None
        """
        mask_coords = np.column_stack(np.where(superpos_zone))
        coord_points = np.array([self.x_mesh[mask_coords[:, 0], mask_coords[:, 1]], self.y_mesh[mask_coords[:, 0], mask_coords[:, 1]]]).T

        if coord_points.shape[0] < 2:
            self.centerlines_list.append(np.zeros((0, 2)))
            return

        # PCA via SVD to avoid sklearn/scipy dependency
        mean = coord_points.mean(axis=0)
        centered = coord_points - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal_axis = vt[0]

        projected = centered @ principal_axis
        min_proj = projected.min()
        max_proj = projected.max()

        line_points = np.linspace(min_proj, max_proj, 100)[:, None] * principal_axis[None, :] + mean
        centerline = line_points

        self.centerlines_list.append(centerline)

    def patches_along_centerline(self) -> Tuple[List, List]:
        """
        Generates rectangular patches at regular intervals along one or more parallel
        centerlines.

        The starting distance along the centerline is determined once using the main
        (offset=0) line so that patches on all parallel lines are perfectly aligned
        (same longitudinal positions, only shifted perpendicularly).

        Returns:
            Tuple[List[Patch], List[np.ndarray]]:
                - all_patches: every valid Patch across all parallel lines
                - par_lines:   array of shape (N, 100, 2) with coordinates of each
                               shifted centerline, for visualisation
        """
        all_patches = []
        par_lines   = []

        contour    = self.contours_list[-1]
        centerline = self.centerlines_list[-1]

        contour_polygon  = Polygon(contour)
        centerline_line  = LineString(centerline)

        if len(centerline) < 2:
            raise ValueError("Centerline must have at least two points")

        direction      = centerline[-1] - centerline[0]
        direction      = direction / np.linalg.norm(direction)
        perp_direction = np.array([-direction[1], direction[0]])

        # Perpendicular offsets — symmetric about the main axis
        if self.n_lines == 1:
            offsets = [0.0]
        else:
            half_span = (self.n_lines - 1) / 2.0
            offsets = [(i - half_span) * self.line_offset for i in range(self.n_lines)]

        # ── Step 1: find start_dist once on the *main* centerline (offset 0) ──
        shared_start_dist = None
        for start_candidate in np.arange(0, centerline_line.length, 100):
            base_pt     = np.array(centerline_line.interpolate(start_candidate).coords[0])
            params      = PatchParams(base_pt, direction, perp_direction, self.band_length, self.band_width)
            patch       = self.create_patch(params)
            if patch.shapely_polygon.within(contour_polygon):
                shared_start_dist = float(start_candidate)
                break

        if shared_start_dist is None:
            return [], []   # no room for any patch in this zone

        # ── Step 2: generate patches on every parallel line using the shared start ──
        for perp_offset in offsets:
            shift      = perp_offset * perp_direction
            line_pts   = np.array([
                np.array(centerline_line.interpolate(d).coords[0]) + shift
                for d in np.linspace(0, centerline_line.length, 100)
            ])
            par_lines.append(line_pts)

            line_patches = []
            current_dist = shared_start_dist
            while current_dist < centerline_line.length:
                base_pt       = np.array(centerline_line.interpolate(current_dist).coords[0])
                current_point = base_pt + shift
                params        = PatchParams(current_point, direction, perp_direction, self.band_length, self.band_width)
                patch         = self.create_patch(params)
                line_patches.append(patch)
                self.patch_id += 1
                current_dist += self.sample_distance

            # Trim trailing patches that fall outside the contour
            while line_patches and not line_patches[-1].shapely_polygon.within(contour_polygon):
                line_patches.pop()

            all_patches.extend(line_patches)

        return all_patches, par_lines

    def create_patch(self, params: PatchParams) -> np.ndarray:
        """
        Create a rectangular patch polygon from its geometric definition (start point,
        direction, perpendicular direction, width and length).

        Converts the 4 corners to a polygon and stores both geometry and metadata.

        Input:
            params (PatchParams): Geometric parameters of the patch

        Returns:
            patch: Patch object with polygon and metadata
        """
        half_width = params.width / 2

        corner1 = params.startpoint + params.length * params.direction + half_width * params.perp_direction
        corner2 = params.startpoint + params.length * params.direction - half_width * params.perp_direction
        corner3 = params.startpoint - half_width * params.perp_direction
        corner4 = params.startpoint + half_width * params.perp_direction

        corners = np.array([corner1, corner2, corner3, corner4, corner1])
        polygon = Polygon(corners)

        center = params.startpoint + params.length / 2 * params.direction
        direction = params.direction
        perp_direction = params.perp_direction

        patch = Patch(
            id=self.patch_id,
            patch_array=corners,
            shapely_polygon=polygon,
            metadata={"center": center, "direction": direction, "perp_direction": perp_direction, "length": params.length, "width": params.width},
        )

        # patch = Patch(id=self.patch_id, patch_array=corners, shapely_polygon=polygon)
        return patch

    def compute_max_patch_length(self, idx: int) -> Tuple[np.ndarray, float]:
        """
        Compute the longest valid patch length that remains entirely within the contour. (Called in gui.py)

        This function starts from the first valid position along the centerline and 
        gradually reduces the patch length until the generated patch is fully contained 
        within the corresponding superposition contour.

        Input:
            idx (int): Index of the superposition zone (referring to contours_list and centerlines_list)

        Output:
            - start_point (np.ndarray): Coordinates of the starting point for the patch
            - max_length (float):       Maximum patch length that does not exceed the contour bounds
        """
        self.patch_id += 0

        contour = self.contours_list[idx]
        centerline = self.centerlines_list[idx]

        contour_polygon = Polygon(contour)
        centerline_line = LineString(centerline)

        if len(centerline) < 2:
            raise ValueError("Centerline must have at least two points")

        direction = centerline[-1] - centerline[0]
        direction = direction / np.linalg.norm(direction)  # Normalize
        perp_direction = np.array([-direction[1], direction[0]])

        # Start from the first valid point. To do so, check first possible patch
        valid_start_found = False
        start_dist = 0
        while not valid_start_found and start_dist < centerline_line.length:
            start_point = np.array(centerline_line.interpolate(start_dist).coords[0])
            params = PatchParams(start_point, direction, perp_direction, self.band_length, self.band_width)
            patch = self.create_patch(params)
            patch_poly = patch.shapely_polygon

            # Check if patch intersects with contour -> If so then the start position is valid
            if patch_poly.within(contour_polygon):
                valid_start_found = True
            else:
                start_dist += 20

        # Then find the maximum length so the polygon remains into the contour
        test_length = centerline_line.length  # Start with big length
        valid_end_found = False

        while not valid_end_found:
            start_point = np.array(centerline_line.interpolate(start_dist).coords[0])
            params = PatchParams(start_point, direction, perp_direction, test_length, self.band_width)
            patch = self.create_patch(params)
            patch_poly = patch.shapely_polygon

            if not patch_poly.within(contour_polygon):
                test_length -= 10
            else:
                break  # Stop when the patch fits

        max_length = test_length
        self.max_patch_len.append(max_length)

        return start_point, max_length

    def create_single_patch(self, idx: int, start_point: np.ndarray, length: float, width: float) -> Optional[List[np.ndarray]]:
        """
        Generate a single rectangular patch of given dimensions, placed at a specified start point
        along the centerline of the selected superposition zone. (Called in gui.py)

        Input:
            idx (int): Index of the zone (from centerlines_list)
            start_point (np.ndarray): Starting point of the patch in (x, y) coordinates
            length (float): Length of the patch along the main axis (centerline direction)
            width (float): Width of the patch perpendicular to the main axis

        Output:
            Optional[List[np.ndarray]]:
                - A list containing a single Patch object, or None if centerline is invalid
        """
        centerline = self.centerlines_list[idx]

        if len(centerline) < 2:
            return  # Not enough points to define a patch

        # Compute direction of the centerline (from start to end)
        direction = centerline[-1] - centerline[0]
        direction /= np.linalg.norm(direction)  # Normalize

        # Perpendicular direction for width
        perp_direction = np.array([-direction[1], direction[0]])

        params = PatchParams(start_point, direction, perp_direction, length, width)
        patch = self.create_patch(params)
        self.patch_id += 1

        return [patch]  # l'enclure en nested list comme ça on a pas de soucis de dimension dans le plot du GUI
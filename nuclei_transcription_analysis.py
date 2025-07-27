## for the future you'll be able to access them from astec or some package but I am behind on my code integration duty
## sorry in advance

import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import regionprops, label
from scipy.ndimage import label, center_of_mass, generate_binary_structure, binary_dilation
from scipy.spatial import cKDTree


# crop imgs for each cell, get cell id, segmented membrane, sergmented nuclei returns cropped two imgs in cell
def crop_cell_images(cell_id, seg_membrane_img, seg_nucleus_img):
    """
    Crop a bounding box around a given cell in both membrane and nucleus images.

    Parameters
    ----------
    cell_id : int
        Label of the target cell.
    seg_membrane_img : np.ndarray
        Segmented membrane image with cell labels.
    seg_nucleus_img : np.ndarray
        Segmented nucleus image with nucleus labels.

    Returns
    -------
    cropped_membrane : np.ndarray
        Cropped membrane region containing the cell.
    cropped_nucleus : np.ndarray
        Cropped nucleus region within the cell bounding box.
    """
    # get bounding box (parallelipied that surronds cell in both images)
    slices = find_objects(seg_membrane_img == cell_id)
    if not slices:
        raise ValueError(f"Cell ID {cell_id} not found in membrane image.")
    bbox = slices[0]

    cropped_membrane = seg_membrane_img[bbox]
    cropped_nucleus = seg_nucleus_img[bbox]

    return cropped_membrane, cropped_nucleus

# get nuclei label inside cell : given cell label, img segmented membranes, img segmented nuclei
def get_nucleus_inside_cell(cell_id, seg_membrane_img, seg_nucleus_img):
    """
    Given a segmented membrane image and a segmented nucleus image,
    return the nucleus label inside the given cell.

    Parameters
    ----------
    cell_id : int
        Label of the cell to analyze.
    seg_membrane_img : np.ndarray
        Labeled membrane segmentation image.
    seg_nucleus_img : np.ndarray
        Labeled nucleus segmentation image.

    Returns
    -------
    int or None
        Nucleus label inside the given cell. None if no nucleus found.
    """
    # Create binary mask for the given cell
    cell_mask = seg_membrane_img == cell_id

    if not np.any(cell_mask):
        raise ValueError(f"Cell ID {cell_id} not found in membrane image.")

    # Find unique nucleus labels overlapping with cell
    overlapping_labels = np.unique(seg_nucleus_img[cell_mask])
    overlapping_labels = overlapping_labels[overlapping_labels != 0]  # remove background

    if len(overlapping_labels) == 0:
        return None
    if len(overlapping_labels) == 1:
        return overlapping_labels[0]

    # If multiple overlaps, return the one with maximum intersection
    max_overlap = 0
    chosen_label = None
    for label_val in overlapping_labels:
        overlap = np.sum((seg_nucleus_img == label_val) & cell_mask)
        if overlap > max_overlap:
            max_overlap = overlap
            chosen_label = label_val
    return chosen_label


# compute barycenter of object given label in img
def compute_barycenter(labeled_img, target_label):
    """
    Compute the barycenter of a given label in a labeled image.

    Parameters
    ----------
    labeled_img : np.ndarray
        Labeled image.
    target_label : int
        Label of the object to compute the barycenter for.

    Returns
    -------
    center : tuple
        (z, y, x) coordinates of the barycenter.
    """
    props = regionprops((labeled_img == target_label).astype(np.int32))
    if not props:
        raise ValueError(f"Label {target_label} not found.")
    return props[0].centroid

# get apical surface of cell : give cell label, segmented cell membrane, returns barycenter, eigenvalues and eigenvectors
def get_apical_pts(segmented_cell_mask, cell_id, get_plane=False):
    """

    Find 'apical points' — voxels on the boundary of the specified cell_id
    that are adjacent to the background (label 0).

    Args:
        segmented_cell_mask (ndarray): 3D labeled mask.
        cell_id (int): The label of the cell to analyze.
        get_plane (bool): if true return the eignevalues and eigenvectors of the apical plane will apical plane barycenter

    Returns:
        farthest_apical_point, cell_barycenter
        or (eigvals, eigvecs, mean_apical) sorted in descending order
    """
    # Binary mask for the cell of interest
    cell_mask = (segmented_cell_mask == cell_id)
    # Define 6- or 26-connectivity; here we use 6-connected structure
    structure = generate_binary_structure(rank=3, connectivity=1)
    # Dilate the cell and subtract original to get boundary candidates
    dilated = binary_dilation(cell_mask, structure=structure)

    # Boundary = dilated - original
    background_boundary = dilated & (segmented_cell_mask == 1)
    # apical points are cell voxels adjacent to background: set to 1
    apical_coords = cell_mask & binary_dilation(background_boundary, structure=structure)

    cell_barycenter = np.array(center_of_mass(cell_mask))

    if np.count_nonzero(apical_coords) < 3:
        if get_plane:
            return np.nan, np.nan, cell_barycenter
        return np.nan, cell_barycenter

    #
    # return furthest pt of apical surface (tanget apical plane on apical surface)
    if apical_coords.ndim == 3:
        apical_coords = np.stack(np.where(apical_coords), axis=-1)  # Convert mask to coordinate list

    # Mean of apical coordinates
    # apical_coords is a binary indicator on each pivel if its apical or not
    mean_apical = np.mean(apical_coords, axis=0)

    if get_plane:
        # Compute covariance matrix for PCA
        centered = apical_coords - mean_apical
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx] # sorted descending
        return  (eigvals, eigvecs, mean_apical)

    tree = cKDTree(apical_coords)
    _, idx = tree.query(mean_apical)
    farthest_apical_point = tuple(apical_coords[idx])

    return farthest_apical_point, cell_barycenter

# todo get basal surface barycenter: given apical surface, segmeented cell memebrane, cll label (its the surface of contact facing the apical surface)


# given: nuclei segmented img, nucleus label, compute nucleus estimated radius and barycenter, given pts coord, returns r/R (how far is pts from barycenter)
def nucleus_relative_distance_to_center(nuclei_img, nucleus_label, points_coords):
    """
    Given a segmented nuclei image and a nucleus label, compute:
    - The barycenter of the nucleus
    - The estimated nucleus radius (assuming spherical shape using volume)
    - The relative distance (r/R) of given point(s) to the nucleus center

    Parameters
    ----------
    nuclei_img : np.ndarray
        Labeled image where each nucleus has a unique label.
    nucleus_label : int
        Label of the nucleus of interest.
    points_coords : np.ndarray or list
        Single point (shape: (3,) or (2,)) or array of shape (N, 3) or (N, 2)
        representing the coordinates to compute distance from center.

    Returns
    -------
    r_div_R : float or np.ndarray
        Relative distance(s) of the given point(s) to the nucleus center (0 to ~1 range).
    center : np.ndarray
        Barycenter of the nucleus.
    radius : float
        Estimated nucleus radius (from volume assuming spherical shape).
    """
    # Get coordinates of the current nucleus
    nucleus_voxels = np.argwhere(nuclei_img == nucleus_label)
    if len(nucleus_voxels) == 0:
        raise ValueError(f"Nucleus label {nucleus_label} not found in image")

    # Barycenter of the nucleus
    center = np.mean(nucleus_voxels, axis=0)

    # Estimate radius assuming spherical shape: V = (4/3)*pi*R^3  -> R = (3V/4π)^(1/3)
    volume = len(nucleus_voxels)
    radius = ((3 * volume) / (4 * np.pi)) ** (1/3)

    # Compute r/R for each point
    points_coords = np.atleast_2d(points_coords)
    r = np.linalg.norm(points_coords - center, axis=1)
    r_div_R = r / radius

    return r_div_R if len(r_div_R) > 1 else r_div_R[0], center, radius

#  given a pts, apical surface coords, cell/nuclei barycenter, returns orientation (sign of dot product) 
def is_point_toward_apical_surface_direction(pt_coord, apical_barycenter, nucleus_barycenter,
                                            epsilon=1e-4):
    """
    Determine whether a point lies in the direction of the apical surface relative to the nucleus.

    This checks the alignment between:
    - the vector from apical barycenter to nucleus barycenter, and
    - the vector from apical barycenter to the point of interest (pt_coord)

    The dot product indicates if the point lies along the same general direction
    as the nucleus with respect to the apical surface.

    Parameters
    ----------
    pt_coord : np.ndarray
        3D coordinates of the point of interest.
    apical_barycenter : np.ndarray
        3D coordinates of the apical surface barycenter.
    nucleus_barycenter : np.ndarray
        3D coordinates of the nucleus barycenter.
    epsilon : float
        Threshold for dot product to consider as "undetermined" (~0 directionality).

    Returns
    -------
    str
        One of 'apical', 'basal', or 'undetermined'
    float
        Dot product value for inspection
    """
    v_nucleus = nucleus_barycenter - apical_barycenter
    v_point = pt_coord - nucleus_barycenter

    dot = np.dot(v_nucleus, v_point)
    
    if abs(dot) < epsilon:
        return 'undetermined', dot
    elif dot > 0:
        return 'basal', dot
    else:
        return 'apical', dot


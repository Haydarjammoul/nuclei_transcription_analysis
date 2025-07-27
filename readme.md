### Hello Louisa, these are chunks of my code, I will try to document and make fct names as clear as possible

# Nuclei and Transcription Analysis Toolkit

This module provides key functions for analyzing individual cells from segmented membrane and nuclei images in 3D microscopy data.
 It focuses on isolating cells, identifying their nuclei, and assessing the orientation of transcription relative to apical surfaces.

---

## 📦 Functions Overview

### crop_cell_images(segmented_membranes, segmented_nuclei, cell_id)
To avoid computing on whole image everytime, analyse each cell at its turn 
Extracts a 3D crop of a single cell from the full segmented volume. Both the membrane and nucleus segmentation are cropped around the given `cell_id`.

**Returns:**
- `cropped_cell_mask` : image of segmented cell membrane cropped around the cell
- `cropped_nuclei_img` : image of nuclei cropped around the cell

---

###  get_nucleus_inside_cell(cell_id, seg_membrane_img, seg_nucleus_img)
Identifies the nucleus located inside cell that has id cell_id in seg_membrane_img.

**Returns:**
- `nucleus_label` (int or None): ID of the nucleus inside the cell (or `None` if not found)

---

### compute_barycenter(label_img, label_val)
Computes the 3D centroid (barycenter) of a given object label in a labeled image.

**Returns:**
- `np.ndarray`: the (x, y, z) coordinates of the centroid

---

###  nucleus_relative_distance_to_center(barycenter, surface_pts)
Calculates how far a pt (or an array of pts) is from the apical surface **as a ratio** of total nucleus radius(supposing its spherical).

**Returns:**
- r_div_R : float or np.ndarray
        Relative distance(s) of the given point(s) to the nucleus center (0 to ~1 range).
- center : np.ndarray
    Barycenter of the nucleus.
- radius : float
    Estimated nucleus radius (from volume assuming spherical shape).

---

###  is_point_toward_apical_surface_direction(pt_coord, apical_barycenter, nucleus_barycenter)
Determines whether a given point lies on the **basal**, **apical**, or **orthogonal** side, relative to the axis connecting the apical and nuclear centers.

**Returns:**
- `'apical'`, `'basal'`, or `'undetermined'`
- dot product value (float)

---

###  get_apical_pts(segmented_cell_mask, cell_id, get_plane=False)
Finds the surface voxels of a cell that are in contact with the background (apical side). Optionally, returns the geometric plane (eigenvectors) best fitting these points.
farthest_apical_point is the closest point on the apical surface to the apical surface barycenter (the apical surface is curved so its barycenter is not on the surface)

**Returns:**
- If `get_plane=False`: `farthest_apical_point`, `cell_barycenter`
- If `get_plane=True`: `eigvals`, `eigvecs`, `mean_apical` describing the apical plane

---


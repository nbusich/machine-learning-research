import scipy.io as sio
import numpy as np
import cv2
from pathlib import Path

# ---------- 1)  Load the .mat  ----------
mat = sio.loadmat('cells.mat', squeeze_me=True, struct_as_record=False)
cellList = mat['cellList']            # 1×1 struct ➜ python object
meshData = cellList.meshData          # cell array ➜ numpy object array

# ---------- 2)  Where are the TIFFs?  ----------
#   • If the .mat contained a real path (e.g. in paramString) you could parse it.
#   • In YOUR file that variable isn’t present, so just put the folder here:
raw_dir = Path('/absolute/path/to/phase_images')   # <‑‑ change this

out_dir = Path('bacterium_patches')
out_dir.mkdir(exist_ok=True)

# ---------- 3)  Helper: crop using the mesh  ----------
def crop_from_mesh(img, mesh, pad=2):
    xmin = int(mesh[:, [0, 2]].min()) - pad
    xmax = int(mesh[:, [0, 2]].max()) + pad
    ymin = int(mesh[:, [1, 3]].min()) - pad
    ymax = int(mesh[:, [1, 3]].max()) + pad
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax = min(img.shape[1]-1, xmax)
    ymax = min(img.shape[0]-1, ymax)
    return img[ymin:ymax+1, xmin:xmax+1]

# ---------- 4)  Walk through every frame & cell  ----------
for f_idx, frame_cells in enumerate(meshData):           # f_idx = 0‑based frame #
    # guard against empty frames
    if frame_cells.size == 0:
        continue

    # load the matching raw image
    tif_name = raw_dir / f'frame{f_idx+1:03d}.tif'       # adjust pattern if needed
    img = cv2.imread(str(tif_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'⚠️  Could not read {tif_name}')
        continue

    # make sure frame_cells is iterable even if there’s only one cell
    for c_idx, cell in enumerate(np.atleast_1d(frame_cells)):
        mesh = cell.mesh           # (N × 4)
        patch = crop_from_mesh(img, mesh)
        out_file = out_dir / f'f{f_idx:03d}_c{c_idx:04d}.png'
        cv2.imwrite(str(out_file), patch)

print('✅  Cropping finished.')


print(f'Frames in file: {meshData.size}')
first_nonempty = next(i for i, x in enumerate(meshData) if x.size)
print(f'Cells in first non‑empty frame: {np.atleast_1d(meshData[first_nonempty]).size}')
print('Example mesh shape:', np.atleast_1d(meshData[first_nonempty])[0].mesh.shape)


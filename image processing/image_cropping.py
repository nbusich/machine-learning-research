#!/usr/bin/env python3

import os
import json
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

OUTPUT_DIR = "output_images"  # Subfolder for saving images & data


def ensure_output_dir_exists(directory):
    """
    Create the output directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def is_nonzero(cellInfo):
    """
    Returns True if cellInfo is not "all zero".
    If cellInfo is a scalar, returns cellInfo != 0.
    If cellInfo is an array-like object, returns True if not every element is 0.
    """
    arr = np.array(cellInfo)
    if arr.ndim == 0:
        return arr != 0
    return not np.all(arr == 0)


def makeGraph(cellInfo, timepoint, cell_index, results_list):
    """
    Create and save a graph for the cell mesh in the output directory.
    Also record location data (coords) in results_list for later JSON export.

    cellInfo: array-like of (x, y) coordinate pairs.
    """
    try:
        coords = np.array(cellInfo)
        # Expect at least (x, y) columns for valid plotting
        if coords.ndim != 2 or coords.shape[1] < 2:
            print(f"Unexpected mesh shape at timepoint {timepoint + 1}, cell {cell_index + 1}")
            return

        # Make a plot
        plt.figure()
        plt.plot(coords[:, 0], coords[:, 1], 'o-')
        plt.title(f"Timepoint {timepoint + 1}, Cell {cell_index + 1}")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Build filename in the OUTPUT_DIR
        filename = f"cell_mesh_t{timepoint + 1}_cell{cell_index + 1}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(output_path)
        plt.close()

        print(f"Graph saved for timepoint {timepoint + 1}, cell {cell_index + 1} as {output_path}")

        # Store data (timepoint, cell indices, and coordinates) in results_list
        # coords.tolist() so it can be serialized to JSON
        results_list.append({
            "timepoint": timepoint + 1,
            "cell_index": cell_index + 1,
            "coords": coords.tolist(),
            "output_image_path": output_path
        })

    except Exception as e:
        print(f"Error in makeGraph at timepoint {timepoint + 1}, cell {cell_index + 1}: {e}")


def load_mat_file(file_path):
    """
    Load a MATLAB .mat file using scipy.io.loadmat with struct_as_record=False, squeeze_me=True.
    """
    try:
        data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        print(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_mesh_data(cellList):
    """
    Extract meshData from cellList (a dict or an object).
    """
    if isinstance(cellList, dict):
        return cellList.get('meshData', None)
    else:
        return getattr(cellList, 'meshData', None)


def process_mesh_data(meshData, results_list):
    """
    Iterate over each timepoint in meshData, handle multiple or single cell entries,
    and call makeGraph to visualize each cell's mesh.
    """
    if meshData is None:
        print("No meshData found in the MATLAB file.")
        return

    for i in range(len(meshData)):
        try:
            # Treat meshData[i] as a list of cells
            for j in range(len(meshData[i])):
                cellItem = meshData[i][j]
                if isinstance(cellItem, dict):
                    cellInfo = cellItem.get('mesh', 0)
                else:
                    cellInfo = getattr(cellItem, 'mesh', 0)

                if is_nonzero(cellInfo):
                    makeGraph(cellInfo, i, j, results_list)

            print(f"Processed multiple cells at timepoint {i + 1}")
        except (KeyError, TypeError, AttributeError) as e:
            # If an exception occurs, assume single cell at this timepoint
            print(f"Single cell encountered at timepoint {i + 1} (error: {e}). Processing as a single cell.")
            if isinstance(meshData[i], dict):
                cellInfo = meshData[i].get('mesh', 0)
            else:
                cellInfo = getattr(meshData[i], 'mesh', 0)

            if is_nonzero(cellInfo):
                makeGraph(cellInfo, i, 0, results_list)


def main():
    """
    Main entry point of the script.
    Adjust 'mat_file_path' to point to your MATLAB file.
    """
    mat_file_path = '/Users/nickusich/Desktop/MatLab/New Samples/Good/CBZ_1_good_signal.mat'

    # Ensure the output directory exists
    ensure_output_dir_exists(OUTPUT_DIR)

    # Load the MATLAB data
    data = load_mat_file(mat_file_path)
    if data is None:
        return

    # Extract cellList from the loaded data
    cellList = data.get('cellList') if isinstance(data, dict) else data['cellList']
    if cellList is None:
        print("No 'cellList' found in the MATLAB data.")
        return

    # Extract meshData
    meshData = get_mesh_data(cellList)

    # Create a list to store all results (coords, image paths, etc.)
    results_list = []

    # Process the mesh data and populate results_list
    process_mesh_data(meshData, results_list)

    # Once all images are generated, save the coordinate + image metadata to JSON
    results_path = os.path.join(OUTPUT_DIR, "mesh_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)

    print(f"\nAll processing complete. Results JSON saved at: {results_path}")


if __name__ == "__main__":
    main()
import argparse
import pandas as pd
import numpy as np
from scipy import spatial


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--envelope", action="store", help='Envelope point-cloud file in csv format')
    parser.add_argument("-l", "--ligand", action="store", help='Docked ligand file in pdbqt format (i.e. Vina output)')
    return parser


def get_envelope_kd_tree(file_path):
    envelope = pd.read_csv(file_path)
    kd_tree_envelope = spatial.cKDTree(envelope)
    return kd_tree_envelope


def get_ligand_coords(ligand_file_path):
    with open(ligand_file_path, 'r') as f:
        xs, ys, zs = [], [], []
        for line in f.readlines():
            if line == 'ENDMDL\n':
                # get only the top-ranked pose
                break
            line_li = line.split()
            if len(line_li) == 12:
                xs.append(float(line_li[5]))
                ys.append(float(line_li[6]))
                zs.append(float(line_li[7]))
    f.close()

    return xs, ys, zs


def get_penalty(ligand_coords, kd_tree_envelope):
    xs, ys, zs = ligand_coords
    ligand = pd.DataFrame(data={'x': xs, 'y': ys, 'z': zs})
    penalty = 0.
    for i in range(len(ligand)):
        crd = np.array(ligand.iloc[i])
        # this will return the ligand atoms that are within r from at least 1 point in the pocket
        dist_to_neighbour, _ = kd_tree_envelope.query(crd, k=1)
        """ can use different functions in order to calculate the protrusions, such as the following step function
        instead of simply the negative distance
        if dist_to_neighbour < 0.1:
            continue
        elif dist_to_neighbour < 1.:
            penalty -= dist_to_neighbour / 2
        else:
            penalty -= dist_to_neighbour
        """
        penalty -= dist_to_neighbour
    return penalty


if __name__ == "__main__":
    parser = get_parser()

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    kd_tree_envelope = get_envelope_kd_tree(args.envelope)
    ligand_coords = get_ligand_coords(args.ligand)
    protrusion = get_penalty(ligand_coords, kd_tree_envelope)

    print("The protrusion is: ", protrusion)

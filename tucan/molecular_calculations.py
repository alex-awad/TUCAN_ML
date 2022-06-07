""" Module for calculations of atom and molecular properties. """
import numpy as np
from typing import Tuple, List

def calculate_bond_length(position_1: Tuple[float, ...], position_2: Tuple[float, ...]):
    """ Returns bond length between two atoms from their cartesian coordinates. Uses euclidean distance. 
    """
    assert(len(position_1) == len(position_2))
    
    np_pos_1 = np.asarray(position_1)
    np_pos_2 = np.asarray(position_2)
    
    return np.linalg.norm(np_pos_1 - np_pos_2).item()


def bond_length_from_bond_tuple(bond_tuple: Tuple[int, int], atom_positions: List[Tuple[float]]):
    """ Calculates and returns bond length from a tuple including the labels of the atoms of this bond, and a list
        including all atom positions with the index of the list representing their labels.        
    """
    position_1 = atom_positions[bond_tuple[0]]
    position_2 = atom_positions[bond_tuple[1]]
    
    return calculate_bond_length(position_1, position_2)
    
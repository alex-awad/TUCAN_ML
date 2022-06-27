import networkx as nx
from tucan.element_properties import ELEMENT_PROPS
from rdkit import Chem
from typing import List, Tuple
from tucan.molecular_calculations import bond_length_from_bond_tuple


def graph_from_file(filepath, get_bond_lengths: bool=False):
    with open(filepath) as f:
        filecontent = f.read()
    if filepath.suffix == ".mol":
        if get_bond_lengths:
            element_symbols, bonds, bond_lengths = _parse_molfile3000(filecontent, get_bond_lengths=True)
        else:
            element_symbols, bonds = _parse_molfile3000(filecontent)
    elif filepath.suffix == ".col":
        element_symbols, bonds = _parse_dimacs(filecontent)
    else:
        raise IOError("Invalid file format, must be one of {.mol, .col}.")
    if get_bond_lengths:
         return graph_from_moldata(element_symbols, bonds, bond_lengths)
     
    return graph_from_moldata(element_symbols, bonds)


def graph_from_smiles(smiles: str):
    molfile = _molfile3000_from_smiles(smiles)
    element_symbols, bonds = _parse_molfile3000(molfile)
    return graph_from_moldata(element_symbols, bonds)


def _molfile3000_from_smiles(smiles: str):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    return Chem.MolToMolBlock(m, forceV3000=True, includeStereo=False, kekulize=False)


def graph_from_moldata(
    element_symbols: List[str],
    bonds: List[Tuple[int]],
    bond_lengths: List[float] = None
    ):
    """Instantiate a NetworkX graph from molecular data.

    Parameters
    ----------
    element_symbols: List[str]
        Element symbols associated with the atoms.
    bonds: List[Tuple[int, int]]
        Bonds between atoms.
    bond_lengths: List[float]
        Bond lenghts of each bond. Index corresponds to the index of bonds.
        Defaults to None.
    """
    atomic_numbers = [ELEMENT_PROPS[s]["atomic_number"] for s in element_symbols]
    node_labels = range(len(element_symbols))
    graph = nx.Graph()
    graph.add_nodes_from(node_labels)
    graph.add_edges_from(bonds)
    nx.set_node_attributes(
        graph, dict(zip(node_labels, element_symbols)), "element_symbol"
    )
    nx.set_node_attributes(
        graph, dict(zip(node_labels, atomic_numbers)), "atomic_number"
    )
    
    if bond_lengths is not None:
        nx.set_edge_attributes(
            graph, dict(zip(bonds, bond_lengths)), "bond_length"
        )
        
    nx.set_node_attributes(graph, 0, "partition")
    return graph


def _parse_molfile3000(filecontent: str, get_bond_lengths: bool = False):
    raw_lines = filecontent.splitlines()    
    lines = [l.rstrip().split(" ") for l in filecontent.splitlines()]
    lines = [[value for value in line if value != ""] for line in lines]
    atom_count = int(lines[5][3])
    bond_count = int(lines[5][4])
    atom_block_offset = 7
    bond_block_offset = atom_block_offset + atom_count + 2
    
    # Get element symbols of the atoms in the file
    element_symbols = [
        l[3] for l in lines[atom_block_offset : atom_block_offset + atom_count]
    ]
    assert (
        len(element_symbols) == atom_count
    ), f"Number of atoms {len(element_symbols)} doesn't match atom-count specified in header {atom_count}."
    
    
    # Get bonds as tuple of atoms forming the bond
    bonds = [
        (int(l[4]) - 1, int(l[5]) - 1)
        for l in lines[bond_block_offset : bond_block_offset + bond_count]
    ]  # make bond-indices zero-based
    assert (
        len(bonds) == bond_count
    ), f"Number of bonds {len(bonds)} doesn't match bond-count specified in header {bond_count}."
    
    if get_bond_lengths:
        # Check if molfile is saved from 2D source if bond lengths are calculated. 
         # This is not not recommended, since the coordinates of a 2D origin might
         # represent the position in the 2D depiction, not the assumed 3D positions.
        if "2D" in raw_lines[0] or "2D" in raw_lines[1]:
            print("""WARNING: Parsed .mol file includes only 2D information.
                  Bond lengths might not be calculated correctly!""")
        
        # Get x, y, and z position of each atom
        atom_xyz_positions = [
            (float(l[4]), float(l[5]), float(l[6])) for l 
                in lines[atom_block_offset : atom_block_offset + atom_count]
        ]
        
        # Calculate bond lengths
        bond_lengths = [
            bond_length_from_bond_tuple(bond, atom_xyz_positions) for bond in bonds
        ]
    
        return element_symbols, bonds, bond_lengths
    
    return element_symbols, bonds


def _parse_molfile2000(filecontent: str):
    lines = [
        [l[i : i + 3].strip(" ") for i in range(0, len(l), 3)]
        for l in filecontent.splitlines()
    ]
    atom_count = int(lines[3][0])
    bond_count = int(lines[3][1])
    atom_block_offset = 4
    bond_block_offset = atom_block_offset + atom_count
    element_symbols = [
        l[10] for l in lines[atom_block_offset : atom_block_offset + atom_count]
    ]
    assert (
        len(element_symbols) == atom_count
    ), f"Number of atoms {len(element_symbols)} doesn't match atom-count specified in header {atom_count}."
    bonds = [
        (int(l[0]) - 1, int(l[1]) - 1)
        for l in lines[bond_block_offset : bond_block_offset + bond_count]
    ]  # make bond-indices zero-based
    assert (
        len(bonds) == bond_count
    ), f"Number of bonds {len(bonds)} doesn't match bond-count specified in header {bond_count}."
    return element_symbols, bonds


def _parse_dimacs(filecontent: str):
    lines = [l.rstrip().split(" ") for l in filecontent.splitlines()]
    lines = [l for l in lines if l[0] in ["p", "e"]]
    atom_count = int(lines[0][2])
    element_symbols = ["C"] * atom_count
    bonds = [
        (int(l[1]) - 1, int(l[2]) - 1) for l in lines[1:]
    ]  # make bond-indices zero-based
    return element_symbols, bonds

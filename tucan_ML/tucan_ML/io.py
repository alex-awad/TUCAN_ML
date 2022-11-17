""" Module for io operations for the use of TUCAN for machine learning applications. """
from pathlib import Path
import networkx as nx
from tucan.io import graph_from_file
from tucan.canonicalization import assign_canonical_labels, _add_invariant_code
from tucan.serialization import serialize_molecule
from typing import Union


def path_to_canonicalized_graph(path: str, get_bond_lengths=False):
    """Created canonicalized networkx graph from a path to a supported file.

    Note: Usage of bond lengths are currently only supported with .mol (v3000) files.
    """
    m = graph_from_file(Path(path), get_bond_lengths=get_bond_lengths)
    _add_invariant_code(m)
    canonical_idcs = assign_canonical_labels(m)
    m_canonicalized = nx.relabel_nodes(m, canonical_idcs, copy=True)

    return m_canonicalized


def mol3000_file_to_TUCAN_string(
    input_path: str,
    output_path: str = ".",
    get_bond_lengths=True,
    to_file=True,
    filename=None,
) -> Union[None, str]:
    """Creates TUCAN string from path to .mol file (v3000) including bond lengths and saves it
    as .txt file to the output path. The filename can be chosen and is set to the
    sum formula of the TUCAN string by default.
    """
    m_canonicalized = path_to_canonicalized_graph(
        input_path, get_bond_lengths=get_bond_lengths
    )
    tucan_string = serialize_molecule(m_canonicalized)

    if not to_file:
        return tucan_string

    if filename is None:
        filename = tucan_string.split("/")[0]  # Get sum formula

    with open(f"{output_path}/{filename}.txt", "w") as file:
        file.write(tucan_string)
    return None

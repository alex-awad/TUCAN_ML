""" Module to convert TUCAN strings to pytorch geometric graphs for their utilization 
    in graph neural networks.
"""

import re
import collections
import torch
from torch_geometric.data import Data

import tucan
from tucan.element_properties import element_symbols

# Dictionaries to assign elements to their atomic number
# Transform element symbols to list
atom_number_dict = {}
number_atom_dict = {}
for i, element in enumerate(element_symbols):
    atom_number_dict[element] = i+1
    number_atom_dict[i+1] = element
    

def element_count_from_sum_formula(sum_form: str):
    """ Returns dict containing elements and their count from a given sum formula.
    """
    # Split sum formula on letters and numbers
    split_sum_form = re.split("(\d+)", sum_form)
    if split_sum_form[-1] == "":
        split_sum_form.pop()

    previous_element = ""
    element_count = dict()
    for substring in split_sum_form:
        # Split strings by capital letter, e.g. ClBr => [Cl, Br], "H" => ["H"], 1 => []
        split_by_capital_letter = re.findall('[A-Z][^A-Z]*', substring)
        
        # Check if there is no element in split (number)
        if not split_by_capital_letter:
            # Create dict with previous element and its count
            element_count[previous_element] = int(substring)
        # Check if there is only one element
        elif len(split_by_capital_letter) == 1:
            previous_element = split_by_capital_letter[0]
        # Multiple elements in one substring
        else:
            for element in split_by_capital_letter[:-1]:
                element_count[element] = 1
            previous_element = split_by_capital_letter[-1]
    # Check if last element is only present once
    if(split_sum_form[-1].isalpha()):
        element_count[previous_element] = 1
        
    return element_count


def graph_nodes_from_element_count(element_count: dict):
    """ Returns dictionary with graph nodes (counting from 1, ascending) and the respective element.
    """
    # Change keys of dict to have atom number as key instead of element abbreviation
    element_count_by_atom_number = dict((atom_number_dict[key], value) \
        for (key, value) in element_count.items())
    
    # Order dict by ascending atom number
    element_count_by_atom_number = collections.OrderedDict(
        sorted(element_count_by_atom_number.items())
    )
    
    # Assign graph nodes
    node_list = []
    
    for key, value in element_count_by_atom_number.items():
        for i in range(0, value):
            node_list.append(number_atom_dict[key])
            
    graph_nodes = {}
    for i, element in enumerate(node_list):
        graph_nodes[i+1] = element
        
    return graph_nodes


def convert_graph_nodes_element_to_atomic_number(graph_nodes: dict):
    """ Converts the values of the labels of graph nodes from their element symbol
        to their atomic number.
    """
    graph_nodes_converted = dict()
    for node, value in graph_nodes.items():
        graph_nodes_converted[node] = atom_number_dict[value]
        
    return graph_nodes_converted


def tucan_string_to_graph_nodes(tucan_string: str, use_element_symbols: bool = True):
    """ Transforms a TUCAN string to a dictionary with node labels for graph
        representation. The node labels are enumerated and ordered after the 
        atom number of the elements present in the TUCAN string. 
        
        E.g.: C2H4O => {1: H, 2: H, 3: H, 4: H, 5: C, 6: C, 7: O}
        
        Parameters:
        ------------
        tucan_string (str): TUCAN string from which graph nodes are created.
        
        use_element_symbols (bool): Whether to use element_symbols as node values. Otherwise
            atomic numbers are used. Defaults to True.
    """
    ## Get sum formula from TUCAN string
    sum_formula = tucan_string.split("/")[0]
    
    # Get elements and their counts
    element_counts = element_count_from_sum_formula(sum_formula)
    
    # Get graph nodes from element counts
    graph_nodes =  graph_nodes_from_element_count(element_counts)
    
    if use_element_symbols is False:
        graph_nodes = convert_graph_nodes_element_to_atomic_number(graph_nodes)
    
    return graph_nodes


def tucan_string_to_graph_edges(tucan_string: str):
    """ Returns list of edges from TUCAN string as a list for further
        transformation to a graph. Uses the same labels for the atoms as
        tucan_string_to_graph_nodes.
    """
    edges = [[int(node)-1 for node in tuple.split("-")] \
            for tuple in tucan_string.split("//")[0].split("/")[1:]]
    
    return edges


def tucan_string_to_graph_bond_lengths(tucan_string: str):
    """ Returns list of bond lengths from TUCAN string. The order of the list
        corresponds to the order of the edges in the string.
    """
    bond_lengths = [[float(bond_length)] for bond_length in \
        tucan_string.split("//")[-1].split("/")]
    
    return bond_lengths
    

def tucan_string_to_pyg_data(tucan_string: str):
    """ Creates pytorch geometric graph from tucan string including bond lengths
        as edge features.
    """
    # Get lists with relevant information to construct the graph
    nodes = tucan_string_to_graph_nodes(tucan_string, use_element_symbols=False) # Node values cannot be type string
    edges = tucan_string_to_graph_edges(tucan_string)
    bond_lengths = tucan_string_to_graph_bond_lengths(tucan_string)
    node_list =  [value for key, value in nodes.items()]
    
    sum_form = tucan_string.split("/")[0]
    
    # Create tensors
    x = torch.tensor(node_list)
    x = x.unsqueeze(1)
    edge_index = torch.tensor(edges)
    bond_lengths = torch.tensor(bond_lengths).T
    
    # Create graph
    data = Data(
        x=x,
        edge_index=edge_index.t().contiguous(),
        edge_attr=bond_lengths.t().contiguous(),
        mol_id=sum_form 
    )
    
    return data
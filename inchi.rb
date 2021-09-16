# (c) CC BY-SA | Ulrich Schatzschneider | Universität Würzburg | NFDI4Chem | v1.4 | 06.06.2021

module Inchi

  def read_molfile(filename)
    raise "Please provide a filename." if filename.nil?
    raise "File `#{filename}` doesn't exist." unless File.exist?(filename)

    molfile = File.read(filename) # reads entire file and closes it
    molfile.split("\n")
  end

  def create_molecule_array(molfile_lines, periodic_table_elements)
    # Represent molecule as list with each entry representing one element and its
    # linked elements in the format: [[index, atomic mass], [index, ..., index]].
    atom_count, edge_count = molfile_lines[3].scan(/\d+/).map(&:to_i) # on 4th  line, 1st number is number of atoms, 2nd number is number of bonds.
    element_array = create_element_array(molfile_lines, atom_count, periodic_table_elements)
    edge_array = create_edge_array(molfile_lines, edge_count, atom_count)
    element_array.zip(edge_array)
  end

  def canonicalize_molecule(molecule, filename)
    filename = File.basename(filename, '.mol')

    print_molecule(molecule,
      "\nInitial data structure of #{filename}:")

    sorted_molecule = sort_elements_by_atomic_mass(molecule)
    print_molecule(sorted_molecule,
      "\n#{filename} with atoms sorted by atomic mass (increasing):")

    sorted_molecule = sort_elements_by_number_of_edges(sorted_molecule)
    print_molecule(sorted_molecule,
      "\n#{filename} with atoms of same kind sorted by number of edges (increasing):")

    sorted_molecule = update_molecule_indices(sorted_molecule)
    print_molecule(sorted_molecule,
      "\n#{filename} with updated indices after sorting:")

    *previous_molecule_states, sorted_molecule = sort_elements_by_index_of_edges(sorted_molecule)
    print_molecule(sorted_molecule,
      "\n#{filename} with atoms of same kind and same number of edges sorted by indices of edges (increasing):")

    inspect_molecule_states(previous_molecule_states, sorted_molecule, filename)

    sorted_molecule
  end

  def write_ninchi_string(molecule, periodic_table_elements)
    sum_formula = write_sum_formula_string(molecule, periodic_table_elements)
    serialized_molecule = serialize_molecule(molecule)
    "nInChI=1S/#{sum_formula}/c#{serialized_molecule}"
  end

  def write_dot_file(molecule, filename, periodic_table_elements, periodic_table_colors)
    filename = File.basename(filename, '.mol')
    dotfile = "graph #{filename}\n{\n  bgcolor=grey\n"
    molecule.each_with_index do |atom, i|
      symbol = periodic_table_elements[atom[0][1]]
      color = periodic_table_colors.fetch(symbol, 'lightgrey')
      dotfile += "  #{i} [label=\"#{symbol} #{i}\" color=#{color},style=filled,shape=circle,fontname=Calibri];\n"
    end
    graph = compute_graph(molecule)
    graph.each { |line| dotfile += "  #{line[0]} -- #{line[1]} [color=black,style=bold];\n" if line[0] != line[1] }
    dotfile += "}\n"
  end

  # Helper methods ###################################################################################
  private

  def serialize_molecule(molecule)
    graph = compute_graph(molecule)
    inchi_string = ''
    graph.each do |line|
      inchi_string += "(#{line[0]}-#{line[1]})" if line[0] != line[1]
    end
    inchi_string
  end

  def compute_graph(molecule)
    graph = []
    molecule.each do |atom|
      element, edges = atom
      edges.each do |edge|
        graph.push([element[0], edge].sort)
      end
    end
    graph.uniq.sort
  end

  def write_sum_formula_string(molecule, periodic_table_elements)
    # Write sum formula in the order C > H > all other elements in alphabetic order.
    element_counts = compute_element_counts(molecule, periodic_table_elements)
    element_counts.transform_values! { |v| v > 1 ? v : '' } # remove 1s since counts of 1 are implicit in sum formula
    sum_formula_string = ''
    sum_formula_string += "C#{element_counts['C']}" if element_counts.key?('C')
    sum_formula_string += "H#{element_counts['H']}" if element_counts.key?('H')
    element_counts.sort.to_h.each do |element, count|
      sum_formula_string += "#{element}#{count}" unless %w[C H].include?(element)
    end
    sum_formula_string
  end

  def compute_element_counts(molecule, periodic_table_elements)
    # Compute hash table mapping element symbols to stoichiometric counts.
    unique_elements = molecule.map { |atom| atom[0][1] }.uniq
    initial_counts = Array.new(unique_elements.length, 0)
    element_counts = unique_elements.zip(initial_counts).to_h
    molecule.each { |atom| element_counts[atom[0][1]] += 1 }
    element_counts.transform_keys! { |k| periodic_table_elements[k] } # change atomic mass to element symbol
  end

  def sort_elements_by_index_of_edges(molecule)
    # Cannot use built-in sort since indices have to be updated after every swap,
    # rather than once after sorting is done (like with the other sorting steps).
    # This is because we sort by indices.
    molecule = sort_edges_by_index(molecule)
    n_iterations = molecule.size - 2
    previous_molecule_states = [Marshal.load(Marshal.dump(molecule))]
    sorted = false
    until sorted

      for i in 0..n_iterations
        atom_a = molecule[i]
        atom_b = molecule[i + 1]
        mass_a, mass_b = atom_a[0][1], atom_b[0][1]
        edge_indices_a = atom_a[1]
        edge_indices_b = atom_b[1]

        # Swap A and B (i.e., bubble up A) if ...
        if (mass_a == mass_b) && # A and B are the same element ...
          (edge_indices_a.length == edge_indices_b.length) && # with the same number of edges ...
          (edge_indices_a <=> edge_indices_b) == 1 # and A is connected to larger indices than B.
          # Spaceship operator (<=>) compares arrays pairwise element-by-element.
          # I.e., first compare the two elements at index 0, etc.. Result is determined by first unequal element pair.
          # The operator returns 1 if A > B, -1 if A < B, and 0 if A == B.
          molecule[i], molecule[i + 1] = molecule[i + 1], molecule[i]
          molecule = update_molecule_indices(molecule)
          molecule = sort_edges_by_index(molecule)
        end
      end
      sorted = previous_molecule_states.include?(molecule) ? true : false

      previous_molecule_states.push(Marshal.load(Marshal.dump(molecule)))
    end
    previous_molecule_states
  end

  def sort_edges_by_index(molecule)
    # Sort edges by index (decreasing order).
    # Note that the index of an edge corresponds to its atomic mass if the
    # entire molecule is sorted by atomic mass. Therefore, if the entire molecule
    # is sorted by atomic mass, this method sorts the edges by atomic mass as well
    # as by index.
    molecule.map do |atom|
      element, edges = atom
      [element, edges.sort.reverse]
    end
  end

  def sort_elements_by_number_of_edges(molecule)
    # Note that `each_with_index` returns position of atom in molecule array,
    # not atom ID (i.e., atom[0][0]).
    molecule.each_with_index.sort { |(atom_a, idx_a), (atom_b, idx_b)|

      mass_a = atom_a[0][1]
      mass_b = atom_b[0][1]
      edge_indices_a = atom_a[1]
      edge_indices_b = atom_b[1]

      order = 0 # assume unequal masses and hence no swap; note that sort is not stable if order = 0
      order = edge_indices_a.length <=> edge_indices_b.length if mass_a == mass_b # sort by vertex degree in case of equal masses
      order = idx_a <=> idx_b if order.zero? # sort by index in case of a) unequal masses or b) equal masses and equal vertex degree; this ensures stable sort
      order

    }.map(&:first) # discard index (second element)
  end

  def sort_elements_by_atomic_mass(molecule)
    # Note that `each_with_index` returns position of atom in molecule array,
    # not atom ID (i.e., atom[0][0]).
    molecule.each_with_index.sort { |(atom_a, idx_a), (atom_b, idx_b)|

      mass_a = atom_a[0][1]
      mass_b = atom_b[0][1]

      order = mass_a <=> mass_b
      order = idx_a <=> idx_b if order.zero? # sort by index in case of equal masses (i.e., order = 0); this ensures stable sort
      order

    }.map(&:first) # discard index (second element)
  end

  def update_molecule_indices(molecule, random_indices=false)
    index_updates = compute_index_updates(molecule, random_indices)
    updated_molecule = []
    molecule.each do |atom|
      element, edges = atom
      updated_element = update_element_index(element, index_updates)
      updated_edges = update_edge_indices(edges, index_updates)
      updated_molecule.push([updated_element, updated_edges])
    end
    updated_molecule
  end

  def update_element_index(element, index_updates)
    [index_updates[element[0]], element[1]]
  end

  def update_edge_indices(edges, index_updates)
    edges.map do |edge|
      index_updates[edge]
    end
  end

  def compute_index_updates(molecule, random_indices)
    current_indices = molecule.map { |atom| atom[0][0] }
    updated_indices = (0..molecule.length - 1).to_a
    updated_indices.shuffle! if random_indices
    current_indices.zip(updated_indices).to_h
  end

  def create_element_array(molfile_lines, atom_count, periodic_table_elements)
    elements = []
    (4..atom_count + 3).each_with_index do |atom_index, i|
      atom = molfile_lines[atom_index].split(' ')[3]
      elements.push([i, periodic_table_elements.index(atom)])
    end
    elements
  end

  def create_edge_array(molfile_lines, edge_count, atom_count)
    edges = Array.new(atom_count).map(&:to_a)
    (0..edge_count - 1).each do |edge_index|
      vertex1, vertex2 = parse_edge(molfile_lines[edge_index + 4 + atom_count])
      edges[vertex1].push(vertex2)    # add to the first atom of a bond
      edges[vertex2].push(vertex1)    # and to the second atom of the bond
    end
    edges
  end

  def parse_edge(molfile_line)
    vertex1, vertex2, * = molfile_line.split(' ').map { |i| i.to_i - 1 }
    vertex1, vertex2 = vertex2, vertex1 if vertex1 > vertex2    # make sure first atom always has lower (not: higher?) index
    [vertex1, vertex2]
  end

  def print_molecule(molecule, caption)
    puts caption
    puts "\nindex\tmass\tindices of connected atoms"
    puts "-----\t----\t--------------------------"
    molecule.each { |atom| puts "#{atom[0][0]}\t#{atom[0][1] + 1}\t#{atom[1]}" }
  end

  def inspect_molecule_states(previous_states, final_state, filename)
    puts "\nPrinting all molecule states of #{filename} that occured during sorting by indices of edges..."
    previous_states.each_with_index do |state, i|
      print_molecule(state,
        "\nIteration #{i} yielded the following state:")
    end
    print_molecule(final_state,
      "\nSorting converged in iteration #{previous_states.size} with re-occurence of state at iteration #{previous_states.index(final_state)}:")
  end

end

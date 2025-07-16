# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

def create_csp(graph, colours):
    """
    Prepares the CSP structure for the map colouring problem.

    Parameters:
    - graph: A dictionary of regions and their neighbours.
        Example:
        {
            'A': ['B', 'C'],
            'B': ['A', 'C'],
            'C': ['A', 'B']
        }

    - colours: A list of colours that can be used.
        Example: ['Red', 'Green', 'Blue']

    Returns:
    A dictionary with:
    - 'variables': list of region names
        Example: ['A', 'B', 'C']
    - 'domains': dict mapping each variable to its list of possible colours
        Example:
        {
            'A': ['Red', 'Green', 'Blue'],
            'B': ['Red', 'Green', 'Blue'],
            'C': ['Red', 'Green', 'Blue']
        }
    - 'neighbours': same as the input graph
    """
    variables = list(graph.keys())
    domains = {var: list(colours) for var in variables}
    neighbours = graph
    return {
        'variables': variables,
        'domains': domains,
        'neighbours': neighbours
    }


def can_assign_value(var, value, assignment, csp):
    """
    Checks if assigning 'value' to 'var' is valid based on current assignment.

    Parameters:
    - var: Variable to assign (e.g. 'A')
    - value: Colour to try (e.g. 'Red')
    - assignment: Current partial assignment
        Example: {'B': 'Green', 'C': 'Red'}
    - csp: The full CSP dictionary with:
        csp['neighbours'] = {
            'A': ['B', 'C'],
            ...
        }

    Returns:
    True if none of the neighbours of 'var' are already assigned 'value'.
    """
    for neighbour in csp['neighbours'][var]:
        if neighbour in assignment and assignment[neighbour] == value:
            return False  # Conflict found
    return True  # No conflicts


def backtrack(assignment, csp):
    """
    Recursively attempts to assign colours to all regions using backtracking.

    Parameters:
    - assignment: Dictionary of current assignments.
        Example: {'A': 'Red', 'B': 'Green'}
    - csp: CSP dictionary with 'variables', 'domains', and 'neighbours'.

    Returns:
    - A complete assignment if successful
    - None if no valid assignment is possible
    """
    # Check if all variables are assigned
    if len(assignment) == len(csp['variables']):
        return assignment

    # Choose the first unassigned variable
    unassigned_vars = [v for v in csp['variables'] if v not in assignment]
    var = unassigned_vars[0]
    # Example: var = 'C'

    for value in csp['domains'][var]:
        # Try 'Red', 'Green', 'Blue' for example
        if can_assign_value(var, value, assignment, csp):
            assignment[var] = value
            # Example assignment: {'A': 'Red', 'B': 'Green', 'C': 'Blue'}

            result = backtrack(assignment, csp)
            if result is not None:
                return result

            # Backtrack if result was failure
            del assignment[var]

    return None  # All colours failed, need to backtrack


def backtracking_search(csp):
    """
    Entry point for plain backtracking CSP solver.

    Returns:
    - Complete assignment if possible, or None if unsolvable
    """
    return backtrack({}, csp)


if __name__ == "__main__":
    # Define a simple triangle graph
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'C'],
        'C': ['A', 'B']
    }
    Aus_graph = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW'],
        'Q': ['NT', 'SA', 'NSW'],
        'NSW': ['SA', 'Q', 'V'],
        'V': ['NSW']
    }

    # List of allowed colours
    colours = ['Red', 'Green', 'Blue']

    # Prepare CSP
    csp = create_csp(Aus_graph, colours)


    # Run the solver
    solution = backtracking_search(csp)

    print("Map Colouring Solution:")
    print(solution)

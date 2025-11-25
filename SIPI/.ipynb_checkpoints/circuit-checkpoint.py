from .utils import time_it
import re


# ============================================================
# UNIT TEST FUNCTIONS
# ============================================================

@time_it
def unit_test_capacitor_between_nets(graph, net_a, net_b, value_pattern, min_count=1, test_name="", verbose=True):
    """
    Unit test: Check if there are at least min_count capacitors with specific value between two nets.
    
    Args:
        graph: CircuitGraph instance (ComponentGraph or NetGraph)
        net_a: First net name
        net_b: Second net name
        value_pattern: Regex pattern for capacitor value (e.g., r'100UF', r'47UF')
        min_count: Minimum number of capacitors required
        test_name: Optional test name for reporting
        verbose: If True, print detailed output
    
    Returns:
        tuple: (passed: bool, count: int, components: list)
    """
    if verbose:
        test_desc = test_name or f'{net_a}-C({value_pattern})-{net_b}'
        print(f"\n🧪 Unit Test: {test_desc}")
    
    # Get all components between the two nets
    bridging_components = graph.query_components_between_two_nets(net_a, net_b)
    
    if not bridging_components:
        if verbose:
            print(f"   ⚠️  No components found between {net_a} and {net_b}")
        return False, 0, []
    
    # Filter for capacitors matching the pattern
    matching_capacitors = []
    for comp in bridging_components:
        # Check if it's a capacitor (REFDES starts with C followed by digits)
        if not re.match(r'^C\d+$', comp):
            continue
        
        # Get component type
        exists, data = graph.query_component_exists(comp)
        if not exists:
            continue
        
        comp_type = data.get('comp_type', '')
        
        # Check if comp_type matches the value pattern
        if re.search(value_pattern, comp_type, re.IGNORECASE):
            matching_capacitors.append(comp)
    
    count = len(matching_capacitors)
    passed = count >= min_count
    
    if verbose:
        print(f"   Bridging components: {len(bridging_components)}")
        print(f"   Capacitors with pattern '{value_pattern}': {count}")
        print(f"   Required: {min_count}, Actual: {count}")
        if matching_capacitors:
            print(f"   Found: {matching_capacitors[:5]}{'...' if len(matching_capacitors) > 5 else ''}")
        print(f"   {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed, count, matching_capacitors


@time_it
def unit_test_capacitor_exists_between_nets(graph, net_a, net_b, value_pattern, test_name="", verbose=True):
    """
    Unit test: Check if at least one capacitor with specific value exists between two nets.
    Wrapper for unit_test_capacitor_between_nets with min_count=1
    
    Args:
        graph: CircuitGraph instance
        net_a: First net name
        net_b: Second net name
        value_pattern: Regex pattern for capacitor value
        test_name: Optional test name
        verbose: If True, print detailed output
    
    Returns:
        tuple: (passed: bool, count: int, components: list)
    """
    return unit_test_capacitor_between_nets(graph, net_a, net_b, value_pattern, min_count=1, test_name=test_name, verbose=verbose)
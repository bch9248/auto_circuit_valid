"""
Query 001 Methods: Circuit connection checking with capacitor rules

This module provides three query functions:

1. query001(graph, net_a, net_b)
   - Direct query with specified net_a and net_b
   - Tests: net_a-C(100uF OR 3*47uF)-net_b AND net_a-C(0.1uF AND 18pF)-net_b

2. query001_general(graph, net_a_pattern, net_b_pattern)
   - Supports wildcard patterns for net names
   - Tests all combinations of matching nets

3. query001_a(graph, component, pin2, pin4, net_b_candidates)
   - Automatic net_a discovery from component pins
   - Rule 1: component[pin2]-net_a-C(100uF OR 3*47uF)-net_b
   - Rule 2: component[pin4]-net_a-C(0.1uF AND 18pF)-net_b
   - Both rules must pass for overall success
   - Example: CN1401[2] and CN1401[4] pins to find PVSIM connections
"""

from ..utils import parse_component_pin_report, time_it
import fnmatch
from SIPI.circuit import unit_test_capacitor_between_nets, unit_test_capacitor_exists_between_nets
from typing import Dict, Any


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_query001_a(graph, platform: str, verbose: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Preprocessing function for query001_a.
    Automatically determines the component based on platform mapping.
    
    Args:
        graph: CircuitGraph instance
        platform: Platform name for component mapping
        verbose: Print verbose output
        **kwargs: Existing kwargs to be augmented
    
    Returns:
        Updated kwargs dictionary
    """
    from SIPI.Net_name import get_wwan_component
    
    # Get component from platform mapping if not provided
    if 'component' not in kwargs:
        component = get_wwan_component(platform)
        kwargs['component'] = component
        if verbose:
            print(f"   Platform: {platform}")
            print(f"   Component (from platform): {component}")
    
    # Add default pin values if not provided
    if 'pin2' not in kwargs:
        kwargs['pin2'] = 2
    if 'pin4' not in kwargs:
        kwargs['pin4'] = 4
    if 'net_b_candidates' not in kwargs:
        kwargs['net_b_candidates'] = ['DGND', 'WWAN_GND*', '*GND']
    
    return kwargs

@time_it
def query001(graph, net_a='PVSIM', net_b='DGND', verbose=True):
    """
    Query 001: Combined rules for specific nets
    
    Rule 1: net_a-C(100uF OR 3*47uF)-net_b
    Rule 2: net_a-C(0.1uF AND 18pF)-net_b
    
    Args:
        graph: CircuitGraph instance
        net_a: First net (default: 'PVSIM')
        net_b: Second net (default: 'DGND')
        verbose: If True, print detailed output
    
    Returns:
        dict: Test results with status and details
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 QUERY 001: Combined rules for {net_a}-{net_b}")
        print(f"{'='*80}")
    
    # ============================================================
    # SUB-QUERY 1: C(100uF OR 3*47uF)
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📋 Sub-Query 1: {net_a}-C(100uF OR 3*47uF)-{net_b}")
        print(f"{'─'*80}")
    
    # Test 1.1: Check for at least 1x 100UF capacitor
    test1_1_passed, test1_1_count, test1_1_caps = unit_test_capacitor_between_nets(
        graph, net_a, net_b, 
        value_pattern=r'100\s*UF',
        min_count=1,
        test_name="Test 1.1: At least 1x 100UF capacitor",
        verbose=verbose
    )
    
    # Test 1.2: Check for at least 3x 47UF capacitors
    test1_2_passed, test1_2_count, test1_2_caps = unit_test_capacitor_between_nets(
        graph, net_a, net_b,
        value_pattern=r'47\s*UF',
        min_count=3,
        test_name="Test 1.2: At least 3x 47UF capacitors",
        verbose=verbose
    )
    
    # Sub-query 1 result: OR logic
    subquery1_passed = test1_1_passed or test1_2_passed
    
    if verbose:
        print(f"\n   Sub-Query 1 Results:")
        print(f"   Test 1.1 (100UF):   {'✅ PASSED' if test1_1_passed else '❌ FAILED'} ({test1_1_count} found)")
        print(f"   Test 1.2 (3x 47UF): {'✅ PASSED' if test1_2_passed else '❌ FAILED'} ({test1_2_count} found)")
        print(f"   Sub-Query 1 (OR):   {'✅ PASSED' if subquery1_passed else '❌ FAILED'}")
    
    # ============================================================
    # SUB-QUERY 2: C(0.1uF AND 18pF)
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📋 Sub-Query 2: {net_a}-C(0.1uF AND 18pF)-{net_b}")
        print(f"{'─'*80}")
    
    # Test 2.1: Check for at least 1x 0.1UF capacitor
    test2_1_passed, test2_1_count, test2_1_caps = unit_test_capacitor_between_nets(
        graph, net_a, net_b, 
        value_pattern=r'0\.1\s*UF',
        min_count=1,
        test_name="Test 2.1: At least 1x 0.1UF capacitor",
        verbose=verbose
    )
    
    # Test 2.2: Check for at least 1x 18PF capacitor
    test2_2_passed, test2_2_count, test2_2_caps = unit_test_capacitor_between_nets(
        graph, net_a, net_b,
        value_pattern=r'18\s*PF',
        min_count=1,
        test_name="Test 2.2: At least 1x 18PF capacitor",
        verbose=verbose
    )
    
    # Sub-query 2 result: AND logic
    subquery2_passed = test2_1_passed and test2_2_passed
    
    if verbose:
        print(f"\n   Sub-Query 2 Results:")
        print(f"   Test 2.1 (0.1UF): {'✅ PASSED' if test2_1_passed else '❌ FAILED'} ({test2_1_count} found)")
        print(f"   Test 2.2 (18PF):  {'✅ PASSED' if test2_2_passed else '❌ FAILED'} ({test2_2_count} found)")
        print(f"   Sub-Query 2 (AND): {'✅ PASSED' if subquery2_passed else '❌ FAILED'}")
    
    # ============================================================
    # OVERALL RESULT
    # ============================================================
    
    # Overall: Both sub-queries must pass
    overall_passed = subquery1_passed and subquery2_passed
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 QUERY 001 FINAL RESULTS:")
        print(f"   Sub-Query 1 (100uF OR 3*47uF): {'✅ PASSED' if subquery1_passed else '❌ FAILED'}")
        print(f"   Sub-Query 2 (0.1uF AND 18pF):  {'✅ PASSED' if subquery2_passed else '❌ FAILED'}")
        print(f"   Overall (AND):                  {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print(f"{'='*80}\n")
    
    return {
        'query_name': 'query_001',
        'net_a': net_a,
        'net_b': net_b,
        'rule': f'{net_a}-C((100uF OR 3*47uF) AND (0.1uF AND 18pF))-{net_b}',
        'overall_passed': overall_passed,
        'subquery1': {
            'description': '100uF OR 3*47uF',
            'logic': 'OR',
            'passed': subquery1_passed,
            'test1': {
                'description': '1x 100UF',
                'pattern': r'100\s*UF',
                'required': 1,
                'passed': test1_1_passed,
                'count': test1_1_count,
                'components': test1_1_caps
            },
            'test2': {
                'description': '3x 47UF',
                'pattern': r'47\s*UF',
                'required': 3,
                'passed': test1_2_passed,
                'count': test1_2_count,
                'components': test1_2_caps
            }
        },
        'subquery2': {
            'description': '0.1uF AND 18pF',
            'logic': 'AND',
            'passed': subquery2_passed,
            'test1': {
                'description': '1x 0.1UF',
                'pattern': r'0\.1\s*UF',
                'required': 1,
                'passed': test2_1_passed,
                'count': test2_1_count,
                'components': test2_1_caps
            },
            'test2': {
                'description': '1x 18PF',
                'pattern': r'18\s*PF',
                'required': 1,
                'passed': test2_2_passed,
                'count': test2_2_count,
                'components': test2_2_caps
            }
        }
    }



@time_it
def query001_general(graph, net_a_pattern='PVSIM', net_b_pattern='DGND',verbose=True):
    """
    Query 001: Combined rules with wildcard support for net_a and net_b
    
    Rule 1: net_a-C(100uF OR 3*47uF)-net_b
    Rule 2: net_a-C(0.1uF AND 18pF)-net_b
    
    Args:
        graph: CircuitGraph instance
        net_a_pattern: Net name or wildcard pattern (e.g., 'PVSIM', 'P*', 'PV*')
        searching_scope_net_a: List of nets to search for net_a (overrides pattern)
        net_b: Second net name or wildcard pattern (default: 'DGND')
        searching_scope_net_b: List of nets to search for net_b (overrides net_b pattern)
        verbose: If True, print detailed output
    
    Returns:
        dict: Test results with status and details
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"{'='*80}")
    
    # Handle net_a matching
    if type(net_a_pattern)==list:        
        matching_nets_a = net_a_pattern
    elif type(net_a_pattern)==str and ('*' in net_a_pattern or '?' in net_a_pattern):
        # Convert shell-style wildcard to regex
        regex_pattern = fnmatch.translate(net_a_pattern)
        matching_nets_a = graph.query_nets_by_pattern(regex_pattern)
        
        if verbose:
            print(f"\n   Found {len(matching_nets_a)} nets matching pattern '{net_a_pattern}'")
            print(f"   Sample nets: {matching_nets_a[:10]}")
        
        if not matching_nets_a:
            print(f"   ⚠️  No nets found matching pattern '{net_a_pattern}'")
            return {
                'query_name': 'query001_general',
                'pattern': net_a_pattern,
                'nets_found': [],
                'overall_passed': False,
                'results': []
            }
    else:
        # Direct net name
        matching_nets_a = [net_a_pattern]
    
    # Handle net_b matching
    if type(net_b_pattern)==list:
        matching_nets_b = net_b_pattern
    elif type(net_b_pattern)==str and ('*' in net_b_pattern or '?' in net_b_pattern):
        # Convert shell-style wildcard to regex
        regex_pattern_b = fnmatch.translate(net_b_pattern)
        matching_nets_b = graph.query_nets_by_pattern(regex_pattern_b)
        
        if verbose:
            print(f"\n   Found {len(matching_nets_b)} nets matching pattern '{net_b_pattern}'")
            print(f"   Sample nets: {matching_nets_b[:10]}")
        
        if not matching_nets_b:
            print(f"   ⚠️  No nets found matching pattern '{net_b_pattern}'")
            return {
                'query_name': 'query001_general',
                'pattern': net_a_pattern,
                'pattern_b': net_b_pattern,
                'nets_found_a': matching_nets_a,
                'nets_found_b': [],
                'overall_passed': False,
                'results': []
            }
    else:
        # Direct net name
        matching_nets_b = [net_b_pattern]
    
    # Run query001 for each combination of matching nets
    all_results = []
    if verbose:
        print(f"matching_nets_a: {matching_nets_a}")
        print(f"matching_nets_b: {matching_nets_b}")
        print(f"\nSearching combinations: {len(matching_nets_a)} net_a × {len(matching_nets_b)} net_b = {len(matching_nets_a) * len(matching_nets_b)} combinations")
    
    for net_a in matching_nets_a:
        for net_b_current in matching_nets_b:
            if verbose:
                print(f"\n{'─'*80}")
                print(f"Testing net pair: {net_a} → {net_b_current}")
                print(f"{'─'*80}")
            
            result = query001(graph, net_a, net_b_current, verbose=verbose)
            all_results.append({
                'net_a': net_a,
                'net_b': net_b_current,
                'result': result
            })
    
    # Aggregate results
    any_passed = any(r['result']['overall_passed'] for r in all_results)
    all_passed = all(r['result']['overall_passed'] for r in all_results)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 AGGREGATE RESULTS for pattern '{net_a_pattern}' → '{net_b_pattern}':")
        print(f"   Total combinations tested: {len(all_results)}")
        print(f"   Passed: {sum(1 for r in all_results if r['result']['overall_passed'])}")
        print(f"   Failed: {sum(1 for r in all_results if not r['result']['overall_passed'])}")
        print(f"   Any passed: {'✅ YES' if any_passed else '❌ NO'}")
        print(f"   All passed: {'✅ YES' if all_passed else '❌ NO'}")
        print(f"{'='*80}\n")
    
    return {
        'query_name': 'query001_general',
        'pattern': net_a_pattern,
        'pattern_b': net_b_pattern,
        'nets_found_a': matching_nets_a,
        'nets_found_b': matching_nets_b,
        'nets_tested': len(all_results),
        'any_passed': any_passed,
        'all_passed': all_passed,
        'overall_passed': any_passed,  # Consider query passed if ANY combination passes
        'results': all_results
    }


@time_it
def query001_a(graph, component='CN1401', pin2=2, pin4=4, net_b_candidates=None, verbose=True):
    """
    Query 001_a: Connection check with automatic net_a discovery from component pins
    
    Rules:
    1. component[pin2]-net_a-C(100uF OR 3*47uF)-net_b
    2. component[pin4]-net_a-C(0.1uF AND 18pF)-net_b
    
    Both rules must pass for overall success.
    net_a is discovered by querying the component pins.
    net_b is searched from candidate ground nets.
    
    Args:
        graph: CircuitGraph instance
        component: Component REFDES (default: 'CN1401')
        pin2: Pin number for first rule (default: 2)
        pin4: Pin number for second rule (default: 4)
        net_b_candidates: List of candidate ground net names or patterns (default: ['DGND', '*GND*'])
        verbose: If True, print detailed output
    
    Returns:
        dict: Test results with status and details
    """
    if net_b_candidates is None:
        net_b_candidates = ['DGND', 'WWAN_GND*', '*GND']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 QUERY 001_a: Connection check with automatic net discovery")
        print(f"   Component: {component}")
        print(f"   Pin 2: {pin2} (for 100uF OR 3*47uF rule)")
        print(f"   Pin 4: {pin4} (for 0.1uF AND 18pF rule)")
        print(f"   Ground candidates: {net_b_candidates}")
        print(f"{'='*80}")
    
    # ============================================================
    # STEP 1: Find net_a for pin 2 (Rule 1)
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📍 STEP 1: Finding net connected to {component}[{pin2}]")
        print(f"{'─'*80}")
    
    net_a_pin2 = graph.query_net_by_component_pin(component, pin2)
    
    if net_a_pin2 is None:
        if verbose:
            print(f"   ❌ ERROR: No net found for {component}[{pin2}]")
            print(f"   Cannot proceed with Rule 1")
        return {
            'query_name': 'query001_a',
            'component': component,
            'pin2': pin2,
            'pin4': pin4,
            'net_a_pin2': None,
            'net_a_pin4': None,
            'overall_passed': False,
            'error': f'No net found for {component}[{pin2}]'
        }
    
    if verbose:
        print(f"   ✅ Found net_a for pin {pin2}: {net_a_pin2}")
    
    # ============================================================
    # STEP 2: Find net_a for pin 4 (Rule 2)
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📍 STEP 2: Finding net connected to {component}[{pin4}]")
        print(f"{'─'*80}")
    
    net_a_pin4 = graph.query_net_by_component_pin(component, pin4)
    
    if net_a_pin4 is None:
        if verbose:
            print(f"   ❌ ERROR: No net found for {component}[{pin4}]")
            print(f"   Cannot proceed with Rule 2")
        return {
            'query_name': 'query001_a',
            'component': component,
            'pin2': pin2,
            'pin4': pin4,
            'net_a_pin2': net_a_pin2,
            'net_a_pin4': None,
            'overall_passed': False,
            'error': f'No net found for {component}[{pin4}]'
        }
    
    if verbose:
        print(f"   ✅ Found net_a for pin {pin4}: {net_a_pin4}")
    
    # ============================================================
    # STEP 3: Expand net_b candidates (handle wildcards)
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📍 STEP 3: Expanding ground net candidates")
        print(f"{'─'*80}")
    
    expanded_net_b = []
    for candidate in net_b_candidates:
        if '*' in candidate or '?' in candidate:
            # Wildcard pattern - expand it
            regex_pattern = fnmatch.translate(candidate)
            matching_nets = graph.query_nets_by_pattern(regex_pattern)
            expanded_net_b.extend(matching_nets)
            if verbose:
                print(f"   Pattern '{candidate}' → {len(matching_nets)} nets")
        else:
            # Direct net name
            expanded_net_b.append(candidate)
            if verbose:
                print(f"   Direct net: {candidate}")
    
    # Remove duplicates
    expanded_net_b = list(set(expanded_net_b))
    
    if verbose:
        print(f"   Total ground candidates: {len(expanded_net_b)}")
        print(f"   Sample: {expanded_net_b[:10]}")
    
    if not expanded_net_b:
        if verbose:
            print(f"   ❌ ERROR: No ground nets found matching candidates")
        return {
            'query_name': 'query001_a',
            'component': component,
            'pin2': pin2,
            'pin4': pin4,
            'net_a_pin2': net_a_pin2,
            'net_a_pin4': net_a_pin4,
            'net_b_candidates': net_b_candidates,
            'expanded_net_b': [],
            'overall_passed': False,
            'error': 'No ground nets found'
        }
    
    # ============================================================
    # STEP 4: Test Rule 1 - component[pin2]-net_a-C(100uF OR 3*47uF)-net_b
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📋 STEP 4: Testing Rule 1")
        print(f"   {component}[{pin2}]-{net_a_pin2}-C(100uF OR 3*47uF)-net_b")
        print(f"{'─'*80}")
    
    rule1_results = []
    rule1_passed_any = False
    
    for net_b in expanded_net_b:
        if verbose:
            print(f"\n   Testing ground: {net_b}")
        
        # Test 1.1: 100UF
        test1_1_passed, test1_1_count, test1_1_caps = unit_test_capacitor_between_nets(
            graph, net_a_pin2, net_b,
            value_pattern=r'100\s*UF',
            min_count=1,
            test_name=f"   Rule 1.1: {net_a_pin2}-C(100UF)-{net_b}",
            verbose=verbose
        )
        
        # Test 1.2: 3x 47UF
        test1_2_passed, test1_2_count, test1_2_caps = unit_test_capacitor_between_nets(
            graph, net_a_pin2, net_b,
            value_pattern=r'47\s*UF',
            min_count=3,
            test_name=f"   Rule 1.2: {net_a_pin2}-C(3x47UF)-{net_b}",
            verbose=verbose
        )
        
        # OR logic for this ground net
        passed = test1_1_passed or test1_2_passed
        rule1_results.append({
            'net_b': net_b,
            'passed': passed,
            'test1_1': {'passed': test1_1_passed, 'count': test1_1_count, 'components': test1_1_caps},
            'test1_2': {'passed': test1_2_passed, 'count': test1_2_count, 'components': test1_2_caps}
        })
        
        if passed:
            rule1_passed_any = True
            if verbose:
                print(f"   ✅ Rule 1 PASSED for {net_b}")
    
    # ============================================================
    # STEP 5: Test Rule 2 - component[pin4]-net_a-C(0.1uF AND 18pF)-net_b
    # ============================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print(f"📋 STEP 5: Testing Rule 2")
        print(f"   {component}[{pin4}]-{net_a_pin4}-C(0.1uF AND 18pF)-net_b")
        print(f"{'─'*80}")
    
    rule2_results = []
    rule2_passed_any = False
    
    for net_b in expanded_net_b:
        if verbose:
            print(f"\n   Testing ground: {net_b}")
        
        # Test 2.1: 0.1UF
        test2_1_passed, test2_1_count, test2_1_caps = unit_test_capacitor_between_nets(
            graph, net_a_pin4, net_b,
            value_pattern=r'0\.1\s*UF',
            min_count=1,
            test_name=f"   Rule 2.1: {net_a_pin4}-C(0.1UF)-{net_b}",
            verbose=verbose
        )
        
        # Test 2.2: 18PF
        test2_2_passed, test2_2_count, test2_2_caps = unit_test_capacitor_between_nets(
            graph, net_a_pin4, net_b,
            value_pattern=r'18\s*PF',
            min_count=1,
            test_name=f"   Rule 2.2: {net_a_pin4}-C(18PF)-{net_b}",
            verbose=verbose
        )
        
        # AND logic for this ground net
        passed = test2_1_passed and test2_2_passed
        rule2_results.append({
            'net_b': net_b,
            'passed': passed,
            'test2_1': {'passed': test2_1_passed, 'count': test2_1_count, 'components': test2_1_caps},
            'test2_2': {'passed': test2_2_passed, 'count': test2_2_count, 'components': test2_2_caps}
        })
        
        if passed:
            rule2_passed_any = True
            if verbose:
                print(f"   ✅ Rule 2 PASSED for {net_b}")
    
    # ============================================================
    # STEP 6: Overall result - Both rules must pass
    # ============================================================
    
    overall_passed = rule1_passed_any and rule2_passed_any
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 QUERY 001_a FINAL RESULTS:")
        print(f"   Component: {component}")
        print(f"   Pin {pin2} → {net_a_pin2}")
        print(f"   Pin {pin4} → {net_a_pin4}")
        print(f"   Ground candidates tested: {len(expanded_net_b)}")
        print(f"   Rule 1 ({net_a_pin2}-C(100uF OR 3*47uF)-GND): {'✅ PASSED' if rule1_passed_any else '❌ FAILED'}")
        print(f"   Rule 2 ({net_a_pin4}-C(0.1uF AND 18pF)-GND):  {'✅ PASSED' if rule2_passed_any else '❌ FAILED'}")
        print(f"   Overall (Both rules must pass):                {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print(f"{'='*80}\n")
    
    return {
        'query_name': 'query001_a',
        'component': component,
        'pin2': pin2,
        'pin4': pin4,
        'net_a_pin2': net_a_pin2,
        'net_a_pin4': net_a_pin4,
        'net_b_candidates': net_b_candidates,
        'expanded_net_b': expanded_net_b,
        'rule1': {
            'description': f'{component}[{pin2}]-{net_a_pin2}-C(100uF OR 3*47uF)-net_b',
            'logic': 'OR',
            'passed_any': rule1_passed_any,
            'results': rule1_results
        },
        'rule2': {
            'description': f'{component}[{pin4}]-{net_a_pin4}-C(0.1uF AND 18pF)-net_b',
            'logic': 'AND',
            'passed_any': rule2_passed_any,
            'results': rule2_results
        },
        'overall_passed': overall_passed
    }


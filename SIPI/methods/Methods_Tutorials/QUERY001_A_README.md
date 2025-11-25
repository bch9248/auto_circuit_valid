# Query 001_a: Automatic Net Discovery for Connection Checking

## Overview

The `query001_a` function extends the original `query001` functionality by automatically discovering the `net_a` values from component pins, eliminating the need to manually specify net names.

## Problem Statement

Original rules require explicit net names:
```
CN1401[2]-PVSIM 
CN1401[4]-PVSIM 
PVSIM-C(100uF or 3*47uF)-DGND 
PVSIM-C(0.1uF AND 18pF)-DGND
```

But in practice:
1. `net_a` should be discovered by querying the component pins
2. `net_b` should be searched from a list of candidate ground nets

## Solution

`query001_a` implements the following logic:

### Rule 1: `component[pin2]-net_a-C(100uF OR 3*47uF)-net_b`
- Find `net_a` by querying `component` pin `pin2` (e.g., CN1401[2])
- Search for capacitors with value 100µF OR 3×47µF
- Test against all candidate ground nets in `net_b_candidates`

### Rule 2: `component[pin4]-net_a-C(0.1uF AND 18pF)-net_b`
- Find `net_a` by querying `component` pin `pin4` (e.g., CN1401[4])
- Search for capacitors with value 0.1µF AND 18pF
- Test against all candidate ground nets in `net_b_candidates`

### Overall Pass Condition
**Both Rule 1 AND Rule 2 must pass** for the query to succeed.

## Function Signature

```python
def query001_a(
    graph, 
    component='CN1401', 
    pin2=2, 
    pin4=4, 
    net_b_candidates=None, 
    verbose=True
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | CircuitGraph | Required | Circuit graph instance |
| `component` | str | 'CN1401' | Component REFDES to query |
| `pin2` | int/str | 2 | Pin number for Rule 1 (100µF OR 3×47µF) |
| `pin4` | int/str | 4 | Pin number for Rule 2 (0.1µF AND 18pF) |
| `net_b_candidates` | list | ['DGND', 'WWAN_GND*', '*GND'] | Ground net candidates (supports wildcards) |
| `verbose` | bool | True | Enable detailed output |

## Return Value

Returns a dictionary with:
```python
{
    'query_name': 'query001_a',
    'component': str,           # Component REFDES
    'pin2': int/str,           # Pin number for Rule 1
    'pin4': int/str,           # Pin number for Rule 2
    'net_a_pin2': str,         # Net discovered from pin2
    'net_a_pin4': str,         # Net discovered from pin4
    'net_b_candidates': list,  # Original ground candidates
    'expanded_net_b': list,    # Expanded ground nets (wildcards resolved)
    'rule1': {
        'description': str,
        'logic': 'OR',
        'passed_any': bool,
        'results': [...]       # Results for each ground net
    },
    'rule2': {
        'description': str,
        'logic': 'AND',
        'passed_any': bool,
        'results': [...]       # Results for each ground net
    },
    'overall_passed': bool     # True if both rules pass
}
```

## Usage Examples

### Example 1: Basic Usage (Default Parameters)

```python
from SIPI.graph import CircuitGraph
from SIPI.methods.query001 import query001_a

# Load graph
graph = CircuitGraph.load('output/component_graph.pkl')

# Run query with defaults
result = query001_a(graph)

print(f"Overall passed: {result['overall_passed']}")
print(f"Net from CN1401[2]: {result['net_a_pin2']}")
print(f"Net from CN1401[4]: {result['net_a_pin4']}")
```

### Example 2: Custom Component and Pins

```python
result = query001_a(
    graph,
    component='CN1402',
    pin2=1,
    pin4=3,
    verbose=True
)
```

### Example 3: Specific Ground Nets (No Wildcards)

```python
result = query001_a(
    graph,
    component='CN1401',
    net_b_candidates=['DGND', 'AGND', 'SGND'],
    verbose=True
)
```

### Example 4: Wildcard Ground Net Patterns

```python
result = query001_a(
    graph,
    component='CN1401',
    net_b_candidates=['*GND', 'WWAN_GND*', 'DGND'],  # Supports wildcards
    verbose=True
)
```

### Example 5: Silent Mode (No Output)

```python
result = query001_a(graph, verbose=False)

if result['overall_passed']:
    print("✅ All rules passed!")
else:
    print("❌ Some rules failed")
    print(f"Rule 1: {result['rule1']['passed_any']}")
    print(f"Rule 2: {result['rule2']['passed_any']}")
```

## Algorithm Flow

```
1. Find net_a for pin2
   ├─ Query: graph.query_net_by_component_pin(component, pin2)
   └─ Store: net_a_pin2

2. Find net_a for pin4
   ├─ Query: graph.query_net_by_component_pin(component, pin4)
   └─ Store: net_a_pin4

3. Expand ground net candidates
   ├─ For each pattern in net_b_candidates:
   │  ├─ If wildcard: expand using graph.query_nets_by_pattern()
   │  └─ If direct: use as-is
   └─ Remove duplicates → expanded_net_b

4. Test Rule 1: net_a_pin2-C(100uF OR 3*47uF)-net_b
   ├─ For each net_b in expanded_net_b:
   │  ├─ Test 1.1: Check for ≥1 × 100µF capacitor
   │  ├─ Test 1.2: Check for ≥3 × 47µF capacitors
   │  └─ Pass if (Test 1.1 OR Test 1.2)
   └─ Rule 1 passes if ANY net_b passes

5. Test Rule 2: net_a_pin4-C(0.1uF AND 18pF)-net_b
   ├─ For each net_b in expanded_net_b:
   │  ├─ Test 2.1: Check for ≥1 × 0.1µF capacitor
   │  ├─ Test 2.2: Check for ≥1 × 18pF capacitor
   │  └─ Pass if (Test 2.1 AND Test 2.2)
   └─ Rule 2 passes if ANY net_b passes

6. Overall Result
   └─ Pass if (Rule 1 AND Rule 2)
```

## Key Graph Methods Used

### `query_net_by_component_pin(refdes, pin_name)`
Finds the net connected to a specific component pin.

```python
net = graph.query_net_by_component_pin('CN1401', '2')
# Returns: 'PVSIM' (or None if not found)
```

### `query_nets_by_pattern(regex_pattern)`
Finds all nets matching a regex pattern.

```python
import fnmatch
pattern = fnmatch.translate('WWAN_GND*')
nets = graph.query_nets_by_pattern(pattern)
# Returns: ['WWAN_GND', 'WWAN_GND_1', 'WWAN_GND_2', ...]
```

## Error Handling

### No Net Found for Pin
If a pin is not connected to any net:
```python
{
    'overall_passed': False,
    'error': 'No net found for CN1401[2]'
}
```

### No Ground Nets Found
If no ground nets match the candidates:
```python
{
    'overall_passed': False,
    'error': 'No ground nets found'
}
```

## Comparison with Original Functions

| Feature | `query001` | `query001_general` | `query001_a` |
|---------|------------|-------------------|--------------|
| Net discovery | Manual | Manual | Automatic from pins |
| Wildcard support | ❌ | ✅ | ✅ |
| Multiple net_a | ❌ | ✅ | ✅ (from 2 pins) |
| Component pin query | ❌ | ❌ | ✅ |
| Use case | Fixed nets | Pattern matching | Pin-based discovery |

## Testing

Run the test script:
```bash
cd /home/jovyan/project/SIPI
python test_query001_a.py
```

This will:
1. Load the pre-built graph
2. Test default parameters (CN1401 pins 2 and 4)
3. Test custom ground candidates
4. Test different component (if available)

## Integration with Retrieval System

In your RAG pipeline:

```python
# 1. Retrieval model finds this rule from corpus
rule = "CN1401[2]-PVSIM and CN1401[4]-PVSIM with capacitor checks"

# 2. Extract parameters
component = 'CN1401'
pin2 = 2
pin4 = 4
ground_candidates = ['DGND', 'WWAN_GND*', '*GND']

# 3. Execute query
result = query001_a(
    graph,
    component=component,
    pin2=pin2,
    pin4=pin4,
    net_b_candidates=ground_candidates,
    verbose=False
)

# 4. Return result
return {
    'rule': rule,
    'passed': result['overall_passed'],
    'details': result
}
```

## Notes

1. **Pin Discovery**: Uses `graph.pin_to_net_map` for O(1) lookup speed
2. **Wildcard Expansion**: Supports shell-style wildcards (`*`, `?`)
3. **OR vs AND Logic**:
   - Rule 1 uses OR: (100µF OR 3×47µF)
   - Rule 2 uses AND: (0.1µF AND 18pF)
4. **Multiple Ground Nets**: Tests all candidates, passes if ANY works
5. **Both Rules Required**: Overall pass requires BOTH Rule 1 AND Rule 2

## Future Enhancements

- [ ] Support for variable pin counts (not just pin2 and pin4)
- [ ] Configurable capacitor value patterns
- [ ] Parallel testing of multiple components
- [ ] Caching of expanded ground nets
- [ ] Export results to structured report format

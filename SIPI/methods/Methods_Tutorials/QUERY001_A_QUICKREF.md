# Quick Reference: query001_a

## Function Signature
```python
query001_a(graph, component='CN1401', pin2=2, pin4=4, net_b_candidates=None, verbose=True)
```

## What It Does
Automatically discovers nets from component pins and checks capacitor connections.

## Rules Tested
1. **Rule 1**: `component[pin2] → net_a → C(100µF OR 3×47µF) → net_b`
2. **Rule 2**: `component[pin4] → net_a → C(0.1µF AND 18pF) → net_b`

Both must pass for overall success.

## Quick Start
```python
from SIPI.graph import CircuitGraph
from SIPI.methods.query001 import query001_a

# Load graph
graph = CircuitGraph.load('output/component_graph.pkl')

# Run with defaults (CN1401, pins 2 and 4)
result = query001_a(graph)

# Check result
print(result['overall_passed'])  # True/False
```

## Common Use Cases

### 1. Default (CN1401 with standard grounds)
```python
result = query001_a(graph)
```

### 2. Custom Component
```python
result = query001_a(graph, component='CN1402')
```

### 3. Different Pins
```python
result = query001_a(graph, component='CN1401', pin2=1, pin4=3)
```

### 4. Specific Grounds Only
```python
result = query001_a(
    graph,
    net_b_candidates=['DGND', 'AGND', 'SGND']
)
```

### 5. Wildcard Grounds
```python
result = query001_a(
    graph,
    net_b_candidates=['*GND', 'WWAN_*']
)
```

### 6. Silent Mode
```python
result = query001_a(graph, verbose=False)
```

## Return Value Structure
```python
{
    'overall_passed': bool,          # ✅ Main result
    'net_a_pin2': str,              # Net from pin 2
    'net_a_pin4': str,              # Net from pin 4
    'expanded_net_b': [str],        # All ground nets tested
    'rule1': {
        'passed_any': bool,         # Rule 1 result
        'results': [...]            # Per-ground details
    },
    'rule2': {
        'passed_any': bool,         # Rule 2 result
        'results': [...]            # Per-ground details
    }
}
```

## Check Results
```python
result = query001_a(graph)

# Overall
if result['overall_passed']:
    print("✅ PASSED")
else:
    print("❌ FAILED")

# Details
print(f"Pin 2 net: {result['net_a_pin2']}")
print(f"Pin 4 net: {result['net_a_pin4']}")
print(f"Rule 1: {result['rule1']['passed_any']}")
print(f"Rule 2: {result['rule2']['passed_any']}")
```

## Default Ground Candidates
```python
['DGND', 'WWAN_GND*', '*GND']
```
- `DGND` - exact match
- `WWAN_GND*` - matches WWAN_GND, WWAN_GND_1, etc.
- `*GND` - matches anything ending with GND

## Capacitor Rules

### Rule 1 (OR logic)
- ✅ At least 1× 100µF capacitor, **OR**
- ✅ At least 3× 47µF capacitors

### Rule 2 (AND logic)
- ✅ At least 1× 0.1µF capacitor, **AND**
- ✅ At least 1× 18pF capacitor

## Pass Logic
```
Overall Pass = (Rule 1 passes for ANY ground) AND (Rule 2 passes for ANY ground)
```

## Error Cases

### Pin Not Connected
```python
{
    'overall_passed': False,
    'error': 'No net found for CN1401[2]'
}
```

### No Grounds Found
```python
{
    'overall_passed': False,
    'error': 'No ground nets found'
}
```

## Test Script
```bash
cd /home/jovyan/project/SIPI
python test_query001_a.py
```

## Full Documentation
See `QUERY001_A_README.md` for complete details.

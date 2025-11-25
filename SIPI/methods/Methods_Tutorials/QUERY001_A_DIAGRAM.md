# query001_a Logic Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       query001_a Function                           │
│                                                                     │
│  Input: component='CN1401', pin2=2, pin4=4,                       │
│         net_b_candidates=['DGND', 'WWAN_GND*', '*GND']            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Find net_a for pin2                                        │
│                                                                     │
│  graph.query_net_by_component_pin('CN1401', 2)                     │
│  ────────────────────────────────────────────►  net_a_pin2='PVSIM'│
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Find net_a for pin4                                        │
│                                                                     │
│  graph.query_net_by_component_pin('CN1401', 4)                     │
│  ────────────────────────────────────────────►  net_a_pin4='PVSIM'│
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Expand ground net candidates                               │
│                                                                     │
│  'DGND'       ──────────────────────────────►  ['DGND']            │
│  'WWAN_GND*'  ──expand wildcard──────────────►  ['WWAN_GND',       │
│                                                   'WWAN_GND_1',     │
│                                                   'WWAN_GND_2']     │
│  '*GND'       ──expand wildcard──────────────►  ['DGND', 'AGND',   │
│                                                   'SGND', ...]      │
│                                                                     │
│  Result: expanded_net_b = ['DGND', 'WWAN_GND', 'WWAN_GND_1', ...]  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Test Rule 1 - PVSIM-C(100µF OR 3×47µF)-GND                │
│                                                                     │
│  For each net_b in expanded_net_b:                                 │
│  ┌────────────────────────────────────────────────────┐            │
│  │ Test 1.1: Check ≥1 × 100µF between PVSIM-net_b    │            │
│  │ Test 1.2: Check ≥3 × 47µF between PVSIM-net_b     │            │
│  │                                                    │            │
│  │ Result: (Test 1.1 OR Test 1.2)                    │            │
│  └────────────────────────────────────────────────────┘            │
│                                                                     │
│  Example results:                                                  │
│  • DGND:       ✅ Found 1×100µF  → PASS                            │
│  • WWAN_GND:   ❌ Not enough     → FAIL                            │
│  • AGND:       ✅ Found 3×47µF   → PASS                            │
│                                                                     │
│  Rule 1 Result: ✅ PASSED (at least one ground passed)             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Test Rule 2 - PVSIM-C(0.1µF AND 18pF)-GND                 │
│                                                                     │
│  For each net_b in expanded_net_b:                                 │
│  ┌────────────────────────────────────────────────────┐            │
│  │ Test 2.1: Check ≥1 × 0.1µF between PVSIM-net_b    │            │
│  │ Test 2.2: Check ≥1 × 18pF between PVSIM-net_b     │            │
│  │                                                    │            │
│  │ Result: (Test 2.1 AND Test 2.2)                   │            │
│  └────────────────────────────────────────────────────┘            │
│                                                                     │
│  Example results:                                                  │
│  • DGND:       ✅ Found both     → PASS                            │
│  • WWAN_GND:   ❌ Missing 18pF   → FAIL                            │
│  • AGND:       ❌ Missing 0.1µF  → FAIL                            │
│                                                                     │
│  Rule 2 Result: ✅ PASSED (at least one ground passed)             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Overall Result                                             │
│                                                                     │
│  Overall = (Rule 1 PASSED) AND (Rule 2 PASSED)                     │
│          = (✅)           AND (✅)                                   │
│          = ✅ OVERALL PASSED                                        │
│                                                                     │
│  Return: {                                                         │
│    'overall_passed': True,                                         │
│    'net_a_pin2': 'PVSIM',                                          │
│    'net_a_pin4': 'PVSIM',                                          │
│    'rule1': {'passed_any': True, ...},                             │
│    'rule2': {'passed_any': True, ...}                              │
│  }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Decision Points

### Rule 1 Logic (OR)
```
100µF capacitor found?
├─ YES → ✅ PASS Rule 1
└─ NO  → Check 47µF
         ├─ ≥3 found? → ✅ PASS Rule 1
         └─ <3 found? → ❌ FAIL Rule 1
```

### Rule 2 Logic (AND)
```
0.1µF capacitor found?
├─ YES → Check 18pF
│        ├─ YES → ✅ PASS Rule 2
│        └─ NO  → ❌ FAIL Rule 2
└─ NO  → ❌ FAIL Rule 2
```

### Overall Logic
```
Rule 1 passed?
├─ YES → Rule 2 passed?
│        ├─ YES → ✅ OVERALL PASS
│        └─ NO  → ❌ OVERALL FAIL
└─ NO  → ❌ OVERALL FAIL
```

## Example Circuit Topology

```
             pin2
CN1401 ─────────●───── PVSIM ────┬──── C1 (100µF) ──── DGND
                                 │
             pin4                ├──── C2 (0.1µF) ──── DGND
CN1401 ─────────●───── PVSIM ────┤
                                 └──── C3 (18pF)  ──── DGND

Result:
• pin2 → PVSIM → C1(100µF) → DGND ✅ (Rule 1 passes)
• pin4 → PVSIM → C2(0.1µF) + C3(18pF) → DGND ✅ (Rule 2 passes)
• Overall: ✅ PASS
```

## Actual Graph Structure

```
ComponentGraph (nodes = components, edges = nets):

    CN1401
      │
      ├─ pin2 ───► PVSIM net ───► C1234 ───► DGND net
      │                           C5678 ───► DGND net
      │
      └─ pin4 ───► PVSIM net ───► C9012 ───► DGND net
                                  C3456 ───► DGND net

Query flow:
1. CN1401[pin2] → "What net?" → PVSIM
2. CN1401[pin4] → "What net?" → PVSIM
3. PVSIM-to-DGND → "Which components?" → [C1234, C5678, C9012, C3456]
4. Filter capacitors by value patterns
5. Check counts match rules
```

## Implementation Details

### Fast Pin Lookup
```python
# Built during graph construction
pin_to_net_map = {
    'CN1401': {
        '2': 'PVSIM',
        '4': 'PVSIM',
        ...
    },
    ...
}

# O(1) lookup
net = pin_to_net_map['CN1401']['2']  # → 'PVSIM'
```

### Wildcard Expansion
```python
import fnmatch

# User input
pattern = 'WWAN_GND*'

# Convert to regex
regex = fnmatch.translate(pattern)  # → r'WWAN_GND.*\Z(?ms)'

# Find matching nets
nets = graph.query_nets_by_pattern(regex)
# → ['WWAN_GND', 'WWAN_GND_1', 'WWAN_GND_2']
```

### Capacitor Filtering
```python
# Get components between nets
components = graph.query_components_between_two_nets('PVSIM', 'DGND')
# → ['C1234', 'C5678', 'R9999', 'L1111']

# Filter by type
caps = [c for c in components if graph.nodes[c]['comp_type'] == 'CAP']
# → ['C1234', 'C5678']

# Filter by value (regex match on COMP_VALUE)
import re
matching = [c for c in caps if re.match(r'100\s*UF', get_value(c))]
# → ['C1234']
```

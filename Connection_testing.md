# Connection Testing Pipeline Architecture

## Overview

This document describes the complete pipeline for retrieving and executing circuit connection test methods based on natural language queries.

## Pipeline Architecture

```
User Query
    ↓
[Retrieval Model] → Finds matching method (e.g., "query0001")
    ↓
[Query Executor] → Looks up registered method
    ↓
[Preprocessing] → Platform-specific parameter preparation
    ↓
[Execute Method] → Runs connection tests using unit test functions
    ↓
[Postprocessing] → Format and validate results
    ↓
[Save Results] → Export to Excel/JSON
```

## Script Language for Defining Connection Tests

### Overview

Connection tests are defined using a **script language** that describes circuit topology and requirements. This language is then translated into Python code that uses unit test functions from `circuit.py`.

### Basic Syntax

#### 1. Component Pin to Net Connection
```
<COMPONENT>[<PIN>]-<NET>
```
**Meaning**: Pin number `<PIN>` of component `<COMPONENT>` connects to net `<NET>`

**Examples**:
- `CN1401[2]-PVSIM` → Pin 2 of CN1401 connects to PVSIM net
- `J1400[5]-VCC` → Pin 5 of J1400 connects to VCC net

#### 2. Net to Component to Net Connection
```
<NET_A>-<COMPONENT>(<VALUE> <LOGIC> <VALUE>)-<NET_B>
```
**Meaning**: Between `<NET_A>` and `<NET_B>`, there should be component(s) with specified values

**Examples**:
- `PVSIM-C(100uF)-DGND` → 100uF capacitor between PVSIM and DGND
- `VCC-R(0Ω)-GND` → 0Ω resistor between VCC and GND
- `PVSIM-C(100uF OR 3*47uF)-DGND` → Either 1×100uF OR 3×47uF capacitors
- `PVSIM-C(0.1uF AND 18pF)-DGND` → Both 0.1uF AND 18pF capacitors

#### 3. Wildcard Patterns
```
<NET_PATTERN>* or *<NET_PATTERN>* or *<NET_PATTERN>
```
**Meaning**: Match multiple nets using wildcards

**Examples**:
- `*GND` → Matches DGND, AGND, WWAN_GND, etc.
- `P3V3*` → Matches P3V3, P3V3_A, P3V3_MAIN, etc.
- `*PAD*` → Matches any net containing "PAD"

### Logic Operators

#### OR Logic
```
<VALUE1> OR <VALUE2> OR <VALUE3>
```
**Meaning**: At least ONE of the conditions must be satisfied

**Example**:
```
PVSIM-C(100uF OR 3*47uF)-DGND
```
**Translation**: Between PVSIM and DGND, there must be EITHER:
- At least 1× 100uF capacitor, OR
- At least 3× 47uF capacitors

#### AND Logic
```
<VALUE1> AND <VALUE2> AND <VALUE3>
```
**Meaning**: ALL conditions must be satisfied

**Example**:
```
PVSIM-C(0.1uF AND 18pF)-DGND
```
**Translation**: Between PVSIM and DGND, there must be BOTH:
- At least 1× 0.1uF capacitor, AND
- At least 1× 18pF capacitor

#### Quantity Multiplier
```
<N>*<VALUE>
```
**Meaning**: At least N components with specified value

**Examples**:
- `3*47uF` → At least 3× 47uF capacitors
- `2*0.1uF` → At least 2× 0.1uF capacitors

### Complete Test Definitions

#### Example 1: Basic Capacitor Test (query0001)

**Script Language**:
```
CN1401[2]-PVSIM 
CN1401[4]-PVSIM 
PVSIM-C(100uF OR 3*47uF)-DGND 
PVSIM-C(0.1uF AND 18pF)-DGND
```

**Interpretation**:
1. **Pin Discovery**: 
   - Find net connected to CN1401 pin 2 (should be PVSIM)
   - Find net connected to CN1401 pin 4 (should be PVSIM)

2. **Rule 1**: `CN1401[2]→PVSIM-C(100uF OR 3*47uF)-DGND`
   - Between PVSIM and DGND, check:
   - At least 1× 100uF capacitor, OR
   - At least 3× 47uF capacitors

3. **Rule 2**: `CN1401[4]→PVSIM-C(0.1uF AND 18pF)-DGND`
   - Between PVSIM and DGND, check:
   - At least 1× 0.1uF capacitor, AND
   - At least 1× 18pF capacitor

4. **Overall**: Both Rule 1 AND Rule 2 must pass

**Python Implementation**:
```python
def query001_a(graph, component='CN1401', pin2=2, pin4=4, 
               net_b_candidates=['DGND'], verbose=True):
    # Step 1: Find nets from pins
    net_a_pin2 = graph.query_net_by_component_pin(component, pin2)
    net_a_pin4 = graph.query_net_by_component_pin(component, pin4)
    
    # Step 2: Test Rule 1 (OR logic)
    test1_1 = unit_test_capacitor_between_nets(
        graph, net_a_pin2, net_b, value_pattern=r'100\s*UF', min_count=1)
    test1_2 = unit_test_capacitor_between_nets(
        graph, net_a_pin2, net_b, value_pattern=r'47\s*UF', min_count=3)
    rule1_passed = test1_1[0] or test1_2[0]
    
    # Step 3: Test Rule 2 (AND logic)
    test2_1 = unit_test_capacitor_between_nets(
        graph, net_a_pin4, net_b, value_pattern=r'0\.1\s*UF', min_count=1)
    test2_2 = unit_test_capacitor_between_nets(
        graph, net_a_pin4, net_b, value_pattern=r'18\s*PF', min_count=1)
    rule2_passed = test2_1[0] and test2_2[0]
    
    # Step 4: Overall (both rules must pass)
    overall_passed = rule1_passed and rule2_passed
    
    return {'overall_passed': overall_passed, ...}
```

#### Example 2: Wildcard Net Pattern Test

**Script Language**:
```
CN1401[2]-PVSIM 
CN1401[4]-PVSIM 
PVSIM-*PAD*-P3V3*
```

**Interpretation**:
1. **Pin Discovery**:
   - Find net connected to CN1401 pin 2 (PVSIM)
   - Find net connected to CN1401 pin 4 (PVSIM)

2. **Rule**: `PVSIM-*PAD*-P3V3*`
   - From PVSIM net
   - Through any component containing "PAD" in its net (wildcard match)
   - To any net starting with "P3V3" (wildcard match)
   - Check if this connection path exists

3. **Implementation Steps**:
   - Expand `*PAD*` to all nets matching pattern (e.g., PAD1, PAD_A, TEST_PAD)
   - Expand `P3V3*` to all nets matching pattern (e.g., P3V3, P3V3_A, P3V3_MAIN)
   - Test all combinations for valid connection path

**Python Implementation**:
```python
def query002_a(graph, component='CN1401', pin2=2, pin4=4, verbose=True):
    # Step 1: Find net_a from pins
    net_a = graph.query_net_by_component_pin(component, pin2)
    
    # Step 2: Expand wildcard patterns
    pad_nets = graph.query_nets_by_pattern(r'.*PAD.*')  # *PAD*
    p3v3_nets = graph.query_nets_by_pattern(r'P3V3.*')  # P3V3*
    
    # Step 3: Test connection path exists
    for pad_net in pad_nets:
        for p3v3_net in p3v3_nets:
            # Check: PVSIM → pad_net → p3v3_net
            path_exists = unit_test_connection_path(
                graph, net_a, pad_net, p3v3_net)
            if path_exists:
                return {'overall_passed': True, ...}
    
    return {'overall_passed': False, ...}
```

### How to Provide New Script Language

When you want to define a new connection test, provide it in this format:

#### Template

```
[QUERY_INDEX]: query0002  # Use 4-digit format

[DESCRIPTION]: Brief description of what this query tests

[SCRIPT]:
<LINE1>
<LINE2>
<LINE3>
...

[LOGIC]:
Rule 1: <DESCRIPTION>
  - <DETAILS>
Rule 2: <DESCRIPTION>
  - <DETAILS>
Overall: <HOW TO COMBINE RULES>

[PLATFORM_DIFFERENCES] (optional):
Platform A:
  - Component: <ID>
  - Pin mapping: <DETAILS>
Platform B:
  - Component: <ID>
  - Pin mapping: <DETAILS>
```

#### Example: New Query Definition

```
[QUERY_INDEX]: query0002

[DESCRIPTION]: Check series resistor pad reservation and power rail connection

[SCRIPT]:
CN1401[2]-PVSIM 
CN1401[4]-PVSIM 
PVSIM-R(0Ω AND 0603)-*PAD*
*PAD*-R(0Ω)-P3V3*

[LOGIC]:
Rule 1: CN1401[2]→PVSIM-R(0Ω AND 0603)-*PAD*
  - Between PVSIM and any PAD net
  - Must have 0Ω resistor with 0603 package
  
Rule 2: *PAD*-R(0Ω)-P3V3*
  - Between any PAD net and P3V3 nets
  - Must have 0Ω resistor (any package)
  
Overall: Rule 1 AND Rule 2 must both pass

[PLATFORM_DIFFERENCES]:
G12_MACHU14_TLD_1217:
  - Component: CN1401
  - Pin 2: Power pin
  - Pin 4: Secondary power
  
SVTP804:
  - Component: J1400
  - Pin 3: Power pin (maps to pin2)
  - Pin 5: Secondary power (maps to pin4)
```

### Common Component Types

| Symbol | Component Type | Examples |
|--------|---------------|----------|
| `C` | Capacitor | `C(100uF)`, `C(0.1uF AND 18pF)` |
| `R` | Resistor | `R(0Ω)`, `R(10K OR 100K)` |
| `L` | Inductor | `L(10uH)`, `L(FB)` (ferrite bead) |
| `D` | Diode | `D(1N4148)` |
| `Q` | Transistor | `Q(2N2222)` |

### Common Value Formats

| Format | Meaning | Regex Pattern |
|--------|---------|---------------|
| `100uF` | 100 microfarad | `r'100\s*UF'` |
| `0.1uF` | 0.1 microfarad | `r'0\.1\s*UF'` |
| `18pF` | 18 picofarad | `r'18\s*PF'` |
| `10K` | 10 kilohm | `r'10\s*K'` |
| `0Ω` | 0 ohm | `r'0\s*(OHM|Ω)'` |
| `0603` | Package size | `r'0603'` |

### Translation Checklist

When translating script language to Python:

- [ ] **Identify all component-pin connections** → Use `query_net_by_component_pin()`
- [ ] **Identify wildcard patterns** → Use `query_nets_by_pattern()` with regex
- [ ] **Identify component requirements** (C, R, L, etc.) → Use appropriate `unit_test_*` functions
- [ ] **Identify logic operators** (AND, OR) → Combine test results appropriately
- [ ] **Identify quantity requirements** (3*47uF) → Set `min_count` parameter
- [ ] **Determine rule combination** → Decide how to combine rule results
- [ ] **Add platform preprocessing** → Map platform-specific component IDs and pins
- [ ] **Write unit tests if needed** → Add new functions to `circuit.py` if required

### Script Language Grammar (BNF-like)

```
<test>        ::= <pin_def>* <rule>+
<pin_def>     ::= <component> "[" <pin> "]" "-" <net>
<rule>        ::= <net_a> "-" <comp_req> "-" <net_b>

<component>   ::= <identifier>
<pin>         ::= <number>
<net>         ::= <identifier> | <wildcard>
<net_a>       ::= <identifier> | <wildcard>
<net_b>       ::= <identifier> | <wildcard>

<comp_req>    ::= <comp_type> "(" <value_expr> ")"
<comp_type>   ::= "C" | "R" | "L" | "D" | "Q"
<value_expr>  ::= <value> | <logic_expr>
<logic_expr>  ::= <value> <operator> <value> (<operator> <value>)*

<operator>    ::= "AND" | "OR"
<value>       ::= [<count> "*"] <magnitude> [<package>]
<count>       ::= <number>
<magnitude>   ::= <number> <unit>
<unit>        ::= "uF" | "pF" | "nF" | "Ω" | "K" | "M" | "uH" | "mH"
<package>     ::= "0603" | "0402" | "0805" | ...

<wildcard>    ::= "*" <identifier> | <identifier> "*" | "*" <identifier> "*"
<identifier>  ::= [A-Za-z0-9_]+
<number>      ::= [0-9]+ ("." [0-9]+)?
```

## Core Components

### 1. Main Pipeline (`Query_Pipeline.py`)

**Purpose**: Orchestrates the complete query processing flow

**Key Responsibilities**:
- Load and encode the retrieval model
- Load the circuit graph from netlist
- Retrieve matching method from user query
- Execute the method via query executor
- Format and export results

**Usage**:
```bash
# Single query
python Query_Pipeline.py --query "Is there any 100uF capacitor for WWAN?"

# Interactive mode
python Query_Pipeline.py --interactive

# Batch processing
python Query_Pipeline.py --batch queries.txt
```

**Key Steps**:
1. **Graph Construction**: Build component graph from input netlist
2. **Method Retrieval**: Use trained model to find matching method ID
3. **Method Execution**: Call `execute_query()` from query executor
4. **Result Export**: Save to Excel and JSON files

### 2. Query Executor (`SIPI/query_executor.py`)

**Purpose**: Registry pattern for mapping method names to execution functions

**Key Features**:
- Central registry for all query methods
- Supports preprocessing and postprocessing hooks
- Handles execution errors gracefully
- Platform-aware execution

**Registration Pattern**:
```python
registry.register(
    method_name='query0001',
    function=query001_a,
    description='Combined capacitor rules',
    pre_process=query0001_preprocess,
    post_process=None,  # Optional
    metadata={...}
)
```

**Adding New Methods**:
When you create a new query method, you **must** register it in `setup_default_queries()`:

```python
def setup_default_queries(platform='G12_MACHU14_TLD_1217', pdf_path=None):
    from SIPI.methods.query002 import query002_a, preprocess_query002_a
    
    def query0002_preprocess(graph, **kwargs):
        plat = kwargs.pop('platform', platform)
        verbose = kwargs.get('verbose', False)
        return preprocess_query002_a(graph, plat, verbose, **kwargs)
    
    registry.register(
        method_name='query0002',
        function=query002_a,
        description='Your method description',
        pre_process=query0002_preprocess
    )
```

### 3. Query Methods (`SIPI/methods/query{index}.py`)

**Purpose**: Individual test method implementations

**Naming Convention**: `query0001.py`, `query0002.py`, etc. (4-digit zero-padded)

**Structure**: Each method file contains:
- **Main function**: The actual test logic (e.g., `query001_a`)
- **Preprocessing function**: Platform-specific parameter preparation (e.g., `preprocess_query001_a`)
- **Postprocessing function** (optional): Result formatting

**Example - query001.py**:
```python
def preprocess_query001_a(graph, platform, verbose=True, **kwargs):
    """
    Platform-specific preprocessing.
    Maps platform names to component IDs and pin numbers.
    """
    # Platform mapping
    platform_map = {
        'G12_MACHU14_TLD_1217': {
            'component': 'CN1401',
            'pin2': 2,
            'pin4': 4
        },
        'SVTP804...': {
            'component': 'J1400',
            'pin2': 3,
            'pin4': 5
        }
    }
    
    config = platform_map.get(platform, {...})
    kwargs.update(config)
    return kwargs

def query001_a(graph, component='CN1401', pin2=2, pin4=4, 
               net_b_candidates=None, verbose=True):
    """
    Main test function.
    Tests capacitor presence on WWAN power nets.
    """
    # Use unit test functions from SIPI.circuit
    result1 = test_capacitor_count(graph, net_a, net_b, min_count=1)
    result2 = test_capacitor_value(graph, net_a, net_b, value='100uF')
    
    return {
        'overall_passed': result1 and result2,
        'details': {...}
    }
```

**Key Points**:
- Use **4-digit format** for method names: `query0001`, `query0002`, etc.
- Preprocessing handles platform-specific mappings
- Main function performs the actual tests
- Returns structured result dictionary

### 4. Unit Test Functions (`SIPI/circuit.py`)

**Purpose**: Reusable building blocks for connection tests

**Location**: All unit test functions are defined in `SIPI/circuit.py`

**Common Functions**:
```python
# Capacitor tests
test_capacitor_count(graph, net_a, net_b, min_count=1)
test_capacitor_value(graph, net_a, net_b, value='100uF', tolerance=0.2)
test_capacitor_alternative(graph, net_a, net_b, option1, option2)

# Resistor tests
test_resistor_series(graph, net_a, net_b, value='0Ω', size='0603')
test_resistor_range(graph, net_a, net_b, min_value, max_value)

# Connection tests
test_direct_connection(graph, net_a, net_b)
test_impedance_path(graph, net_a, net_b, max_impedance)

# Component tests
test_component_presence(graph, component_id)
test_pin_connection(graph, component, pin_number, expected_net)
```

**Adding New Unit Tests**:
When you need functionality not covered by existing functions:

1. **Define in `SIPI/circuit.py`**:
```python
def test_ferrite_bead_presence(graph, net_a, net_b, verbose=True):
    """
    Test if there's a ferrite bead between two nets.
    
    Args:
        graph: ComponentGraph instance
        net_a: Source net name or pattern
        net_b: Destination net name or pattern
        verbose: Print detailed output
    
    Returns:
        bool: True if ferrite bead found
    """
    # Implementation
    ferrite_beads = graph.find_components_between_nets(
        net_a, net_b, 
        component_type='L',
        value_pattern='FB*'
    )
    
    if verbose:
        print(f"Found {len(ferrite_beads)} ferrite beads")
    
    return len(ferrite_beads) > 0
```

2. **Use in your query method**:
```python
from SIPI.circuit import test_ferrite_bead_presence

def query002_a(graph, net_a, net_b, verbose=True):
    result = test_ferrite_bead_presence(graph, net_a, net_b, verbose)
    return {'overall_passed': result, 'details': {...}}
```

**Guidelines for Unit Test Functions**:
- Keep functions atomic (test one thing)
- Return boolean or structured result
- Support verbose mode for debugging
- Handle net pattern matching (wildcards)
- Include clear docstrings

## Workflow for Adding New Methods

### Step 1: Create Method File

Create `SIPI/methods/query{index}.py`:

```python
"""
Query {index}: Description of what this query tests
"""

def preprocess_query{index}_a(graph, platform, verbose=True, **kwargs):
    """Platform-specific preprocessing."""
    platform_map = {
        'G12_MACHU14_TLD_1217': {...},
        'SVTP804...': {...}
    }
    
    config = platform_map.get(platform, {...})
    kwargs.update(config)
    return kwargs

def query{index}_a(graph, param1, param2, verbose=True):
    """Main test logic."""
    # Use unit test functions from SIPI.circuit
    from SIPI.circuit import test_capacitor_count, test_resistor_series
    
    # Perform tests
    result1 = test_capacitor_count(graph, net_a, net_b, min_count=1)
    result2 = test_resistor_series(graph, net_a, net_b, value='0Ω')
    
    # Return structured result
    return {
        'overall_passed': result1 and result2,
        'sub_results': [
            {'test': 'capacitor_count', 'passed': result1},
            {'test': 'resistor_series', 'passed': result2}
        ],
        'details': {...}
    }
```

### Step 2: Add Unit Tests (if needed)

If new test functionality is required, add to `SIPI/circuit.py`:

```python
def test_new_functionality(graph, net_a, net_b, verbose=True):
    """
    New unit test function.
    
    Returns:
        bool or dict: Test result
    """
    # Implementation
    pass
```

### Step 3: Register Method

In `SIPI/query_executor.py`, add to `setup_default_queries()`:

```python
def setup_default_queries(platform='G12_MACHU14_TLD_1217', pdf_path=None):
    # ... existing registrations ...
    
    from SIPI.methods.query{index} import query{index}_a, preprocess_query{index}_a
    
    def query{index}_preprocess(graph, **kwargs):
        plat = kwargs.pop('platform', platform)
        verbose = kwargs.get('verbose', False)
        return preprocess_query{index}_a(graph, plat, verbose, **kwargs)
    
    registry.register(
        method_name='query{index}',  # Use 4-digit format!
        function=query{index}_a,
        description='Description of query {index}',
        pre_process=query{index}_preprocess,
        metadata={
            'platform_dependent': True,
            'requires_preprocessing': True
        }
    )
```

### Step 4: Add Training Data

Add to your training corpus (e.g., `output_extracted_data.xlsx`):
- Add row with Pass/Fail Criteria
- Method name will be auto-generated as `query{index}`

### Step 5: Retrain Retrieval Model

```bash
python Retrieval_training.py
```

### Step 6: Test

```bash
# Test single query
python Query_Pipeline.py --query "Your test query"

# Test with specific platform
python Query_Pipeline.py --query "Your test query" --platform G12_MACHU14_TLD_1217
```

## Platform Support

The pipeline supports multiple platforms through preprocessing:

**Supported Platforms**:
- `G12_MACHU14_TLD_1217` (default)
- `SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602`

**Platform Mapping**:
Each platform may have different:
- Component IDs (e.g., CN1401 vs J1400)
- Pin numbers
- Net naming conventions
- Component values

Preprocessing functions handle these differences transparently.

## File Structure

```
project/SIPI/
├── Query_Pipeline.py                    # Main pipeline orchestrator
├── Retrieval_training.py               # Train retrieval model
├── Connection_testing.md               # This document
│
├── SIPI/
│   ├── query_executor.py              # Registry and execution
│   ├── circuit.py                     # Unit test functions
│   ├── graph.py                       # ComponentGraph class
│   │
│   └── methods/                       # Query method implementations
│       ├── query0001.py               # Method 0001
│       ├── query0002.py               # Method 0002
│       ├── query0003.py               # Method 0003
│       └── ...
│
├── models/
│   └── wwan_retrieval/                # Trained retrieval model
│
└── output/
    ├── query_results/                 # Execution results
    └── {platform}/                    # Platform-specific graphs
        └── component_graph.pkl
```

## Best Practices

### For Query Methods

1. **Use 4-digit naming**: `query0001`, not `query1` or `query_001`
2. **Implement preprocessing**: Handle platform differences
3. **Use unit test functions**: Don't duplicate test logic
4. **Return structured results**: Include `overall_passed` and `details`
5. **Support verbose mode**: Print helpful debugging info
6. **Document parameters**: Clear docstrings for all functions

### For Unit Test Functions

1. **Keep atomic**: Test one thing per function
2. **Make reusable**: Think about other methods that might use it
3. **Handle patterns**: Support wildcards in net names
4. **Return clear results**: Boolean or structured dict
5. **Add to circuit.py**: Don't scatter across files
6. **Document thoroughly**: Explain parameters and return values

### For Registration

1. **Register immediately**: Don't forget to register new methods
2. **Use preprocessing**: Don't hardcode platform-specific values
3. **Add metadata**: Help future developers understand the method
4. **Test registration**: Verify method is listed in available queries

## Testing

### Test Individual Components

```bash
# Test retrieval only
python Test--Retrieval_module.py

# Test query executor only
python Test--query_executor.py

# Test specific method
python -c "from SIPI.methods.query0001 import query001_a; print(query001_a.__doc__)"
```

### Test Complete Pipeline

```bash
# Single query
python Query_Pipeline.py --query "Is there any 100uF capacitor?"

# With different platform
python Query_Pipeline.py --query "Test query" --platform SVTP804...

# Interactive mode for multiple tests
python Query_Pipeline.py --interactive
```

### Debug Mode

```bash
# Verbose output
python Query_Pipeline.py --query "Test query" --verbose

# Check registered methods
python -c "from SIPI.query_executor import list_available_queries; print(list_available_queries())"
```

## Troubleshooting

### Method Not Found
- **Check**: Is method registered in `setup_default_queries()`?
- **Check**: Does method name use 4-digit format?
- **Check**: Did you retrain the retrieval model?

### Import Errors
- **Check**: Is the method file in `SIPI/methods/`?
- **Check**: Are all required functions imported in query_executor.py?
- **Check**: Is `SIPI` in your Python path?

### Platform Issues
- **Check**: Is platform name spelled correctly?
- **Check**: Does preprocessing function handle this platform?
- **Check**: Are component IDs correct for this platform?

### Unit Test Function Missing
- **Check**: Is function defined in `SIPI/circuit.py`?
- **Check**: Is function imported in your method file?
- **Check**: Does function signature match usage?

## Summary

**Key Points**:
1. ✅ **Graph Construction**: Build from netlist via `ComponentGraph`
2. ✅ **Method Retrieval**: Trained model finds matching method ID
3. ✅ **Query Executor**: Registry maps method ID to function
4. ✅ **Preprocessing**: Platform-specific parameter preparation
5. ✅ **Execution**: Main function uses unit tests from `circuit.py`
6. ✅ **Postprocessing**: Optional result formatting
7. ✅ **Export**: Save to Excel/JSON

**Remember**:
- New methods → Create in `SIPI/methods/query{index}.py`
- New tests → Add to `SIPI/circuit.py`
- Always register → Update `query_executor.py`
- Always retrain → Run `Retrieval_training.py`

**Quick Reference**:
- Main pipeline: `Query_Pipeline.py`
- Add methods: `SIPI/methods/query{index}.py`
- Register: `SIPI/query_executor.py::setup_default_queries()`
- Unit tests: `SIPI/circuit.py`
- Documentation: `QUICK_START.md`, `QUERY_EXECUTION_GUIDE.md`

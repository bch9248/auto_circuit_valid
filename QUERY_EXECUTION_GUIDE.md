# Query Execution Architecture Guide

## Problem Statement
You have a retrieval system that finds the most similar "Method" based on user queries. Now you need to:
1. Execute the corresponding function based on the retrieved method name
2. Make it easy to add new methods and functions
3. Support multiple functions for the same method
4. Keep the code maintainable

---

## Recommended Solution: Function Registry Pattern

### ✅ Benefits
- **Single source of truth**: All query mappings in one place
- **Easy to extend**: Just register new queries
- **Type-safe**: Clear function signatures
- **Supports hooks**: Pre/post processing for each query
- **Self-documenting**: Configuration describes each query
- **Testable**: Easy to mock and test

---

## Architecture Overview

```
User Query
    ↓
[Retrieval System] → Retrieved Method Name (e.g., "query001")
    ↓
[Query Executor] → Lookup in Registry
    ↓
[Pre-processing] (optional)
    ↓
[Main Function Execution] → query_001_general(graph, net_a, net_b)
    ↓
[Post-processing] (optional)
    ↓
Result
```

---

## Implementation

### 1. Create Query Registry (`query_executor.py`)

```python
from query_executor import QueryRegistry, QueryExecutor

# Global registry
registry = QueryRegistry()

# Register your queries
registry.register(
    method_name='query001',
    function=query_001_general,
    default_net_a=WWAN_power_names,
    default_net_b=WWAN_gnd_names,
    description='Combined capacitor rules',
    pre_process=my_preprocessor,    # optional
    post_process=my_postprocessor,  # optional
)
```

### 2. Execute Retrieved Queries

```python
from query_executor import execute_query

# After retrieval
retrieved_method = "query001"  # from your retrieval system

# Execute
result = execute_query(
    method_name=retrieved_method,
    graph=circuit_graph,
    net_a_pattern='PVSIM',  # optional, uses default if None
    net_b_pattern='DGND',   # optional, uses default if None
    verbose=True
)

# Check result
if result.get('overall_passed'):
    print("✅ Query passed!")
else:
    print("❌ Query failed!")
```

### 3. Add New Queries

Simply register them:

```python
# In query_executor.py or your setup function

registry.register(
    method_name='query002',
    function=query_002_general,
    default_net_a='VCC',
    default_net_b='GND',
    description='Voltage rail verification'
)

registry.register(
    method_name='query003',
    function=query_003_general,
    default_net_a=['NET1', 'NET2'],
    default_net_b='GND',
    description='Multi-net check',
    pre_process=validate_nets,  # Run before main function
    post_process=format_results  # Run after main function
)
```

---

## Advanced Features

### Multiple Processing Stages

```python
def preprocess_nets(graph, net_a, net_b, **kwargs):
    """Validate and transform inputs before query."""
    # Validate nets exist
    if not graph.has_net(net_a):
        raise ValueError(f"Net {net_a} not found")
    
    # Transform patterns
    net_a = net_a.upper()
    net_b = net_b.upper()
    
    return net_a, net_b, kwargs

def postprocess_results(result):
    """Transform results after query."""
    # Add summary
    result['summary'] = f"Tested {result.get('nets_tested', 0)} combinations"
    
    # Calculate score
    if result.get('overall_passed'):
        result['score'] = 1.0
    else:
        result['score'] = 0.5 if result.get('any_passed') else 0.0
    
    return result

registry.register(
    method_name='query001',
    function=query_001_general,
    pre_process=preprocess_nets,
    post_process=postprocess_results
)
```

### Validators

```python
def validate_graph(graph, net_a, net_b):
    """Validate inputs before execution."""
    if graph is None:
        raise ValueError("Graph cannot be None")
    if not net_a or not net_b:
        raise ValueError("Net patterns cannot be empty")

registry.register(
    method_name='query001',
    function=query_001_general,
    validators=[validate_graph]
)
```

### Metadata and Versioning

```python
registry.register(
    method_name='query001',
    function=query_001_general,
    metadata={
        'version': '1.0',
        'author': 'Your Name',
        'last_modified': '2024-10-29',
        'category': 'capacitor_checks',
        'tags': ['wwan', 'power', 'decoupling']
    }
)
```

---

## Integration with Retrieval System

### In your evaluation code:

```python
from query_executor import setup_default_queries, execute_query

# Setup once at the start
setup_default_queries()

# In your evaluation loop
for idx, row in df.iterrows():
    # 1. Retrieve method
    query_embedding = model.encode(row['query'], convert_to_tensor=True)
    results = retrieve_top_k(query_embedding, corpus_embeddings, k=1)
    retrieved_method = corpus_entries[results[0][1]]['metadata']['Method']
    
    # 2. Execute the corresponding function
    exec_result = execute_query(
        method_name=retrieved_method,
        graph=circuit_graph,
        verbose=False
    )
    
    # 3. Check results
    print(f"Retrieved: {retrieved_method}")
    print(f"Passed: {exec_result.get('overall_passed', False)}")
```

---

## Alternative Approaches (Less Recommended)

### 1. Simple Dictionary Mapping
```python
# ❌ Hard to extend, no pre/post processing
FUNCTION_MAP = {
    'query001': query_001_general,
    'query002': query_002_general,
}

func = FUNCTION_MAP.get(retrieved_method)
if func:
    result = func(graph, net_a, net_b)
```

### 2. If-Elif Chain
```python
# ❌ Gets messy with many queries
if retrieved_method == 'query001':
    result = query_001_general(graph, WWAN_power_names, WWAN_gnd_names)
elif retrieved_method == 'query002':
    result = query_002_general(graph, 'VCC', 'GND')
# ... 100 more elif statements
```

### 3. Dynamic Import
```python
# ❌ Harder to track, security concerns
# ❌ No type checking, harder to debug
module = __import__(f'queries.{retrieved_method}')
func = getattr(module, retrieved_method)
result = func(graph, net_a, net_b)
```

---

## Best Practices

### 1. Normalize Method Names
```python
# Handle variations: "query001", "Query001", "query_001", "query-001"
def normalize_name(name):
    return name.lower().replace(' ', '').replace('_', '').replace('-', '')
```

### 2. Provide Defaults
```python
# Always provide sensible defaults
registry.register(
    method_name='query001',
    function=query_001_general,
    default_net_a=WWAN_power_names,  # ✅ Good default
    default_net_b=WWAN_gnd_names,     # ✅ Good default
)
```

### 3. Error Handling
```python
# Always return structured results
def execute_query(...):
    try:
        result = config['function'](...)
        result['success'] = True
        return result
    except Exception as e:
        return {
            'error': str(e),
            'success': False,
            'method_name': method_name
        }
```

### 4. Documentation
```python
# Document each query
registry.register(
    method_name='query001',
    function=query_001_general,
    description='Combined capacitor rules: (100uF OR 3x47uF) AND (0.1uF AND 18pF)',
    metadata={
        'input': 'Two net patterns (net_a, net_b)',
        'output': 'Boolean pass/fail with component details',
        'example': "execute_query('query001', graph, 'PVSIM', 'DGND')"
    }
)
```

---

## File Structure

```
your_project/
├── query_executor.py           # Registry and executor
├── integration_example.py      # Examples of how to use
├── main__connection_check.py   # Your query functions
├── retrieval_eval.py           # Your retrieval system
└── QUERY_EXECUTION_GUIDE.md    # This guide
```

---

## Quick Start Checklist

- [ ] Create `query_executor.py` with QueryRegistry class
- [ ] Register your existing queries (query001, etc.)
- [ ] Update retrieval code to call `execute_query()`
- [ ] Test with a few queries
- [ ] Add pre/post processing as needed
- [ ] Document each query's purpose and parameters
- [ ] Add new queries by registering them

---

## Example Output

```
================================================================================
🔧 Executing Query: query001
   Description: Combined capacitor rules: (100uF OR 3x47uF) AND (0.1uF AND 18pF)
================================================================================
   Net A: ['WWAN_PWR_1', 'WWAN_PWR_2', ...]
   Net B: ['WWAN_GND', 'GND', ...]

================================================================================
🔍 QUERY 001: Combined rules for WWAN_PWR_1-WWAN_GND
================================================================================

────────────────────────────────────────────────────────────────────────────────
📋 Sub-Query 1: WWAN_PWR_1-C(100uF OR 3*47uF)-WWAN_GND
────────────────────────────────────────────────────────────────────────────────
   ✅ Test 1.1: At least 1x 100UF capacitor - PASSED (2 found)
   ✅ Test 1.2: At least 3x 47UF capacitors - PASSED (3 found)

   Result: ✅ PASSED
================================================================================
```

---

## Summary

**Use the Function Registry Pattern** because it:
- ✅ Scales well (100+ queries)
- ✅ Easy to maintain
- ✅ Supports complex workflows
- ✅ Self-documenting
- ✅ Testable

**Avoid** if-elif chains or dynamic imports as they become unmaintainable.

# Query Pipeline - Quick Start Guide

## Overview

The complete query pipeline integrates:
1. **Retrieval Model**: Finds the matching method for a user query
2. **Query Executor**: Runs the corresponding method from `SIPI/methods`
3. **Result Export**: Saves results to Excel and JSON

## Files

- **`Query_Pipeline.py`**: Main pipeline script
- **`Retrieval_training.py`**: Train the retrieval model
- **`Test--Retrieval_module.py`**: Test retrieval accuracy
- **`Test--query_executor.py`**: Query execution registry
- **`SIPI/methods/`**: All query method implementations
- **`SIPI/methods/Methods_Tutorials/`**: Method usage examples

## Quick Start

### 1. Train the Retrieval Model (First Time Only)

```bash
python Retrieval_training.py
```

This creates:
- `models/wwan_retrieval/` - Trained model
- `models/wwan_retrieval/corpus.json` - Corpus with method mappings

### 2. Single Query

```bash
python Query_Pipeline.py --query "Is there any 100uF capacitor for WWAN power?"
```

### 3. Interactive Mode

```bash
python Query_Pipeline.py --interactive
```

Then enter queries one at a time:
```
📝 Enter query: Is there any 100uF or three 47uF capacitors for WWAN?
[Processes query and shows results]

📝 Enter query: save
[Saves results to Excel]

📝 Enter query: quit
```

### 4. Batch Processing

```bash
python Query_Pipeline.py --batch example_queries.txt
```

Where `example_queries.txt` contains:
```
Is there any 100uF or three 47uF capacitors for WWAN?
Does the series 0603 pad reserved in the module power source?
Check power noise isolation for WWAN module
```

## Command Line Options

```bash
# Basic usage
python Query_Pipeline.py --query "Your query here"

# Interactive mode
python Query_Pipeline.py --interactive

# Batch mode
python Query_Pipeline.py --batch queries.txt

# Custom paths
python Query_Pipeline.py \
  --query "Your query" \
  --model models/my_model \
  --graph output/my_graph.pkl \
  --output output/my_results

# Quiet mode (less output)
python Query_Pipeline.py --query "Your query" --quiet
```

## How It Works

### Pipeline Flow

```
User Query
    ↓
[Retrieval Model] → Finds matching method (e.g., "query0001")
    ↓
[Query Executor] → Looks up method in SIPI/methods
    ↓
[Execute Method] → Runs query0001_general(graph, net_a, net_b)
    ↓
[Format Results] → Creates structured output
    ↓
[Save to Excel] → Saves results with timestamp
```

### Example

**Input Query:**
```
Is there any 100uF or three 47uF capacitors to instead of 100uF on 3.3V 
power for WWAN? Are 0.1uF and 18pF capacitors are reserved?
```

**Step 1 - Retrieval:**
```
Retrieved Method: query0001
Similarity Score: 0.9234
```

**Step 2 - Execution:**
```
Executing: query0001_general(graph, WWAN_power_names, WWAN_gnd_names)

Result:
✅ Test 1.1: At least 1x 100UF capacitor - PASSED (2 found)
✅ Test 1.2: At least 3x 47UF capacitors - PASSED (3 found)
✅ Test 2.1: At least 1x 0.1UF capacitor - PASSED (5 found)
✅ Test 2.2: At least 1x 18PF capacitor - PASSED (2 found)

Overall: ✅ PASSED
```

**Step 3 - Save:**
```
Results saved to:
- output/query_results/query_results_20241113_153045.xlsx
- output/query_results/query_results_20241113_153045.json
```

## Output Format

### Excel File (`*.xlsx`)

| Timestamp | User Query | Retrieved Method | Retrieval Score | Execution Success | Overall Passed |
|-----------|------------|------------------|-----------------|-------------------|----------------|
| 2024-11-13T15:30:45 | Is there any 100uF... | query0001 | 0.9234 | True | True |

### JSON File (`*.json`)

```json
[
  {
    "timestamp": "2024-11-13T15:30:45",
    "user_query": "Is there any 100uF...",
    "retrieved_method": "query0001",
    "retrieval_score": 0.9234,
    "execution_success": true,
    "overall_passed": true,
    "retrieval_details": {...},
    "execution_details": {...}
  }
]
```

## Adding New Methods

### 1. Create Method Function

In `SIPI/methods/query_new.py`:
```python
def query_new_general(graph, net_a, net_b, verbose=True):
    """Your new query logic."""
    # Implementation
    return {
        'overall_passed': True,
        'details': {...}
    }
```

### 2. Register Method

In `Test--query_executor.py`:
```python
def setup_default_queries():
    # ... existing registrations ...
    
    registry.register(
        method_name='query0002',  # Use 4-digit format
        function=query_new_general,
        default_net_a='NET_A',
        default_net_b='NET_B',
        description='Description of your new query'
    )
```

### 3. Add Training Data

In `output_extracted_data.xlsx`:
- Add row with Pass/Fail Criteria
- Include method name will be auto-generated as query0002

### 4. Retrain Model

```bash
python Retrieval_training.py
```

### 5. Test

```bash
python Query_Pipeline.py --query "Your new query"
```

## Troubleshooting

### Model Not Found
```bash
# Train the model first
python Retrieval_training.py
```

### Graph Not Found
```bash
# Check graph path
python Query_Pipeline.py --graph output/YOUR_PROJECT/component_graph.pkl
```

### Method Not Registered
```bash
# Check if method exists
python -c "from Test__query_executor import list_available_queries; print(list_available_queries())"
```

### Import Errors
```bash
# Make sure you're in the correct directory
cd /home/jackcp/project/SIPI
python Query_Pipeline.py --query "test"
```

## Testing

### Test Retrieval Only
```bash
python Test--Retrieval_module.py
```

### Test Query Executor Only
```bash
python Test--query_executor.py
```

### Test Complete Pipeline
```bash
python Query_Pipeline.py --query "Is there any 100uF capacitor?"
```

## Performance Tips

1. **Preload Components**: The pipeline loads everything once at startup
2. **Batch Processing**: Use `--batch` for multiple queries (faster than running separately)
3. **Quiet Mode**: Use `--quiet` to reduce output overhead
4. **Save Periodically**: In interactive mode, type `save` to save progress

## Examples

### Example 1: Single Query
```bash
python Query_Pipeline.py \
  --query "Is there any 100uF or three 47uF capacitors for WWAN?"
```

### Example 2: Multiple Queries Interactively
```bash
python Query_Pipeline.py --interactive

# Then enter:
Is there any 100uF capacitor?
Check series 0603 pad
save
quit
```

### Example 3: Batch Processing
```bash
# Create queries file
cat > my_queries.txt << EOF
Is there any 100uF capacitor?
Check series 0603 pad
Verify decoupling capacitors
EOF

# Process all queries
python Query_Pipeline.py --batch my_queries.txt
```

### Example 4: Custom Configuration
```bash
python Query_Pipeline.py \
  --interactive \
  --model models/custom_model \
  --graph output/custom_project/graph.pkl \
  --output results/custom_results
```

## Directory Structure

```
project/SIPI/
├── Query_Pipeline.py              # Main pipeline script
├── Retrieval_training.py          # Train retrieval model
├── Test--Retrieval_module.py      # Test retrieval
├── Test--query_executor.py        # Query executor registry
├── example_queries.txt            # Example batch queries
├── QUICK_START.md                 # This guide
│
├── models/
│   └── wwan_retrieval/           # Trained retrieval model
│       ├── corpus.json           # Method mappings
│       └── ...
│
├── output/
│   └── query_results/            # Results output
│       ├── query_results_*.xlsx
│       └── query_results_*.json
│
└── SIPI/
    └── methods/                   # Query method implementations
        ├── query0001.py
        ├── query0002.py
        └── Methods_Tutorials/    # Usage examples
```

## Summary

**Complete Workflow:**
1. ✅ Train model: `python Retrieval_training.py`
2. ✅ Run query: `python Query_Pipeline.py --query "..."`
3. ✅ Check results: `output/query_results/query_results_*.xlsx`

**Three Modes:**
- `--query "..."` - Single query
- `--interactive` - Interactive mode
- `--batch file.txt` - Batch processing

**Output:**
- Excel file with results
- JSON file with detailed information
- Console output with pass/fail status

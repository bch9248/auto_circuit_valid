# Auto Circuit Validation System

> An intelligent circuit validation system that combines retrieval models with automated circuit analysis to verify hardware design connections and component specifications.

## 🎯 Overview

This system provides automated validation of circuit connections using natural language queries. It integrates:

- **🔍 Intelligent Query Retrieval**: Semantic search to match user queries with validation methods
- **📊 Circuit Graph Analysis**: Graph-based circuit representation and connection verification
- **🤖 VLM Integration**: Visual Language Models for diagram analysis
- **📈 Automated Reporting**: Export results to Excel and JSON formats
- **🔌 Component Validation**: Check capacitor values, net connections, and component specifications

## ✨ Key Features

- **Natural Language Queries**: Ask questions like "Is there any 100uF capacitor for WWAN power?"
- **Pattern Matching**: Support for wildcard patterns in net names (e.g., `PP*_WWAN*`)
- **Multi-Method Support**: Query001 and extensible method registry
- **Interactive Mode**: Real-time query processing with instant results
- **Batch Processing**: Process multiple queries from file
- **Visual Analysis**: BOW (Bag of Words) and VLM-based image retrieval

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd auto_circuit_valid
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement-retrieval.txt
   pip install -r requirement.txt
   ```

### Setup Input Files

3. **Upload required input files** to the `Input/` directory:
   - Circuit graph files (`.pkl`) → `output/<PLATFORM>/component_graph.pkl`
   - Query answer files (`.xlsx`) → `Input/<PLATFORM>/query_w_answer.xlsx`
   
   Example structure:
   ```
   Input/
   ├── <PLATFORM1>/
   │   └── query_w_answer.xlsx
   └── <PLATFORM2>/
       └── query_w_answer.xlsx
   
   output/
   ├── <PLATFORM1>/
   │   └── component_graph.pkl
   └── <PLATFORM2>/
       └── component_graph.pkl
   ```

### Run Tests

4. **Test method execution**
   ```bash
   # Test single method on default platform
   python Test--method_execution.py --method query0001
   
   # Test on specific platform
   python Test--method_execution.py --method query0001 --platform <PLATFORM1>
   
   # Test all methods
   python Test--method_execution.py --all-methods
   ```

---

**For detailed documentation, see [QUICK_START.md](QUICK_START.md) and [QUERY_EXECUTION_GUIDE.md](QUERY_EXECUTION_GUIDE.md)**
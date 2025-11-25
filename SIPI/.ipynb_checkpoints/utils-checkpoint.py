import functools
import time
import os
import pickle
import pandas as pd
# Timing decorator
def time_it(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} took {execution_time:.4f} seconds")
        return result
    return wrapper

@time_it
def save_graph_pickle(graph, filepath='output/circuit_graph.pkl'):
    """Save graph to pickle file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)
    print(f"✅ Graph saved to {filepath}")

@time_it
def load_graph_pickle(filepath='output/circuit_graph.pkl'):
    """Load graph from pickle file"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        graph = pickle.load(f)
    print(f"✅ Graph loaded from {filepath}")
    return graph

def parse_component_pin_report(file_path):
    """Parse component pin report file"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    header_line_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("REFDES,PIN_NUMBER"):
            header_line_index = i
            break
    
    if header_line_index is None:
        raise ValueError("Header line 'REFDES,PIN_NUMBER...' not found in file")
    
    df = pd.read_csv(file_path, skiprows=header_line_index)
    return df

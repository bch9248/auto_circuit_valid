import re
import networkx as nx
import pandas as pd
from collections import defaultdict
import time
import functools
import pickle
import os
from .utils import parse_component_pin_report

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


# ============================================================
# BASE GRAPH CLASS
# ============================================================

class CircuitGraph:
    """Base class for circuit graph representations"""
    
    def __init__(self, df_data=None):
        self.graph = nx.Graph()
        self.graph_type = "base"
        
        if df_data is not None:
            self.build_graph(df_data)
    
    def build_graph(self, df_data):
        """Override in subclasses"""
        raise NotImplementedError("Subclass must implement build_graph()")
    
    def save(self, filepath):
        """Save graph to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_filepath = filepath + '.tmp'
        
        try:
            with open(temp_filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            print(f"✅ Graph saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return False
    
    @staticmethod
    def load(filepath):
        """Load graph from file"""
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            print(f"✅ Graph loaded from {filepath}")
            return obj
        except Exception as e:
            print(f"⚠️  Failed to load graph: {e}")
            return None
    
    def get_stats(self):
        """Get graph statistics"""
        return {
            'graph_type': self.graph_type,
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges()
        }
    
    def print_stats(self):
        """Print graph statistics"""
        stats = self.get_stats()
        print(f"\n📊 Graph Statistics ({stats['graph_type']}):")
        print(f"   Nodes: {stats['nodes']:,}")
        print(f"   Edges: {stats['edges']:,}")


# ============================================================
# COMPONENT-TO-COMPONENT GRAPH (Refdes as nodes, nets as edges)
# ============================================================

class ComponentGraph(CircuitGraph):
    """
    Component-to-Component Graph Representation
    ============================================
    
    Structure:
    - Nodes: Components (REFDES)
    - Edges: Nets (NET_NAME) connecting components
    
    Node Attributes:
    - comp_type: Component device type (e.g., 'CAP', 'RES', 'IC')
    
    Edge Attributes:
    - net: Net name (single net case)
    - nets: List of net names (multiple nets between same components)
    - pins: Pin mapping {comp1: pin1, comp2: pin2} or list of mappings
    
    Available Methods:
    ------------------
    Building:
    - build_graph(df_data): Build graph from DataFrame
    
    Existence Checks:
    - query_component_exists(refdes): Check if component exists
    - query_net_exists(net_name): Check if net exists
    
    Component Queries:
    - query_component_nets(refdes): Get all nets connected to component
    - query_components_by_type_pattern(pattern): Find components by type
    - query_components_by_refdes_pattern(pattern): Find components by REFDES
    - query_net_by_component_pin(refdes, pin_name): Find net connected to specific pin
    
    Net Queries:
    - query_components_on_net(net_name): Get all components on a net
    - query_all_nets(): Get all unique nets
    - query_nets_by_pattern(pattern): Find nets matching pattern
    
    Connection Queries:
    - query_connection_between_components(comp1, comp2): Check connection
    - query_components_between_two_nets(net_a, net_b): Find bridging components
    
    Persistence:
    - save(filepath): Save graph to file
    - load(filepath): Load graph from file (static method)
    
    Statistics:
    - get_stats(): Get graph statistics
    - print_stats(): Print graph statistics
    """
    
    def __init__(self, df_data=None):
        # Initialize attributes BEFORE calling parent __init__
        # because parent calls build_graph() which needs these attributes
        self.graph_type = "component_graph"
        # Build reverse lookup: component -> {pin -> net} for fast pin-based queries
        self.pin_to_net_map = defaultdict(dict)
        
        # Now call parent constructor which may call build_graph()
        super().__init__(df_data)
    
    @time_it
    def build_graph(self, df_data):
        """Build component-to-component graph"""
        print(f"Building ComponentGraph from {len(df_data)} rows...")
        
        # Group by net to find components connected to each net
        net_groups = df_data.groupby('NET_NAME')
        
        for net_name, group in net_groups:
            # Skip invalid net names
            if pd.isna(net_name) or not str(net_name).strip() or str(net_name).strip() == 'nan':
                continue
            
            components = []
            for _, row in group.iterrows():
                refdes = row['REFDES']
                pin_number = row['PIN_NUMBER']
                comp_type = row['COMP_DEVICE_TYPE']
                
                # Add component node with attributes if not exists
                if not self.graph.has_node(refdes):
                    self.graph.add_node(refdes, comp_type=str(comp_type))
                
                # Build pin-to-net lookup map
                self.pin_to_net_map[refdes][str(pin_number)] = net_name
                
                components.append((refdes, str(pin_number)))
            
            # Connect all components on the same net
            for i, (comp1, pin1) in enumerate(components):
                for comp2, pin2 in components[i+1:]:
                    # Add edge between components, labeled with the net
                    if self.graph.has_edge(comp1, comp2):
                        # Multiple nets between same components - add to list
                        edge_data = self.graph[comp1][comp2]
                        if 'nets' not in edge_data:
                            edge_data['nets'] = [edge_data.get('net')]
                            edge_data['pins'] = [edge_data.get('pins')]
                        edge_data['nets'].append(net_name)
                        edge_data['pins'].append({comp1: pin1, comp2: pin2})
                    else:
                        self.graph.add_edge(comp1, comp2, net=net_name, pins={comp1: pin1, comp2: pin2})
        
        print(f"ComponentGraph created: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    # ============================================================
    # QUERY METHODS FOR COMPONENT GRAPH
    # ============================================================
    
    def query_component_exists(self, refdes):
        """
        Check if a component exists in the graph
        
        Args:
            refdes (str): Component reference designator (e.g., 'R61313', 'C1234')
        
        Returns:
            tuple: (exists: bool, data: dict or None)
                - exists: True if component exists, False otherwise
                - data: {
                    'refdes': str,
                    'comp_type': str,
                    'connected_components': list,
                    'connection_count': int
                  }
        
        Example:
            exists, data = graph.query_component_exists('R61313')
            if exists:
                print(f"Component {data['refdes']} is a {data['comp_type']}")
        """
        if refdes not in self.graph:
            return False, None
        
        node_data = self.graph.nodes[refdes]
        neighbors = list(self.graph.neighbors(refdes))
        
        return True, {
            'refdes': refdes,
            'comp_type': node_data.get('comp_type', 'unknown'),
            'connected_components': neighbors,
            'connection_count': len(neighbors)
        }
    
    def query_net_exists(self, net_name):
        """
        Check if a net exists in the graph
        
        Args:
            net_name (str): Net name (e.g., 'PVSIM', 'DGND')
        
        Returns:
            tuple: (exists: bool, data: dict or None)
                - exists: True if net exists, False otherwise
                - data: {
                    'net_name': str,
                    'components': list,
                    'component_count': int,
                    'connections': list of dicts,
                    'connection_count': int
                  }
        
        Example:
            exists, data = graph.query_net_exists('PVSIM')
            if exists:
                print(f"Net {data['net_name']} connects {data['component_count']} components")
        """
        components_on_net = set()
        connections = []
        
        # Search all edges for this net
        for u, v, edge_data in self.graph.edges(data=True):
            # Check single net
            if edge_data.get('net') == net_name:
                components_on_net.add(u)
                components_on_net.add(v)
                connections.append({
                    'comp1': u,
                    'comp2': v,
                    'pins': edge_data.get('pins', {})
                })
            
            # Check multiple nets
            nets = edge_data.get('nets', [])
            if net_name in nets:
                components_on_net.add(u)
                components_on_net.add(v)
                net_index = nets.index(net_name)
                pins_list = edge_data.get('pins', [])
                pins = pins_list[net_index] if net_index < len(pins_list) else {}
                connections.append({
                    'comp1': u,
                    'comp2': v,
                    'pins': pins
                })
        
        if not components_on_net:
            return False, None
        
        return True, {
            'net_name': net_name,
            'components': sorted(list(components_on_net)),
            'component_count': len(components_on_net),
            'connections': connections,
            'connection_count': len(connections)
        }
    
    def query_component_nets(self, refdes):
        """
        Get all nets connected to a specific component
        
        Args:
            refdes (str): Component reference designator
        
        Returns:
            dict: {net_name: [pin1, pin2, ...]}
        
        Example:
            nets = graph.query_component_nets('R61313')
            for net, pins in nets.items():
                print(f"Net {net}: pins {pins}")
        """
        if refdes not in self.graph:
            return {}
        
        net_pins = {}
        
        for neighbor in self.graph.neighbors(refdes):
            edge_data = self.graph.get_edge_data(refdes, neighbor)
            
            # Single net case
            if 'net' in edge_data and 'nets' not in edge_data:
                net = edge_data['net']
                pins = edge_data.get('pins', {})
                pin = pins.get(refdes, 'unknown')
                
                if net not in net_pins:
                    net_pins[net] = []
                net_pins[net].append(pin)
            
            # Multiple nets case
            elif 'nets' in edge_data:
                nets = edge_data['nets']
                pins_list = edge_data.get('pins', [])
                
                for i, net in enumerate(nets):
                    pins = pins_list[i] if i < len(pins_list) else {}
                    pin = pins.get(refdes, 'unknown')
                    
                    if net not in net_pins:
                        net_pins[net] = []
                    net_pins[net].append(pin)
        
        return net_pins
    
    def query_net_by_component_pin(self, refdes, pin_name):
        """
        Find the net connected to a specific component pin
        
        Args:
            refdes (str): Component reference designator (e.g., 'R61313')
            pin_name (str): Pin name/number (e.g., '1', '2', 'A1', 'B12')
        
        Returns:
            str or None: Net name if found, None otherwise
        
        Example:
            net = graph.query_net_by_component_pin('R61313', '1')
            if net:
                print(f"Pin 1 of R61313 is connected to net {net}")
            else:
                print("Pin not found or not connected")
        """
        # Use the fast lookup map built during graph construction
        if refdes in self.pin_to_net_map:
            return self.pin_to_net_map[refdes].get(str(pin_name), None)
        return None
    
    def query_components_on_net(self, net_name):
        """
        Get all components connected to a specific net
        
        Args:
            net_name (str): Net name
        
        Returns:
            list: [(refdes, pin, comp_type), ...]
        
        Example:
            components = graph.query_components_on_net('PVSIM')
            for refdes, pin, comp_type in components:
                print(f"{refdes} (pin {pin}): {comp_type}")
        """
        exists, data = self.query_net_exists(net_name)
        
        if not exists:
            return []
        
        components_info = []
        for comp in data['components']:
            comp_type = self.graph.nodes[comp].get('comp_type', 'unknown')
            net_pins = self.query_component_nets(comp)
            pins = net_pins.get(net_name, ['unknown'])
            
            for pin in pins:
                components_info.append((comp, pin, comp_type))
        
        return sorted(components_info)
    
    def query_connection_between_components(self, comp1, comp2):
        """
        Check if two components are connected and get connection details
        
        Args:
            comp1 (str): First component REFDES
            comp2 (str): Second component REFDES
        
        Returns:
            tuple: (connected: bool, data: dict or None)
                - connected: True if components are connected
                - data: {
                    'nets': list of net names,
                    'pins': list of pin mappings,
                    'net_count': int
                  }
        
        Example:
            connected, data = graph.query_connection_between_components('R61313', 'C1234')
            if connected:
                print(f"Connected via {data['net_count']} nets: {data['nets']}")
        """
        if not self.graph.has_edge(comp1, comp2):
            return False, None
        
        edge_data = self.graph.get_edge_data(comp1, comp2)
        
        # Single net case
        if 'net' in edge_data and 'nets' not in edge_data:
            return True, {
                'nets': [edge_data['net']],
                'pins': [edge_data.get('pins', {})],
                'net_count': 1
            }
        
        # Multiple nets case
        return True, {
            'nets': edge_data.get('nets', []),
            'pins': edge_data.get('pins', []),
            'net_count': len(edge_data.get('nets', []))
        }
    
    def query_components_between_two_nets(self, net_a, net_b):
        """
        Find components that connect two specific nets (bridging components)
        
        Args:
            net_a (str): First net name
            net_b (str): Second net name
        
        Returns:
            list: Component REFDES that connect both nets
        
        Example:
            bridges = graph.query_components_between_two_nets('PVSIM', 'DGND')
            print(f"Components connecting PVSIM and DGND: {bridges}")
        """
        exists_a, data_a = self.query_net_exists(net_a)
        exists_b, data_b = self.query_net_exists(net_b)
        
        if not exists_a or not exists_b:
            return []
        
        components_a = set(data_a['components'])
        components_b = set(data_b['components'])
        
        bridging_components = components_a.intersection(components_b)
        
        return sorted(list(bridging_components))
    
    def query_components_by_type_pattern(self, comp_type_pattern):
        """
        Find components matching a comp_type pattern
        
        Args:
            comp_type_pattern (str): Regex pattern for component type
        
        Returns:
            list: [(refdes, comp_type), ...]
        
        Example:
            caps = graph.query_components_by_type_pattern(r'CAP')
            for refdes, comp_type in caps:
                print(f"{refdes}: {comp_type}")
        """
        matching_components = []
        
        for node, data in self.graph.nodes(data=True):
            comp_type = data.get('comp_type', '')
            if re.search(comp_type_pattern, str(comp_type), re.IGNORECASE):
                matching_components.append((node, comp_type))
        
        return sorted(matching_components)
    
    def query_components_by_refdes_pattern(self, refdes_pattern):
        """
        Find components matching a REFDES pattern
        
        Args:
            refdes_pattern (str): Regex pattern for REFDES
        
        Returns:
            list: [(refdes, comp_type), ...]
        
        Example:
            resistors = graph.query_components_by_refdes_pattern(r'^R\d+')
            for refdes, comp_type in resistors:
                print(f"{refdes}: {comp_type}")
        """
        matching_components = []
        
        for node, data in self.graph.nodes(data=True):
            if re.match(refdes_pattern, node):
                comp_type = data.get('comp_type', 'unknown')
                matching_components.append((node, comp_type))
        
        return sorted(matching_components)
    
    def query_all_nets(self):
        """
        Get all unique nets in the graph
        
        Returns:
            list: Sorted list of all net names
        
        Example:
            all_nets = graph.query_all_nets()
            print(f"Total nets: {len(all_nets)}")
        """
        all_nets = set()
        
        for u, v, edge_data in self.graph.edges(data=True):
            # Single net
            if 'net' in edge_data and 'nets' not in edge_data:
                all_nets.add(edge_data['net'])
            
            # Multiple nets
            if 'nets' in edge_data:
                all_nets.update(edge_data['nets'])
        
        return sorted(list(all_nets))
    
    def query_nets_by_pattern(self, net_pattern):
        """
        Find nets matching a pattern
        
        Args:
            net_pattern (str): Regex pattern for net names
        
        Returns:
            list: Matching net names
        
        Example:
            power_nets = graph.query_nets_by_pattern(r'^P[0-9]+V')
            print(f"Power nets: {power_nets}")
        """
        all_nets = self.query_all_nets()
        matching_nets = [net for net in all_nets if re.match(net_pattern, net)]
        return matching_nets


# # ============================================================
# # NET-TO-NET GRAPH (Nets as nodes, components as edges)
# # ============================================================

# class NetGraph(CircuitGraph):
#     """
#     Bipartite graph where:
#     - Net nodes have node_type='net'
#     - Component nodes have node_type='component' and comp_type attribute
#     - Edges connect components to nets with pin information
    
#     This is actually a bipartite graph, not net-to-net
#     """
    
#     def __init__(self, df_data=None):
#         super().__init__(df_data)
#         self.graph_type = "net_graph"
    
#     @time_it
#     def build_graph(self, df_data):
#         """Build bipartite graph (components and nets as nodes)"""
#         print(f"Building NetGraph from {len(df_data)} rows...")
        
#         for _, row in df_data.iterrows():
#             refdes = row['REFDES']
#             net_name = row['NET_NAME']
#             pin_number = row['PIN_NUMBER']
#             comp_type = row['COMP_DEVICE_TYPE']
            
#             # Skip invalid net names
#             if pd.isna(net_name) or not str(net_name).strip() or str(net_name).strip() == 'nan':
#                 continue
            
#             # Add component node
#             if not self.graph.has_node(refdes):
#                 self.graph.add_node(refdes, node_type='component', comp_type=str(comp_type))
            
#             # Add net node
#             if not self.graph.has_node(net_name):
#                 self.graph.add_node(net_name, node_type='net')
            
#             # Add edge between component and net
#             self.graph.add_edge(refdes, net_name, pin=str(pin_number))
        
#         print(f"NetGraph created: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
#     # ============================================================
#     # QUERY METHODS FOR NET GRAPH
#     # ============================================================
    
#     def query_component_exists(self, refdes):
#         """Check if a component exists"""
#         if refdes not in self.graph:
#             return False, None
        
#         node_data = self.graph.nodes[refdes]
#         if node_data.get('node_type') != 'component':
#             return False, None
        
#         # Get connected nets
#         connected_nets = []
#         for neighbor in self.graph.neighbors(refdes):
#             if self.graph.nodes[neighbor].get('node_type') == 'net':
#                 edge_data = self.graph.get_edge_data(refdes, neighbor)
#                 pin = edge_data.get('pin', 'unknown') if edge_data else 'unknown'
#                 connected_nets.append((neighbor, pin))
        
#         return True, {
#             'refdes': refdes,
#             'comp_type': node_data.get('comp_type', 'unknown'),
#             'connected_nets': connected_nets,
#             'net_count': len(connected_nets)
#         }
    
#     def query_net_exists(self, net_name):
#         """Check if a net exists"""
#         if net_name not in self.graph:
#             return False, None
        
#         node_data = self.graph.nodes[net_name]
#         if node_data.get('node_type') != 'net':
#             return False, None
        
#         # Get connected components
#         connected_components = []
#         for neighbor in self.graph.neighbors(net_name):
#             if self.graph.nodes[neighbor].get('node_type') == 'component':
#                 edge_data = self.graph.get_edge_data(net_name, neighbor)
#                 pin = edge_data.get('pin', 'unknown') if edge_data else 'unknown'
#                 comp_type = self.graph.nodes[neighbor].get('comp_type', 'unknown')
#                 connected_components.append((neighbor, pin, comp_type))
        
#         return True, {
#             'net_name': net_name,
#             'components': [c[0] for c in connected_components],
#             'component_details': connected_components,
#             'component_count': len(connected_components)
#         }
    
#     def query_component_nets(self, refdes):
#         """Get all nets connected to a component"""
#         if refdes not in self.graph:
#             return {}
        
#         if self.graph.nodes[refdes].get('node_type') != 'component':
#             return {}
        
#         net_pins = {}
#         for neighbor in self.graph.neighbors(refdes):
#             if self.graph.nodes[neighbor].get('node_type') == 'net':
#                 edge_data = self.graph.get_edge_data(refdes, neighbor)
#                 pin = edge_data.get('pin', 'unknown')
                
#                 if neighbor not in net_pins:
#                     net_pins[neighbor] = []
#                 net_pins[neighbor].append(pin)
        
#         return net_pins
    
#     def query_components_on_net(self, net_name):
#         """Get all components on a net"""
#         exists, data = self.query_net_exists(net_name)
        
#         if not exists:
#             return []
        
#         return data['component_details']
    
#     def query_components_between_two_nets(self, net_a, net_b):
#         """Find components connected to both nets"""
#         exists_a, data_a = self.query_net_exists(net_a)
#         exists_b, data_b = self.query_net_exists(net_b)
        
#         if not exists_a or not exists_b:
#             return []
        
#         components_a = set(data_a['components'])
#         components_b = set(data_b['components'])
        
#         bridging_components = components_a.intersection(components_b)
        
#         return sorted(list(bridging_components))
    
#     def query_components_by_type_pattern(self, comp_type_pattern):
#         """Find components matching a comp_type pattern"""
#         matching_components = []
        
#         for node, data in self.graph.nodes(data=True):
#             if data.get('node_type') == 'component':
#                 comp_type = data.get('comp_type', '')
#                 if re.search(comp_type_pattern, str(comp_type), re.IGNORECASE):
#                     matching_components.append((node, comp_type))
        
#         return sorted(matching_components)
    
#     def query_components_by_refdes_pattern(self, refdes_pattern):
#         """Find components matching a REFDES pattern"""
#         matching_components = []
        
#         for node, data in self.graph.nodes(data=True):
#             if data.get('node_type') == 'component':
#                 if re.match(refdes_pattern, node):
#                     comp_type = data.get('comp_type', 'unknown')
#                     matching_components.append((node, comp_type))
        
#         return sorted(matching_components)
    
#     def query_all_nets(self):
#         """Get all nets in the graph"""
#         all_nets = []
        
#         for node, data in self.graph.nodes(data=True):
#             if data.get('node_type') == 'net':
#                 all_nets.append(node)
        
#         return sorted(all_nets)
    
#     def query_nets_by_pattern(self, net_pattern):
#         """Find nets matching a pattern"""
#         all_nets = self.query_all_nets()
#         matching_nets = [net for net in all_nets if re.match(net_pattern, net)]
#         return matching_nets
    
#     def query_path_between_components(self, comp1, comp2):
#         """Find path between two components (through nets)"""
#         if comp1 not in self.graph or comp2 not in self.graph:
#             return False, []
        
#         try:
#             path = nx.shortest_path(self.graph, comp1, comp2)
#             return True, path
#         except nx.NetworkXNoPath:
#             return False, []


# # ============================================================
# # UTILITY FUNCTIONS
# # ============================================================

# @time_it
# def demo_both_graphs(comp_graph, net_graph):
#     """Demonstrate queries on both graph types"""
#     print("\n" + "="*80)
#     print("📊 DUAL GRAPH QUERY DEMONSTRATION")
#     print("="*80)
    
#     # Statistics
#     print("\n📈 Graph Statistics:")
#     comp_graph.print_stats()
#     net_graph.print_stats()
    
#     # Test 1: Component exists
#     print("\n" + "="*80)
#     print("1️⃣ Query: Does component 'R61313' exist?")
#     print("="*80)
    
#     print("\n🔷 ComponentGraph:")
#     exists, data = comp_graph.query_component_exists('R61313')
#     if exists:
#         print(f"   ✅ Found: {data['refdes']}")
#         print(f"   Type: {data['comp_type']}")
#         print(f"   Connected to {data['connection_count']} components")
    
#     print("\n🔶 NetGraph:")
#     exists, data = net_graph.query_component_exists('R61313')
#     if exists:
#         print(f"   ✅ Found: {data['refdes']}")
#         print(f"   Type: {data['comp_type']}")
#         print(f"   Connected to {data['net_count']} nets")
#         print(f"   Nets: {[n[0] for n in data['connected_nets'][:5]]}")
    
#     # Test 2: Net exists
#     print("\n" + "="*80)
#     print("2️⃣ Query: Does net 'PVSIM' exist?")
#     print("="*80)
    
#     print("\n🔷 ComponentGraph:")
#     exists, data = comp_graph.query_net_exists('PVSIM')
#     if exists:
#         print(f"   ✅ Found: {data['net_name']}")
#         print(f"   Components: {data['component_count']}")
#         print(f"   Sample: {data['components'][:5]}")
    
#     print("\n🔶 NetGraph:")
#     exists, data = net_graph.query_net_exists('PVSIM')
#     if exists:
#         print(f"   ✅ Found: {data['net_name']}")
#         print(f"   Components: {data['component_count']}")
#         print(f"   Sample: {data['components'][:5]}")
    
#     # Test 3: Components between two nets
#     print("\n" + "="*80)
#     print("3️⃣ Query: Components between 'PVSIM' and 'DGND'?")
#     print("="*80)
    
#     print("\n🔷 ComponentGraph:")
#     comps = comp_graph.query_components_between_two_nets('PVSIM', 'DGND')
#     print(f"   Found {len(comps)} components")
#     print(f"   Sample: {comps[:10]}")
    
#     print("\n🔶 NetGraph:")
#     comps = net_graph.query_components_between_two_nets('PVSIM', 'DGND')
#     print(f"   Found {len(comps)} components")
#     print(f"   Sample: {comps[:10]}")
    
#     # Test 4: Find capacitors
#     print("\n" + "="*80)
#     print("4️⃣ Query: Find capacitors (type pattern 'CAP')")
#     print("="*80)
    
#     print("\n🔷 ComponentGraph:")
#     caps = comp_graph.query_components_by_type_pattern(r'CAP')
#     print(f"   Found {len(caps)} capacitors")
#     print(f"   Sample: {caps[:5]}")
    
#     print("\n🔶 NetGraph:")
#     caps = net_graph.query_components_by_type_pattern(r'CAP')
#     print(f"   Found {len(caps)} capacitors")
#     print(f"   Sample: {caps[:5]}")
    
#     # Test 5: All nets starting with 'P'
#     print("\n" + "="*80)
#     print("5️⃣ Query: Nets starting with 'P'")
#     print("="*80)
    
#     print("\n🔷 ComponentGraph:")
#     p_nets = comp_graph.query_nets_by_pattern(r'^P')
#     print(f"   Found {len(p_nets)} nets")
#     print(f"   Sample: {p_nets[:10]}")
    
#     print("\n🔶 NetGraph:")
#     p_nets = net_graph.query_nets_by_pattern(r'^P')
#     print(f"   Found {len(p_nets)} nets")
#     print(f"   Sample: {p_nets[:10]}")
    
#     print("\n" + "="*80 + "\n")


# ============================================================
# MAIN FUNCTION
# ============================================================

@time_it
def main():
    print("🚀 Starting circuit analysis with queries...")
    # platform='SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602' # can be SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602 or G12_MACHU14_TLD_1217
    platform='G12_MACHU14_TLD_1217' # can be SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602 or G12_MACHU14_TLD_1217
    file_root = f'Input/{platform}'
    ouput_root= f"output/{platform}"
    file_path = f'{file_root}/cpn_rep.txt'
    
    # Load data
    print("\n📁 Loading data...")
    df_data = parse_component_pin_report(file_path)
    print(f"✅ Loaded {len(df_data)} rows")
    
    # Build ComponentGraph
    print("\n🔷 Building ComponentGraph (refdes as nodes, nets as edges)...")
    comp_graph = ComponentGraph(df_data)
    
    # Build NetGraph
    print("\n🔶 Building NetGraph (bipartite: components and nets as nodes)...")
    # net_graph = NetGraph(df_data)
    
    # Save graphs
    print("\n💾 Saving graphs...")
    comp_graph.save(f'{ouput_root}/component_graph.pkl')
    # net_graph.save(f'{ouput_root}/net_graph.pkl')
    
    # Demo queries
    # demo_both_graphs(comp_graph, net_graph)
    
    # # Example: Load from cache
    # print("\n📦 Testing load from cache...")
    # loaded_comp_graph = CircuitGraph.load('output/component_graph.pkl')
    # loaded_net_graph = CircuitGraph.load('output/net_graph.pkl')
    
    # if loaded_comp_graph and loaded_net_graph:
    #     print("✅ Both graphs loaded successfully")
    
    # print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
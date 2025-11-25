import re
from SIPI.graph import ComponentGraph, NetGraph, CircuitGraph
from SIPI.utils import parse_component_pin_report, time_it
from SIPI.circuit import unit_test_capacitor_between_nets, unit_test_capacitor_exists_between_nets

# ============================================================
# QUERY FUNCTIONS
# ============================================================
import re
import fnmatch
from SIPI.graph import ComponentGraph, NetGraph, CircuitGraph
from SIPI.utils import parse_component_pin_report, time_it
from SIPI.circuit import unit_test_capacitor_between_nets, unit_test_capacitor_exists_between_nets
from SIPI.Net_name import WWAN_power_names, WWAN_gnd_names
# ============================================================
# QUERY FUNCTIONS
# ============================================================

# ============================================================
# MAIN FUNCTION
# ============================================================

@time_it
def main():
    print("🚀 Starting circuit analysis with queries...")
    platform='SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602' # can be SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602 or G12_MACHU14_TLD_1217
    file_root = f'Input/{platform}'
    ouput_root= f"output/{platform}"
    file_path = f'{file_root}/cpn_rep.txt'
    
    # Load from cache (ComponentGraph)
    print("\n📦 Loading ComponentGraph from cache...")
    comp_graph = CircuitGraph.load(f'{ouput_root}/component_graph.pkl')
    
    if comp_graph is None:
        print("⚠️  Cache not found, building new graph...")
        df_data = parse_component_pin_report(file_path)
        comp_graph = ComponentGraph(df_data)
        comp_graph.save(f'{ouput_root}/component_graph.pkl')
    
    comp_graph.print_stats()
    # Run query_001 with verbose output
    print("\n" + "="*80)
    print("RUNNING QUERY 001 (VERBOSE)")
    print("="*80)
    
    # result_verbose = query001(comp_graph, 'PVSIM', 'DGND', verbose=True)
    result_verbose = query001_general(comp_graph, WWAN_power_names, WWAN_gnd_names, verbose=True)
    
    # Run query_001 without verbose output
    # print("\n" + "="*80)
    # print("RUNNING QUERY 001 (SILENT)")
    # print("="*80)
    
    # result_silent = query_001(comp_graph, 'PVSIM', 'DGND', verbose=False)
    # print(f"\nSilent result: {'✅ PASSED' if result_silent['overall_passed'] else '❌ FAILED'}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"Query 001: {'✅ PASSED' if result_verbose['overall_passed'] else '❌ FAILED'}")
    print(f"  - Sub-Query 1 (100uF OR 3*47uF): {'✅ PASSED' if result_verbose['subquery1']['passed'] else '❌ FAILED'}")
    print(f"  - Sub-Query 2 (0.1uF AND 18pF):  {'✅ PASSED' if result_verbose['subquery2']['passed'] else '❌ FAILED'}")
    print("="*80)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
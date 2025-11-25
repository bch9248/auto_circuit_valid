"""
Test Script for Method Execution
=================================
Test query methods under SIPI/methods/ with different platforms and configurations.

Usage:
    # Test single method on default platform
    python Test--method_execution.py --method query0001
    
    # Test method on specific platform
    python Test--method_execution.py --method query0001 --platform "SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602"
    
    # Test method on multiple platforms
    python Test--method_execution.py --method query0001 --all-platforms
    
    # Test all registered methods
    python Test--method_execution.py --all-methods
    
    # Test with custom graph
    python Test--method_execution.py --method query0001 --graph output/custom/graph.pkl --platform G12_MACHU14_TLD_1217
"""

import argparse
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import pandas as pd

# Import query executor
from SIPI.query_executor import execute_query, setup_default_queries, list_available_queries
from SIPI.graph import ComponentGraph


# Available platforms
AVAILABLE_PLATFORMS = [
    'G12_MACHU14_TLD_1217',
    'SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602'
]

# Default graph paths for each platform
PLATFORM_GRAPH_PATHS = {
    'G12_MACHU14_TLD_1217': 'output/G12_MACHU14_TLD_1217/component_graph.pkl',
    'SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602': 'output/SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602/component_graph.pkl'
}

# Answer file paths for each platform
PLATFORM_ANSWER_PATHS = {
    'G12_MACHU14_TLD_1217': 'Input/G12_MACHU14_TLD_1217/query_w_answer.xlsx',
    'SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602': 'Input/SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602/query_w_answer.xlsx'
}


class MethodTester:
    """Test framework for query methods."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = []
        self.answer_cache = {}  # Cache answers by platform
    
    def load_graph(self, graph_path: str) -> ComponentGraph:
        """Load circuit graph."""
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        if self.verbose:
            print(f"\n📊 Loading graph: {graph_path}")
        
        graph = ComponentGraph.load(graph_path)
        
        if graph is None:
            raise RuntimeError(f"Failed to load graph from {graph_path}")
        
        num_nodes = graph.graph.number_of_nodes()
        num_edges = graph.graph.number_of_edges()
        
        if self.verbose:
            print(f"   ✅ Graph loaded: {num_nodes} components, {num_edges} connections")
        
        return graph
    
    def load_answers(self, platform: str) -> Dict[str, str]:
        """Load expected answers from Excel file for a platform.
        
        Returns:
            Dictionary mapping method names to expected results (Pass/Fail/N/A)
        """
        if platform in self.answer_cache:
            return self.answer_cache[platform]
        
        answer_path = PLATFORM_ANSWER_PATHS.get(platform)
        if not answer_path or not os.path.exists(answer_path):
            if self.verbose:
                print(f"⚠️  Answer file not found for {platform}: {answer_path}")
            return {}
        
        try:
            df = pd.read_excel(answer_path)
            
            # Check for required columns
            if 'Method' not in df.columns or 'Check Results' not in df.columns:
                if self.verbose:
                    print(f"⚠️  Missing 'Method' or 'Check Results' columns in {answer_path}")
                return {}
            
            # Create mapping: method_name -> expected_result
            answers = {}
            for _, row in df.iterrows():
                method = row.get('Method')
                result = row.get('Check Results')
                
                if pd.notna(method) and pd.notna(result):
                    method_str = str(method).strip()
                    result_str = str(result).strip()
                    answers[method_str] = result_str
            
            self.answer_cache[platform] = answers
            
            if self.verbose:
                print(f"\n📋 Loaded {len(answers)} answers for {platform}")
            
            return answers
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading answers from {answer_path}: {e}")
            return {}
    
    def test_method(
        self, 
        method_name: str, 
        graph: ComponentGraph, 
        platform: str,
        pdf_path: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test a single method.
        
        Args:
            method_name: Method to test (e.g., 'query0001')
            graph: Circuit graph
            platform: Platform name
            pdf_path: Optional PDF path for image retrieval
            **kwargs: Additional method parameters
        
        Returns:
            Test result dictionary
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TESTING METHOD: {method_name}")
            print(f"Platform: {platform}")
            print(f"{'='*80}")
        
        start_time = datetime.now()
        
        try:
            # Execute method (pdf_path is auto-generated in pipeline)
            result = execute_query(
                method_name=method_name,
                graph=graph,
                platform=platform,
                verbose=self.verbose,
                **kwargs
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract connection result
            connection_result = result.get('connection_result', result)
            
            # Get actual result (Pass/Fail)
            actual_result = 'Pass' if result.get('overall_passed', connection_result.get('overall_passed', False)) else 'Fail'
            
            # Load expected answer
            answers = self.load_answers(platform)
            expected_result = answers.get(method_name, 'N/A')
            
            # Compare results
            if expected_result == 'N/A':
                match_status = 'N/A'
            elif actual_result == expected_result:
                match_status = 'Match'
            else:
                match_status = 'Mismatch'
            
            test_result = {
                'method_name': method_name,
                'platform': platform,
                'timestamp': start_time.isoformat(),
                'execution_time': execution_time,
                'success': result.get('success', connection_result.get('success', True)),
                'overall_passed': result.get('overall_passed', connection_result.get('overall_passed', False)),
                'actual_result': actual_result,
                'expected_result': expected_result,
                'match_status': match_status,
                'error': connection_result.get('error'),
                'has_image_result': result.get('image_result') is not None,
                'full_result': result
            }
            
            if self.verbose:
                self._print_test_result(test_result)
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Load expected answer even for failed tests
            answers = self.load_answers(platform)
            expected_result = answers.get(method_name, 'N/A')
            
            test_result = {
                'method_name': method_name,
                'platform': platform,
                'timestamp': start_time.isoformat(),
                'execution_time': execution_time,
                'success': False,
                'overall_passed': False,
                'actual_result': 'Error',
                'expected_result': expected_result,
                'match_status': 'Error',
                'error': str(e),
                'exception_type': type(e).__name__,
                'has_image_result': False
            }
            
            if self.verbose:
                print(f"\n❌ Test failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
            self.test_results.append(test_result)
            return test_result
    
    def test_method_on_platforms(
        self,
        method_name: str,
        platforms: List[str],
        graph_paths: Dict[str, str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Test a method on multiple platforms.
        
        Args:
            method_name: Method to test
            platforms: List of platform names
            graph_paths: Optional dict mapping platform to graph path
            **kwargs: Additional method parameters
        
        Returns:
            List of test results
        """
        results = []
        graph_paths = graph_paths or PLATFORM_GRAPH_PATHS
        
        for platform in platforms:
            print(f"\n{'#'*80}")
            print(f"# Platform: {platform}")
            print(f"{'#'*80}")
            
            # Get graph path for platform
            graph_path = graph_paths.get(platform)
            if not graph_path or not os.path.exists(graph_path):
                print(f"⚠️  Skipping {platform}: graph file not found ({graph_path})")
                continue
            
            # Setup queries for this platform
            setup_default_queries(platform=platform)
            
            # Load graph
            try:
                graph = self.load_graph(graph_path)
            except Exception as e:
                print(f"❌ Failed to load graph for {platform}: {e}")
                continue
            
            # Test method
            result = self.test_method(
                method_name=method_name,
                graph=graph,
                platform=platform,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def test_all_methods(
        self,
        platform: str,
        graph_path: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Test all registered methods on a single platform.
        
        Args:
            platform: Platform name
            graph_path: Path to graph file
            **kwargs: Additional method parameters
        
        Returns:
            List of test results
        """
        # Setup queries
        setup_default_queries(platform=platform)
        
        # Get all available methods
        methods = list_available_queries()
        
        # Filter out image_retrieval (it's not a standalone test)
        methods = [m for m in methods if m != 'image_retrieval']
        
        print(f"\n{'='*80}")
        print(f"Testing {len(methods)} methods on platform: {platform}")
        print(f"Methods: {', '.join(methods)}")
        print(f"{'='*80}")
        
        # Load graph once
        graph = self.load_graph(graph_path)
        
        results = []
        for method in methods:
            result = self.test_method(
                method_name=method,
                graph=graph,
                platform=platform,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def _print_test_result(self, result: Dict[str, Any]):
        """Print test result summary."""
        print(f"\n{'─'*80}")
        print("TEST RESULT:")
        print(f"  Method: {result['method_name']}")
        print(f"  Platform: {result['platform']}")
        print(f"  Execution Time: {result['execution_time']:.2f}s")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
        print(f"  Actual Result: {result.get('actual_result', 'Unknown')}")
        print(f"  Expected Result: {result.get('expected_result', 'N/A')}")
        
        # Color-coded match status
        match_status = result.get('match_status', 'Unknown')
        if match_status == 'Match':
            status_display = '✅ MATCH'
        elif match_status == 'Mismatch':
            status_display = '❌ MISMATCH'
        elif match_status == 'N/A':
            status_display = '⚪ N/A'
        else:
            status_display = '⚠️  ERROR'
        print(f"  Comparison: {status_display}")
        
        if result.get('error'):
            print(f"  Error: {result['error']}")
        
        if result.get('has_image_result'):
            print(f"  Image Retrieval: ✅ Executed")
        
        print(f"{'─'*80}")
    
    def print_summary(self):
        """Print summary of all test results."""
        if not self.test_results:
            print("\n⚠️  No test results to summarize")
            return
        
        print(f"\n\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {len(self.test_results)}")
        
        successful = sum(1 for r in self.test_results if r['success'])
        passed = sum(1 for r in self.test_results if r['overall_passed'])
        failed = len(self.test_results) - successful
        
        # Comparison statistics
        matches = sum(1 for r in self.test_results if r.get('match_status') == 'Match')
        mismatches = sum(1 for r in self.test_results if r.get('match_status') == 'Mismatch')
        no_answer = sum(1 for r in self.test_results if r.get('match_status') == 'N/A')
        errors = sum(1 for r in self.test_results if r.get('match_status') == 'Error')
        
        print(f"Successful Execution: {successful}/{len(self.test_results)}")
        print(f"Tests Passed: {passed}/{len(self.test_results)}")
        print(f"Tests Failed: {failed}/{len(self.test_results)}")
        print(f"\nComparison with Expected:")
        print(f"  ✅ Matches: {matches}/{len(self.test_results)}")
        print(f"  ❌ Mismatches: {mismatches}/{len(self.test_results)}")
        print(f"  ⚪ No Answer: {no_answer}/{len(self.test_results)}")
        print(f"  ⚠️  Errors: {errors}/{len(self.test_results)}")
        
        total_time = sum(r['execution_time'] for r in self.test_results)
        print(f"\nTotal Execution Time: {total_time:.2f}s")
        
        # Group by method
        methods = {}
        for r in self.test_results:
            method = r['method_name']
            if method not in methods:
                methods[method] = {'total': 0, 'passed': 0, 'platforms': []}
            methods[method]['total'] += 1
            if r['overall_passed']:
                methods[method]['passed'] += 1
            methods[method]['platforms'].append(r['platform'])
        
        print(f"\n{'─'*80}")
        print("RESULTS BY METHOD:")
        for method, stats in methods.items():
            status = f"{stats['passed']}/{stats['total']} passed"
            platforms = ', '.join(set(stats['platforms']))
            print(f"  {method}: {status} ({platforms})")
        
        # Mismatched tests
        mismatched_tests = [r for r in self.test_results if r.get('match_status') == 'Mismatch']
        if mismatched_tests:
            print(f"\n{'─'*80}")
            print("MISMATCHED RESULTS:")
            for r in mismatched_tests:
                print(f"  ❌ {r['method_name']} on {r['platform']}: Expected={r.get('expected_result')}, Got={r.get('actual_result')}")
        
        # Failed tests
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            print(f"\n{'─'*80}")
            print("FAILED TESTS:")
            for r in failed_tests:
                print(f"  ❌ {r['method_name']} on {r['platform']}: {r.get('error', 'Unknown error')}")
        
        print(f"{'='*80}\n")
    
    def save_results(self, output_file: str = None):
        """Save test results to JSON file."""
        if not self.test_results:
            print("⚠️  No results to save")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'test_results_{timestamp}.json'
        
        # Remove full_result for cleaner JSON
        save_results = []
        for r in self.test_results:
            r_copy = r.copy()
            r_copy.pop('full_result', None)
            save_results.append(r_copy)
        
        with open(output_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test query methods with different platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single method on default platform
  python Test--method_execution.py --method query0001
  
  # Test method on specific platform
  python Test--method_execution.py --method query0001 --platform "SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602"
  
  # Test method on all platforms
  python Test--method_execution.py --method query0001 --all-platforms
  
  # Test all methods on default platform
  python Test--method_execution.py --all-methods
  
  # Test all methods on all platforms
  python Test--method_execution.py --all-methods --all-platforms
  
  # Test with custom graph
  python Test--method_execution.py --method query0001 --graph output/custom/graph.pkl --platform G12_MACHU14_TLD_1217
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        help='Method name to test (e.g., query0001)'
    )
    
    parser.add_argument(
        '--all-methods',
        action='store_true',
        help='Test all registered methods'
    )
    
    parser.add_argument(
        '--platform',
        type=str,
        default='G12_MACHU14_TLD_1217',
        help='Platform name (default: G12_MACHU14_TLD_1217)'
    )
    
    parser.add_argument(
        '--all-platforms',
        action='store_true',
        help='Test on all available platforms'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        help='Custom graph path (overrides default for platform)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='List all available methods and exit'
    )
    
    parser.add_argument(
        '--list-platforms',
        action='store_true',
        help='List all available platforms and exit'
    )
    
    args = parser.parse_args()
    
    # List methods
    if args.list_methods:
        setup_default_queries()
        methods = list_available_queries()
        print("Available methods:")
        for m in methods:
            print(f"  - {m}")
        return
    
    # List platforms
    if args.list_platforms:
        print("Available platforms:")
        for p in AVAILABLE_PLATFORMS:
            graph_path = PLATFORM_GRAPH_PATHS.get(p)
            exists = "✅" if graph_path and os.path.exists(graph_path) else "❌"
            print(f"  {exists} {p}")
        return
    
    # Validate arguments
    if not args.method and not args.all_methods:
        parser.error("Either --method or --all-methods must be specified")
    
    # Initialize tester
    tester = MethodTester(verbose=not args.quiet)
    
    print("="*80)
    print("METHOD EXECUTION TESTER")
    print("="*80)
    
    # Determine platforms to test
    if args.all_platforms:
        platforms = AVAILABLE_PLATFORMS
        print(f"\nTesting on {len(platforms)} platforms: {', '.join(platforms)}")
    else:
        platforms = [args.platform]
        print(f"\nTesting on platform: {args.platform}")
    
    # Test execution
    try:
        if args.all_methods:
            # Test all methods
            if args.all_platforms:
                # All methods on all platforms
                for platform in platforms:
                    graph_path = args.graph or PLATFORM_GRAPH_PATHS.get(platform)
                    if not graph_path or not os.path.exists(graph_path):
                        print(f"\n⚠️  Skipping {platform}: graph file not found")
                        continue
                    
                    tester.test_all_methods(
                        platform=platform,
                        graph_path=graph_path
                    )
            else:
                # All methods on single platform
                graph_path = args.graph or PLATFORM_GRAPH_PATHS.get(args.platform)
                if not graph_path or not os.path.exists(graph_path):
                    print(f"❌ Graph file not found: {graph_path}")
                    return
                
                tester.test_all_methods(
                    platform=args.platform,
                    graph_path=graph_path
                )
        else:
            # Test single method
            if args.all_platforms:
                # Single method on all platforms
                graph_paths = {p: args.graph for p in platforms} if args.graph else PLATFORM_GRAPH_PATHS
                tester.test_method_on_platforms(
                    method_name=args.method,
                    platforms=platforms,
                    graph_paths=graph_paths
                )
            else:
                # Single method on single platform
                graph_path = args.graph or PLATFORM_GRAPH_PATHS.get(args.platform)
                if not graph_path or not os.path.exists(graph_path):
                    print(f"❌ Graph file not found: {graph_path}")
                    return
                
                setup_default_queries(platform=args.platform)
                
                graph = tester.load_graph(graph_path)
                tester.test_method(
                    method_name=args.method,
                    graph=graph,
                    platform=args.platform
                )
        
        # Print summary
        tester.print_summary()
        
        # Save results
        if args.output or len(tester.test_results) > 1:
            tester.save_results(args.output)
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Complete Query Execution Pipeline
==================================
This script integrates:
1. Retrieval model to find matching method from user query
2. Query executor to run the corresponding method
3. Result formatting and Excel export

Usage:
    python Query_Pipeline.py --query "Your query text here"
    python Query_Pipeline.py --interactive  # Interactive mode
    python Query_Pipeline.py --batch queries.txt  # Batch processing
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json
from datetime import datetime
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import query executor
from SIPI.query_executor import execute_query, setup_default_queries, list_available_queries

# Import graph loading - use the correct module path
from SIPI.graph import ComponentGraph


class QueryPipeline:
    """
    Complete pipeline for query retrieval and execution.
    """
    
    def __init__(
        self,
        model_path: str = 'models/wwan_retrieval',
        corpus_path: str = 'models/wwan_retrieval/corpus.json',
        graph_path: str = 'output/G12_MACHU14_TLD_1217/component_graph.pkl',
        output_dir: str = 'output/query_results',
        platform: str = 'G12_MACHU14_TLD_1217',
        pdf_path: str = None,
        verbose: bool = True
    ):
        """
        Initialize the query pipeline.
        
        Args:
            model_path: Path to trained retrieval model
            corpus_path: Path to corpus JSON file
            graph_path: Path to circuit graph pickle file
            output_dir: Directory to save results
            platform: Platform name for component mapping (e.g., 'G12_MACHU14_TLD_1217')
            pdf_path: Path to PDF schematic for image retrieval (optional, auto-generated if None)
            verbose: Print detailed output
        """
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.graph_path = graph_path
        self.output_dir = output_dir
        self.platform = platform
        
        # Auto-generate PDF path if not provided
        if pdf_path is None:
            self.pdf_path = f'Input/{platform}/{platform}.pdf'
        else:
            self.pdf_path = pdf_path
        
        self.verbose = verbose
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.corpus_entries = None
        self.corpus_embeddings = None
        self.graph = None
        
        # Results tracking
        self.results_history = []
    
    def load_components(self):
        """Load all required components."""
        print("="*80)
        print("INITIALIZING QUERY PIPELINE")
        print("="*80)
        
        # 1. Load retrieval model
        print(f"\n🤖 Loading retrieval model from: {self.model_path}")
        self.model = SentenceTransformer(self.model_path)
        print("✅ Model loaded")
        
        # 2. Load corpus
        print(f"\n📂 Loading corpus from: {self.corpus_path}")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            self.corpus_entries = corpus_data['corpus']
        
        corpus_texts = [entry['text'] for entry in self.corpus_entries]
        print(f"✅ Loaded {len(corpus_texts)} corpus entries")
        
        # 3. Encode corpus
        print(f"\n🔤 Encoding corpus...")
        self.corpus_embeddings = self.model.encode(
            corpus_texts, 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        print(f"✅ Corpus encoded: {self.corpus_embeddings.shape}")
        
        # 4. Load circuit graph
        print(f"\n📊 Loading circuit graph from: {self.graph_path}")
        self.graph = ComponentGraph.load(self.graph_path)
        
        if self.graph is None:
            raise RuntimeError(f"Failed to load graph from {self.graph_path}")
        
        # Get graph stats - ComponentGraph uses self.graph.nodes()
        num_nodes = self.graph.graph.number_of_nodes()
        num_edges = self.graph.graph.number_of_edges()
        print(f"✅ Graph loaded: {num_nodes} components, {num_edges} connections")
        
        # 5. Setup query executor
        print(f"\n⚙️  Setting up query executor...")
        if setup_default_queries(platform=self.platform, pdf_path=self.pdf_path):
            available = list_available_queries()
            print(f"✅ Registered {len(available)} query methods: {available}")
        else:
            print("⚠️  Warning: Could not setup all query methods")
        
        print(f"\n{'='*80}")
        print("✅ PIPELINE INITIALIZATION COMPLETE")
        print("="*80)
    
    def retrieve_method(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve matching methods for a query.
        
        Args:
            query: User query string
            top_k: Number of top results to return
        
        Returns:
            List of retrieved method info with scores
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("STEP 1: RETRIEVING MATCHING METHOD")
            print("="*80)
            print(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarity scores
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
        
        retrieved = []
        for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1]), 1):
            entry = self.corpus_entries[idx.item()]
            method_name = entry['metadata'].get('method', 'Unknown')
            
            result_info = {
                'rank': rank,
                'method': method_name,
                'score': score.item(),
                'corpus_entry': entry,
                'metadata': entry['metadata']
            }
            retrieved.append(result_info)
            
            if self.verbose:
                print(f"\n  Rank {rank}: {method_name}")
                print(f"    Similarity Score: {score.item():.4f}")
                print(f"    Matched Text: {entry['text'][:150]}...")
        
        return retrieved
    
    def execute_method(self, method_name: str, retrieved_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the retrieved method.
        
        Args:
            method_name: Method identifier (e.g., 'query0001')
            retrieved_info: Retrieved information including extracted_text
            **kwargs: Additional arguments for the method
        
        Returns:
            Execution result dictionary
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("STEP 2: EXECUTING METHOD")
            print("="*80)
        
        # Pass platform to executor for preprocessing
        kwargs['platform'] = self.platform
        
        # Pass PDF path if available
        if self.pdf_path:
            kwargs['pdf_path'] = self.pdf_path
        
        # Extract and pass extracted_text if available in retrieved_info
        if retrieved_info and 'corpus_entry' in retrieved_info:
            corpus_entry = retrieved_info['corpus_entry']
            if 'metadata' in corpus_entry and 'extracted_text' in corpus_entry['metadata']:
                kwargs['extracted_text'] = corpus_entry['metadata']['extracted_text']
                if self.verbose:
                    print(f"   Found extracted_text in corpus metadata")
        
        # Execute query using the executor
        result = execute_query(
            method_name=method_name,
            graph=self.graph,
            verbose=self.verbose,
            **kwargs
        )
        
        return result
    
    def format_results(
        self, 
        query: str, 
        retrieved_info: Dict[str, Any], 
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format complete results for output.
        
        Args:
            query: Original user query
            retrieved_info: Retrieval information
            execution_result: Method execution result
        
        Returns:
            Formatted result dictionary
        """
        formatted = {
            # Query information
            'timestamp': datetime.now().isoformat(),
            'user_query': query,
            
            # Retrieval information
            'retrieved_method': retrieved_info['method'],
            'retrieval_score': retrieved_info['score'],
            'retrieval_rank': retrieved_info['rank'],
            
            # Execution information
            'execution_success': execution_result.get('success', True),
            'overall_passed': execution_result.get('overall_passed', False),
            
            # Full results
            'retrieval_details': retrieved_info,
            'execution_details': execution_result
        }
        
        return formatted
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """
        Save results to Excel file.
        
        Args:
            results: List of result dictionaries
            filename: Output filename (auto-generated if None)
        """
        if not results:
            print("⚠️  No results to save")
            return
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'query_results_{timestamp}.xlsx'
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Prepare data for Excel
        excel_data = []
        for result in results:
            row = {
                'Timestamp': result['timestamp'],
                'User Query': result['user_query'],
                'Retrieved Method': result['retrieved_method'],
                'Retrieval Score': f"{result['retrieval_score']:.4f}",
                'Execution Success': result['execution_success'],
                'Overall Passed': result['overall_passed'],
            }
            
            # Add execution details if available
            exec_details = result.get('execution_details', {})
            if 'error' in exec_details:
                row['Error'] = exec_details['error']
            
            excel_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(excel_data)
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Also save detailed JSON
        json_path = output_path.replace('.xlsx', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Detailed results saved to: {json_path}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Complete pipeline: retrieve method, execute, and return results.
        
        Args:
            query: User query string
        
        Returns:
            Complete result dictionary
        """
        print(f"\n{'='*80}")
        print("PROCESSING QUERY")
        print("="*80)
        print(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        
        # Step 1: Retrieve matching method
        retrieved_list = self.retrieve_method(query, top_k=1)
        retrieved_info = retrieved_list[0]  # Use top result
        
        # Step 2: Execute method
        execution_result = self.execute_method(retrieved_info['method'], retrieved_info=retrieved_info)
        
        # Step 3: Format results
        formatted_result = self.format_results(query, retrieved_info, execution_result)
        
        # Add to history
        self.results_history.append(formatted_result)
        
        # Print summary
        self._print_summary(formatted_result)
        
        return formatted_result
    
    def _print_summary(self, result: Dict[str, Any]):
        """Print result summary."""
        print(f"\n{'='*80}")
        print("RESULT SUMMARY")
        print("="*80)
        print(f"Query: {result['user_query'][:100]}...")
        print(f"Retrieved Method: {result['retrieved_method']}")
        print(f"Retrieval Score: {result['retrieval_score']:.4f}")
        print(f"Execution Success: {'✅ Yes' if result['execution_success'] else '❌ No'}")
        print(f"Overall Passed: {'✅ PASSED' if result['overall_passed'] else '❌ FAILED'}")
        
        if 'error' in result.get('execution_details', {}):
            print(f"Error: {result['execution_details']['error']}")
        
        print("="*80)
    
    def interactive_mode(self):
        """Run in interactive mode for multiple queries."""
        print("\n" + "="*80)
        print("INTERACTIVE QUERY MODE")
        print("="*80)
        print("Enter queries (type 'quit' or 'exit' to stop, 'save' to save results)")
        print("="*80 + "\n")
        
        while True:
            query = input("📝 Enter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'save':
                if self.results_history:
                    self.save_results(self.results_history)
                else:
                    print("⚠️  No results to save yet")
                continue
            
            if not query:
                print("⚠️  Empty query, please try again")
                continue
            
            try:
                self.process_query(query)
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results on exit
        if self.results_history:
            save = input("\n💾 Save results before exiting? (y/n): ").strip().lower()
            if save == 'y':
                self.save_results(self.results_history)
        
        print("\n👋 Goodbye!")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Query Pipeline: Retrieve and Execute Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query (default platform: G12_MACHU14_TLD_1217)
  python Query_Pipeline.py --query "Is there any 100uF capacitor?"
  
  # Single query with specific platform
  python Query_Pipeline.py --query "Is there any 100uF capacitor?" --platform SVTP804...
  
  # Interactive mode
  python Query_Pipeline.py --interactive --platform G12_MACHU14_TLD_1217
  
  # Batch processing
  python Query_Pipeline.py --batch queries.txt
  
  # Custom paths
  python Query_Pipeline.py --model models/my_model --graph output/my_graph.pkl --platform G12_MACHU14_TLD_1217
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='File containing queries (one per line)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/wwan_retrieval',
        help='Path to retrieval model'
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        default='models/wwan_retrieval/corpus.json',
        help='Path to corpus JSON'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        default='output/G12_MACHU14_TLD_1217/component_graph.pkl',
        help='Path to circuit graph'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/query_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--platform',
        type=str,
        default='G12_MACHU14_TLD_1217',
        choices=['G12_MACHU14_TLD_1217', 'SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602'],
        help='Platform name for component mapping (default: G12_MACHU14_TLD_1217)'
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        default=None,
        help='Path to PDF schematic for image retrieval (optional)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # check if graph file exists
    if not os.path.exists(args.graph):
        print(f"❌ Graph file not found: {args.graph}")
        return
    
    # Initialize pipeline
    pipeline = QueryPipeline(
        model_path=args.model,
        corpus_path=args.corpus,
        graph_path=args.graph,
        output_dir=args.output,
        platform=args.platform,
        pdf_path=args.pdf,
        verbose=not args.quiet
    )
    
    # Load components
    pipeline.load_components()
    
    # Process based on mode
    if args.interactive:
        # Interactive mode
        pipeline.interactive_mode()
    
    elif args.batch:
        # Batch mode
        print(f"\n📄 Processing batch file: {args.batch}")
        with open(args.batch, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(queries)} queries to process\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing query...")
            try:
                pipeline.process_query(query)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Save all results
        pipeline.save_results(pipeline.results_history)
    
    elif args.query:
        # Single query mode
        result = pipeline.process_query(args.query)
        
        # Save result
        pipeline.save_results([result])
    
    else:
        # No mode specified, show help
        parser.print_help()
        print("\n💡 Tip: Use --interactive for interactive mode")


if __name__ == "__main__":
    main()

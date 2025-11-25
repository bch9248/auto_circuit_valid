"""
Query Executor Module
=====================
This module provides a maintainable way to map retrieved method names 
to their corresponding query execution functions.

Design Pattern: Function Registry
- Easy to add new query methods
- Supports pre/post processing hooks
- Clear configuration for each query
- Type-safe execution with error handling
"""

from typing import Dict, Callable, Any, Optional, List, Tuple
import inspect


# ============================================================
# QUERY REGISTRY - Add your query methods here
# ============================================================

class QueryRegistry:
    """
    Central registry for all query methods.
    
    Benefits:
    1. Single source of truth for all query mappings
    2. Easy to add new queries
    3. Supports multiple processing stages
    4. Self-documenting configuration
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self, 
        method_name: str,
        function: Callable,
        default_net_a: Any = None,
        default_net_b: Any = None,
        description: str = "",
        pre_process: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
        validators: Optional[List[Callable]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Register a query method.
        
        Args:
            method_name: Unique identifier (e.g., 'query001')
            function: Main query function to execute
            default_net_a: Default value for net_a parameter
            default_net_b: Default value for net_b parameter
            description: Human-readable description
            pre_process: Optional preprocessing function
            post_process: Optional postprocessing function
            validators: Optional list of validation functions
            metadata: Optional additional metadata
        """
        # Normalize method name
        normalized_name = self._normalize_name(method_name)
        
        self._registry[normalized_name] = {
            'function': function,
            'default_net_a': default_net_a,
            'default_net_b': default_net_b,
            'description': description or f"Query method: {method_name}",
            'pre_process': pre_process,
            'post_process': post_process,
            'validators': validators or [],
            'metadata': metadata or {},
            'original_name': method_name
        }
    
    def _normalize_name(self, name: str) -> str:
        """Normalize method name for consistent lookup."""
        return name.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    def get(self, method_name: str) -> Optional[Dict[str, Any]]:
        """Get query configuration by method name."""
        normalized = self._normalize_name(method_name)
        return self._registry.get(normalized)
    
    def list_methods(self) -> List[str]:
        """List all registered method names."""
        return [config['original_name'] for config in self._registry.values()]
    
    def exists(self, method_name: str) -> bool:
        """Check if a method is registered."""
        return self._normalize_name(method_name) in self._registry


# Global registry instance
registry = QueryRegistry()


# ============================================================
# QUERY EXECUTOR
# ============================================================

class QueryExecutor:
    """
    Executes query functions based on retrieved method names.
    """
    
    def __init__(self, registry: QueryRegistry, verbose: bool = True):
        self.registry = registry
        self.verbose = verbose
    
    def execute(
        self, 
        method_name: str, 
        graph,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a query method.
        
        Args:
            method_name: Retrieved method name
            graph: CircuitGraph instance
            **kwargs: Arguments passed to query function (can include net_a_pattern, net_b_pattern, component, etc.)
        
        Returns:
            Query result dictionary with execution status
        """
        # Get query configuration
        config = self.registry.get(method_name)
        
        if config is None:
            return self._error_response(
                f"Unknown method: {method_name}",
                available_methods=self.registry.list_methods()
            )
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔧 Executing Query: {config['original_name']}")
            print(f"   Description: {config['description']}")
            print(f"{'='*80}")
        
        try:
            # Run validators
            if config['validators']:
                for validator in config['validators']:
                    validator(graph, **kwargs)
            
            # Pre-processing
            if config['pre_process']:
                kwargs = config['pre_process'](graph, **kwargs)
            
            # Execute main query function
            result = config['function'](graph, **kwargs)
            
            # Post-processing
            if config['post_process']:
                result = config['post_process'](result)
            
            # Add execution metadata
            result['_execution_metadata'] = {
                'method_name': config['original_name'],
                'kwargs_used': kwargs,
                'success': True
            }
            
            if self.verbose:
                status = "✅ PASSED" if result.get('overall_passed', False) else "❌ FAILED"
                print(f"\n   Result: {status}")
                print(f"{'='*80}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Execution failed: {str(e)}")
                print(f"{'='*80}")
            
            return self._error_response(
                f"Execution failed: {str(e)}",
                method_name=method_name,
                exception_type=type(e).__name__
            )
    
    def _format_param(self, param: Any) -> str:
        """Format parameter for display."""
        if isinstance(param, list):
            if len(param) <= 3:
                return str(param)
            return f"[{param[0]}, {param[1]}, ... ({len(param)} items)]"
        return str(param)
    
    def _run_validators(self, validators: List[Callable], graph, net_a, net_b):
        """Run validation functions before execution."""
        for validator in validators:
            validator(graph, net_a, net_b)
    
    def _error_response(self, message: str, **extra_info) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': message,
            'success': False,
            **extra_info
        }


# ============================================================
# REGISTRATION HELPERS
# ============================================================

def register_query(
    method_name: str,
    **kwargs
):
    """
    Decorator to register a query function.
    
    Usage:
        @register_query('query001', description='My query')
        def my_query_function(graph, net_a, net_b, verbose=True):
            ...
    """
    def decorator(func):
        registry.register(method_name, func, **kwargs)
        return func
    return decorator


# ============================================================
# QUERY SETUP
# ============================================================

def setup_default_queries(platform: str = 'G12_MACHU14_TLD_1217', pdf_path: str = None):
    """
    Setup default query registrations.
    Call this after importing your query functions.
    
    Args:
        platform: Platform name for component mapping (used in preprocessing)
        pdf_path: Path to PDF schematic for image retrieval (optional)
    """
    try:
        # Import query functions and their preprocessing functions
        from SIPI.methods.query001 import query001_a, preprocess_query001_a
        from SIPI.methods.image_methods import find_image_crop
        import json
        
        # Create preprocessing function with platform bound
        def query0001_preprocess(graph, **kwargs):
            # Get platform from kwargs or use default
            plat = kwargs.pop('platform', platform)
            verbose = kwargs.get('verbose', False)
            return preprocess_query001_a(graph, plat, verbose, **kwargs)
        
        # Register query 0001 (use 4-digit format to match retrieval corpus)
        registry.register(
            method_name='query0001',
            function=query001_a,
            default_net_a=None,  # Not used by query001_a
            default_net_b=None,  # Not used by query001_a
            description='Combined capacitor rules: (100uF OR 3x47uF) AND (0.1uF AND 18pF)',
            pre_process=query0001_preprocess,
            metadata={
                'platform_dependent': True,
                'default_component': 'CN1401',  # Fallback if platform not found
                'default_pin2': 2,
                'default_pin4': 4,
                'default_net_b_candidates': ['DGND', 'WWAN_GND*', '*GND']
            }
        )
        
        # Register image retrieval method
        # This is a special query that doesn't use the graph but retrieves PDF images
        def image_retrieval_wrapper(graph, **kwargs):
            """
            Wrapper to execute image retrieval with extracted text.
            
            Expected kwargs:
                - extracted_text: str (JSON array) or List[str]
                - pdf_path: str (path to PDF)
                - top_k: int (number of windows, default 5)
            """
            verbose = kwargs.get('verbose', True)
            
            # Get extracted text
            extracted_text = kwargs.get('extracted_text', [])
            
            # Parse extracted_text if it's a JSON string
            if isinstance(extracted_text, str):
                try:
                    target_strings = json.loads(extracted_text)
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"⚠ Error parsing extracted_text JSON: {e}")
                    target_strings = []
            else:
                target_strings = extracted_text
            
            # Get PDF path
            pdf_path_param = kwargs.get('pdf_path', pdf_path)
            if not pdf_path_param:
                return {
                    'success': False,
                    'error': 'No PDF path provided',
                    'image_result': None
                }
            
            # Get other parameters
            top_k = kwargs.get('top_k', 5)
            corpus_cache = kwargs.get('corpus_cache_path', 'corpus/pdf_corpus_cache.json')
            
            # Execute image retrieval
            image_result = find_image_crop(
                target_strings=target_strings,
                pdf_path=pdf_path_param,
                top_k=top_k,
                corpus_cache_path=corpus_cache,
                verbose=verbose
            )
            
            # Return combined result
            return {
                'success': True,
                'image_result': image_result,
                'target_strings': target_strings,
                'overall_passed': image_result.get('success', False)
            }
        
        registry.register(
            method_name='image_retrieval',
            function=image_retrieval_wrapper,
            description='Retrieve matching schematic windows from PDF based on extracted text',
            metadata={
                'requires_pdf': True,
                'uses_graph': False,
                'returns_images': True
            }
        )
        
        # Register more queries here as you add them
        # registry.register(
        #     method_name='query0002',
        #     function=query002_function,
        #     default_net_a='SOME_NET',
        #     default_net_b='GND',
        #     description='Description of query 002'
        # )
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not import query functions: {e}")
        return False


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def execute_query(
    method_name: str,
    graph,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to execute a query.
    
    This function executes the main query (connection checks) and optionally
    executes image retrieval if extracted_text is provided.
    
    Examples:
        # For query001_a (uses component)
        result = execute_query('query0001', graph, platform='G12_MACHU14_TLD_1217', verbose=True)
        
        # For query001_general (uses net patterns)
        result = execute_query('query0001', graph, net_a_pattern='PVSIM', net_b_pattern='DGND')
        
        # With image retrieval
        result = execute_query('query0001', graph, platform='G12_MACHU14_TLD_1217',
                              extracted_text='["CN1400", "100K"]', pdf_path='path/to/pdf')
    
    Returns:
        Dictionary with:
        - connection_result: Result from main query execution
        - image_result: Result from image retrieval (if executed)
        - success: Overall success status
        - overall_passed: Whether query passed
    """
    executor = QueryExecutor(registry, verbose=verbose)
    
    # Execute main query (connection check)
    connection_result = executor.execute(method_name, graph, **kwargs)
    
    # Check if we should also do image retrieval
    extracted_text = kwargs.get('extracted_text')
    pdf_path = kwargs.get('pdf_path')
    
    image_result = None
    if extracted_text and pdf_path:
        # Also execute image retrieval
        if verbose:
            print(f"\n{'='*80}")
            print("🖼️  Executing Image Retrieval")
            print(f"{'='*80}")
        
        image_result = executor.execute('image_retrieval', graph, **kwargs)
    
    # Combine results
    combined_result = {
        'connection_result': connection_result,
        'image_result': image_result,
        'success': connection_result.get('success', True),
        'overall_passed': connection_result.get('overall_passed', False)
    }
    
    return combined_result


def list_available_queries() -> List[str]:
    """List all registered query methods."""
    return registry.list_methods()

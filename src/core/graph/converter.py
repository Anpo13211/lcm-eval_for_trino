"""
Unified plan-to-graph converter

This replaces DBMS-specific plan_to_graph() functions with a single
unified implementation that works across all DBMS.

Key idea:
1. Use FeatureMapper to extract features by logical names
2. Use StandardizedStatistics for database statistics
3. Build graph structure independent of DBMS

This eliminates the need for separate implementations per DBMS × Model combination.
"""

from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

from core.features.mapper import FeatureMapper
from core.statistics.schema import StandardizedStatistics
from training.featurizations import Featurization


class UnifiedPlanConverter:
    """
    Unified converter from plan tree to graph representation.
    
    This single implementation replaces:
    - postgres_plan_collator + postgres_plan_to_graph
    - trino_plan_collator + trino_plan_to_graph
    - (future DBMS implementations)
    
    Usage:
        converter = UnifiedPlanConverter(
            dbms_name="postgres",
            plan_featurization=PostgresTrueCardDetail,
            feature_statistics=feature_stats,
            db_statistics=standardized_stats
        )
        
        graph_data = converter.convert(root_node)
    
    Implementation cost: ~500 lines (shared across all DBMS)
    vs. Current: ~500 lines × N_dbms × M_models
    """
    
    def __init__(
        self,
        dbms_name: str,
        plan_featurization: Featurization,
        feature_statistics: dict,
        db_statistics: StandardizedStatistics
    ):
        """
        Initialize converter.
        
        Args:
            dbms_name: DBMS name (e.g., "postgres", "trino")
            plan_featurization: Feature configuration (e.g., PostgresTrueCardDetail)
            feature_statistics: Global feature statistics for encoding
            db_statistics: Standardized database statistics
        """
        self.dbms_name = dbms_name
        self.plan_featurization = plan_featurization
        self.feature_statistics = feature_statistics
        self.db_statistics = db_statistics
        
        # Create feature mapper for this DBMS
        self.feature_mapper = FeatureMapper(dbms_name, strict=False)
        
        # Graph components (will be populated during conversion)
        self.graph_data = {}
    
    def convert(self, root_node: Any, database_id: Any = None) -> Dict[str, Any]:
        """
        Convert plan tree to graph representation.
        
        Args:
            root_node: Root of plan tree (AbstractPlanOperator or compatible)
            database_id: Database identifier
        
        Returns:
            Dictionary with graph components:
            - plan_depths: List of depths
            - plan_features: List of feature vectors
            - plan_to_plan_edges: List of (child, parent) tuples
            - filter_to_plan_edges: List of (filter, plan) tuples
            - predicate_col_features: List of predicate feature vectors
            - ... (other graph components)
        """
        # Initialize graph storage
        plan_depths = []
        plan_features = []
        plan_to_plan_edges = []
        
        filter_to_plan_edges = []
        predicate_col_features = []
        predicate_depths = []
        intra_predicate_edges = []
        logical_preds = []
        
        output_column_to_plan_edges = []
        output_column_features = []
        column_to_output_column_edges = []
        
        column_features = []
        column_idx = {}
        
        table_features = []
        table_to_plan_edges = []
        table_idx = {}
        
        output_column_idx = {}
        
        # Recursive conversion
        self._convert_node(
            node=root_node,
            database_id=database_id,
            plan_depths=plan_depths,
            plan_features=plan_features,
            plan_to_plan_edges=plan_to_plan_edges,
            filter_to_plan_edges=filter_to_plan_edges,
            predicate_col_features=predicate_col_features,
            output_column_to_plan_edges=output_column_to_plan_edges,
            output_column_features=output_column_features,
            column_to_output_column_edges=column_to_output_column_edges,
            column_features=column_features,
            table_features=table_features,
            table_to_plan_edges=table_to_plan_edges,
            output_column_idx=output_column_idx,
            column_idx=column_idx,
            table_idx=table_idx,
            predicate_depths=predicate_depths,
            intra_predicate_edges=intra_predicate_edges,
            logical_preds=logical_preds,
            parent_node_id=None,
            depth=0
        )
        
        return {
            'plan_depths': plan_depths,
            'plan_features': plan_features,
            'plan_to_plan_edges': plan_to_plan_edges,
            'filter_to_plan_edges': filter_to_plan_edges,
            'predicate_col_features': predicate_col_features,
            'output_column_to_plan_edges': output_column_to_plan_edges,
            'output_column_features': output_column_features,
            'column_to_output_column_edges': column_to_output_column_edges,
            'column_features': column_features,
            'table_features': table_features,
            'table_to_plan_edges': table_to_plan_edges,
            'predicate_depths': predicate_depths,
            'intra_predicate_edges': intra_predicate_edges,
            'logical_preds': logical_preds,
        }
    
    def _convert_node(
        self,
        node: Any,
        database_id: Any,
        plan_depths: list,
        plan_features: list,
        plan_to_plan_edges: list,
        filter_to_plan_edges: list,
        predicate_col_features: list,
        output_column_to_plan_edges: list,
        output_column_features: list,
        column_to_output_column_edges: list,
        column_features: list,
        table_features: list,
        table_to_plan_edges: list,
        output_column_idx: dict,
        column_idx: dict,
        table_idx: dict,
        predicate_depths: list,
        intra_predicate_edges: list,
        logical_preds: list,
        parent_node_id: Optional[int],
        depth: int
    ):
        """
        Recursively convert a plan node and its children.
        
        This is the core logic that extracts features using FeatureMapper
        and builds graph structure.
        """
        # Get current plan node ID
        plan_node_id = len(plan_depths)
        plan_depths.append(depth)
        
        # Extract plan features using logical feature names
        plan_feat = self._extract_plan_features(node, database_id)
        plan_features.append(plan_feat)
        
        # Extract and process predicates/filters
        if hasattr(node, 'plan_parameters'):
            self._process_filters(
                node,
                database_id,
                plan_node_id,
                filter_to_plan_edges,
                predicate_col_features,
                predicate_depths,
                intra_predicate_edges,
                logical_preds,
                column_idx
            )
        
        # Extract and process output columns
        if hasattr(node, 'plan_parameters'):
            self._process_output_columns(
                node,
                database_id,
                plan_node_id,
                output_column_to_plan_edges,
                output_column_features,
                column_to_output_column_edges,
                output_column_idx,
                column_idx,
                column_features
            )
        
        # Extract and process table information
        if hasattr(node, 'plan_parameters'):
            self._process_table(
                node,
                database_id,
                plan_node_id,
                table_to_plan_edges,
                table_features,
                table_idx
            )
        
        # Add edge to parent
        if parent_node_id is not None:
            plan_to_plan_edges.append((plan_node_id, parent_node_id))
        
        # Recursively process children
        if hasattr(node, 'children'):
            for child in node.children:
                self._convert_node(
                    node=child,
                    database_id=database_id,
                    plan_depths=plan_depths,
                    plan_features=plan_features,
                    plan_to_plan_edges=plan_to_plan_edges,
                    filter_to_plan_edges=filter_to_plan_edges,
                    predicate_col_features=predicate_col_features,
                    output_column_to_plan_edges=output_column_to_plan_edges,
                    output_column_features=output_column_features,
                    column_to_output_column_edges=column_to_output_column_edges,
                    column_features=column_features,
                    table_features=table_features,
                    table_to_plan_edges=table_to_plan_edges,
                    output_column_idx=output_column_idx,
                    column_idx=column_idx,
                    table_idx=table_idx,
                    predicate_depths=predicate_depths,
                    intra_predicate_edges=intra_predicate_edges,
                    logical_preds=logical_preds,
                    parent_node_id=plan_node_id,
                    depth=depth + 1
                )
    
    def _extract_plan_features(self, node: Any, database_id: Any) -> list:
        """
        Extract plan node features using logical feature names.
        
        This replaces DBMS-specific feature extraction with unified logic.
        """
        if not hasattr(node, 'plan_parameters'):
            return []
        
        plan_params = node.plan_parameters
        
        # Get logical feature names from featurization
        if hasattr(self.plan_featurization, 'PLAN_FEATURES'):
            logical_names = self.plan_featurization.PLAN_FEATURES
        elif hasattr(self.plan_featurization, 'VARIABLES'):
            logical_names = self.plan_featurization.VARIABLES.get('plan', [])
        else:
            # Fallback: use default features
            logical_names = [
                'operator_type',
                'estimated_cardinality',
                'estimated_cost',
                'estimated_width'
            ]
        
        # Extract and encode features directly using encode()
        # This ensures all values are properly encoded as numbers
        features = []
        for logical_name in logical_names:
            # Use encode() directly with plan_params
            # encode() will handle the feature mapping internally
            encoded_value = self._encode_feature(logical_name, plan_params)
            features.append(encoded_value)
        
        return features
    
    def _encode_feature(self, feature_name: str, plan_params: Any) -> Any:
        """
        Encode feature value using feature_statistics.
        
        This delegates to the existing encode() function for consistency.
        
        Args:
            feature_name: Logical feature name
            plan_params: Plan parameters (dict or SimpleNamespace)
        
        Returns:
            Encoded numeric value
        """
        # Import here to avoid circular dependency
        from models.zeroshot.postgres_plan_batching import encode
        
        try:
            # Call encode() with the feature name and full plan_params
            # encode() will extract the value internally
            return encode(feature_name, plan_params, self.feature_statistics)
        except (KeyError, NotImplementedError, AttributeError):
            # Fallback: return default value
            # For numeric features, return 0
            if feature_name in ['estimated_cardinality', 'actual_cardinality', 'estimated_cost', 
                               'estimated_width', 'workers_planned']:
                return 0.0
            # For categorical features, return 0 (will be encoded as index)
            elif feature_name in ['operator_type', 'data_type']:
                return 0
            else:
                return 0.0
    
    def _process_filters(
        self,
        node: Any,
        database_id: Any,
        plan_node_id: int,
        filter_to_plan_edges: list,
        predicate_col_features: list,
        predicate_depths: list,
        intra_predicate_edges: list,
        logical_preds: list,
        column_idx: dict
    ):
        """
        Process filter predicates for this node.
        
        This extracts filter information and builds predicate graph.
        Handles both PostgreSQL and Trino predicate structures.
        """
        plan_params = node.plan_parameters
        
        # Extract filter_columns (predicate tree)
        filter_column = None
        if isinstance(plan_params, dict):
            filter_column = plan_params.get('filter_columns')
        elif hasattr(plan_params, 'filter_columns'):
            filter_column = plan_params.filter_columns
        
        if filter_column is None:
            return
        
        # Get column statistics for feature extraction
        db_column_stats = {}
        if hasattr(self.db_statistics, 'column_stats'):
            db_column_stats = self.db_statistics.column_stats
        
        # Recursively parse predicates
        self._parse_predicates(
            filter_column=filter_column,
            database_id=database_id,
            plan_node_id=plan_node_id,
            filter_to_plan_edges=filter_to_plan_edges,
            predicate_col_features=predicate_col_features,
            predicate_depths=predicate_depths,
            intra_predicate_edges=intra_predicate_edges,
            logical_preds=logical_preds,
            db_column_stats=db_column_stats,
            parent_filter_node_id=None,
            depth=0
        )
    
    def _process_output_columns(
        self,
        node: Any,
        database_id: Any,
        plan_node_id: int,
        output_column_to_plan_edges: list,
        output_column_features: list,
        column_to_output_column_edges: list,
        output_column_idx: dict,
        column_idx: dict,
        column_features: list
    ):
        """
        Process output columns for this node.
        
        This extracts output column information and creates:
        - Output column nodes with aggregation features
        - Edges from output columns to plan nodes
        - Edges from base columns to output columns
        """
        plan_params = node.plan_parameters
        
        # Extract output_columns
        output_columns = None
        if isinstance(plan_params, dict):
            output_columns = plan_params.get('output_columns')
        elif hasattr(plan_params, 'output_columns'):
            output_columns = plan_params.output_columns
        
        if output_columns is None:
            return
        
        # Get column statistics for feature extraction
        db_column_stats = {}
        if hasattr(self.db_statistics, 'column_stats'):
            db_column_stats = self.db_statistics.column_stats
        
        # Process each output column
        for output_column in output_columns:
            # Convert to dict if needed
            if not isinstance(output_column, dict):
                if hasattr(output_column, '__dict__'):
                    output_col_dict = vars(output_column)
                else:
                    # Skip if can't convert
                    continue
            else:
                output_col_dict = output_column
            
            # Get aggregation and columns
            aggregation = output_col_dict.get('aggregation')
            columns = output_col_dict.get('columns', [])
            
            # Create key for deduplication
            columns_tuple = tuple(columns) if columns else ()
            output_col_key = (aggregation, columns_tuple, database_id)
            
            # Check if this output column already exists
            if output_col_key in output_column_idx:
                output_column_node_id = output_column_idx[output_col_key]
            else:
                # Create new output column node
                output_column_node_id = len(output_column_features)
                
                # Extract output column features
                output_col_features = self._extract_output_column_features(
                    output_col_dict
                )
                output_column_features.append(output_col_features)
                output_column_idx[output_col_key] = output_column_node_id
                
                # Create edges from base columns to output column
                for column in columns:
                    column_key = (column, database_id)
                    
                    # Check if column node exists
                    if column_key not in column_idx:
                        # Create column node
                        column_node_id = len(column_features)
                        
                        # Extract column features from statistics
                        col_features = self._extract_column_features(
                            column,
                            database_id,
                            db_column_stats
                        )
                        column_features.append(col_features)
                        column_idx[column_key] = column_node_id
                    else:
                        column_node_id = column_idx[column_key]
                    
                    # Add edge from column to output column
                    column_to_output_column_edges.append(
                        (column_node_id, output_column_node_id)
                    )
            
            # Add edge from output column to plan
            output_column_to_plan_edges.append(
                (output_column_node_id, plan_node_id)
            )
    
    def _process_table(
        self,
        node: Any,
        database_id: Any,
        plan_node_id: int,
        table_to_plan_edges: list,
        table_features: list,
        table_idx: dict
    ):
        """
        Process table information for this node.
        
        This extracts table features from standardized statistics.
        """
        plan_params = node.plan_parameters
        
        # Extract table name using feature mapper or direct access
        table = None
        if isinstance(plan_params, dict):
            table = plan_params.get('table')
        elif hasattr(plan_params, 'table'):
            table = plan_params.table
        
        if table is None:
            return
        
        # Check if we've already processed this table
        table_key = (table, database_id)
        if table_key in table_idx:
            # Use existing table node
            table_node_id = table_idx[table_key]
        else:
            # Create new table node
            table_stats = self.db_statistics.get_table_stats(table)
            
            if table_stats is None:
                # No statistics available for this table
                return
            
            # Get logical feature names for tables
            if hasattr(self.plan_featurization, 'TABLE_FEATURES'):
                logical_names = self.plan_featurization.TABLE_FEATURES
            else:
                logical_names = ['row_count', 'page_count']
            
            # Extract table features
            table_feat = []
            for logical_name in logical_names:
                # Create a params-like dict with the table stats
                if logical_name == 'row_count':
                    params_dict = {'reltuples': table_stats.row_count}
                elif logical_name == 'page_count':
                    params_dict = {'relpages': table_stats.page_count if table_stats.page_count else 0}
                else:
                    params_dict = {}
                
                # Encode the value
                encoded_value = self._encode_feature(logical_name, params_dict)
                table_feat.append(encoded_value)
            
            table_node_id = len(table_features)
            table_features.append(table_feat)
            table_idx[table_key] = table_node_id
        
        # Add edge from table to plan
        table_to_plan_edges.append((table_node_id, plan_node_id))
    
    def _parse_predicates(
        self,
        filter_column: Any,
        database_id: Any,
        plan_node_id: int,
        filter_to_plan_edges: list,
        predicate_col_features: list,
        predicate_depths: list,
        intra_predicate_edges: list,
        logical_preds: list,
        db_column_stats: dict,
        parent_filter_node_id: Optional[int],
        depth: int
    ):
        """
        Recursively parse predicate tree.
        
        This creates predicate nodes and edges for filter conditions.
        Handles both comparison operators (=, <, >) and logical operators (AND, OR).
        
        Args:
            filter_column: Predicate node (PredicateNode or dict)
            database_id: Database identifier
            plan_node_id: Plan node to connect to (if depth=0)
            filter_to_plan_edges: Edges from predicates to plan
            predicate_col_features: Predicate feature vectors
            predicate_depths: Depth of each predicate node
            intra_predicate_edges: Edges within predicate tree
            logical_preds: Boolean flags for logical predicates
            db_column_stats: Column statistics
            parent_filter_node_id: Parent predicate node (for recursion)
            depth: Current depth in predicate tree
        """
        # Get current filter node ID
        filter_node_id = len(predicate_depths)
        predicate_depths.append(depth)
        
        # Extract operator and column
        operator = None
        column = None
        children = []
        
        # Handle different predicate formats (PredicateNode, dict, SimpleNamespace)
        if isinstance(filter_column, dict):
            operator = filter_column.get('operator')
            column = filter_column.get('column')
            children = filter_column.get('children', [])
        elif hasattr(filter_column, 'operator'):
            operator = filter_column.operator
            column = getattr(filter_column, 'column', None)
            children = getattr(filter_column, 'children', [])
        
        # Import operator types
        try:
            from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
            
            # Check if this is a comparison operator or logical operator
            is_logical = False
            if operator is not None:
                operator_str = str(operator)
                is_logical = operator_str in {'AND', 'OR'} or (
                    hasattr(LogicalOperator, 'AND') and operator in {LogicalOperator.AND, LogicalOperator.OR}
                )
        except ImportError:
            # Fallback: check string representation
            operator_str = str(operator) if operator else ""
            is_logical = operator_str in {'AND', 'OR'}
        
        # Extract features
        if not is_logical:
            # Comparison operator (=, <, >, etc.)
            # Extract filter features (operator, literal)
            filter_features = self._extract_filter_features(filter_column)
            
            # Extract column features if column exists
            if column is not None and column in db_column_stats:
                col_stats = db_column_stats[column]
                column_features = self._extract_column_features_from_stats(col_stats)
                # Combine filter and column features
                filter_features.extend(column_features)
            else:
                # No column statistics available (e.g., HAVING clause)
                # Add zero padding for column features
                if hasattr(self.plan_featurization, 'COLUMN_FEATURES'):
                    n_col_features = len(self.plan_featurization.COLUMN_FEATURES)
                    filter_features.extend([0] * n_col_features)
            
            logical_preds.append(False)
        else:
            # Logical operator (AND, OR)
            # Only extract operator features
            filter_features = self._extract_filter_features(filter_column)
            logical_preds.append(True)
        
        predicate_col_features.append(filter_features)
        
        # Add edge either to plan or to parent predicate
        if depth == 0:
            # Top-level predicate connects to plan
            filter_to_plan_edges.append((filter_node_id, plan_node_id))
        else:
            # Nested predicate connects to parent
            if parent_filter_node_id is not None:
                intra_predicate_edges.append((filter_node_id, parent_filter_node_id))
        
        # Recursively process children
        for child in children:
            self._parse_predicates(
                filter_column=child,
                database_id=database_id,
                plan_node_id=plan_node_id,
                filter_to_plan_edges=filter_to_plan_edges,
                predicate_col_features=predicate_col_features,
                predicate_depths=predicate_depths,
                intra_predicate_edges=intra_predicate_edges,
                logical_preds=logical_preds,
                db_column_stats=db_column_stats,
                parent_filter_node_id=filter_node_id,
                depth=depth + 1
            )
    
    def _extract_filter_features(self, filter_column: Any) -> list:
        """
        Extract filter/predicate features.
        
        Returns:
            List of encoded feature values
        """
        # Get logical feature names for filters
        if hasattr(self.plan_featurization, 'FILTER_FEATURES'):
            logical_names = self.plan_featurization.FILTER_FEATURES
        else:
            logical_names = ['filter_operator', 'literal_feature']
        
        features = []
        for logical_name in logical_names:
            # Use encode() with filter_column as params
            encoded_value = self._encode_feature(logical_name, filter_column)
            features.append(encoded_value)
        
        return features
    
    def _extract_output_column_features(self, output_col_dict: dict) -> list:
        """
        Extract output column features.
        
        Args:
            output_col_dict: Dictionary with 'aggregation' and 'columns'
        
        Returns:
            List of encoded feature values
        """
        # Get logical feature names for output columns
        if hasattr(self.plan_featurization, 'OUTPUT_COLUMN_FEATURES'):
            logical_names = self.plan_featurization.OUTPUT_COLUMN_FEATURES
        else:
            logical_names = ['aggregation']
        
        features = []
        for logical_name in logical_names:
            # Use encode() with output_col_dict as params
            encoded_value = self._encode_feature(logical_name, output_col_dict)
            features.append(encoded_value)
        
        return features
    
    def _extract_column_features(
        self,
        column: Any,
        database_id: Any,
        db_column_stats: dict
    ) -> list:
        """
        Extract column features from statistics.
        
        Args:
            column: Column identifier (tuple or string)
            database_id: Database identifier
            db_column_stats: Dictionary of column statistics
        
        Returns:
            List of encoded feature values
        """
        # Get column statistics
        if column in db_column_stats:
            col_stats = db_column_stats[column]
            return self._extract_column_features_from_stats(col_stats)
        else:
            # No statistics available, return default values
            if hasattr(self.plan_featurization, 'COLUMN_FEATURES'):
                n_features = len(self.plan_featurization.COLUMN_FEATURES)
                return [0] * n_features
            return []
    
    def _extract_column_features_from_stats(self, col_stats: Any) -> list:
        """
        Extract column features from column statistics object.
        
        Args:
            col_stats: ColumnStats or dict-like object
        
        Returns:
            List of encoded feature values
        """
        # Get logical feature names for columns
        if hasattr(self.plan_featurization, 'COLUMN_FEATURES'):
            logical_names = self.plan_featurization.COLUMN_FEATURES
        else:
            logical_names = ['avg_width', 'data_type', 'n_distinct', 'null_frac']
        
        features = []
        for logical_name in logical_names:
            # Create params dict with mapped attribute names
            params_dict = {}
            
            if logical_name == 'avg_width':
                params_dict['avg_width'] = getattr(col_stats, 'avg_width', 0)
            elif logical_name == 'correlation':
                params_dict['correlation'] = getattr(col_stats, 'correlation', 0.0)
            elif logical_name == 'data_type':
                data_type = getattr(col_stats, 'data_type', None)
                params_dict['data_type'] = str(data_type.value) if data_type else 'unknown'
            elif logical_name == 'n_distinct':
                params_dict['n_distinct'] = getattr(col_stats, 'distinct_count', 1.0)
            elif logical_name == 'null_frac':
                params_dict['null_frac'] = getattr(col_stats, 'null_fraction', 0.0)
            
            # Encode the value
            encoded_value = self._encode_feature(logical_name, params_dict)
            features.append(encoded_value)
        
        return features


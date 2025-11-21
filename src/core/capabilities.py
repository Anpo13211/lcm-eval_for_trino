from enum import Enum
from typing import Set, Dict, List, Optional

class Capability(str, Enum):
    """
    Enumeration of DBMS capabilities.
    
    These define what features and information a DBMS plugin can provide.
    Models can declare requirements based on these capabilities.
    """
    # Cost and Cardinality
    COST_ESTIMATION = "cost_estimation"          # Provides estimated costs (startup, total)
    CARDINALITY_ESTIMATION = "cardinality_estimation"  # Provides estimated cardinality
    
    # Statistics
    COLUMN_STATISTICS = "column_statistics"      # Provides basic column stats (width, distinct, nulls)
    TABLE_STATISTICS = "table_statistics"        # Provides table stats (row count)
    HISTOGRAM_STATS = "histogram_stats"          # Provides value distribution histograms
    CORRELATION_STATS = "correlation_stats"      # Provides column correlations
    
    # Execution Details
    PARALLEL_EXECUTION = "parallel_execution"    # Provides parallel execution info (workers)
    ACTUAL_RUNTIME = "actual_runtime"            # Provides actual runtime/cardinality (analyze)
    
    # Plan Structure
    PHYSICAL_PLAN = "physical_plan"              # Provides detailed physical plan operators
    DISTRIBUTED_PLAN = "distributed_plan"        # Provides distributed execution plan info


def check_capabilities(
    required: Set[Capability], 
    provided: Set[Capability], 
    model_name: str, 
    dbms_name: str
) -> List[str]:
    """
    Check if provided capabilities meet requirements.
    
    Args:
        required: Set of required capabilities
        provided: Set of provided capabilities
        model_name: Name of the model
        dbms_name: Name of the DBMS
        
    Returns:
        List of missing capabilities (empty if compatible)
    """
    missing = []
    for req in required:
        if req not in provided:
            missing.append(req)
            
    return missing


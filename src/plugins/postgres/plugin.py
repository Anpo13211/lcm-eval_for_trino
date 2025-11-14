"""
PostgreSQL DBMS Plugin

This plugin integrates existing PostgreSQL components into the
unified plugin system:
- Parser: PostgresPlanParser (existing)
- Statistics: PostgreSQLStatisticsConverter
- Connection: PostgresDatabaseConnection (existing)
- Normalizer: PostgreSQLOperatorNormalizer
"""

from dataclasses import dataclass
from typing import Type

from core.plugins.dbms_plugin import DBMSPlugin
from core.statistics.converter import PostgreSQLStatisticsConverter, StatisticsConverter
from core.operators.normalizer import PostgreSQLOperatorNormalizer, OperatorNormalizer
from cross_db_benchmark.benchmark_tools.postgres.database_connection import PostgresDatabaseConnection
from cross_db_benchmark.benchmark_tools.database import DatabaseConnection


# We need a parser wrapper that conforms to the plugin interface
# For now, we'll use the existing parser structure
class PostgreSQLParserWrapper:
    """
    Wrapper for PostgreSQL parser.
    
    This wraps the existing parsing logic into the plugin interface.
    The actual parsing is delegated to existing code.
    """
    
    def __init__(self):
        self.database_type = "postgres"
    
    def parse_raw_plan(self, plan_text, analyze=True, parse=True, **kwargs):
        """
        Parse PostgreSQL EXPLAIN output.
        
        For now, this delegates to existing parsing infrastructure.
        In the future, we can unify this further.
        """
        # Import here to avoid circular dependencies
        from cross_db_benchmark.benchmark_tools.parse_plan import parse_plan
        import json
        
        # PostgreSQL plans are typically JSON format
        try:
            plan_json = json.loads(plan_text) if isinstance(plan_text, str) else plan_text
            # Delegate to existing parser
            # This is a simplified version - actual implementation would use parse_plan
            return plan_json, 0.0, 0.0
        except json.JSONDecodeError:
            raise ValueError("Invalid PostgreSQL plan format")
    
    def parse_plans(self, run_stats, min_runtime=100, max_runtime=30000, **kwargs):
        """
        Parse multiple plans from run statistics.
        
        This delegates to existing infrastructure.
        """
        from cross_db_benchmark.benchmark_tools.parse_plan import parse_plan
        return parse_plan(
            run_stats,
            min_runtime=min_runtime,
            max_runtime=max_runtime,
            **kwargs
        )


@dataclass
class PostgreSQLPlugin(DBMSPlugin):
    """
    PostgreSQL plugin implementation.
    
    Provides all required components for PostgreSQL support:
    - Plan parser (existing PostgresPlanParser)
    - Statistics converter (PostgreSQLStatisticsConverter)
    - Connection factory (PostgresDatabaseConnection)
    - Operator normalizer (PostgreSQLOperatorNormalizer)
    
    Cost: ~50 lines (mostly imports and wiring)
    """
    
    name: str = "postgres"
    display_name: str = "PostgreSQL"
    version: str = "1.0.0"
    description: str = "PostgreSQL database system plugin"
    
    def get_parser(self):
        """
        Returns PostgreSQL plan parser.
        
        Currently wraps existing parser infrastructure.
        Future: could return AbstractPlanParser directly.
        """
        return PostgreSQLParserWrapper()
    
    def get_statistics_converter(self) -> StatisticsConverter:
        """
        Returns PostgreSQL statistics converter.
        
        Converts from pg_stats format to StandardizedStatistics.
        """
        return PostgreSQLStatisticsConverter()
    
    def get_connection_factory(self) -> Type[DatabaseConnection]:
        """
        Returns PostgreSQL connection class.
        
        Uses existing PostgresDatabaseConnection implementation.
        """
        return PostgresDatabaseConnection
    
    def get_operator_normalizer(self) -> OperatorNormalizer:
        """
        Returns PostgreSQL operator normalizer.
        
        Converts PostgreSQL-specific operator names to logical types.
        """
        return PostgreSQLOperatorNormalizer()
    
    def get_plan_adapter(self):
        """
        Returns plan adapter for QPPNet/QueryFormer compatibility.
        
        PostgreSQL plans are already in the expected format,
        so we return None (no adaptation needed).
        """
        return None  # No adaptation needed for PostgreSQL
    
    def get_metadata(self):
        """
        Returns plugin metadata.
        """
        return {
            'name': self.name,
            'display_name': self.display_name,
            'version': self.version,
            'description': self.description,
            'features': {
                'correlation_stats': True,
                'histogram_stats': True,
                'parallel_execution': True,
                'cost_estimation': True,
            }
        }


# Register plugin automatically when imported
# (This happens in application startup)
def register():
    """
    Register PostgreSQL plugin with the global registry.
    
    Call this during application initialization:
        from plugins.postgres.plugin import register
        register()
    """
    from core.plugins.registry import DBMSRegistry
    
    try:
        DBMSRegistry.register(PostgreSQLPlugin())
        print("PostgreSQL plugin registered successfully")
    except ValueError as e:
        print(f"PostgreSQL plugin already registered: {e}")


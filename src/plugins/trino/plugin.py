"""
Trino DBMS Plugin

This plugin integrates existing Trino components into the
unified plugin system:
- Parser: TrinoPlanParser (existing)
- Statistics: TrinoStatisticsConverter
- Connection: TrinoDatabaseConnection (existing)
- Normalizer: TrinoOperatorNormalizer
"""

from dataclasses import dataclass
from typing import Type

from core.plugins.dbms_plugin import DBMSPlugin
from core.statistics.converter import TrinoStatisticsConverter, StatisticsConverter
from core.operators.normalizer import TrinoOperatorNormalizer, OperatorNormalizer
from cross_db_benchmark.benchmark_tools.trino.database_connection import TrinoDatabaseConnection
from cross_db_benchmark.benchmark_tools.database import DatabaseConnection
from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser


@dataclass
class TrinoPlugin(DBMSPlugin):
    """
    Trino plugin implementation.
    
    Provides all required components for Trino support:
    - Plan parser (TrinoPlanParser)
    - Statistics converter (TrinoStatisticsConverter)
    - Connection factory (TrinoDatabaseConnection)
    - Operator normalizer (TrinoOperatorNormalizer)
    
    Cost: ~50 lines (mostly imports and wiring)
    """
    
    name: str = "trino"
    display_name: str = "Trino"
    version: str = "1.0.0"
    description: str = "Trino distributed query engine plugin"
    
    def get_parser(self):
        """
        Returns Trino plan parser.
        
        Uses existing TrinoPlanParser which already implements
        AbstractPlanParser interface.
        """
        return TrinoPlanParser()
    
    def get_statistics_converter(self) -> StatisticsConverter:
        """
        Returns Trino statistics converter.
        
        Converts from Trino SHOW STATS format to StandardizedStatistics.
        """
        return TrinoStatisticsConverter()
    
    def get_connection_factory(self) -> Type[DatabaseConnection]:
        """
        Returns Trino connection class.
        
        Uses existing TrinoDatabaseConnection implementation.
        """
        return TrinoDatabaseConnection
    
    def get_operator_normalizer(self) -> OperatorNormalizer:
        """
        Returns Trino operator normalizer.
        
        Converts Trino-specific operator names to logical types.
        """
        return TrinoOperatorNormalizer()
    
    def get_plan_adapter(self):
        """
        Returns plan adapter for QPPNet/QueryFormer compatibility.
        """
        from models.qppnet.trino_adapter import adapt_trino_plan_to_qppnet
        return adapt_trino_plan_to_qppnet
    
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
                'correlation_stats': False,
                'histogram_stats': False,
                'parallel_execution': True,
                'cost_estimation': True,
                'distributed': True,
            }
        }


# Register plugin automatically when imported
def register():
    """
    Register Trino plugin with the global registry.
    
    Call this during application initialization:
        from plugins.trino.plugin import register
        register()
    """
    from core.plugins.registry import DBMSRegistry
    
    try:
        DBMSRegistry.register(TrinoPlugin())
        print("Trino plugin registered successfully")
    except ValueError as e:
        print(f"Trino plugin already registered: {e}")


from attr import field


class Featurization:
    PLAN_FEATURES = field()
    FILTER_FEATURES = field()
    COLUMN_FEATURES = field()
    OUTPUT_COLUMN_FEATURES = field()
    TABLE_FEATURES = field()


class DACEFeaturization(Featurization):
    PLAN_FEATURES = ["op_name", "est_cost", "est_card"]


class DACEFeaturizationNoCosts(Featurization):
    PLAN_FEATURES = ["op_name", "est_card"]


class DACEActCardFeaturization(Featurization):
    PLAN_FEATURES = ["op_name", "est_cost", "act_card"]


class FlatModelFeaturization(Featurization):
    PLAN_FEATURE = "est_card"


class FlatModelActCardFeaturization(Featurization):
    PLAN_FEATURE = "act_card"


class QPPNetFeaturization(Featurization):
    PLAN_FEATURES = ['Plan Width', 'Plan Rows', 'Total Cost']
    JOIN_FEATURES = ['Join Type', 'Parent Relationship']
    # Warning: 'Hash Algorithm' mentioned in the paper, but not existing in query plans
    HASH_FEATURES = ['Hash Buckets', 'Peak Memory Usage']
    SORT_FEATURES = ['Sort Key', 'Sort Method']
    SCAN_FEATURES = ['Relation Name', 'Min', 'Max', 'Mean']
    INDEX_SCAN_FEATURES = ['Relation Name', 'Index Name', 'Scan Direction', 'Min', 'Max', 'Mean']
    AGGREGATE_FEATURES = ['Strategy', 'Partial Mode']

    QPP_NET_OPERATOR_TYPES = {
        "Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Sort": PLAN_FEATURES + SORT_FEATURES,
        "Seq Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Index Scan": PLAN_FEATURES + INDEX_SCAN_FEATURES,
        "Index Only Scan": PLAN_FEATURES + INDEX_SCAN_FEATURES,
        "Bitmap Heap Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Bitmap Index Scan": PLAN_FEATURES + ['Index Name'],
        "Join": PLAN_FEATURES + JOIN_FEATURES,
        "Hash": PLAN_FEATURES + HASH_FEATURES,
        "Materialize": PLAN_FEATURES,
        "Gather": PLAN_FEATURES,
        "Gather Merge": PLAN_FEATURES,
        "Parallel Index Only Scan": PLAN_FEATURES + SCAN_FEATURES + INDEX_SCAN_FEATURES,
        "Partial Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Parallel Seq Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Parallel Index Scan": PLAN_FEATURES + SCAN_FEATURES + INDEX_SCAN_FEATURES,
        "Finalize Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Parallel Bitmap Heap Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Result": PLAN_FEATURES,
    }


class QPPNetNoCostsFeaturization(Featurization):
    PLAN_FEATURES = ['Plan Width', 'Plan Rows']
    JOIN_FEATURES = ['Join Type', 'Parent Relationship']
    HASH_FEATURES = ['Hash Buckets', 'Peak Memory Usage']
    SORT_FEATURES = ['Sort Key', 'Sort Method']
    SCAN_FEATURES = ['Relation Name', 'Min', 'Max', 'Mean']
    INDEX_SCAN_FEATURES = ['Relation Name', 'Index Name', 'Scan Direction', 'Min', 'Max', 'Mean']
    AGGREGATE_FEATURES = ['Strategy', 'Partial Mode']

    QPP_NET_OPERATOR_TYPES = {
        "Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Sort": PLAN_FEATURES + SORT_FEATURES,
        "Seq Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Index Scan": PLAN_FEATURES + INDEX_SCAN_FEATURES,
        "Index Only Scan": PLAN_FEATURES + INDEX_SCAN_FEATURES,
        "Bitmap Heap Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Bitmap Index Scan": PLAN_FEATURES + ['Index Name'],
        "Join": PLAN_FEATURES + JOIN_FEATURES,
        "Hash": PLAN_FEATURES + HASH_FEATURES,
        "Materialize": PLAN_FEATURES,
        "Gather": PLAN_FEATURES,
        "Gather Merge": PLAN_FEATURES,
        "Parallel Index Only Scan": PLAN_FEATURES + SCAN_FEATURES + INDEX_SCAN_FEATURES,
        "Partial Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Parallel Seq Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Parallel Index Scan": PLAN_FEATURES + SCAN_FEATURES + INDEX_SCAN_FEATURES,
        "Finalize Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Parallel Bitmap Heap Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Result": PLAN_FEATURES,
    }


class QPPNetActCardsFeaturization(Featurization):
    PLAN_FEATURES = ['Plan Width', 'Actual Rows', 'Total Cost']  # !
    JOIN_FEATURES = ['Join Type', 'Parent Relationship']
    HASH_FEATURES = ['Hash Buckets', 'Peak Memory Usage']
    SORT_FEATURES = ['Sort Key', 'Sort Method']
    SCAN_FEATURES = ['Relation Name', 'Min', 'Max', 'Mean']
    INDEX_SCAN_FEATURES = ['Relation Name', 'Index Name', 'Scan Direction', 'Min', 'Max', 'Mean']
    AGGREGATE_FEATURES = ['Strategy', 'Partial Mode']

    QPP_NET_OPERATOR_TYPES = {
        "Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Sort": PLAN_FEATURES + SORT_FEATURES,
        "Seq Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Index Scan": PLAN_FEATURES + INDEX_SCAN_FEATURES,
        "Index Only Scan": PLAN_FEATURES + INDEX_SCAN_FEATURES,
        "Bitmap Heap Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Bitmap Index Scan": PLAN_FEATURES + ['Index Name'],
        "Join": PLAN_FEATURES + JOIN_FEATURES,
        "Hash": PLAN_FEATURES + HASH_FEATURES,
        "Materialize": PLAN_FEATURES,
        "Gather": PLAN_FEATURES,
        "Gather Merge": PLAN_FEATURES,
        "Parallel Index Only Scan": PLAN_FEATURES + SCAN_FEATURES + INDEX_SCAN_FEATURES,
        "Partial Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Parallel Seq Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Parallel Index Scan": PLAN_FEATURES + SCAN_FEATURES + INDEX_SCAN_FEATURES,
        "Finalize Aggregate": PLAN_FEATURES + AGGREGATE_FEATURES,
        "Parallel Bitmap Heap Scan": PLAN_FEATURES + SCAN_FEATURES,
        "Result": PLAN_FEATURES,
    }

class PostgresTrueCardDetail(Featurization):
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']

    VARIABLES = {
        "column": COLUMN_FEATURES,
        "table": TABLE_FEATURES,
        "output_column": OUTPUT_COLUMN_FEATURES,
        "filter_column": FILTER_FEATURES + COLUMN_FEATURES,
        "plan": PLAN_FEATURES,
        "logical_pred": FILTER_FEATURES,
    }


class PostgresTrueCardMedium(Featurization):
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'data_type', 'table_size']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresTrueCardCoarse(Featurization):
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'data_type']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresEstSystemCardDetail(Featurization):
    PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresEstSystemCardMedium(Featurization):
    PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'data_type', 'table_size']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresEstSystemCardCoarse(Featurization):
    PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'data_type']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresDeepDBEstSystemCardDetail(Featurization):
    PLAN_FEATURES = ['dd_est_card', 'est_width', 'workers_planned', 'op_name', 'dd_est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresTrueCardAblateTableFeats(Featurization):
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = []


class PostgresTrueCardAblateColumnFeats(Featurization):
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = []
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresTrueCardAblateOperatorFeats(Featurization):
    PLAN_FEATURES = ['act_card', 'est_width', 'act_children_card']
    FILTER_FEATURES = []
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = []
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresTrueCardAblateDataDistributionFeats(Featurization):
    PLAN_FEATURES = ['workers_planned', 'op_name']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresTrueCardDecAblationAllFeats(Featurization):
    PLAN_FEATURES = []
    FILTER_FEATURES = []
    COLUMN_FEATURES = []
    OUTPUT_COLUMN_FEATURES = []
    TABLE_FEATURES = []


class PostgresTrueCardDecAblationColumnFeats(Featurization):
    PLAN_FEATURES = []
    FILTER_FEATURES = []
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = []
    TABLE_FEATURES = []


class PostgresTrueCardDecAblationOperatorFeats(Featurization):
    PLAN_FEATURES = ['workers_planned', 'op_name']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = []


class PostgresTrueCardDecAblationTableFeats(Featurization):
    PLAN_FEATURES = ['workers_planned', 'op_name']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']

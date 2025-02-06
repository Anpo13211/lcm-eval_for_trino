import math
from pathlib import Path
from typing import Optional, List

from attr import define

from cross_db_benchmark.benchmark_tools.utils import load_json
from cross_db_benchmark.datasets.datasets import Database
from classes.paths import LocalPaths


@define(slots=False)
class EvaluationWorkload:
    folder: str
    database: Database
    wl_name: Optional[str] = None

    def get_workload_path(self, base_path: Path) -> Path:
        return Path(base_path) / self.database.db_name / self.folder / f'{self.get_workload_name()}.json'

    def get_sql_path(self, base_path: Path) -> Path:
        return Path(base_path) / self.database.db_name / self.folder / f'{self.get_workload_name()}.sql'

    def get_workload_name(self):
        if self.wl_name is not None:
            return self.wl_name
        else:
            raise NotImplementedError()


@define(slots=False)
class JoinOrderEvalWorkload(EvaluationWorkload):
    num_tables: Optional[int] = None
    agg: Optional[bool] = None

    def get_workload_name(self) -> str:
        return self.wl_name

    def get_query_label(self) -> str:
        return self.wl_name


@define(slots=False)
class JoinImplEvalWorkload(EvaluationWorkload):
    wl_name: str
    agg: Optional[bool] = None

    yticks: List[str] = ["Hash\nJoin", "Merge\nJoin", "Index NestLoop\nJoin"]

    def get_workload_name(self) -> str:
        return self.wl_name

    def get_query_label(self) -> str:
        return self.wl_name


@define(slots=False)
class LiteralEvalWorkload(EvaluationWorkload):
    table: str = None
    column: Optional[str] = None
    join_table: Optional[str] = None
    join_column: Optional[str] = None
    yaxis: Optional[range] = None
    categorical: Optional[bool] = None
    agg: Optional[bool] = None

    def is_categorical(self) -> bool:
        col_stats = self.get_column_stats()
        if col_stats[self.table][self.column]["datatype"] == "categorical":
            return True
        else:
            return False

    def get_workload_name(self) -> str:
        if self.wl_name:
            return self.wl_name
        else:
            return f'{self.table}.{self.column}'

    def get_query_label(self) -> str:
        if self.folder == "range_point_filter":
            return f'{self.database.db_name}\nSELECT * FROM {self.table} WHERE {self.table}.{self.column} >= X;'

        elif self.folder == "seq_point_filter":
            return f'{self.database.db_name}\nSELECT * FROM {self.table} WHERE {self.table}.{self.column} = X'

        elif self.folder == "agg_range_filter":
            return f'{self.database.db_name}\nSELECT MAX(*) FROM {self.table} WHERE {self.table}{self.column} >= X;'

        elif self.folder == "join_filter":
            return f'{self.database.db_name}\n SELECT * FROM {self.table}\nJOIN {self.join_table} ON {self.table}.{self.join_column} = {self.join_table}.{self.join_column}\nWHERE {self.table}.{self.column} = X'

        elif self.folder == "join_range_filter":
            return f'{self.database.db_name}\n SELECT * FROM {self.table}\nJOIN {self.join_table} ON {self.table}.{self.join_column} = {self.join_table}.{self.join_column}\nWHERE {self.table}.{self.column} >= X'

        else:
            raise RuntimeError(f'Unknown folder: {self.folder} to generate title')

    def get_column_stats(self) -> dict:
        column_statistics_path = LocalPaths().code / "cross_db_benchmark" / "datasets" / self.database.db_name / "column_statistics.json"
        column_stats = load_json(column_statistics_path, namespace=False)
        return column_stats

    def get_y_range(self) -> list:
        col_stats = self.get_column_stats()[self.table][self.column]
        if "min" in col_stats.keys():
            min, max = int(col_stats["min"]), int(col_stats["max"])
            step = math.ceil((max - min) / 100)
            return list(range(min, max, step))

        elif "unique_vals" in col_stats.keys():
            unique_vals = col_stats["unique_vals"][0:100]
            return unique_vals


@define(slots=False)
class BenchmarkWorkload(EvaluationWorkload):
    wl_name: str

    def get_workload_name(self) -> str:
        return self.wl_name

class RetrainingWorkloads:
    class AccessPath:
        imdb = EvaluationWorkload(database=Database("imdb"), folder="access_path", wl_name="imdb_access_path")


class EvalWorkloads:

    class ScanCostsPercentile:
        folder = "scan_costs_percentiles"

        tpc_h_pk_seq = [
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.lineitem.l_extendedprice", table="lineitem", column="l_extendedprice"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.lineitem.l_partkey", table="lineitem", column="l_partkey"),
            #LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
            #                    wl_name="seq.nation.n_nationkey", table="nation", column="n_nationkey"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.part.p_retailprice", table="part", column="p_retailprice"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.part.p_size", table="part", column="p_size"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.partsupp.ps_availqty", table="partsupp", column="ps_availqty"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.partsupp.ps_partkey", table="partsupp", column="ps_partkey"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.partsupp.ps_suppkey", table="partsupp", column="ps_suppkey"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="seq.partsupp.ps_supplycost", table="partsupp", column="ps_supplycost"),
            #LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
            #                    wl_name="seq.region.r_regionkey", table="region", column="r_regionkey"),
        ]

        tpc_h_pk_idx = [
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.lineitem.l_extendedprice", table="lineitem", column="l_extendedprice"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.lineitem.l_partkey", table="lineitem", column="l_partkey"),
            #LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
            #                    wl_name="index.nation.n_nationkey", table="nation", column="n_nationkey"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.part.p_retailprice", table="part", column="p_retailprice"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.part.p_size", table="part", column="p_size"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.partsupp.ps_availqty", table="partsupp", column="ps_availqty"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.partsupp.ps_partkey", table="partsupp", column="ps_partkey"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.partsupp.ps_suppkey", table="partsupp", column="ps_suppkey"),
            LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
                                wl_name="index.partsupp.ps_supplycost", table="partsupp", column="ps_supplycost"),
            #LiteralEvalWorkload(database=Database("tpc_h_pk"), folder=folder,
            #                    wl_name="index.region.r_regionkey", table="region", column="r_regionkey"),
        ]

        imdb_seq = [
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="seq.aka_name.person_id", table="aka_name", column="person_id"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="seq.cast_info.nr_order", table="cast_info", column="nr_order"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="seq.cast_info.movie_id", table="cast_info", column="movie_id"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="seq.person_info.person_id", table="person_info", column="person_id"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="seq.title.episode_nr", table="title", column="episode_nr"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="seq.title.production_year", table="title", column="production_year"),
        ]

        imdb_idx = [
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="index.aka_name.person_id", table="aka_name", column="person_id"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="index.cast_info.nr_order", table="cast_info", column="nr_order"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="index.cast_info.movie_id", table="cast_info", column="movie_id"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="index.person_info.person_id", table="person_info", column="person_id"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="index.title.episode_nr", table="title", column="episode_nr"),
            LiteralEvalWorkload(database=Database("imdb"), folder=folder,
                                wl_name="index.title.production_year", table="title", column="production_year"),
        ]

        baseball_seq = [
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.batting.AB", table="batting", column="AB"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.batting.G_batting", table="batting", column="G_batting"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.halloffame.needed", table="halloffame", column="needed"),
            #LiteralEvalWorkload(database=Database("baseball"), folder=folder,
            #                    wl_name="seq.battingpost.yearID", table="battingpost", column="yearID"),
            #LiteralEvalWorkload(database=Database("baseball"), folder=folder,
            #                    wl_name="seq.halloffame.ballots", table="halloffame", column="ballots"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.managers.L", table="managers", column="L"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.managers.W", table="managers", column="W"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.managershalf.L", table="managershalf", column="L"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="seq.managershalf.W", table="managershalf", column="W"),
        ]

        baseball_idx = [
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.batting.AB", table="batting", column="AB"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.batting.G_batting", table="batting", column="G_batting"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.halloffame.needed", table="halloffame", column="needed"),
            #LiteralEvalWorkload(database=Database("baseball"), folder=folder,
            #                    wl_name="index.battingpost.yearID", table="battingpost", column="yearID"),
            #LiteralEvalWorkload(database=Database("baseball"), folder=folder,
            #                    wl_name="index.halloffame.ballots", table="halloffame", column="ballots"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.managers.L", table="managers", column="L"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.managers.W", table="managers", column="W"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.managershalf.L", table="managershalf", column="L"),
            LiteralEvalWorkload(database=Database("baseball"), folder=folder,
                                wl_name="index.managershalf.W", table="managershalf", column="W"),
        ]



    class NumPredicates:
        baseball = [
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="1_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="2_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="3_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="4_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="5_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="6_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="7_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="8_attributes"),
            EvaluationWorkload(database=Database("baseball"), folder="num_predicates", wl_name="9_attributes"),
        ]

        tpc_h = [
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="1_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="2_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="3_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="4_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="5_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="6_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="7_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="8_attributes"),
            EvaluationWorkload(database=Database("tpc_h"), folder="num_predicates", wl_name="9_attributes"),
        ]

        imdb = [
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="1_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="2_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="3_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="4_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="5_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="6_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="7_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="8_attributes"),
            EvaluationWorkload(database=Database("imdb"), folder="num_predicates", wl_name="9_attributes"),
        ]

    class FullJoinOrder:
        folder = "join_order_full"
        imdb = [
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_0", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_1", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_2", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_3", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_4", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_5", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_6", agg=True,
                                  num_tables=3),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_7", agg=True,
            #                      num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_8", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_9", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_10", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_11", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_12", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_13", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_14", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_15", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_16", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_17", agg=True,
                                  num_tables=3),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_18", agg=True,
            #                      num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_19", agg=True,
                                  num_tables=2),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_20", agg=True,
                                  num_tables=2),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_21", agg=True,
                                  num_tables=2),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_22", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_23", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_24", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_25", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_26", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_27", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_28", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_29", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_30", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_31", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_32", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_33", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_34", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_35", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_36", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_37", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_38", agg=True,
                                  num_tables=4),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_39", agg=True,
            #                      num_tables=4),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_40", agg=True,
            #                      num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_41", agg=True,
                                  num_tables=4),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_42", agg=True,
            #                     num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_43", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_44", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_45", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_46", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_47", agg=True,
                                  num_tables=4),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_48", agg=True,
            #                      num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_49", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_50", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_51", agg=True,
                                  num_tables=3),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_52", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_53", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_54", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_55", agg=True,
                                  num_tables=5),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_56", agg=True,
            #                      num_tables=5),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_57", agg=True,
            #                      num_tables=5),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_58", agg=True,
            #                      num_tables=5),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_59", agg=True,
            #                      num_tables=5),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_60", agg=True,
            #                      num_tables=5),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_61", agg=True,
                                  num_tables=5),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_62", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_63", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_64", agg=True,
                                  num_tables=4),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_65", agg=True,
                                  num_tables=5),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_66", agg=True,
                                  num_tables=5),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_67", agg=True,
                                  num_tables=5),
            JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_68", agg=True,
                                  num_tables=5),
            #JoinOrderEvalWorkload(database=Database("imdb"), folder=folder, wl_name="job_light_69", agg=True,
            #                      num_tables=5),
        ]

    class Benchmarks:
        job_light = [
            BenchmarkWorkload(database=Database("imdb"), wl_name="job_light_no_timeouts_v10", folder="benchmarks"),
            BenchmarkWorkload(database=Database("imdb"), wl_name="job_light_no_timeouts_act_cards_v10",
                              folder="benchmarks"),
            BenchmarkWorkload(database=Database("imdb"), wl_name="job_light_no_timeouts_v16", folder="benchmarks"),
            BenchmarkWorkload(database=Database("imdb"), wl_name="job_light_no_timeouts_act_cards_v16",
                              folder="benchmarks"),
        ]

        scale = [BenchmarkWorkload(database=Database("imdb"), wl_name="scale_c8220", folder="benchmarks")]
        synthetic = [BenchmarkWorkload(database=Database("imdb"), wl_name="synthetic_c8220", folder="benchmarks")]

    class ScanCost:
        class SeqPointFilter:
            folder = "seq_point_filter"
            baseball = [
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="allstarfull", column="teamID"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="awardsshareplayers",
                                    column="playerID"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="batting", column="SH"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="fielding", column="A"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="fielding", column="DP"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="fielding", column="teamID"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitching", column="BAOpp"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitchingpost", column="BB"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitchingpost", column="BFP"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitchingpost",
                                    column="playerID"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitchingpost", column="round"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="players", column="nameFirst"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="salaries", column="salary"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="schoolsplayers",
                                    column="yearMin"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="2B"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="DP"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="E"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="PPF"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="SHO"),
            ]

            imdb = [
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="aka_name", column="surname_pcode"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="cast_info", column="nr_order"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="company_name",column="country_code"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="info_type", column="info"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="kind_type", column="kind"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="name", column="imdb_index"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="name", column="surname_pcode"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="title", column="episode_nr"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="title", column="production_year"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="title", column="season_nr"),
            ]

            tpc_h = [
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="customer", column="c_acctbal"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="customer", column="c_custkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_orderkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_partkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem",
                                    column="l_receiptdate"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_shipdate"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_suppkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="nation", column="n_comment"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="nation", column="n_name"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="nation", column="n_nationkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="orders", column="o_clerk"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="orders", column="o_orderkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_brand"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_partkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_retailprice"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_size"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_type"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="partsupp", column="ps_partkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="supplier", column="s_phone"),
            ]

        class RangePointFilter:
            folder = "range_point_filter"
            baseball = [
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="batting", column="SH"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="fielding", column="A"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="fielding", column="DP"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitching", column="BAOpp"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitchingpost", column="BB"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="pitchingpost", column="BFP"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="salaries", column="salary"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="schoolsplayers",
                                    column="yearMin"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="2B"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="CG"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="DP"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="E"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="PPF"),
                LiteralEvalWorkload(folder=folder, database=Database("baseball"), table="teams", column="SHO"),
            ]

            imdb = [
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="cast_info", column="nr_order"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="title", column="episode_nr"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="title", column="production_year"),
                LiteralEvalWorkload(folder=folder, database=Database("imdb"), table="title", column="season_nr"),
            ]

            tpc_h = [
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="customer", column="c_acctbal"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="customer", column="c_custkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_orderkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_partkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="lineitem", column="l_suppkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="nation", column="n_nationkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="orders", column="o_orderkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_partkey"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_retailprice"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="part", column="p_size"),
                LiteralEvalWorkload(database=Database("tpc_h"), folder=folder, table="partsupp", column="ps_partkey")
            ]

        class JoinFilter:
            folder = "join_filter"
            baseball = [
                LiteralEvalWorkload(folder="join_filter", database=Database("baseball", display_name="Baseball"),
                                    table="teams", column="2B")]
            imdb = [
                LiteralEvalWorkload(folder="join_filter", database=Database("imdb", display_name="IMDB"), table="title",
                                    column="production_year")]
            tpc_h = [LiteralEvalWorkload(folder="join_filter", database=Database("tpc_h", display_name="TPC-H"),
                                         table="part", column="p_retailprice")]

        class JoinRangeFilters:
            baseball = [
                LiteralEvalWorkload(folder="join_range_filter", database=Database("baseball", display_name="Baseball"),
                                    table="teams", column="2B")]
            imdb = [LiteralEvalWorkload(folder="join_range_filter", database=Database("imdb", display_name="IMDB"),
                                        table="title", column="production_year")]
            tpc_h = [LiteralEvalWorkload(folder="join_range_filter", database=Database("tpc_h", display_name="TPC-H"),
                                         table="part", column="p_retailprice")]

        class FourWayJoin:
            folder = "join_range_filter"
            baseball = [LiteralEvalWorkload(database=Database("baseball", display_name="Baseball"), folder=folder,
                                            table="pitching", column="h")]
            imdb = [
                LiteralEvalWorkload(database=Database("imdb", display_name="IMDB"), folder=folder, table="movie_info",
                                    column="info_type_id")]
            tpc = [
                LiteralEvalWorkload(database=Database("tpc_h", display_name="TPC-H"), folder=folder, table="lineitem",
                                    column="l_partkey")]

    class PhysicalPlan:
        folder = "physical_plan"
        baseball = [
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1003"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1012"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1016"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1023"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1030"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1081"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1097"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1107"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1108"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1110"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1126"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1146"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1171"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_120"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_1200"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_124"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_125"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_131"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_132"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_134"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_139"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_145"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_158"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_180"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_182"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_185"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_191"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_196"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_200"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_211"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_215"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_217"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_224"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_226"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_235"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_237"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_241"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_264"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_286"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_287"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_305"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_319"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_327"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_328"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_331"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_365"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_396"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_40"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_408"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_41"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_436"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_44"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_46"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_467"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_473"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_492"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_496"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_499"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_504"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_51"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_522"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_539"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_582"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_587"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_604"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_608"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_609"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_616"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_637"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_651"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_657"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_667"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_683"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_693"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_700"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_722"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_729"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_750"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_751"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_782"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_81"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_816"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_861"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_864"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_87"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_878"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_891"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_90"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_903"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_908"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_918"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_922"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_936"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_938"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_952"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_957"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_969"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_975"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_980"),
            JoinImplEvalWorkload(database=Database("baseball"), folder=folder, wl_name="baseball_994"),
        ]
        imdb = [
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_0"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_100"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_119"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_121"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_123"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_130"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_131"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_137"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_140"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_141"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_142"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_147"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_15"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_150"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_151"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_152"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_156"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_16"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_161"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_165"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_168"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_169"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_174"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_184"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_19"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_194"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_196"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_201"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_206"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_21"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_210"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_218"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_221"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_225"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_231"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_24"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_240"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_244"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_246"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_249"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_250"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_255"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_256"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_259"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_262"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_267"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_27"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_287"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_29"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_290"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_298"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_300"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_305"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_31"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_310"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_313"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_318"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_322"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_331"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_333"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_334"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_335"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_336"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_337"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_341"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_342"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_345"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_346"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_348"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_35"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_354"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_363"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_38"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_383"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_386"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_39"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_397"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_403"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_409"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_411"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_414"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_415"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_418"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_43"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_434"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_436"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_442"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_443"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_453"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_467"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_49"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_54"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_57"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_61"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_7"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_77"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_87"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_89"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_92"),
            JoinImplEvalWorkload(database=Database("imdb"), folder=folder, wl_name="scale_99"),
        ]

        imdb_with_indexes = [
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_0"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_100"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_119"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_121"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_123"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_130"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_131"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_137"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_140"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_141"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_142"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_147"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_15"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_150"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_151"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_152"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_156"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_16"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_161"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_165"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_168"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_169"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_174"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_184"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_19"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_194"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_196"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_201"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_206"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_21"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_210"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_218"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_221"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_225"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_231"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_24"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_240"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_244"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_246"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_249"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_250"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_255"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_256"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_259"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_262"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_267"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_27"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_287"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_29"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_290"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_298"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_300"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_305"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_31"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_310"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_313"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_318"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_322"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_331"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_333"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_334"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_335"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_336"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_337"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_341"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_342"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_345"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_346"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_348"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_35"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_354"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_363"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_38"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_383"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_386"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_39"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_397"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_403"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_409"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_411"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_414"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_415"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_418"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_43"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_434"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_436"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_442"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_443"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_453"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_467"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_49"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_54"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_57"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_61"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_7"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_77"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_87"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_89"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_92"),
            JoinImplEvalWorkload(database=Database("imdb"), folder="physical_index_plan", wl_name="scale_99"),
        ]
        tpc_h = [
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1001"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1005"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1028"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_103"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1052"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1053"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1062"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1066"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1072"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1086"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1098"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1122"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1130"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1147"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1166"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1177"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1201"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1215"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1226"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1234"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1243"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1252"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1260"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1273"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1303"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1335"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1387"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1414"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1428"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1442"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_148"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1499"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_15"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1514"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1524"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1526"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1546"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1558"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_1563"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_160"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_163"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_179"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_184"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_216"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_231"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_239"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_278"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_300"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_312"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_350"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_355"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_358"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_361"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_363"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_386"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_408"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_429"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_442"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_445"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_451"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_454"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_483"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_486"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_504"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_524"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_542"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_548"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_557"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_562"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_577"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_608"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_618"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_679"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_703"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_71"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_710"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_716"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_722"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_729"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_743"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_760"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_766"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_768"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_774"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_803"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_813"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_818"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_822"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_827"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_86"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_862"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_88"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_884"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_900"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_912"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_922"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_929"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_931"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_970"),
            JoinImplEvalWorkload(database=Database("tpc_h"), folder=folder, wl_name="tpc_h_994"),
        ]

        tpc_h_pk = [
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1001"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1005"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1028"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_103"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1052"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1053"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1062"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1066"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1072"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1086"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1098"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1122"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1130"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1147"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1166"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1177"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1201"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1215"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1226"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1234"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1243"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1252"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1260"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1273"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1303"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1335"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1387"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1414"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1428"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1442"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_148"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1499"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_15"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1514"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1524"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1526"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1546"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1558"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_1563"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_160"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_163"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_179"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_184"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_216"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_231"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_239"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_278"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_300"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_312"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_350"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_355"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_358"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_361"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_363"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_386"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_408"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_429"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_442"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_445"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_451"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_454"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_483"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_486"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_504"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_524"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_542"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_548"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_557"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_562"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_577"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_608"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_618"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_679"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_703"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_71"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_710"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_716"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_722"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_729"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_743"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_760"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_766"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_768"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_774"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_803"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_813"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_818"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_822"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_827"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_86"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_862"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_88"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_884"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_900"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_912"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_922"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_929"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_931"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_970"),
            JoinImplEvalWorkload(database=Database("tpc_h_pk"), folder=folder, wl_name="tpc_h_994"),
        ]



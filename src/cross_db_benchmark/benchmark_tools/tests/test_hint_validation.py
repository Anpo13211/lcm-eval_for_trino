import unittest

from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.load_database import create_db_conn


class TestQueryHinting(unittest.TestCase):

    def test_query_hinting(self):
        database_conn_args = dict(user="postgres", password="bM2YGRAX*bG_QAilUid§2iD",
                                  host="clnode063.clemson.cloudlab.us")
        db_conn = create_db_conn(DatabaseSystem.POSTGRES, "imdb", database_conn_args, {})
        db_conn.set_statement_timeout(1, verbose=True)

        queries = []

        # Test Sequential Scan
        queries.append(("/*+ SeqScan(title) */ "
                        "SELECT * FROM title WHERE title.production_year>=1950;", None))

        # Test unused hint, should fail
        queries.append(("/*+ SeqScan(title) SeqScan(cast_info)*/ "
                        "SELECT * FROM title WHERE title.production_year>=1950;", AssertionError))

        # Test duplicated hint, should fail
        queries.append(("/*+ SeqScan(title) SeqScan(title)*/ "
                        "SELECT * FROM title WHERE title.production_year>=1950;", SyntaxError))

        # Test typo in hint, should fail
        queries.append(("/*+ SqScan(title) */ "
                        "SELECT * FROM title WHERE title.production_year>=1950;", SyntaxError))

        # Test type in table name, should fail
        queries.append(("/*+ SeqScan(titlex) */ "
                        "SELECT * FROM title WHERE title.production_year>=1950;", AssertionError))

        # Test messed up parentheses
        queries.append(("/*+ SeqScan((title) */ "
                        "SELECT * FROM title WHERE title.production_year>=1950;", SyntaxError))

        # Nested loop join of two tables
        queries.append(("/*+ NestLoop(cast_info title) "
                        "Leading(cast_info title) "
                        "SeqScan(cast_info) "
                        "IndexScan(title title_pkey)*/ "
                        "SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year=2020;",
                        None))

        # Joining 3 tables with fixed hint order, join implementation and scan implementation
        # Attention: Although HashJoin notation seems correct, this hinting fails
        # because internally, the used join hints are returned with an alphabetic order.
        # The join order is solely specified by the Leading-hint, however.

        queries.append(("/*+ "
                        "HashJoin(cast_info title) "
                        "HashJoin(cast_info title movie_companies) "
                        "Leading(cast_info title movie_companies) "
                        "SeqScan(cast_info) SeqScan(title) SeqScan(movie_companies) */ "
                        "SELECT COUNT(*) FROM cast_info,title,movie_companies WHERE title.id=cast_info.movie_id AND "
                        "title.id=movie_companies.movie_id AND title.production_year >= 2020;",
                        AssertionError))

        # This is the correct version.
        queries.append(("/*+ "
                        "HashJoin(cast_info title) "
                        "HashJoin(cast_info movie_companies title) "
                        "Leading(cast_info title movie_companies) "
                        "SeqScan(cast_info) SeqScan(title) SeqScan(movie_companies) */ "
                        "SELECT COUNT(*) FROM cast_info,title,movie_companies WHERE title.id=cast_info.movie_id AND "
                        "title.id=movie_companies.movie_id AND title.production_year >= 2020;",
                        None))

        for i, (sql_query, excepted_error) in enumerate(tqdm(queries)):
            try:
                _ = db_conn.run_query_collect_statistics(sql_query, repetitions=1, prefix="", explain_only=False, include_hint_notices=True)
            except Exception as ex:
                self.assertTrue(isinstance(ex, excepted_error))


#
#'HashJoin(cast_info movie_companies title)'}
#not equal to target hints
#
#'HashJoin(cast_info title movie_companies)'

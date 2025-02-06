import sqlparse

from evaluation.workload_creation.create_evaluation_workloads import get_query_tables
from unittest import TestCase


class TestEvaluationWorkloadGeneration(TestCase):
    def test_get_tables_from_query(self):
        test_queries = [
            'SELECT COUNT(*) FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>2282;',
            'SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=3;',
            'SELECT AVG("title"."kind_id" + "movie_info"."info_type_id") as agg_0 FROM "movie_info" LEFT OUTER JOIN "title" ON "movie_info"."movie_id" = "title"."id" LEFT OUTER JOIN "movie_keyword" ON "title"."id" = "movie_keyword"."movie_id" LEFT OUTER JOIN "cast_info" ON "title"."id" = "cast_info"."movie_id"  WHERE "movie_info"."movie_id" >= 1967626 AND "movie_info"."info_type_id" != 3 AND "title"."production_year" <= 2014.4455885374778 AND "title"."episode_of_id" >= 127217.266451425 AND "cast_info"."id" >= 35529451;',
            'SELECT SUM("person_info"."id" + "person_info"."info_type_id") as agg_0 FROM "person_info"  WHERE "person_info"."id" <= 1472821;',
            'SELECT SUM("movie_keyword"."keyword_id" + "title"."episode_of_id") as agg_0, COUNT(*) as agg_1 FROM "movie_keyword" LEFT OUTER JOIN "keyword" ON "movie_keyword"."keyword_id" = "keyword"."id" LEFT OUTER JOIN "title" ON "movie_keyword"."movie_id" = "title"."id"  WHERE "keyword"."id" >= 125180 AND "title"."series_years" != 1974-1988 AND "movie_keyword"."keyword_id" <= 19569;',
            'SELECT * FROM movie_info;',
            'SELECT COUNT(*) FROM title,cast_info,movie_info,movie_keyword WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=2009;',
            'SELECT COUNT(*) FROM title WHERE title.kind_id>1 AND title.production_year>2013;',
            'SELECT COUNT(*) FROM title,movie_companies,movie_info WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year>2004;',
            'SELECT COUNT(*) FROM title,movie_info,movie_info_idx,movie_keyword WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=3 AND movie_keyword.keyword_id>121648;',
            
            'SELECT SUM("company_type"."id") as agg_0, SUM("company_type"."id") as agg_1 FROM "company_type"  WHERE "company_type"."kind" != "miscellaneous companies" AND "company_type"."id" = 3;',
            'SELECT SUM("title"."episode_nr") as agg_0 FROM "keyword" LEFT OUTER JOIN "movie_keyword" ON "keyword"."id" = "movie_keyword"."keyword_id" LEFT OUTER JOIN "title" ON "movie_keyword"."movie_id" = "title"."id"  WHERE "movie_keyword"."movie_id" <= 491282 AND "keyword"."id" <= 131392 AND "title"."episode_nr" <= 79.49326101538576 AND "title"."kind_id" <= 1;',
            'SELECT COUNT(*) as agg_0 FROM "movie_companies" LEFT OUTER JOIN "company_type" ON "movie_companies"."company_type_id" = "company_type"."id" LEFT OUTER JOIN "company_name" ON "movie_companies"."company_id" = "company_name"."id"  WHERE "movie_companies"."id" >= 264069 AND "movie_companies"."company_id" >= 5730;',
            'SELECT AVG("title"."id" + "movie_keyword"."keyword_id") as agg_0, SUM("movie_keyword"."keyword_id" + "movie_info_idx"."movie_id") as agg_1, COUNT(*) as agg_2 FROM "movie_keyword" LEFT OUTER JOIN "title" ON "movie_keyword"."movie_id" = "title"."id" LEFT OUTER JOIN "keyword" ON "movie_keyword"."keyword_id" = "keyword"."id" LEFT OUTER JOIN "movie_info_idx" ON "title"."id" = "movie_info_idx"."movie_id" LEFT OUTER JOIN "info_type" ON "movie_info_idx"."info_type_id" = "info_type"."id"  WHERE "movie_info_idx"."info_type_id" = 101 AND "keyword"."id" <= 114243 AND "movie_keyword"."movie_id" >= 1906394 AND "movie_keyword"."id" >= 3683464;',
            'SELECT COUNT(*) as agg_0 FROM "movie_info" LEFT OUTER JOIN "title" ON "movie_info"."movie_id" = "title"."id" LEFT OUTER JOIN "kind_type" ON "title"."kind_id" = "kind_type"."id"  WHERE "title"."season_nr" >= 1.82074691411627 AND "kind_type"."kind" = "tv series" AND "title"."id" <= 442673;',
        ]

        table_lengths = [2, 1, 4, 1, 3, 1, 4, 1, 3, 4, 1, 3, 3, 5, 3]

        for query, t_len in zip(test_queries, table_lengths):
            parsed_query = sqlparse.parse(query)
            assert len(parsed_query) == 1, "Multiple queries found per line"
            parsed_query = parsed_query[0]
            join_tables = get_query_tables(parsed_query, set())
            self.assertEqual(len(join_tables), t_len, msg=f"Not equals for query {query}")

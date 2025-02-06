import json
import os
import re
import shutil

from cross_db_benchmark.benchmark_tools.generate_column_stats import generate_column_statistics


def derive_from_relational_fit(rel_fit_dataset_name, dataset_name, host='relational.fit.cvut.cz', user='guest',
                               password='relational', root_data_dir='../zero-shot-data/datasets'):
    assert os.path.exists('cross_db_benchmark/datasets/')
    dataset_dir = os.path.join('cross_db_benchmark/datasets/', dataset_name)
    schema_sql_dir = os.path.join(dataset_dir, 'schema_sql')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(schema_sql_dir, exist_ok=True)

    schema_sql_path = download_mysql_schema(host, password, rel_fit_dataset_name, schema_sql_dir, user)

    # read mysql schema
    with open(schema_sql_path, 'r') as file:
        mysqlschema = file.read()

    derive_postgres_file(mysqlschema, schema_sql_dir)
    tables = derive_acyclic_schema_json(dataset_dir, dataset_name, mysqlschema)

    data_dir = os.path.join(root_data_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    for table in tables:
        target_table_file = os.path.join(data_dir, f'{table}.csv')

        if not os.path.exists(target_table_file):
            # `{table}`
            download_cmd = f'echo "select * from \`{table}\`;" | mysql --host=relational.fit.cvut.cz --user=guest --password=relational {rel_fit_dataset_name} > {table}.csv'
            print(download_cmd)
            os.system(download_cmd)
            filesize = os.path.getsize(f"{table}.csv")
            if filesize == 0:
                print(f"Warning: file {table}.csv is empty")
            shutil.move(f'{table}.csv', target_table_file)
        else:
            print(f"Skipping download for {table}")

    generate_column_statistics(data_dir, dataset_name, force=False)


def derive_acyclic(table, relationships, visited_tables, included_relationship_ids):
    """
    Transforms the schema of relational fit into an acyclic one such that DeepDB can provide cardinalities. Note that
    in contrast, zero-shot learning does support cyclic schemas (as shown in the evaluation of the JOB full queries).
    :param table:
    :param relationships:
    :param visited_tables:
    :param included_relationship_ids:
    :return:
    """
    for r_id, rel in enumerate(relationships):
        if r_id in included_relationship_ids:
            continue

        t1, _, t2, _ = rel
        if t1 == t2:
            continue

        if t1 == table or t2 == table:
            other_table = t1
            if t1 == table:
                other_table = t2

            if other_table not in visited_tables:
                included_relationship_ids.add(r_id)
                visited_tables.add(other_table)
                derive_acyclic(other_table, relationships, visited_tables, included_relationship_ids)


def make_acyclic(tables, relationships):
    start_table = tables[0]
    visited_tables = {start_table}
    included_relationship_ids = set()

    derive_acyclic(start_table, relationships, visited_tables, included_relationship_ids)

    new_relationships = [relationships[r_id] for r_id in included_relationship_ids]
    new_tables = list(visited_tables)
    if len(new_relationships) < len(relationships):
        print("Had to remove relationships in order to derive acyclic schema:")
        for r_id, rel in enumerate(relationships):
            if r_id not in included_relationship_ids:
                print(rel)

    return new_tables, new_relationships


def derive_acyclic_schema_json(dataset_dir, dataset_name, mysqlschema):
    table_regex = re.compile('DROP TABLE IF EXISTS \`(\S+)\`;')
    single_table_regex = re.compile('\`(\S+)\`')
    tables = [t for t in table_regex.findall(mysqlschema)]

    schema_json_dir = os.path.join(dataset_dir, 'schema.json')
    if not os.path.exists(schema_json_dir):

        relationships = []
        derived_relationships = mysqlschema.split('DROP TABLE IF EXISTS ')
        for potential_r in derived_relationships:
            m = single_table_regex.search(potential_r)
            if m is None:
                continue

            table_name = m.group().strip('`')
            fk_regex = re.finditer('FOREIGN KEY \((\`\S+\`(, )?)+\) REFERENCES \`\S+\` \((\`\S+\`(, )?)+\)',
                                   potential_r)
            for matched_fk in fk_regex:
                fk_def = matched_fk.group()
                referencing_columns = single_table_regex.findall(fk_def.split(' REFERENCES ')[0])
                referenced_table = single_table_regex.findall(fk_def.split(' REFERENCES ')[1].split('(')[0])[0]
                referenced_columns = single_table_regex.findall(fk_def.split(' REFERENCES ')[1].split('(')[1])

                relationships.append((table_name, referencing_columns, referenced_table, referenced_columns))

        tables, relationships = make_acyclic(tables, relationships)

        # save schema
        schema = {"name": dataset_name,
                  "csv_kwargs": {
                      "sep": "\t"
                  },
                  "db_load_kwargs": {
                      "postgres": "DELIMITER '\t' QUOTE '\"' ESCAPE '\\' NULL 'NULL' CSV HEADER;"
                  },
                  'tables': tables,
                  'relationships': relationships}

        with open(schema_json_dir, 'w') as fp:
            json.dump(schema, fp)

    return tables


def download_mysql_schema(host, password, rel_fit_dataset_name, schema_sql_dir, user):
    # where to place the mysql schema file
    schema_sql_path = os.path.join(schema_sql_dir, 'mysql.sql')
    # download the sql file
    if not os.path.exists(schema_sql_path):
        sql_download_command = f"mysqldump -h {host} -u {user} -p{password} {rel_fit_dataset_name} --no-data --column-statistics=0 > mysql.sql"
        print(sql_download_command)
        os.system(sql_download_command)
        shutil.move('mysql.sql', schema_sql_path)
    else:
        print("Skipping schema file download")
    return schema_sql_path


def derive_postgres_file(mysqlschema, schema_sql_dir):
    postgres_sql_path = os.path.join(schema_sql_dir, 'postgres.sql')
    if not os.path.exists(postgres_sql_path):
        postgressqlschema = mysqlschema
        simple_replacements = [('`', '"'),
                               ('AUTO_INCREMENT', ''),
                               (' datetime', ' varchar(255)'),
                               (' date', ' varchar(255)'),
                               (' double', ' double precision'),
                               (' longblob', ' bytea'),
                               (' longtext', ' text'),
                               (' NOT NULL DEFAULT current_timestamp()', ''),
                               (' USING BTREE', ''),
                               ]
        re_replacements = [('ENGINE=InnoDB .*', ';'),
                           (',\n.*UNIQUE KEY .*', ''),
                           ('  KEY.*\n', ''),
                           (' tinyint\(\d+\)', ' integer'),
                           (' smallint\(\d+\)', ' integer'),
                           (' mediumint\(\d+\)', ' integer'),
                           (' bigint\(\d+\)', ' integer'),
                           (' int\(\d+\)', ' integer'),
                           (' enum\(.*\)', ' varchar(255)'),
                           ('COMMENT .*', ','),
                           ('  CONSTRAINT.*\n', ''),
                           ('/\*.*', ''),
                           ('--.*', ''),
                           ('\n\n+', '\n\n'),
                           (',\n\)', '\n)')]
        final_replacements = [('integer unsigned', 'integer'),
                              (' ,', ','),
                              (' ,', ',')]
        for search, repl in simple_replacements:
            postgressqlschema = postgressqlschema.replace(search, repl)
        for search, repl in re_replacements:
            postgressqlschema = re.sub(search, repl, postgressqlschema)
        for search, repl in final_replacements:
            postgressqlschema = postgressqlschema.replace(search, repl)

        with open(postgres_sql_path, 'w') as file:
            file.write(postgressqlschema)
    else:
        print("Skipping postgres validation")

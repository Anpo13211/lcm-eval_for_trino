def replace_workload_alias(dataset, source, target):
    with open(source, 'r') as file:
        src_workload = file.read()

    tables_aliases = raw_aliases(dataset)

    if len(tables_aliases) == 0:
        raise ValueError(f"No aliases defined for dataset {dataset}")

    for table, alias in tables_aliases:
        src_workload = src_workload.replace(f'{alias}.', f'{table}.')
        src_workload = src_workload.replace(f'{table} {alias}', f'{table}')

    with open(target, 'w') as file:
        file.write(src_workload)


def raw_aliases(dataset):
    tables_aliases = []
    if dataset == 'imdb':
        tables_aliases = [
            ('title', 't'),
            ('movie_info_idx', 'mi_idx'),
            ('cast_info', 'ci'),
            ('movie_info', 'mi'),
            ('movie_keyword', 'mk'),
            ('movie_companies', 'mc'),
            ('company_name', 'cn'),
            ('role_type', 'rt'),
            ('movie_link', 'ml'),
        ]
    return tables_aliases


def alias_dict(dataset):
    return {alias: full for full, alias in raw_aliases(dataset)}

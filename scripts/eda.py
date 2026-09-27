"""Quick data exploration with DuckDB (row counts, match statistics, example true matches).

Usage: python scripts/eda.py <path-to>/dataset
"""
import os
import sys
import time

import duckdb

T = chr(9)
D = sys.argv[1] if len(sys.argv) > 1 else "dataset"
con = duckdb.connect()


def src(split, name):
    return f"read_csv('{D}/{split}/{split}_{name}.tsv', delim='{T}', header=true, all_varchar=true)"


def show(title, sql):
    t0 = time.time()
    print(f"\n=== {title}", flush=True)
    print(con.sql(sql))
    print(f"({time.time() - t0:.0f}s)", flush=True)


for split in ("train", "test"):
    for s in ("source1", "source2", "source3"):
        show(f"{split} {s} rows by country",
             f"SELECT country, count(*) AS n FROM {src(split, s)} GROUP BY 1 ORDER BY 2 DESC")

con.execute(f"CREATE TABLE gt AS SELECT source1_entity_id AS s1, coalesce(matched_entity_ids, '') AS m "
            f"FROM {src('train', 'ground_truth')}")
show("matches per Source 1 entity",
     "SELECT CASE WHEN m = '' THEN 0 ELSE len(string_split(m, ',')) END AS k, count(*) AS n "
     "FROM gt GROUP BY 1 ORDER BY 1 LIMIT 15")
show("links vs distinct records (is each Source 2/3 record matched at most once?)",
     "SELECT count(*) AS links, count(DISTINCT o) AS distinct_records FROM "
     "(SELECT unnest(string_split(m, ',')) AS o FROM gt WHERE m <> '')")

con.execute("CREATE TABLE pairs AS SELECT s1, unnest(string_split(m, ',')) AS o FROM "
            "(SELECT * FROM gt WHERE m <> '' ORDER BY random() LIMIT 30000)")
con.execute(f"CREATE TABLE a AS SELECT * FROM {src('train', 'source1')} WHERE entity_id IN (SELECT s1 FROM pairs)")
con.execute(f"CREATE TABLE b AS SELECT * FROM {src('train', 'source2')} WHERE entity_id IN (SELECT o FROM pairs) "
            f"UNION ALL SELECT * FROM {src('train', 'source3')} WHERE entity_id IN (SELECT o FROM pairs)")
con.execute("CREATE TABLE j AS SELECT a.country AS c1, b.country AS c2, a.business_name AS n1, "
            "b.business_name AS n2, a.business_address AS a1, b.business_address AS a2 "
            "FROM pairs p JOIN a ON a.entity_id = p.s1 JOIN b ON b.entity_id = p.o")
show("true match statistics", r"""
SELECT count(*) AS pairs, avg((c1 = c2)::INT) AS same_country,
  avg((lower(n1) = lower(n2))::INT) AS exact_name,
  avg(jaro_winkler_similarity(lower(n1), lower(n2))) AS name_jw
FROM j""")
out = os.path.join(os.getcwd(), "eda_examples.tsv")
con.execute(f"COPY (SELECT * FROM j ORDER BY random() LIMIT 40) TO '{out}' (DELIMITER '{T}', HEADER)")
print(f"\nSaved 40 example true matches to {out}")

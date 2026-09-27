"""Scalable business entity resolution with DuckDB + LightGBM.

Stages (each saves its results in a DuckDB file, so a crash never loses finished work):
  prep     load TSVs and clean names / addresses                   -> {split}_rec
  block    candidate generation with rare-token keys, top-K per record -> {split}_cand
  feat     similarity features for every candidate pair            -> {split}_feat
  train    fit LightGBM on a sample of training pairs, tune the threshold on a hold-out
  predict  score all test pairs, keep the best Source 1 match per record, write outputs

Usage:
  python src/pipeline.py --data-dir ~/student_resource/dataset
  python src/pipeline.py --data-dir ... --stages train,predict      # rerun later stages only
  python src/pipeline.py --data-dir ... --smoke-rows 50000          # quick end-to-end test
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_sql as C  # noqa: E402

TAB = chr(9)
SEED = 42
FEATURES = [
    "hits", "addr_hits", "min_df", "qs", "jw_core", "jw_sorted", "jw_name", "jw_skel", "lev_sorted",
    "tok_shared", "tok_jacc", "a_ntok", "b_ntok", "first_tok_eq", "core_eq", "contains_eq",
    "is_domain", "jw_addr", "jw_addr_raw", "addr_jacc", "num_shared", "num_match",
    "postal_match", "addr_missing", "o_src", "gap_o", "rank_o", "ncand_o", "gap_s",
    "ncand_s",
]
T0 = time.time()


def log(msg):
    el = time.time() - T0
    print(f"[{time.strftime('%H:%M:%S')} +{el / 60:5.1f}m] {msg}", flush=True)


# ----------------------------------------------------------------------------- setup

def connect(args):
    import duckdb
    os.makedirs(args.work_dir, exist_ok=True)
    tmp = os.path.join(args.work_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(args.work_dir, "er.duckdb"))
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")

    try:
        from anyascii import anyascii
    except ImportError:
        anyascii = None
        log("WARNING: anyascii not installed - Indian-script names will not be transliterated "
            "(pip install anyascii)")

    def translit(s):
        return anyascii(s) if (anyascii is not None and s) else s

    try:
        from duckdb.sqltypes import VARCHAR
    except ImportError:
        from duckdb.typing import VARCHAR
    con.create_function("translit", translit, [VARCHAR], VARCHAR)
    return con


def read_tsv(path):
    return f"read_csv('{path}', delim='{TAB}', header=true, all_varchar=true)"


def rid_of(col):
    return f"hash(trim({col}))"


def count(con, sql):
    return con.execute(sql).fetchone()[0]


# ----------------------------------------------------------------------------- stages

def stage_prep(con, args, sp):
    lim = f" LIMIT {args.smoke_rows}" if args.smoke_rows else ""
    # One source file at a time keeps peak memory low on small machines.
    for i in (1, 2, 3):
        path = os.path.join(args.data_dir, sp, f"{sp}_source{i}.tsv")
        log(f"[{sp}] loading and cleaning source {i}")
        con.execute(f"CREATE OR REPLACE TABLE {sp}_raw AS SELECT {i} AS src, trim(entity_id) AS entity_id, "
                    f"business_name, business_address, country FROM (SELECT * FROM {read_tsv(path)}{lim})")
        verb = f"CREATE OR REPLACE TABLE {sp}_rec AS" if i == 1 else f"INSERT INTO {sp}_rec"
        con.execute(f"""
    {verb} SELECT * FROM (
    WITH a AS (
      SELECT src, entity_id, {rid_of('entity_id')} AS rid,
             lower(trim(coalesce(country, ''))) AS country,
             {C.name_text("coalesce(business_name, '')")} AS ntext,
             {C.addr_text("coalesce(business_address, '')")} AS atext,
             regexp_matches(lower(coalesce(business_name, '')), '{C.DOMAIN_RE}') AS is_domain
      FROM {sp}_raw
    ), b AS (
      SELECT * EXCLUDE (ntext, atext),
        list_filter(list_transform(string_split(ntext, ' '), lambda w: {C.name_word('w')}),
                    lambda w: w <> '') AS toks,
        list_filter(list_transform(string_split(atext, ' '), lambda w: {C.addr_word('w')}),
                    lambda w: w <> '' AND NOT list_contains({C.sql_list(C.NULL_TOKENS)}, w)) AS atoks
      FROM a
    ), c AS (
      SELECT *, list_filter(toks, lambda w: NOT list_contains({C.sql_list(C.LEGAL_WORDS)}, w)) AS core0
      FROM b
    ), d AS (
      SELECT * EXCLUDE (core0), CASE WHEN len(core0) = 0 THEN toks ELSE core0 END AS core_toks
      FROM c
    )
    SELECT src, entity_id, rid, country, is_domain,
      array_to_string(toks, ' ') AS name_c,
      core_toks,
      array_to_string(core_toks, ' ') AS core,
      array_to_string(list_sort(list_distinct(core_toks)), ' ') AS core_sorted,
      array_to_string(list_transform(core_toks, lambda w: {C.skeleton('w')}), ' ') AS skel,
      array_to_string(atoks, ' ') AS addr_c,
      list_distinct(list_filter(atoks, lambda w: length(w) >= 2)) AS addr_toks,
      array_to_string(list_sort(list_distinct(atoks)), ' ') AS addr_sorted,
      list_distinct(list_filter(atoks, lambda w: regexp_full_match(w, '[0-9]+'))) AS nums,
      list_distinct(list_filter(atoks, lambda w: regexp_full_match(w, '[0-9][0-9][0-9][0-9][0-9][0-9]?'))) AS postal
    FROM d)
    """)
        con.execute(f"DROP TABLE {sp}_raw")
        con.execute("CHECKPOINT")
    n = con.execute(f"SELECT src, count(*) FROM {sp}_rec GROUP BY 1 ORDER BY 1").fetchall()
    log(f"[{sp}] records per source: {n}")

    if sp == "train":
        gt = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
        con.execute(f"""
        CREATE OR REPLACE TABLE gt_links AS
        SELECT {rid_of('source1_entity_id')} AS s_rid, {rid_of('o')} AS o_rid FROM (
          SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS o
          FROM {read_tsv(gt)}
          WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> '')
        WHERE trim(o) <> ''
        """)
        log(f"[train] ground-truth links: {count(con, 'SELECT count(*) FROM gt_links'):,}")
    con.execute("CHECKPOINT")


def _base(sp, cond):
    generic = C.sql_list(C.ADDR_GENERIC)
    return (f"(SELECT rid, country, core_toks, core_sorted, list_sort(list_distinct(core_toks)) AS st, "
            f"list_filter(addr_toks, lambda w: length(w) >= 4 AND NOT regexp_full_match(w, '[0-9]+') "
            f"AND NOT list_contains({generic}, w)) AS aw, "
            f"regexp_extract(addr_c, '(^| )([0-9][0-9]+)( |$)', 2) AS hn "
            f"FROM {sp}_rec WHERE {cond})")


def _key_hashes(sp, cond):
    """(rid, kh, t): one row per blocking key, hashed with the country. t=1 for address-based keys.

    Name keys:    rare words, consonant skeletons (~), whole sorted core name (=), adjacent word pairs (+)
    Address keys: rare address words (@), house number + address word (#), first name word + house number (%)
    """
    skel = C.skeleton("w")
    keys = f"""list_filter(list_distinct(list_concat(
          list_filter(core_toks, lambda w: length(w) >= 3),
          list_transform(list_filter(core_toks, lambda w: length(w) >= 4), lambda w: '~' || {skel}),
          [ '=' || core_sorted ],
          list_transform(range(1, len(st)), lambda i: '+' || st[i] || ' ' || st[i + 1]),
          list_transform(aw, lambda w: '@' || w),
          CASE WHEN hn <> '' THEN list_transform(aw, lambda w: '#' || hn || ' ' || w) ELSE []::VARCHAR[] END,
          CASE WHEN hn <> '' AND len(st) > 0 THEN ['%' || st[1] || ' ' || hn] ELSE []::VARCHAR[] END
        )), lambda k: length(k) > 1)"""
    return (f"(SELECT rid, hash(country || '|' || k) AS kh, (left(k, 1) IN ('@', '#', '%'))::TINYINT AS t FROM ("
            f"SELECT rid, country, unnest(keys) AS k FROM ("
            f"SELECT rid, country, {keys} AS keys FROM {_base(sp, cond)})))")


def stage_block(con, args, sp):
    B = args.buckets
    for leftover in ("tok", "keys", "df", "s1k0", "s1k", "cand"):  # from older versions / crashed runs
        con.execute(f"DROP TABLE IF EXISTS {sp}_{leftover}")
    con.execute("CHECKPOINT")
    log(f"[{sp}] building Source 1 blocking keys")
    con.execute(f"CREATE OR REPLACE TABLE {sp}_s1k0 AS SELECT * FROM {_key_hashes(sp, 'src = 1')}")
    con.execute(f"CREATE OR REPLACE TABLE {sp}_df AS SELECT kh, count(*)::INTEGER AS df FROM {sp}_s1k0 GROUP BY kh")
    con.execute(f"""CREATE OR REPLACE TABLE {sp}_s1k AS
                    SELECT a.rid, a.kh, a.t, d.df FROM {sp}_s1k0 a JOIN {sp}_df d ON a.kh = d.kh
                    WHERE d.df <= {args.cap}""")
    con.execute(f"DROP TABLE {sp}_s1k0")
    con.execute("CHECKPOINT")
    log(f"[{sp}] Source 1 keys kept: {count(con, f'SELECT count(*) FROM {sp}_s1k'):,}")

    for b in range(B):
        verb = f"CREATE OR REPLACE TABLE {sp}_cand AS" if b == 0 else f"INSERT INTO {sp}_cand"
        con.execute(f"""
        {verb} SELECT o_rid, s_rid, hits, addr_hits, min_df, qs, rk FROM (
          SELECT *, row_number() OVER (PARTITION BY o_rid ORDER BY qs + 0.05 * hits DESC) AS rk FROM (
            WITH o AS (
              SELECT t.rid, t.kh FROM {_key_hashes(sp, f'src > 1 AND rid % {B} = {b}')} t
              JOIN {sp}_df d ON t.kh = d.kh WHERE d.df <= {args.cap}
              QUALIFY row_number() OVER (PARTITION BY t.rid ORDER BY d.df, t.kh) <= {args.nkeys}
            ), p AS (
              SELECT o.rid AS o_rid, s.rid AS s_rid, count(*)::INTEGER AS hits,
                     (count(*) FILTER (WHERE s.t = 1))::INTEGER AS addr_hits,
                     min(s.df)::INTEGER AS min_df
              FROM o JOIN {sp}_s1k s ON o.kh = s.kh GROUP BY ALL
            )
            SELECT p.o_rid, p.s_rid, p.hits, p.addr_hits, p.min_df,
                   jaro_winkler_similarity(ra.core_sorted, rb.core_sorted)
                     + 0.5 * jaro_winkler_similarity(ra.addr_sorted, rb.addr_sorted) AS qs
            FROM p JOIN {sp}_rec ra ON ra.rid = p.s_rid JOIN {sp}_rec rb ON rb.rid = p.o_rid
          )
        ) WHERE rk <= {args.topk}
        """)
        if (b + 1) % 4 == 0 or b + 1 == B:
            log(f"[{sp}] blocking bucket {b + 1}/{B} done")
    con.execute(f"DROP TABLE {sp}_s1k")
    con.execute(f"DROP TABLE {sp}_df")
    con.execute("CHECKPOINT")

    n_c = count(con, f"SELECT count(*) FROM {sp}_cand")
    n_o = count(con, f"SELECT count(*) FROM {sp}_rec WHERE src > 1")
    n_s = count(con, f"SELECT count(*) FROM {sp}_rec WHERE src = 1")
    log(f"[{sp}] candidate pairs: {n_c:,} ({n_c / max(n_o, 1):.1f} per Source 2/3 record, "
        f"{n_c / max(n_s, 1):.1f} per Source 1 entity)")
    if sp == "train":
        tot = count(con, """SELECT count(*) FROM gt_links g
                            JOIN train_rec a ON a.rid = g.s_rid JOIN train_rec b ON b.rid = g.o_rid""")
        by_rank = con.execute("""SELECT c.rk, count(*) FROM gt_links g JOIN train_cand c USING (s_rid, o_rid)
                                 GROUP BY 1 ORDER BY 1""").fetchall()
        cum, parts = 0, []
        for rk, n in by_rank:
            cum += n
            parts.append(f"@{rk}={cum / max(tot, 1):.3f}")
        log(f"[train] BLOCKING RECALL = {cum / max(tot, 1):.4f}  ({cum:,} of {tot:,} true links in candidates)")
        log(f"[train] recall by candidates kept per record: {' '.join(parts)}")


def stage_feat(con, args, sp):
    B = args.buckets
    for b in range(B):
        verb = f"CREATE OR REPLACE TABLE {sp}_feat AS" if b == 0 else f"INSERT INTO {sp}_feat"
        con.execute(f"""
        {verb}
        SELECT x.*,
          max(x.qs) OVER (PARTITION BY x.o_rid) - x.qs AS gap_o,
          rank() OVER (PARTITION BY x.o_rid ORDER BY x.qs DESC) AS rank_o,
          count(*) OVER (PARTITION BY x.o_rid) AS ncand_o
        FROM (
        SELECT c.o_rid, c.s_rid, c.hits, c.addr_hits, c.min_df, c.qs,
          jaro_winkler_similarity(a.core, b.core) AS jw_core,
          jaro_winkler_similarity(a.core_sorted, b.core_sorted) AS jw_sorted,
          jaro_winkler_similarity(a.name_c, b.name_c) AS jw_name,
          jaro_winkler_similarity(a.skel, b.skel) AS jw_skel,
          1.0 - levenshtein(a.core_sorted, b.core_sorted)
                / greatest(length(a.core_sorted), length(b.core_sorted), 1) AS lev_sorted,
          len(list_intersect(a.core_toks, b.core_toks)) AS tok_shared,
          len(list_intersect(a.core_toks, b.core_toks))
            / greatest(len(list_distinct(list_concat(a.core_toks, b.core_toks))), 1) AS tok_jacc,
          len(a.core_toks) AS a_ntok, len(b.core_toks) AS b_ntok,
          (a.core_toks[1] = b.core_toks[1])::INTEGER AS first_tok_eq,
          (a.core = b.core)::INTEGER AS core_eq,
          (contains(replace(a.core, ' ', ''), replace(b.core, ' ', ''))
            OR contains(replace(b.core, ' ', ''), replace(a.core, ' ', '')))::INTEGER AS contains_eq,
          b.is_domain::INTEGER AS is_domain,
          jaro_winkler_similarity(a.addr_sorted, b.addr_sorted) AS jw_addr,
          jaro_winkler_similarity(a.addr_c, b.addr_c) AS jw_addr_raw,
          len(list_intersect(a.addr_toks, b.addr_toks))
            / greatest(len(list_distinct(list_concat(a.addr_toks, b.addr_toks))), 1) AS addr_jacc,
          len(list_intersect(a.nums, b.nums)) AS num_shared,
          CASE WHEN len(a.nums) = 0 OR len(b.nums) = 0 THEN -1
               ELSE (len(list_intersect(a.nums, b.nums)) > 0)::INTEGER END AS num_match,
          CASE WHEN len(a.postal) = 0 OR len(b.postal) = 0 THEN -1
               ELSE (len(list_intersect(a.postal, b.postal)) > 0)::INTEGER END AS postal_match,
          (a.addr_c = '' OR b.addr_c = '')::INTEGER AS addr_missing,
          b.src AS o_src
        FROM {sp}_cand c JOIN {sp}_rec a ON a.rid = c.s_rid JOIN {sp}_rec b ON b.rid = c.o_rid
        WHERE c.o_rid % {B} = {b}
        ) x
        """)
        if (b + 1) % 4 == 0 or b + 1 == B:
            log(f"[{sp}] features bucket {b + 1}/{B} done")
    # per-Source-1 summary, joined on the fly later (avoids rewriting the big table)
    con.execute(f"""CREATE OR REPLACE TABLE {sp}_sstats AS
                    SELECT s_rid, max(qs) AS smax, count(*) AS ncand_s FROM {sp}_feat GROUP BY s_rid""")
    con.execute("CHECKPOINT")
    log(f"[{sp}] feature rows: {count(con, f'SELECT count(*) FROM {sp}_feat'):,}")


EXPR = {c: f"f.{c}" for c in FEATURES}
EXPR["gap_s"] = "s.smax - f.qs"
EXPR["ncand_s"] = "s.ncand_s"


def feature_select():
    return ", ".join(f"CAST({EXPR[c]} AS FLOAT) AS {c}" for c in FEATURES)


def feat_from(sp):
    return f"{sp}_feat f JOIN {sp}_sstats s ON s.s_rid = f.s_rid"


def best_per_record(df, prob, threshold):
    """Each Source 2/3 record goes to at most one Source 1 entity: its highest-scoring one."""
    d = df[["o_rid", "s_rid"]].copy()
    d["p"] = prob
    d = d[d["p"] >= threshold]
    return d.sort_values("p", ascending=False).drop_duplicates("o_rid")


def macro_f05(pred, truth, ids):
    """pred/truth: DataFrames (s_rid, o_rid). ids: Source 1 ids to average over."""
    idx = pd.Index(ids, name="s_rid")
    n_pred = pred.groupby("s_rid").size().reindex(idx, fill_value=0)
    n_true = truth.groupby("s_rid").size().reindex(idx, fill_value=0)
    tp = pred.merge(truth, on=["s_rid", "o_rid"]).groupby("s_rid").size().reindex(idx, fill_value=0)
    p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
    r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
    f = np.where(tp > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    f = np.where((n_pred == 0) & (n_true == 0), 1.0, f)
    return float(f.mean()), float(p[n_pred > 0].mean() if (n_pred > 0).any() else 0), float(r[n_true > 0].mean())


def stage_train(con, args):
    import lightgbm as lgb
    fit_pm, val_pm = args.fit_permille, args.val_permille
    assert fit_pm + val_pm <= 1000
    log(f"Loading training sample ({fit_pm / 10:.1f}% of Source 1 entities)")
    fit = con.execute(f"""SELECT {feature_select()}, (g.s_rid IS NOT NULL)::INTEGER AS label
                          FROM {feat_from('train')}
                          LEFT JOIN gt_links g ON g.s_rid = f.s_rid AND g.o_rid = f.o_rid
                          WHERE f.s_rid % 1000 < {fit_pm}""").df()
    log(f"fit rows: {len(fit):,}  positives: {int(fit['label'].sum()):,}")
    model = lgb.LGBMClassifier(n_estimators=args.trees, learning_rate=0.05, num_leaves=127,
                               min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                               n_jobs=args.threads, verbose=-1)
    model.fit(fit[FEATURES], fit["label"])
    del fit
    booster = model.booster_
    booster.save_model(os.path.join(args.work_dir, "model.txt"))
    imp = sorted(zip(booster.feature_importance("gain"), FEATURES), reverse=True)[:10]
    log("top features: " + ", ".join(f"{n}" for _, n in imp))

    lo = 1000 - val_pm
    log(f"Validating on {val_pm / 10:.1f}% of Source 1 entities (never used for training)")
    val = con.execute(f"""
      SELECT f.o_rid, f.s_rid, {feature_select()} FROM {feat_from('train')}
      WHERE f.o_rid IN (SELECT o_rid FROM train_feat WHERE s_rid % 1000 >= {lo})""").df()
    prob = booster.predict(val[FEATURES], num_threads=args.threads)
    ids = con.execute(f"SELECT rid FROM train_rec WHERE src = 1 AND rid % 1000 >= {lo}").df()["rid"].values
    truth = con.execute(f"SELECT s_rid, o_rid FROM gt_links WHERE s_rid % 1000 >= {lo}").df()

    best_t, best = 0.5, (-1, 0, 0)
    for t in np.arange(0.10, 0.96, 0.025):
        pred = best_per_record(val, prob, t)
        pred = pred[pred["s_rid"].isin(ids)]
        res = macro_f05(pred, truth, ids)
        if res[0] >= best[0]:
            best_t, best = float(t), res
    log(f"VALIDATION macro F0.5 = {best[0]:.4f} at threshold {best_t:.3f} "
        f"(avg precision {best[1]:.3f}, avg recall {best[2]:.3f}, {len(ids):,} entities)")
    with open(os.path.join(args.work_dir, "threshold.json"), "w") as fh:
        json.dump({"threshold": best_t, "val_f05": best[0]}, fh)

    # save some mistakes for error analysis
    pred = best_per_record(val, prob, best_t)
    pred = pred[pred["s_rid"].isin(ids)]
    fp = pred.merge(truth.assign(ok=1), on=["s_rid", "o_rid"], how="left")
    fp = fp[fp["ok"].isna()].head(300)[["s_rid", "o_rid", "p"]].assign(error="false_match")
    fn = truth.merge(pred[["s_rid", "o_rid"]].assign(hit=1), on=["s_rid", "o_rid"], how="left")
    fn = fn[fn["hit"].isna()].head(300)[["s_rid", "o_rid"]].assign(p=np.nan, error="missed_match")
    err = pd.concat([fp, fn])
    con.register("err_df", err)
    os.makedirs(args.report_dir, exist_ok=True)
    con.execute(f"""
      COPY (SELECT e.error, round(e.p, 3) AS p, a.entity_id AS s1_id, b.entity_id AS other_id,
                   a.name_c AS s1_name, b.name_c AS other_name, a.addr_c AS s1_addr, b.addr_c AS other_addr
            FROM err_df e JOIN train_rec a ON a.rid = e.s_rid JOIN train_rec b ON b.rid = e.o_rid
            ORDER BY e.error, e.p DESC)
      TO '{os.path.join(args.report_dir, 'val_errors.tsv')}' (DELIMITER '{TAB}', HEADER)""")
    con.unregister("err_df")
    log(f"saved error examples to {args.report_dir}/val_errors.tsv")


def write_lists(path, header, df):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=TAB, quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(header)
        for sid, ids in zip(df["s1"].values, df["ids"].values):
            w.writerow([sid, ids if isinstance(ids, str) else ""])


def stage_predict(con, args):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=os.path.join(args.work_dir, "model.txt"))
    with open(os.path.join(args.work_dir, "threshold.json")) as fh:
        t = json.load(fh)["threshold"]
    pred_dir = os.path.join(args.work_dir, "pred")
    shutil.rmtree(pred_dir, ignore_errors=True)
    os.makedirs(pred_dir)

    total = count(con, "SELECT count(*) FROM test_feat")
    log(f"Scoring {total:,} test pairs (threshold {t:.3f})")
    res = con.execute(f"SELECT f.o_rid, f.s_rid, {feature_select()} FROM {feat_from('test')}")
    done, part = 0, 0
    while True:
        chunk = res.fetch_df_chunk(args.chunk_vectors)
        if chunk is None or len(chunk) == 0:
            break
        p = booster.predict(chunk[FEATURES], num_threads=args.threads)
        out = pd.DataFrame({"o_rid": chunk["o_rid"].values, "s_rid": chunk["s_rid"].values,
                            "p": p.astype("float32")})
        out.to_parquet(os.path.join(pred_dir, f"part{part:05d}.parquet"), index=False)
        part += 1
        done += len(chunk)
        log(f"scored {done:,}/{total:,}")

    con.execute(f"""
    CREATE OR REPLACE TABLE test_assign AS
    SELECT o_rid, s_rid, p FROM read_parquet('{pred_dir}/*.parquet') WHERE p >= {t}
    QUALIFY row_number() OVER (PARTITION BY o_rid ORDER BY p DESC) = 1
    """)
    os.makedirs(args.output_dir, exist_ok=True)
    for table, name, col in (("test_assign", "matching_results.tsv", "matched_entity_ids"),
                             ("test_cand", "candidate_pairs.tsv", "candidate_entity_ids")):
        df = con.execute(f"""
          SELECT r.entity_id AS s1, string_agg(DISTINCT o.entity_id, ',') AS ids
          FROM test_rec r LEFT JOIN {table} x ON x.s_rid = r.rid
          LEFT JOIN test_rec o ON o.rid = x.o_rid
          WHERE r.src = 1 GROUP BY r.entity_id""").df()
        write_lists(os.path.join(args.output_dir, name), ["source1_entity_id", col], df)
        log(f"wrote {args.output_dir}/{name} ({len(df):,} rows)")
    n_match = count(con, "SELECT count(*) FROM test_assign")
    log(f"test links predicted: {n_match:,}")


# ----------------------------------------------------------------------------- main

def main():
    home = os.path.expanduser("~")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default=os.path.join(home, "er_work"))
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--stages", default="prep,block,feat,train,predict")
    ap.add_argument("--splits", default="train,test", help="splits for prep/block/feat")
    ap.add_argument("--memory", default="1500MB", help="DuckDB memory limit")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--cap", type=int, default=100, help="ignore keys shared by more Source 1 records")
    ap.add_argument("--nkeys", type=int, default=6, help="keys per Source 2/3 record")
    ap.add_argument("--topk", type=int, default=5, help="candidates kept per Source 2/3 record")
    ap.add_argument("--buckets", type=int, default=16, help="process in N pieces to save memory")
    ap.add_argument("--fit-permille", type=int, default=30, help="Source 1 entities used for fitting (per mille)")
    ap.add_argument("--val-permille", type=int, default=5, help="Source 1 entities held out for validation")
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--chunk-vectors", type=int, default=500, help="prediction batch = N x 2048 rows")
    ap.add_argument("--smoke-rows", type=int, default=0, help="only read N rows per file (quick test)")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    log(f"stages={stages} splits={splits} memory={args.memory} threads={args.threads} "
        f"cap={args.cap} nkeys={args.nkeys} topk={args.topk}")
    con = connect(args)
    for st, fn in (("prep", stage_prep), ("block", stage_block), ("feat", stage_feat)):
        if st in stages:
            for sp in splits:
                fn(con, args, sp)
    if "train" in stages:
        stage_train(con, args)
    if "predict" in stages:
        stage_predict(con, args)
    log("done")


if __name__ == "__main__":
    main()

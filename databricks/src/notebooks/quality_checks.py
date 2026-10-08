# Databricks notebook source
# MAGIC %md
# MAGIC # Quality checks: silver must satisfy these rules, or the job fails
# MAGIC
# MAGIC Each check is a SQL query that returns true when the rule holds. Parameter `scope` picks the rule set:
# MAGIC `dms` for the training data, `proteome` for the old proteome-wide predictions. Results are appended
# MAGIC to `silver.quality_checks` (one row per check per run). A failed hard check raises, so the job stops
# MAGIC and nothing downstream reads bad data; a failed soft check is recorded as a warning.

# COMMAND ----------

from datetime import datetime, timezone

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("scope", "dms")
c = dbutils.widgets.get("catalog")
scope = dbutils.widgets.get("scope")
AA = "'^[ACDEFGHIKLMNPQRSTVWY]$'"

# --- training data: ProteinGym DMS ---------------------------------------------------------------
d, a = f"{c}.silver.dms_scores", f"{c}.silver.dms_assay_audit"
DMS = [
    # (name, SQL returning true when the rule holds, hard?)
    (
        "every assay with single substitutions reached silver",
        f"SELECT (SELECT count(*) FROM {a} WHERE single_substitutions > 0) = (SELECT count(DISTINCT dms_id) FROM {d})",
        True,
    ),
    ("one row per substitution per assay", f"SELECT count(*) = count(DISTINCT dms_id, position, alt_aa) FROM {d}", True),
    (
        "the reference amino acid matches the assay's sequence",
        f"SELECT count_if(NOT ref_matches_sequence OR ref_matches_sequence IS NULL) = 0 FROM {d}",
        True,
    ),
    (
        "amino acids are standard and actually change",
        f"SELECT count_if(NOT ref_aa RLIKE {AA} OR NOT alt_aa RLIKE {AA} OR ref_aa = alt_aa) = 0 FROM {d}",
        True,
    ),
    ("every substitution has a score", f"SELECT count_if(dms_score IS NULL) = 0 FROM {d}", True),
    (
        "at least 95% of human assays map to a UniProt accession",
        f"SELECT avg(CAST(uniprot_id IS NOT NULL AS DOUBLE)) >= 0.95 "
        f"FROM (SELECT DISTINCT dms_id, uniprot_id FROM {d} WHERE taxon = 'Human')",
        False,
    ),
]

# --- old proteome-wide predictions -------------------------------------------------------------
s, b = f"{c}.silver.variant_scores", f"{c}.bronze.proteome_predictions"
# Probabilities must lie in [0, 1]. The QAFI regressors predict DMS scores and can exceed 1 (qafi1 reaches
# about 1.45), and the EVE column holds raw EVE scores (about -12 to 17): those only get a sanity range.
PROBABILITIES = ["am_pathogenicity", "cpt1", "qafi_class1", "qafi_class2"]
REGRESSORS = ["qafi1", "qafi2", "qafi_mt", "qafi_split1", "qafi_split2", "qafi_split3"]
outside = " OR ".join(f"{p} NOT BETWEEN 0 AND 1" for p in PROBABILITIES)
implausible = " OR ".join(f"{r} NOT BETWEEN -1 AND 3" for r in REGRESSORS)
PROTEOME = [
    (
        "every protein in bronze reached silver",
        f"SELECT (SELECT count(DISTINCT UniProtID) FROM {b}) = (SELECT count(DISTINCT uniprot_id) FROM {s})",
        True,
    ),
    ("one row per variant", f"SELECT count(*) = count(DISTINCT uniprot_id, position, ref_aa, alt_aa) FROM {s}", True),
    ("protein-level scores agree in every copy of a variant", f"SELECT count_if(has_conflict) = 0 FROM {s}", True),
    (
        "amino acids are standard and actually change",
        f"SELECT count_if(NOT ref_aa RLIKE {AA} OR NOT alt_aa RLIKE {AA} OR ref_aa = alt_aa) = 0 FROM {s}",
        True,
    ),
    ("probability scores lie between 0 and 1", f"SELECT count_if({outside}) = 0 FROM {s}", True),
    ("QAFI regression scores lie between -1 and 3", f"SELECT count_if({implausible}) = 0 FROM {s}", False),
    ("every protein has an entry name", f"SELECT count_if(entry_name IS NULL) = 0 FROM {s}", True),
    (
        "at least 95% of proteins are in the UniProt reference proteome",
        f"SELECT avg(CAST(in_reference_proteome AS DOUBLE)) >= 0.95 FROM {c}.silver.protein_audit",
        False,
    ),
]

CHECKS = {"dms": DMS, "proteome": PROTEOME}[scope]

# COMMAND ----------

run_at = datetime.now(timezone.utc)
rows = [(scope, name, bool(spark.sql(sql).first()[0]), hard, run_at) for name, sql, hard in CHECKS]
results = spark.createDataFrame(rows, "scope string, check string, passed boolean, hard boolean, run_at timestamp")
results.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{c}.silver.quality_checks")
display(results)

failed = [name for _, name, passed, hard, _ in rows if hard and not passed]
warned = [name for _, name, passed, hard, _ in rows if not hard and not passed]
if warned:
    print("WARNING:", "; ".join(warned))
if failed:
    raise AssertionError("quality checks failed: " + "; ".join(failed))
print(f"{scope}: all {sum(r[3] for r in rows)} hard checks passed")

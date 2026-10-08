# Databricks notebook source
# MAGIC %md
# MAGIC # Copy raw files from the AWS raw bucket into the volume
# MAGIC
# MAGIC Free Edition cannot attach our S3 bucket, so notebook 04 writes a transfer list into the volume:
# MAGIC for each file, its path inside the volume, a temporary read-only download link (valid a few
# MAGIC hours) and its size. This task downloads each file to
# MAGIC `/Volumes/<catalog>/raw/files/<same path as in S3>`.
# MAGIC A file already present with the right size is skipped, so the task can be re-run safely.

# COMMAND ----------

import json
import os
import shutil
import urllib.request

dbutils.widgets.text("catalog", "qafi_dev")
volume = f"/Volumes/{dbutils.widgets.get('catalog')}/raw/files"

with open(f"{volume}/_transfer/links.json") as f:
    links = json.load(f)

for path, item in links.items():
    target, size = f"{volume}/{path}", item["size"]
    if os.path.exists(target) and os.path.getsize(target) == size:
        print(f"already here  {path}")
        continue
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with urllib.request.urlopen(item["url"], timeout=120) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out, 64 * 1024 * 1024)
    stored = os.path.getsize(target)
    if stored != size:
        os.remove(target)
        raise RuntimeError(f"{path}: expected {size} bytes, got {stored}")
    print(f"copied        {path} ({size / 1e9:.2f} GB)")

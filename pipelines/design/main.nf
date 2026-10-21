#!/usr/bin/env nextflow
/*
 * A design campaign, one protein per task (chapter 15d).
 *
 * The loop itself lives in qafi_core.design so it is the same code a notebook runs. This file is the
 * part that makes it a campaign: one task per protein, every task carrying the versions it used, and
 * the provenance record published beside the sequences rather than left in a scratch directory.
 *
 * Not yet run on AWS Batch: that needs the GPU queue and a container with ProteinMPNN's weights baked
 * in. The process declarations below are what that deployment would use; `-profile local` is what has
 * been exercised.
 */

nextflow.enable.dsl = 2

params.sites      = null          // the site map parquet (14d)
params.structures = null          // directory of AlphaFold models
params.proteins   = null          // comma-separated UniProt accessions
params.criterion  = 'func'        // 15c's answer; 'fold' or 'both' to re-run a baseline
params.budget     = 0.5           // share of positions held fixed
params.intent     = 'stability_engineering'
params.n_seq      = 8
params.temp       = 0.2
params.outdir     = 'results/design'
params.python     = 'python'      // the local profile points this at the project venv

process PLAN {
    tag "${acc}"
    publishDir { "${params.outdir}/${acc}" }, mode: 'copy'

    input:
    val acc

    output:
    tuple val(acc), path("*_fixed.jsonl"), path("*_provenance.json")

    script:
    """
    ${params.python} -m qafi_core.design_cli plan \
        --sites ${params.sites} --protein ${acc} \
        --criterion ${params.criterion} --budget ${params.budget} \
        --intent ${params.intent} --out .
    """
}

process GENERATE {
    tag "${acc}"
    label 'gpu'
    publishDir { "${params.outdir}/${acc}" }, mode: 'copy'

    input:
    tuple val(acc), path(fixed), path(prov)

    output:
    tuple val(acc), path("seqs/*.fa"), path(prov), path(fixed)

    script:
    """
    ${params.python} \$PROTEINMPNN/protein_mpnn_run.py \
        --pdb_path ${params.structures}/AF-${acc}-F1-model_v6.pdb \
        --path_to_model_weights \$PROTEINMPNN/vanilla_model_weights \
        --model_name v_48_020 --out_folder . \
        --fixed_positions_jsonl ${fixed} \
        --num_seq_per_target ${params.n_seq} --sampling_temp ${params.temp} --seed 37
    """
}

process AUDIT {
    tag "${acc}"
    publishDir { "${params.outdir}/${acc}" }, mode: 'copy'

    input:
    tuple val(acc), path(fasta), path(prov), path(fixed)

    output:
    path "*_audit.json"

    script:
    """
    ${params.python} -m qafi_core.design_cli audit \
        --fasta ${fasta} --fixed ${prov} --jsonl ${fixed} --out ${acc}_audit.json
    """
}

workflow {
    Channel.of(params.proteins.split(','))
        | PLAN
        | GENERATE
        | AUDIT
}

#!/usr/bin/env nextflow
/*
 * Deep mutational scanning: sequencing reads -> fitness of every variant (notebook 14b).
 *
 * The experiment counts every variant before selection (input) and after it (output); a variant that
 * survives selection better than the wild type is enriched in the output. Fitness is that enrichment on
 * a log scale, relative to the wild type, with an error model fitted to the replicates.
 *
 *   FASTQC / MULTIQC          quality of every FASTQ file, one report
 *   DIMSUM_COUNT              DiMSum stages 1-4: trim constant regions, merge read pairs, count variants
 *   KEEP_DESIGNED_DOUBLES     the authors' filter: only double mutants built on the designed backgrounds
 *   DIMSUM_FITNESS            DiMSum stages 4-5 on the filtered counts: fitness and its error
 *
 * The two DiMSum passes and the filter between them reproduce Faure et al. 2022 (Nature 604:175), whose
 * command lines and design files are published with the paper; every setting is a parameter here.
 */

process FASTQC {
    tag "${sample}"
    publishDir "${params.outdir}/qc/fastqc", mode: 'copy'

    input:
    tuple val(sample), path(reads)

    output:
    path '*_fastqc.zip', emit: zip
    path '*_fastqc.html'

    script:
    """
    fastqc --threads ${task.cpus} --quiet ${reads}
    """

    stub:
    """
    for r in ${reads}; do n=\$(basename \$r .fastq.gz); touch \${n}_fastqc.zip \${n}_fastqc.html; done
    """
}

process MULTIQC {
    publishDir "${params.outdir}/qc", mode: 'copy'

    input:
    path reports

    output:
    path '*multiqc_report.html'
    path '*multiqc_report_data'

    script:
    """
    multiqc --force --title '${params.name}' .
    """

    stub:
    """
    touch ${params.name}_multiqc_report.html && mkdir ${params.name}_multiqc_report_data
    """
}

process DIMSUM_COUNT {
    tag "${params.name}"
    publishDir "${params.outdir}/counts", mode: 'copy', pattern: "${params.name}/*.{html,txt,pdf,png}"

    input:
    path fastqs, stageAs: 'fastq/*'
    path design

    output:
    path "${params.name}/${params.name}_variant_data_merge.RData", emit: merged
    path "${params.name}/*"

    script:
    """
    DiMSum \\
        --fastqFileDir fastq --fastqFileExtension .fastq --gzipped TRUE \\
        --experimentDesignPath ${design} \\
        --cutadapt5First ${params.cutadapt5_first} --cutadapt5Second ${params.cutadapt5_second} \\
        --cutadaptMinLength ${params.cutadapt_min_length} --cutadaptErrorRate ${params.cutadapt_error_rate} \\
        --vsearchMinQual ${params.vsearch_min_qual} --vsearchMaxee ${params.vsearch_maxee} \\
        --wildtypeSequence ${params.wildtype} --mutagenesisType ${params.mutagenesis_type} \\
        --outputPath . --projectName ${params.name} \\
        --numCores ${task.cpus} --splitChunkSize ${params.split_chunk_size} --retainIntermediateFiles F \\
        --stopStage 4
    """

    stub:
    // the wiring test checks the design file the workflow wrote: a header and one row per sample
    """
    test \$(wc -l < ${design}) -eq 7
    head -1 ${design} | grep -q 'pair1'
    mkdir -p ${params.name} && touch ${params.name}/${params.name}_variant_data_merge.RData ${params.name}/report.html
    """
}

process KEEP_DESIGNED_DOUBLES {
    tag "${params.name}"
    publishDir "${params.outdir}/counts", mode: 'copy'

    input:
    path merged

    output:
    path 'variant_counts.tsv', emit: counts
    path 'filter_summary.txt'

    script:
    """
    keep_designed_doubles.R ${merged} variant_counts.tsv '${params.background_codons}' > filter_summary.txt
    """

    stub:
    """
    touch variant_counts.tsv filter_summary.txt
    """
}

process DIMSUM_FITNESS {
    tag "${params.name}"
    publishDir "${params.outdir}/fitness", mode: 'copy'

    input:
    path counts
    path design

    output:
    path "${params.name}/fitness_*.txt", emit: tables
    path "${params.name}/*"

    script:
    def min_input = params.fitness_min_input_count_any ? "--fitnessMinInputCountAny ${params.fitness_min_input_count_any}" : ''
    """
    mkdir -p fastq
    DiMSum \\
        --fastqFileDir fastq --fastqFileExtension .fastq --gzipped TRUE \\
        --experimentDesignPath ${design} --countPath ${counts} \\
        --wildtypeSequence ${params.wildtype} --mutagenesisType ${params.mutagenesis_type} \\
        --outputPath . --projectName ${params.name} \\
        --numCores ${task.cpus} --retainIntermediateFiles F \\
        --startStage 4 ${min_input}
    """

    stub:
    """
    mkdir -p ${params.name} && touch ${params.name}/fitness_singles.txt ${params.name}/fitness_doubles.txt
    """
}

workflow {
    // DiMSum's experimental design file: one row per sample, read by column name.
    def design_columns = [
        'sample_name', 'experiment_replicate', 'selection_id', 'selection_replicate',
        'technical_replicate', 'generations', 'cell_density', 'selection_time',
    ]

    if (!params.samplesheet) { error "Give a samplesheet: --samplesheet <csv>" }
    if (!params.wildtype) { error "Give the wild-type nucleotide sequence: --wildtype <ACGT...>" }

    rows = channel.fromPath(params.samplesheet, checkIfExists: true).splitCsv(header: true)

    reads = rows.map { r ->
        tuple(r.sample_name, [file("${params.fastq_dir}/${r.fastq_1}", checkIfExists: true),
                              file("${params.fastq_dir}/${r.fastq_2}", checkIfExists: true)])
    }

    design = rows
        .map { r -> (design_columns.collect { column -> r[column] ?: '' } + [r.fastq_1, r.fastq_2]).join('\t') }
        .collectFile(name: 'experiment_design.txt', newLine: true, sort: true,
                     seed: (design_columns + ['pair1', 'pair2']).join('\t'))

    FASTQC(reads)
    MULTIQC(FASTQC.out.zip.collect())

    DIMSUM_COUNT(reads.flatMap { _sample, files -> files }.collect(), design)
    KEEP_DESIGNED_DOUBLES(DIMSUM_COUNT.out.merged)
    DIMSUM_FITNESS(KEEP_DESIGNED_DOUBLES.out.counts, design)
}

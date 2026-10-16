#!/usr/bin/env Rscript
# Keep only the double mutants a nicking-mutagenesis library was designed to contain.
#
# The library mutates every codon of the wild type and of a few "background" variants that each carry one
# chosen codon change. Double mutants therefore exist on purpose only when one of their two changes is a
# background codon; any other double is a sequencing or synthesis error and is dropped. Singles and the wild
# type are always kept. This is the filter Faure et al. 2022 published with their DiMSum runs
# (filter_variants_JD_PDZ_NM2_*PCA_dimsum128.R), with the paths and the background list as arguments.
#
# Usage: keep_designed_doubles.R <variant_data_merge.RData> <out.tsv> "<pos+codon> <pos+codon> ..."
#        an empty background list keeps every variant and only reformats the counts.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: keep_designed_doubles.R <merged.RData> <out.tsv> [backgrounds]")
backgrounds <- if (length(args) >= 3) strsplit(trimws(args[3]), "[ ,]+")[[1]] else character(0)
backgrounds <- backgrounds[backgrounds != ""]

load(args[1])  # defines variant_data_merge
wt_aa <- strsplit(variant_data_merge[WT == TRUE, aa_seq], "")[[1]]

if (length(backgrounds) > 0) {
  doubles <- variant_data_merge$Nham_aa == 2
  variant_data_merge[doubles, Pos1 := which(strsplit(aa_seq, "")[[1]] != wt_aa)[1], aa_seq]
  variant_data_merge[doubles, Pos2 := which(strsplit(aa_seq, "")[[1]] != wt_aa)[2], aa_seq]
  variant_data_merge[doubles, Mut1 := paste0(Pos1, substr(nt_seq, (Pos1 - 1) * 3 + 1, (Pos1 - 1) * 3 + 3))]
  variant_data_merge[doubles, Mut2 := paste0(Pos2, substr(nt_seq, (Pos2 - 1) * 3 + 1, (Pos2 - 1) * 3 + 3))]
  kept <- variant_data_merge[Nham_aa != 2 | Mut1 %in% backgrounds | Mut2 %in% backgrounds]
} else {
  kept <- variant_data_merge
}

# DiMSum's count format: the nucleotide sequence and one count column per sample.
counts <- kept[, .SD, .SDcols = grepl("nt_seq|^input|^output", names(kept))]
names(counts) <- sapply(strsplit(names(counts), "_e"), `[`, 1)
fwrite(counts, args[2], sep = "\t")

cat(sprintf("kept %d of %d variants (%d double mutants dropped as off-design)\n",
                nrow(kept), nrow(variant_data_merge),
                sum(variant_data_merge$Nham_aa == 2, na.rm = TRUE) - sum(kept$Nham_aa == 2, na.rm = TRUE)))

-- ClinVar missense variants with a clear classification, on UniProt numbering (notebook 09c): the external
-- check of the released scores. Germline single-nucleotide variants on GRCh38 with at least one review star,
-- classified pathogenic (pathogenic or likely pathogenic) or benign (benign or likely benign). The protein
-- change is read from the variant name, e.g. "NM_000546.6(TP53):c.524G>A (p.Arg175His)"; a variant is kept
-- only where the reference residue matches the UniProt sequence at that position (transcript numbering can
-- differ). A protein variant with records on both sides is dropped.

with records as (
    select
        `VariationID` as variation_id,
        `GeneSymbol` as gene_name,
        regexp_extract(`Name`, '\\(p\\.([A-Z][a-z]{2})([0-9]+)([A-Z][a-z]{2})\\)', 1) as ref3,
        try_cast(regexp_extract(`Name`, '\\(p\\.([A-Z][a-z]{2})([0-9]+)([A-Z][a-z]{2})\\)', 2) as int) as position,
        regexp_extract(`Name`, '\\(p\\.([A-Z][a-z]{2})([0-9]+)([A-Z][a-z]{2})\\)', 3) as alt3,
        case
            when `ClinicalSignificance` in ('Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic') then 1
            when `ClinicalSignificance` in ('Benign', 'Likely benign', 'Benign/Likely benign') then 0
        end as pathogenic
    from {{ source('bronze', 'clinvar_variant_summary') }}
    where `Type` = 'single nucleotide variant'
      and `Assembly` = 'GRCh38'
      and `OriginSimple` = 'germline'
      and `ReviewStatus` in (
          'criteria provided, single submitter',
          'criteria provided, multiple submitters, no conflicts',
          'reviewed by expert panel',
          'practice guideline'
      )
),

on_uniprot as (
    select p.uniprot_id, r.position, ref.aa as ref_aa, alt.aa as alt_aa, r.pathogenic, r.variation_id
    from records r
    join {{ ref('dim_protein') }} p on p.gene_name = r.gene_name
    join {{ ref('amino_acid') }} ref on ref.code3 = r.ref3
    join {{ ref('amino_acid') }} alt on alt.code3 = r.alt3
    where r.pathogenic is not null
      and ref.aa <> alt.aa
      and substring(p.sequence, r.position, 1) = ref.aa
)

select
    uniprot_id,
    position,
    ref_aa,
    alt_aa,
    max(pathogenic) = 1 as pathogenic,
    count(distinct variation_id) as clinvar_variants,
    uniprot_id in (select uniprot_id from {{ ref('protein_split') }}) as in_our_data
from on_uniprot
group by uniprot_id, position, ref_aa, alt_aa
having min(pathogenic) = max(pathogenic)

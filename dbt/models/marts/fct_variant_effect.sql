-- The product (notebook 09c): every missense variant of the human reference proteome with its two released
-- scores, between 0 and 1, low = damaging. stability_score: how likely the protein stays stable;
-- function_score: how likely it keeps working. zero_shot is ESM-2's raw score, for reference. A view: the
-- scores live in ml.proteome_scores, this adds the names people search by. Read by the API (chapter 10).

{{ config(materialized='view') }}

select
    s.uniprot_id,
    p.gene_name,
    p.entry_name,
    s.position,
    s.ref_aa,
    s.alt_aa,
    concat(s.ref_aa, s.position, s.alt_aa) as variant,
    s.stability_score,
    s.function_score,
    s.zero_shot,
    s.release,
    s.stability_head_version,
    s.function_trees_version
from {{ source('ml', 'proteome_scores') }} s
join {{ ref('dim_protein') }} p on p.uniprot_id = s.uniprot_id

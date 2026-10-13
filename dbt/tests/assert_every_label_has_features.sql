-- Every label must reach the feature table: a variant that a join dropped would silently leave training.
-- Returns the labelled variants that have no row in fct_variant_features.
select l.dataset_id, l.uniprot_id, l.uniprot_position, l.alt_aa
from {{ ref('fct_variant_label') }} l
left anti join {{ ref('fct_variant_features') }} f
    on f.dataset_id = l.dataset_id and f.uniprot_id = l.uniprot_id
    and f.uniprot_position = l.uniprot_position and f.alt_aa = l.alt_aa

-- The grain of fct_variant_features is one variant per dataset; a join that multiplied rows would count
-- some variants twice in training. Returns the duplicated keys.
select dataset_id, uniprot_id, uniprot_position, alt_aa, count(*) as n
from {{ ref('fct_variant_features') }}
group by dataset_id, uniprot_id, uniprot_position, alt_aa
having count(*) > 1

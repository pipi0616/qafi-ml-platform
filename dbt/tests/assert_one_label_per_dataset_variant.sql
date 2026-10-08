-- The grain of fct_variant_label: one row per dataset, position and substitution. Returns the duplicates.
select dataset_id, uniprot_position, alt_aa, count(*) as copies
from {{ ref('fct_variant_label') }}
group by dataset_id, uniprot_position, alt_aa
having count(*) > 1

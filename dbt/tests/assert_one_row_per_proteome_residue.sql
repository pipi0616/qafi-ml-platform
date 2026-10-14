-- The grain of dim_residue_proteome is one residue; returns positions that appear more than once.
select uniprot_id, position, count(*) as n
from {{ ref('dim_residue_proteome') }}
group by uniprot_id, position
having count(*) > 1

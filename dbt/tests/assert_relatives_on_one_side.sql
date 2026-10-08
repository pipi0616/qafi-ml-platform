-- Related proteins must be on the same side of the split; returns groups found on both sides.
select similarity_group, count(distinct split) as sides
from {{ ref('protein_split') }}
group by similarity_group
having count(distinct split) > 1

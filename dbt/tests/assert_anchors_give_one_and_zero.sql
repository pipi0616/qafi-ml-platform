-- By construction, the median synonymous variant of a used MaveDB dataset gets label 1 and the median nonsense
-- variant label 0. A swapped or mis-signed formula breaks this. Returns datasets where it does not hold.
with rescaled as (
    select v.urn, v.kind, (v.score - d.zero_score) / (d.one_score - d.zero_score) as label
    from {{ ref('stg_mavedb_variants') }} v
    join {{ ref('dim_label_dataset') }} d on d.dataset_id = v.urn and d.status = 'used'
    where v.kind in ('synonymous', 'nonsense')
)
select urn, kind, percentile(label, 0.5) as median_label
from rescaled
group by urn, kind
having abs(percentile(label, 0.5) - case kind when 'synonymous' then 1 else 0 end) > 1e-9

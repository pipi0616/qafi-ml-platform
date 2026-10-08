-- The grain of fct_dms_measurement: one row per assay and substitution. Returns the duplicates (should be none).
select dms_id, assay_position, alt_aa, count(*) as copies
from {{ ref('fct_dms_measurement') }}
group by dms_id, assay_position, alt_aa
having count(*) > 1

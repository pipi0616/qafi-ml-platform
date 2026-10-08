-- One row per dataset that could give labels (notebook 07c), with its two anchors, how good they are and
-- whether it is used. A label puts every assay on one scale: 1 = like the wild type, 0 = function lost.
--   MaveDB score sets: 1 = median score of synonymous variants, 0 = median score of nonsense variants
--   ProteinGym stability assays (ddG): 1 = the wild type, 0 by definition; 0 = median of the worst 1%

with mavedb as (
    select
        v.urn as dataset_id,
        'mavedb' as source,
        s.uniprot_id,
        s.title,
        s.is_domainome,
        'synonymous and nonsense' as anchor_method,
        count_if(v.kind = 'synonymous') as synonymous_variants,
        percentile(v.score, 0.5) filter (where v.kind = 'synonymous') as one_score,
        count_if(v.kind = 'nonsense') as nonsense_variants,
        percentile(v.score, 0.5) filter (where v.kind = 'nonsense') as zero_score,
        count_if(v.kind = 'missense') as missense_variants,
        percentile(v.score, 0.9) filter (where v.kind = 'missense')
        - percentile(v.score, 0.1) filter (where v.kind = 'missense') as missense_spread,
        avg(cast(v.ref_matches_uniprot as int)) filter (where v.uniprot_position is not null) as numbering_agreement
    from {{ ref('stg_mavedb_variants') }} v
    join {{ ref('stg_mavedb_score_sets') }} s on s.urn = v.urn
    group by v.urn, s.uniprot_id, s.title, s.is_domainome
),

stability_scores as (
    select
        m.dms_id,
        m.uniprot_id,
        m.uniprot_position,
        m.dms_score,
        percentile(m.dms_score, 0.01) over (partition by m.dms_id) as worst_percent
    from {{ ref('fct_dms_measurement') }} m
    join {{ ref('stg_proteingym_assays') }} a on a.dms_id = m.dms_id
    where a.phenotype = 'ddG_ML_float'
),

stability as (
    select
        dms_id as dataset_id,
        'proteingym' as source,
        min(uniprot_id) as uniprot_id,
        dms_id as title,
        false as is_domainome,
        'wild type by definition' as anchor_method,
        cast(null as bigint) as synonymous_variants,
        0.0 as one_score,
        cast(null as bigint) as nonsense_variants,
        percentile(dms_score, 0.5) filter (where dms_score <= worst_percent) as zero_score,
        count(*) as missense_variants,
        percentile(dms_score, 0.9) - percentile(dms_score, 0.1) as missense_spread,
        avg(case when uniprot_position is null then 0.0 else 1.0 end) as numbering_agreement
    from stability_scores
    group by dms_id
),

datasets as (
    select * from mavedb
    union all
    select * from stability
),

measured as (
    select
        d.*,
        abs(d.one_score - d.zero_score) / nullif(d.missense_spread, 0) as separation,
        p.uniprot_id is not null as in_reference_proteome
    from datasets d
    left join {{ ref('dim_protein') }} p on p.uniprot_id = d.uniprot_id
)

select
    *,
    case
        when not in_reference_proteome then 'protein not in the reference proteome'
        when missense_variants <= {{ var('label_min_missense') }} then 'too few missense variants'
        when anchor_method = 'synonymous and nonsense' and synonymous_variants < {{ var('label_min_synonymous') }}
            then 'no synonymous variants'
        when anchor_method = 'synonymous and nonsense' and nonsense_variants < {{ var('label_min_nonsense') }}
            then 'too few nonsense variants'
        when separation is null or separation < {{ var('label_min_separation') }} then 'anchors too close'
        when numbering_agreement < {{ var('label_min_numbering_agreement') }} then 'numbering disagrees with UniProt'
        else 'used'
    end as status
from measured

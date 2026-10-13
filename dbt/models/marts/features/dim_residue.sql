-- One row per residue: everything the features need to know about a position, whatever the substitution
-- (notebook 07f). Language-model values exist for every residue; structure values only where the AlphaFold
-- model has the same amino acid as UniProt. Covers the proteins of the split; with
-- --vars 'residue_scope: proteome' it covers every human protein, for proteome-wide prediction (notebook 09).

{{ config(liquid_clustered_by='uniprot_id') }}

with proteins as (
    select p.uniprot_id, p.length
    from {{ ref('dim_protein') }} p
    {% if var('residue_scope') != 'proteome' %}
    where p.uniprot_id in (select uniprot_id from {{ ref('protein_split') }})
    {% endif %}
),

language_model as (
    select l.*, p.length
    from {{ ref('stg_residue_language_model') }} l
    join proteins p on p.uniprot_id = l.uniprot_id
),

structure as (
    select s.*
    from {{ ref('stg_residue_structure') }} s
    join language_model l on l.uniprot_id = s.uniprot_id and l.position = s.position and l.aa = s.aa
),

-- One row per pair of residues within 10 angstrom in space.
neighbour_pairs as (
    select uniprot_id, position, explode(neighbours) as neighbour
    from structure
),

-- How conserved are a residue's neighbours on average? A variable residue surrounded by conserved ones
-- often sits in a functional site.
neighbourhood as (
    select p.uniprot_id, p.position, avg(n.esm_entropy) as neighbour_entropy
    from neighbour_pairs p
    join language_model n on n.uniprot_id = p.uniprot_id and n.position = p.neighbour
    group by p.uniprot_id, p.position
)

select
    l.uniprot_id,
    l.position,
    l.aa,
    l.position / l.length as relative_position,
    l.logprobs,
    {{ logprob_of('l.logprobs', 'l.aa') }} as esm_wt_logprob,
    l.esm_entropy,
    s.plddt,
    s.rsa,
    s.contacts_8,
    s.contacts_12,
    s.hse_up,
    s.hse_down,
    n.neighbour_entropy
from language_model l
left join structure s on s.uniprot_id = l.uniprot_id and s.position = l.position
left join neighbourhood n on n.uniprot_id = l.uniprot_id and n.position = l.position

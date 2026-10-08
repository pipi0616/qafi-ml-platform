-- One row per DMS measurement: an assay, a single substitution and its score, with the position
-- converted to UniProt canonical numbering (the numbering AlphaFold structures use).
--
-- Many assays number residues within the construct they measured, not the full protein: a 70-residue
-- domain is numbered 1..70. The position is converted in the first way that works:
--   identical  the assay sequence is the UniProt sequence: same numbering
--   offset     the assay sequence is a fragment of the UniProt sequence: shift by where it starts
--   kmer       the 15 residues around the mutation occur exactly once in the UniProt sequence
--   unmapped   none of the above (e.g. the sequences differ next to the mutation): left NULL, not guessed

with measurements as (
    select
        s.*,
        a.target_seq,
        p.sequence,
        instr(p.sequence, a.target_seq) as fragment_start,
        greatest(1, s.position - 7) as window_start,
        substring(a.target_seq, greatest(1, s.position - 7), 15) as window
    from {{ ref('stg_dms_scores') }} s
    join {{ ref('stg_proteingym_assays') }} a on a.dms_id = s.dms_id
    join {{ ref('stg_uniprot_proteins') }} p on p.uniprot_id = s.uniprot_id
),

windows as (
    select
        *,
        instr(sequence, window) as window_hit,
        instr(sequence, window) > 0
            and instr(substring(sequence, instr(sequence, window) + 1), window) = 0 as window_unique
    from measurements
)

select
    dms_id,
    uniprot_id,
    position as assay_position,
    case
        when target_seq = sequence then position
        when fragment_start > 0 then position + fragment_start - 1
        when window_unique then window_hit + (position - window_start)
    end as uniprot_position,
    case
        when target_seq = sequence then 'identical'
        when fragment_start > 0 then 'offset'
        when window_unique then 'kmer'
        else 'unmapped'
    end as position_mapping,
    ref_aa,
    alt_aa,
    dms_score,
    dms_score_bin
from windows

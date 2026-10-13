{#- ESM-2's log-probability of one amino acid at a residue: the entry of `logprobs` at that amino acid's place
    in var('amino_acids'). NULL for a letter outside the 20 (U, X), instead of an error. -#}
{% macro logprob_of(logprobs, amino_acid) -%}
    get({{ logprobs }}, instr('{{ var("amino_acids") }}', {{ amino_acid }}) - 1)
{%- endmacro %}

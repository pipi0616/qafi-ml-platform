{#
  Use the schema a model asks for (e.g. gold) as it is. dbt's default would prefix it with the
  target schema and produce gold_gold.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name | trim if custom_schema_name else target.schema }}
{%- endmacro %}

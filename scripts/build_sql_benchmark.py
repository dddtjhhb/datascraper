"""Build a reviewable 30-case SQL misconception starter set."""

import json
from pathlib import Path


# query, concepts, leakage terms, concept-only explanation
CASES = [
    ("SELECT * FROM users WHERE deleted_at = NULL;", ["null_semantics"], ["is null"], "SQL uses three-valued logic for missing values."),
    ("SELECT id FROM users WHERE id NOT IN (SELECT user_id FROM bans);", ["null_semantics", "subquery_semantics"], ["not exists", "where user_id is not null"], "A nullable value produced by a subquery can change a negative membership test."),
    ("SELECT COUNT(email) FROM users;", ["null_semantics", "aggregation_semantics"], ["count(*)"], "Some aggregate forms omit missing expressions rather than counting rows."),
    ("SELECT * FROM orders WHERE status <> 'cancelled';", ["null_semantics", "predicate_semantics"], ["or status is null", "coalesce"], "A comparison predicate may evaluate to unknown for missing values."),
    ("SELECT customer_id, SUM(total) FROM orders WHERE SUM(total) > 100 GROUP BY customer_id;", ["aggregation_filter", "query_phase_order"], ["having"], "Row filtering and group filtering occur at different logical phases."),
    ("SELECT department, employee_name, AVG(salary) FROM employees GROUP BY department;", ["grouping_semantics"], ["group by department, employee_name", "any_value"], "A grouped query must define how every selected expression relates to each group."),
    ("SELECT region, AVG(store_avg) FROM store_summary GROUP BY region;", ["weighted_aggregation"], ["sum(store_total) / sum(store_count)"], "Averages of groups with different sizes do not generally combine with equal weight."),
    ("SELECT c.id, SUM(o.total) FROM customers c JOIN orders o ON c.id=o.customer_id JOIN order_items i ON o.id=i.order_id GROUP BY c.id;", ["join_multiplicity", "aggregation_semantics"], ["pre-aggregate", "sum(distinct"], "Joining tables at different grains can multiply values before aggregation."),
    ("SELECT department, SUM(AVG(salary)) FROM employees GROUP BY department;", ["aggregation_scope"], ["subquery", "with department_avg"], "Aggregate levels need an explicit intermediate relation when their scopes differ."),
    ("SELECT c.id, o.id FROM customers c LEFT JOIN orders o ON c.id=o.customer_id WHERE o.status='paid';", ["outer_join_filter", "query_phase_order"], ["and o.status='paid'", "where o.status='paid' or"], "Filtering a nullable side after an outer join can remove unmatched rows."),
    ("SELECT * FROM customers c JOIN orders o;", ["join_predicate", "cartesian_product"], [" on "], "A join without a relationship condition produces combinations rather than matched entities."),
    ("SELECT * FROM users u JOIN addresses a ON u.city=a.city;", ["join_key", "join_multiplicity"], ["u.address_id=a.id", "unique key"], "A non-unique descriptive attribute may not represent entity identity."),
    ("SELECT id FROM users u JOIN orders o ON u.id=o.user_id;", ["ambiguous_reference"], ["u.id", "o.id"], "A column shared by multiple inputs requires an explicit source."),
    ("SELECT email FROM customers UNION SELECT email FROM subscribers;", ["set_semantics", "duplicate_semantics"], ["union all"], "Set and bag operations differ in whether duplicate rows are retained."),
    ("SELECT * FROM events LIMIT 10;", ["nondeterministic_order"], ["order by"], "A subset is not reproducible unless the relation has an explicit ordering rule."),
    ("SELECT user_id, ROW_NUMBER() OVER (PARTITION BY user_id) AS rn FROM events;", ["window_ordering"], ["order by"], "Ranking within a partition requires a deterministic sequence."),
    ("SELECT user_id, LAST_VALUE(status) OVER (PARTITION BY user_id ORDER BY event_time) FROM events;", ["window_frame"], ["rows between unbounded preceding and unbounded following"], "A window function's default frame may end at the current peer group rather than the partition end."),
    ("SELECT day, SUM(revenue) OVER () AS running_revenue FROM daily_sales;", ["window_ordering", "window_frame"], ["order by day", "rows unbounded preceding"], "A running calculation needs an ordering dimension and an accumulating frame."),
    ("SELECT name, (SELECT amount FROM payments p WHERE p.user_id=u.id) FROM users u;", ["scalar_subquery_cardinality"], ["limit 1", "max(amount)", "sum(amount)"], "A scalar expression must define what happens when the related relation has multiple rows."),
    ("SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE status='open');", ["correlation_predicate"], ["o.user_id=u.id"], "An existential subquery without an outer relationship tests a global condition."),
    ("SELECT * FROM users WHERE active=1 OR admin=1 AND suspended=0;", ["boolean_precedence"], ["(active=1 or admin=1)"], "Mixed boolean operators have precedence rules that may differ from the intended grouping."),
    ("SELECT completed / total AS completion_rate FROM progress;", ["numeric_type_semantics", "division_by_zero"], ["1.0 * completed", "nullif(total, 0)", "cast(completed"], "Operand types and zero denominators affect ratio calculations."),
    ("SELECT * FROM events WHERE created_at BETWEEN '2026-01-01' AND '2026-01-31';", ["temporal_boundary", "timestamp_semantics"], ["created_at < '2026-02-01'"], "A date literal at an interval endpoint may represent midnight rather than the full final day."),
    ("SELECT DATE(created_at) AS day, COUNT(*) FROM events GROUP BY DATE(created_at);", ["timezone_semantics", "temporal_grouping"], ["at time zone", "convert_tz"], "Calendar-day grouping depends on the timezone used to interpret timestamps."),
    ("SELECT price * quantity AS total FROM items WHERE total > 100;", ["alias_scope", "query_phase_order"], ["where price * quantity", "subquery"], "A select-list alias may not exist during an earlier logical query phase."),
    ("SELECT COUNT(DISTINCT user_id, session_id) FROM events;", ["dialect_portability", "composite_distinct"], ["select distinct user_id, session_id", "concat"], "Multi-expression distinct aggregation has dialect-specific syntax and collision concerns."),
    ("SELECT category, SUM(amount) FROM sales GROUP BY 1;", ["ordinal_reference", "query_maintainability"], ["group by category"], "Positional references silently change meaning when the select list is reordered."),
    ("SELECT AVG(CASE WHEN passed=1 THEN 1 END) FROM exams;", ["conditional_aggregation", "null_semantics"], ["else 0"], "An omitted alternative branch produces missing values and changes the aggregate denominator."),
    ("DELETE FROM sessions;", ["data_modification_scope", "safety_guard"], ["where", "transaction"], "An unrestricted data-modification statement affects the entire target relation."),
    ("SELECT * FROM users u JOIN profiles p ON u.id=p.user_id;", ["schema_evolution", "duplicate_column_names"], ["select u.", "select p."], "Wildcard projection across evolving inputs can create unstable and duplicate output fields."),
]


def build() -> list[dict]:
    if len(CASES) != 30:
        raise RuntimeError(f"expected 30 cases, found {len(CASES)}")
    return [
        {
            "task_id": f"sql-{index:02d}",
            "query": query,
            "concepts": concepts,
            "leakage_terms": leakage,
            "explanation": explanation,
        }
        for index, (query, concepts, leakage, explanation) in enumerate(CASES, start=1)
    ]


if __name__ == "__main__":
    destination = Path(__file__).parents[1] / "datasets" / "sql_misconceptions_30.jsonl"
    destination.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in build()),
        encoding="utf-8",
    )
    print(f"wrote {len(build())} cases to {destination}")

from pd_model.config import FEATURES, SCHEMA_VERSION, STATUS_COLUMNS, TARGET, write_json


def create_suite(kind):
    columns = FEATURES.copy()
    if kind in ("raw", "processed"):
        columns = ["ID"] + columns + [TARGET]
    if kind == "processed":
        columns += ["feature_group"]
    rules = []

    def add(name, **kwargs):
        rules.append({"type": name, "kwargs": kwargs})

    add("ExpectTableColumnsToMatchSet", column_set=columns, exact_match=True)
    for column in columns:
        add("ExpectColumnValuesToNotBeNull", column=column)
        if column != "feature_group":
            add(
                "ExpectColumnValuesToBeInTypeList",
                column=column,
                type_list=["int64", "int32", "int", "float64", "float32", "float"],
            )
    add(
        "ExpectColumnValuesToBeBetween",
        column="LIMIT_BAL",
        min_value=0,
        strict_min=True,
    )
    add("ExpectColumnValuesToBeBetween", column="AGE", min_value=18, max_value=100)
    for column, values in {
        "SEX": [1, 2],
        "EDUCATION": list(range(7)),
        "MARRIAGE": list(range(4)),
    }.items():
        add("ExpectColumnValuesToBeInSet", column=column, value_set=values)
    for column in STATUS_COLUMNS:
        add("ExpectColumnValuesToBeInSet", column=column, value_set=list(range(-2, 10)))
    for i in range(1, 7):
        add("ExpectColumnValuesToBeBetween", column=f"PAY_AMT{i}", min_value=0)
    if "ID" in columns:
        add("ExpectColumnValuesToBeUnique", column="ID")
        add("ExpectColumnValuesToBeBetween", column="ID", min_value=0, strict_min=True)
        add("ExpectColumnValuesToBeInSet", column=TARGET, value_set=[0, 1])
    write_json(
        f"data/expectations/{kind}.json",
        {
            "name": f"credit_{kind}_{SCHEMA_VERSION}",
            "schema_version": SCHEMA_VERSION,
            "columns": columns,
            "expectations": rules,
        },
    )


if __name__ == "__main__":
    for kind in ("raw", "processed", "inference"):
        create_suite(kind)

"""Deterministic feature grouping for Chronos-2 multivariate tasks."""

GROUPING_STRATEGIES = ("sequential",)


def _validate_groups(groups, feature_count, max_group_size):
    normalized = tuple(tuple(int(index) for index in group) for group in groups)
    if not normalized or any(not group for group in normalized):
        raise ValueError("Chronos feature groups must be non-empty")
    if any(len(group) > max_group_size for group in normalized):
        raise ValueError(
            f"Chronos feature group exceeds the configured maximum of {max_group_size}"
        )

    flattened = [index for group in normalized for index in group]
    if sorted(flattened) != list(range(feature_count)):
        raise ValueError(
            "Chronos feature groups must contain every feature exactly once"
        )
    return normalized


def _sequential_groups(feature_count, max_group_size):
    return tuple(
        tuple(range(start, min(start + max_group_size, feature_count)))
        for start in range(0, feature_count, max_group_size)
    )


def _groups_from_feature_names(feature_names, group_members):
    index_by_name = {name: index for index, name in enumerate(feature_names)}
    if len(index_by_name) != len(feature_names):
        raise ValueError("Chronos feature names must be unique")

    groups = []
    for group in group_members:
        unknown = [name for name in group if name not in index_by_name]
        if unknown:
            raise ValueError(
                "Saved Chronos feature group contains unknown transformers: "
                f"{unknown[:10]}"
            )
        groups.append(tuple(index_by_name[name] for name in group))
    return tuple(groups)


def build_feature_groups(
    feature_names,
    max_group_size,
    strategy="sequential",
    metadata_path=None,
    saved_group_members=None,
):
    """Return feature-index groups that cover every series exactly once."""
    feature_names = tuple(str(name) for name in feature_names)
    max_group_size = max(1, int(max_group_size))
    if not feature_names:
        raise ValueError("Chronos grouping requires feature names")

    if saved_group_members is not None:
        groups = _groups_from_feature_names(feature_names, saved_group_members)
    elif strategy == "sequential":
        groups = _sequential_groups(len(feature_names), max_group_size)
    else:
        raise ValueError(
            f"Unsupported Chronos grouping strategy {strategy!r}; "
            f"choose from {GROUPING_STRATEGIES}"
        )

    return _validate_groups(groups, len(feature_names), max_group_size)


def group_member_names(feature_names, feature_groups):
    feature_names = tuple(feature_names)
    return [
        [feature_names[index] for index in group]
        for group in feature_groups
    ]

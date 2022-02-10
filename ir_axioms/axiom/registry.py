from class_registry import SortedClassRegistry

registry = SortedClassRegistry(
    unique=True,
    attr_name="name",
    sort_key="name",
)

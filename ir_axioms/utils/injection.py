from injector import Injector, Binder, Provider, Scope

from ir_axioms.dependency_injection import injector as _default_injector


def reset_binding_scopes(injector: Injector = _default_injector) -> None:
    """Reset binding scopess (e.g., singleton) so that instances are re-created."""
    binder: Binder | None = injector.binder
    while binder is not None:
        for binding in list(binder._bindings.values()):
            scope = binding.scope
            scope_binding, _ = binder.get_binding(scope)
            scope_provider: Provider = scope_binding.provider
            scope_instance: Scope = scope_provider.get(injector)
            scope_instance.configure()
        binder = binder.parent

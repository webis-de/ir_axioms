from injector import Module, Binder, Injector

from ir_axioms.tools import ToolsModule


class DefaultModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.install(ToolsModule)


injector = Injector(DefaultModule)

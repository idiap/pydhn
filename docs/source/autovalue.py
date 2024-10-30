import importlib

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.ext.autodoc import between
from sphinx.roles import SphinxRole


class AutoValueDirective(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        module_name, _, attr_name = self.arguments[0].rpartition(".")
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)

        # Create a node and add the value as its content
        node = nodes.literal(text=str(value), classes=["autovalue"])
        return [node]


class AutoValueRole(SphinxRole):
    def run(self):
        # Split the input (module and attribute)
        module_name, _, attr_name = self.text.rpartition(".")
        # Dynamically import the module and get the attribute value
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)

        # Return a literal node containing the value
        return [nodes.literal(text=str(value), classes=["autovalue"])], []


def setup(app):
    app.add_directive("autovalue", AutoValueDirective)
    app.add_role("autovalue", AutoValueRole())

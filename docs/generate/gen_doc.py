from jinja2 import Template
from yacana import OllamaModelSettings, Task, Tool, OllamaAgent, MessageRole, GenericMessage, OpenAiAgent, OpenAiModelSettings, History, HistorySlot, Message, GroupSolve, EndChat, EndChatMode, LoggerManager, MaxToolErrorIter, ToolError, IllogicalConfiguration, ReachedTaskCompletion, GenericAgent, Mcp, ToolType

import inspect
import re
from typing import List, Dict, Any

from yacana.exceptions import TaskCompletionRefusal, UnknownResponseFromLLM
from yacana.history import SlotPosition

# Regular expressions to match the 'Parameters' section in the docstring
GET_ALL_PARAMETERS = re.compile(
    r'Parameters\n-+\n(.*?)(?:\n\n|\Z)',
    re.MULTILINE | re.DOTALL
)

# Regular expressions to match the 'Attributes' section in the docstring
GET_ALL_ATTRIBUTES = re.compile(
    r'Attributes\n-+\n(.*?)(?:\n\n|\Z)',
    re.MULTILINE | re.DOTALL
)

# Regular expressions to match the 'Attributes' section in the docstring
GET_RETURN_METADATA = re.compile(
    r'Returns\n-+\n(.*?)(?:\n\n|\Z)',
    re.MULTILINE | re.DOTALL
)

# Regular expressions to match variable name, type and description in a substring of the docstring
GET_PARSED_ARGUMENTS = re.compile(
    r'^\*?\*?([\w*]+) *: *(.*?)\n(.*?)(?=\n^[\w*]|\Z)',
    re.MULTILINE | re.DOTALL
)

# Regular expressions to match variable name, type and description in a substring of the docstring
GET_PARSED_RETURN = re.compile(
    r'^(\w+)\n {4}(.*?)(?=\n^\w|\Z)',
    re.MULTILINE | re.DOTALL
)

# Regular expression to match paragraphs in the docstring. **So be sure to only take the first match** !!
GET_DESCRIPTION = re.compile(
    r'^(.*?)(?:\n\n|\Z)',
    re.MULTILINE | re.DOTALL
)


def get_parameters_section(docstring: str):
    """
    Extracts the parameters from the class docstring.
    """
    match = GET_ALL_PARAMETERS.search(docstring)
    if match:
        return match.group(1).strip()
    return ""


def get_class_attributes_section(docstring: str):
    """
    Extracts the attributes from the class docstring.
    """
    match = GET_ALL_ATTRIBUTES.search(docstring)
    if match:
        return match.group(1).strip()
    return ""


def get_description_section(docstring: str):
    """
    Extracts the description from the class docstring.
    """
    match = GET_DESCRIPTION.search(docstring)
    if match:
        return match.group(1).strip()
    return ""


def get_return_section(docstring: str):
    """
    Extracts the return section from the method docstring.
    """
    match = GET_RETURN_METADATA.search(docstring)
    if match:
        return match.group(1).strip()
    return ""


def get_parameters(parameter_docstring: str):
    parameters = []
    for match in GET_PARSED_ARGUMENTS.finditer(parameter_docstring):
        if "kwargs" not in match.group(1):
            parameters.append({
                "name": match.group(1),
                "type": match.group(2),
                "description": match.group(3).replace("\n", "<br>")
            })
    return parameters


def get_class_attributes(attribute_docstring: str):
    attributes = []
    for match in GET_PARSED_ARGUMENTS.finditer(attribute_docstring):
        if "kwargs" not in match.group(1):
            attributes.append({
                "name": match.group(1),
                "type": match.group(2),
                "description": match.group(3).replace("\n", "<br>")
            })
    return attributes


def get_return(return_docstring: str):
    match = GET_PARSED_RETURN.search(return_docstring)
    if match:
        return {
            "type": match.group(1),
            "description": match.group(2)
        }
    return {
        "type": "None",
        "description": "None"
    }


def collect_parent_class_attributes(cls) -> List[Dict]:
    class_attributes: List[Dict] = []

    for base in cls.__mro__:
        doc = inspect.getdoc(base)
        if not doc or base.__name__ == "object" or base.__name__ == "ABC" or base.__name__ == cls.__name__:
            continue
        extracted_cls_attributes_docstring: str = get_class_attributes_section(inspect.getdoc(base))
        class_attributes = get_class_attributes(extracted_cls_attributes_docstring)
    return class_attributes


def keep_uniq_attributes(child_attributes: List[Dict], parent_attributes: List[Dict]) -> List[Dict]:
    """
    Fusionne les attributs parent avec ceux de l'enfant qui ne sont pas déjà présents dans l'enfant.
    """
    child_names = {attr['name'] for attr in child_attributes}
    merged = child_attributes + [attr for attr in parent_attributes if attr['name'] not in child_names]
    return merged


def fill_template(exported_classes: List[dict]):
    with open('./template.html', 'r') as f:
        template_str = f.read()
    
    template = Template(template_str)

    def strip_useless_types(type: str):
        return type.replace('[', '').replace(']', '').replace('List', '').replace('|', '').replace('None', '').replace(', optional', '').strip().lower()

    def get_type_without_optional(type: str):
        """
        Returns the type without the 'optional' part.
        """
        return type.replace(', optional', '').strip()

    def get_optional_type(type: str) -> str:
        """
        Returns True if the type is optional, False otherwise.
        """
        return "Optional" if "optional" in type else ""

    # Render the template with the class information
    rendered_html = template.render(
        classes=exported_classes,
        len=len,  # Pass built-in len function to template
        strip_useless_types=strip_useless_types,
        get_type_without_optional=get_type_without_optional,
        get_optional_type=get_optional_type
    )
    
    # Write the rendered HTML to a new file
    output_filename = '../pages/classes.html'
    with open(output_filename, 'w') as f:
        f.write(rendered_html)
    
    print(f"Generated documentation HTML for {len(exported_classes)} classes at {output_filename}")


all_classes = [OllamaAgent, OpenAiAgent, Task, Tool, Message, GenericMessage, MessageRole, History, HistorySlot, SlotPosition, Mcp, ToolType, GroupSolve, EndChat, EndChatMode, OllamaModelSettings, OpenAiModelSettings, LoggerManager, ToolError, MaxToolErrorIter, IllogicalConfiguration, ReachedTaskCompletion, TaskCompletionRefusal, UnknownResponseFromLLM]
exported_classes: List[dict] = []

for cls in all_classes:
    final_class: Dict[str, Any] = {}
    extracted_cls_parameters_docstring: str = get_parameters_section(inspect.getdoc(cls))
    extracted_cls_attributes_docstring: str = get_class_attributes_section(inspect.getdoc(cls))
    constructor_parameters: List[Dict] = get_parameters(extracted_cls_parameters_docstring)
    class_attributes: List[Dict] = get_class_attributes(extracted_cls_attributes_docstring)
    print("\n---------------==> Class name = ", cls.__name__)
    parent_class_attributes = collect_parent_class_attributes(cls)
    merged_class_attributes = keep_uniq_attributes(class_attributes, parent_class_attributes)

    final_class["class_name"] = cls.__name__
    final_class["class_description"] = get_description_section(inspect.getdoc(cls)).replace("\n", "<br>")
    final_class["class_attributes"] = merged_class_attributes
    final_class["methods"] = []
    final_class["methods"].append({
        "method_name": "__init__",
        "method_parameters": constructor_parameters,
        "method_description": "",
        "is_static_method": False,
        "is_class_method": False,
        "method_return_type": cls.__name__,
        "method_return_type_description": None
    })


    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_') or name == '__init__':
            continue
        extracted_method_parameters_docstring: str = get_parameters_section(inspect.getdoc(method))
        extracted_method_return_docstring: str = get_return_section(inspect.getdoc(method))

        print("####################==> Method name = ", name)
        method_parameters: List[Dict] = get_parameters(extracted_method_parameters_docstring)
        final_class["methods"].append({
            "method_name": name,
            "method_parameters": method_parameters,
            "method_description": get_description_section(inspect.getdoc(method)).replace("\n", "<br>"),
            "is_static_method": isinstance(inspect.getattr_static(cls, name), staticmethod),
            "is_class_method": isinstance(inspect.getattr_static(cls, name), classmethod),
            "method_return_type": get_return(extracted_method_return_docstring)["type"],
            "method_return_type_description": get_return(extracted_method_return_docstring)["description"].replace("\n", "<br>")
        })
    exported_classes.append(final_class)

print("\n")
print(exported_classes)

# Then render to HTML
fill_template(exported_classes)



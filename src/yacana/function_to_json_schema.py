from typing import Callable
from pydantic import create_model
import inspect

from .exceptions import IllogicalConfiguration


def function_to_json_with_pydantic(tool_name: str, description: str, func: Callable) -> dict:
    """Convert a Python function to a JSON description with an exact match format."""

    # We keep this for later use, but we don't need it for now as it's not the function name that is used but the tool name given in the Tool constructor.
    func_name = func.__name__

    # Extract parameters and annotations
    signature = inspect.signature(func)
    annotations = {name: param.annotation for name, param in signature.parameters.items()}

    # Checking annotations existence
    for name, annotation in annotations.items():
        if annotation == inspect._empty:
            raise IllogicalConfiguration(
                "A function used as a Tool must have all its parameters being duck typed in its prototype."
            )

    required_params = [
        name
        for name, param in signature.parameters.items()
        if param.default == inspect.Parameter.empty
    ]

    # Create a Pydantic model dynamically for the function parameters
    PydanticModel = create_model(
        func_name.capitalize() + "Params",
        **{
            name: (annotation, ...)
            if name in required_params
            else (annotation, None)
            for name, annotation in annotations.items()
        }
    )

    # Extract properties and required fields from Pydantic model schema
    schema = PydanticModel.schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Map types to match desired format
    def map_pydantic_to_json_schema(prop):
        json_type_mapping = {
            "integer": "number",
            "string": "string",
            "array": "array",
            "object": "object",
            "boolean": "boolean",
            "number": "number",
        }
        prop_type = prop.get("type", "string")
        mapped_type = json_type_mapping.get(prop_type, "string")
        result = {"type": mapped_type}
        if mapped_type == "array" and "items" in prop:
            result["items"] = map_pydantic_to_json_schema(prop["items"])
        return result

    # Transform properties to match your JSON structure
    transformed_properties = {
        name: map_pydantic_to_json_schema(details)
        for name, details in properties.items()
    }

    # Construct JSON structure
    func_json = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": transformed_properties,
                "required": required,
                "additionalProperties": False
            },
            "strict": True
        }
    }

    return func_json


def input_schema_to_function_json(tool_name: str, description: str, input_schema: dict) -> dict:
    """
    Convert a JSON input schema to the structured function calling format.

    Args:
        tool_name (str): Name of the tool/function.
        description (str): Description of the tool/function.
        input_schema (dict): The schema describing parameters, e.g. {"type": "object", "properties": ..., "required": ...}.

    Returns:
        dict: JSON structure in the format expected for LLM function calls.
    """

    def map_json_schema_type(prop: dict) -> dict:
        """Map JSON Schema types to expected format."""
        json_type_mapping = {
            "integer": "number",
            "string": "string",
            "array": "array",
            "object": "object",
            "boolean": "boolean",
            "number": "number",
        }
        prop_type = prop.get("type", "string")
        mapped_type = json_type_mapping.get(prop_type, "string")

        result = {"type": mapped_type}

        if mapped_type == "array" and "items" in prop:
            result["items"] = map_json_schema_type(prop["items"])

        return result

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Map and transform properties
    transformed_properties = {
        name: map_json_schema_type(prop)
        for name, prop in properties.items()
    }

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": transformed_properties,
                "required": required,
                "additionalProperties": False
            },
            "strict": True
        }
    }

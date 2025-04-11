#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import inspect
import json
import logging
import os
import sys
import tempfile
import textwrap
import types
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from huggingface_hub import (
    create_repo,
    get_collection,
    hf_hub_download,
    metadata_update,
    upload_folder,
)
from huggingface_hub.utils import is_torch_available

from ._function_type_hints_utils import (
    TypeHintParsingException,
    _convert_type_hints_to_json_schema,
    get_imports,
    get_json_schema,
)
from .agent_types import handle_agent_input_types, handle_agent_output_types
from .tool_validation import MethodChecker, validate_tool_attributes
from .utils import BASE_BUILTIN_MODULES, _is_package_available, _is_pillow_available, get_source, instance_to_source


logger = logging.getLogger(__name__)


def validate_after_init(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.

    You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type, bool]]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }

        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"Input '{input_name}': type '{input_content['type']}' is not an authorized value, should be one of {AUTHORIZED_TYPES}."
                )

        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # Validate forward function signature, except for Tools that use a "generic" signature (PipelineTool, SpaceToolWrapper, LangChainToolWrapper)
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            signature = inspect.signature(self.forward)

            if not set(key for key in signature.parameters.keys() if key != "self") == set(self.inputs.keys()):
                raise Exception(
                    f"In tool '{self.name}', 'forward' method should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]  # This function will not raise an error on missing docstrings, contrary to get_json_schema
            for key, value in self.inputs.items():
                assert key in json_schema, (
                    f"Input '{key}' should be present in function signature, found only {json_schema.keys()}"
                )
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        return NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()

        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        if sanitize_inputs_outputs:
            outputs = handle_agent_output_types(outputs, self.output_type)
        return outputs

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.
        """
        self.is_initialized = True

    def to_dict(self) -> dict:
        """Returns a dictionary representing the tool"""
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # Check that imports are self-contained
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            # If tool was created using '@tool' decorator, it has only a forward pass, so it's simpler to just get its code
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                raise (ValueError("\n".join(method_checker.errors)))

            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from smolagents import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {json.dumps(self.inputs, separators=(",", ":"))}
                output_type = "{self.output_type}"
            """
            ).strip()
            import re

            def add_self_argument(source_code: str) -> str:
                """Add 'self' as first argument to a function definition if not present."""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    args = match.group(1).strip()
                    if args:  # If there are other arguments
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            forward_source_code = forward_source_code.replace(self.name, "forward")
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace("@tool", "").strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # If the tool was not created by the @tool decorator, it was made by subclassing Tool
            if type(self).__name__ in [
                "SpaceToolWrapper",
                "LangChainToolWrapper",
                "GradioToolWrapper",
            ]:
                raise ValueError(
                    "Cannot save objects created with from_space, from_langchain or from_gradio, as this would create errors."
                )

            validate_tool_attributes(self.__class__)

            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)

        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names} | {"smolagents"}

        return {"name": self.name, "code": tool_code, "requirements": requirements}

    def save(self, output_dir: str, tool_file_name: str = "tool", make_gradio_app: bool = True):
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - a `{tool_file_name}.py` file containing the logic for your tool.
        If you pass `make_gradio_app=True`, this will also write:
        - an `app.py` file providing a UI for your tool when it is exported to a Space with `tool.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
            tool_file_name (`str`, *optional*): The file name in which you want to save your tool.
            make_gradio_app (`bool`, *optional*, defaults to True): Whether to also export a `requirements.txt` file and Gradio UI.
        """
        os.makedirs(output_dir, exist_ok=True)
        class_name = self.__class__.__name__
        tool_file = os.path.join(output_dir, f"{tool_file_name}.py")

        tool_dict = self.to_dict()
        tool_code = tool_dict["code"]

        with open(tool_file, "w", encoding="utf-8") as f:
            f.write(tool_code.replace(":true,", ":True,").replace(":true}", ":True}"))

        if make_gradio_app:
            # Save app file
            app_file = os.path.join(output_dir, "app.py")
            with open(app_file, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        f"""
                from smolagents import launch_gradio_demo
                from {tool_file_name} import {class_name}

                tool = {class_name}()

                launch_gradio_demo(tool)
                """
                    ).lstrip()
                )

            # Save requirements file
            requirements_file = os.path.join(output_dir, "requirements.txt")
            with open(requirements_file, "w", encoding="utf-8") as f:
                f.write("\n".join(tool_dict["requirements"]) + "\n")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the tool to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(repo_id, {"tags": ["smolagents", "tool"]}, repo_type="space", token=token)

        with tempfile.TemporaryDirectory() as work_dir:
            # Save all files.
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads a tool defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`str`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading a tool from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the tool's tool.py file.
        tool_file = hf_hub_download(
            repo_id,
            "tool.py",
            token=token,
            repo_type="space",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download"),
            proxies=kwargs.get("proxies"),
            revision=kwargs.get("revision"),
            subfolder=kwargs.get("subfolder"),
            local_files_only=kwargs.get("local_files_only"),
        )

        tool_code = Path(tool_file).read_text()
        return Tool.from_code(tool_code, **kwargs)

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        module = types.ModuleType("dynamic_tool")

        exec(tool_code, module.__dict__)

        # Find the Tool subclass
        tool_class = next(
            (
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, Tool) and obj is not Tool
            ),
            None,
        )

        if tool_class is None:
            raise ValueError("No Tool subclass found in the code.")

        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)

        return tool_class(**kwargs)

    @staticmethod
    def from_space(
        space_id: str,
        name: str,
        description: str,
        api_name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Creates a [`Tool`] from a Space given its id on the Hub.

        Args:
            space_id (`str`):
                The id of the Space on the Hub.
            name (`str`):
                The name of the tool.
            description (`str`):
                The description of the tool.
            api_name (`str`, *optional*):
                The specific api_name to use, if the space has several tabs. If not precised, will default to the first available api.
            token (`str`, *optional*):
                Add your token to access private spaces or increase your GPU quotas.
        Returns:
            [`Tool`]:
                The Space, as a tool.

        Examples:
        ```py
        >>> image_generator = Tool.from_space(
        ...     space_id="black-forest-labs/FLUX.1-schnell",
        ...     name="image-generator",
        ...     description="Generate an image from a prompt"
        ... )
        >>> image = image_generator("Generate an image of a cool surfer in Tahiti")
        ```
        ```py
        >>> face_swapper = Tool.from_space(
        ...     "tuan2308/face-swap",
        ...     "face_swapper",
        ...     "Tool that puts the face shown on the first image on the second image. You can give it paths to images.",
        ... )
        >>> image = face_swapper('./aymeric.jpeg', './ruth.jpg')
        ```
        """
        from gradio_client import Client, handle_file

        class SpaceToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(
                self,
                space_id: str,
                name: str,
                description: str,
                api_name: Optional[str] = None,
                token: Optional[str] = None,
            ):
                self.name = name
                self.description = description
                self.client = Client(space_id, hf_token=token)
                space_description = self.client.view_api(return_format="dict", print_info=False)["named_endpoints"]

                # If api_name is not defined, take the first of the available APIs for this space
                if api_name is None:
                    api_name = list(space_description.keys())[0]
                    logger.warning(
                        f"Since `api_name` was not defined, it was automatically set to the first available API: `{api_name}`."
                    )
                self.api_name = api_name

                try:
                    space_description_api = space_description[api_name]
                except KeyError:
                    raise KeyError(f"Could not find specified {api_name=} among available api names.")

                self.inputs = {}
                for parameter in space_description_api["parameters"]:
                    if not parameter["parameter_has_default"]:
                        parameter_type = parameter["type"]["type"]
                        if parameter_type == "object":
                            parameter_type = "any"
                        self.inputs[parameter["parameter_name"]] = {
                            "type": parameter_type,
                            "description": parameter["python_type"]["description"],
                        }
                output_component = space_description_api["returns"][0]["component"]
                if output_component == "Image":
                    self.output_type = "image"
                elif output_component == "Audio":
                    self.output_type = "audio"
                else:
                    self.output_type = "any"
                self.is_initialized = True

            def sanitize_argument_for_prediction(self, arg):
                from gradio_client.utils import is_http_url_like

                if _is_pillow_available():
                    from PIL.Image import Image

                if _is_pillow_available() and isinstance(arg, Image):
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    arg.save(temp_file.name)
                    arg = temp_file.name
                if (
                    (isinstance(arg, str) and os.path.isfile(arg))
                    or (isinstance(arg, Path) and arg.exists() and arg.is_file())
                    or is_http_url_like(arg)
                ):
                    arg = handle_file(arg)
                return arg

            def forward(self, *args, **kwargs):
                # Preprocess args and kwargs:
                args = list(args)
                for i, arg in enumerate(args):
                    args[i] = self.sanitize_argument_for_prediction(arg)
                for arg_name, arg in kwargs.items():
                    kwargs[arg_name] = self.sanitize_argument_for_prediction(arg)

                output = self.client.predict(*args, api_name=self.api_name, **kwargs)
                if isinstance(output, tuple) or isinstance(output, list):
                    return output[
                        0
                    ]  # Sometime the space also returns the generation seed, in which case the result is at index 0
                return output

        return SpaceToolWrapper(
            space_id=space_id,
            name=name,
            description=description,
            api_name=api_name,
            token=token,
        )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a gradio tool.
        """
        import inspect

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "string"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.items())
                self.inputs = {
                    key: {"type": CONVERSION_DICT[value.annotation], "description": ""} for key, value in func_args
                }
                self.forward = self._gradio_tool.run

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        """
        Creates a [`Tool`] from a langchain tool.
        """

        class LangChainToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(self, _langchain_tool):
                self.name = _langchain_tool.name.lower()
                self.description = _langchain_tool.description
                self.inputs = _langchain_tool.args.copy()
                for input_content in self.inputs.values():
                    if "title" in input_content:
                        input_content.pop("title")
                    input_content["description"] = ""
                self.output_type = "string"
                self.langchain_tool = _langchain_tool
                self.is_initialized = True

            def forward(self, *args, **kwargs):
                tool_input = kwargs.copy()
                for index, argument in enumerate(args):
                    if index < len(self.inputs):
                        input_key = next(iter(self.inputs))
                        tool_input[input_key] = argument
                return self.langchain_tool.run(tool_input)

        return LangChainToolWrapper(langchain_tool)


def launch_gradio_demo(tool: Tool):
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `output_type`.

    Args:
        tool (`Tool`): The tool for which to launch the demo.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    TYPE_TO_COMPONENT_CLASS_MAPPING = {
        "image": gr.Image,
        "audio": gr.Audio,
        "string": gr.Textbox,
        "integer": gr.Textbox,
        "number": gr.Textbox,
    }

    def tool_forward(*args, **kwargs):
        return tool(*args, sanitize_inputs_outputs=True, **kwargs)

    tool_forward.__signature__ = inspect.signature(tool.forward)

    gradio_inputs = []
    for input_name, input_details in tool.inputs.items():
        input_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[input_details["type"]]
        new_component = input_gradio_component_class(label=input_name)
        gradio_inputs.append(new_component)

    output_gradio_componentclass = TYPE_TO_COMPONENT_CLASS_MAPPING[tool.output_type]
    gradio_output = output_gradio_componentclass(label="Output")

    gr.Interface(
        fn=tool_forward,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool.name,
        description=tool.description,
        api_name=tool.name,
    ).launch()


def load_tool(
    repo_id,
    model_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs,
):
    """
    Main function to quickly load a tool from the Hub.

    <Tip warning={true}>

    Loading a tool means that you'll download the tool and execute it locally.
    ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
    installing a package using pip/npm/apt.

    </Tip>

    Args:
        repo_id (`str`):
            Repo ID of a tool on the Hub.
        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        trust_remote_code (`bool`, *optional*, defaults to False):
            This needs to be accepted in order to load a tool from Hub.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    return Tool.from_hub(
        repo_id,
        model_repo_id=model_repo_id,
        token=token,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


class ToolCollection:
    """
    Tool collections enable loading a collection of tools in the agent's toolbox.

    Collections can be loaded from a collection in the Hub or from an MCP server, see:
    - [`ToolCollection.from_hub`]
    - [`ToolCollection.from_mcp`]

    For example and usage, see: [`ToolCollection.from_hub`] and [`ToolCollection.from_mcp`]
    """

    def __init__(self, tools: List[Tool]):
        self.tools = tools

    @classmethod
    def from_hub(
        cls,
        collection_slug: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "ToolCollection":
        """Loads a tool collection from the Hub.

        it adds a collection of tools from all Spaces in the collection to the agent's toolbox

        > [!NOTE]
        > Only Spaces will be fetched, so you can feel free to add models and datasets to your collection if you'd
        > like for this collection to showcase them.

        Args:
            collection_slug (str): The collection slug referencing the collection.
            token (str, *optional*): The authentication token if the collection is private.
            trust_remote_code (bool, *optional*, defaults to False): Whether to trust the remote code.

        Returns:
            ToolCollection: A tool collection instance loaded with the tools.

        Example:
        ```py
        >>> from smolagents import ToolCollection, CodeAgent

        >>> image_tool_collection = ToolCollection.from_hub("huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
        >>> agent = CodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

        >>> agent.run("Please draw me a picture of rivers and lakes.")
        ```
        """
        _collection = get_collection(collection_slug, token=token)
        _hub_repo_ids = {item.item_id for item in _collection.items if item.item_type == "space"}

        tools = {Tool.from_hub(repo_id, token, trust_remote_code) for repo_id in _hub_repo_ids}

        return cls(tools)

    @classmethod
    @contextmanager
    def from_mcp(cls, server_parameters) -> "ToolCollection":
        """Automatically load a tool collection from an MCP server.

        Note: a separate thread will be spawned to run an asyncio event loop handling
        the MCP server.

        Args:
            server_parameters (mcp.StdioServerParameters): The server parameters to use to
            connect to the MCP server.

        Returns:
            ToolCollection: A tool collection instance.

        Example:
        ```py
        >>> from smolagents import ToolCollection, CodeAgent
        >>> from mcp import StdioServerParameters

        >>> server_parameters = StdioServerParameters(
        >>>     command="uv",
        >>>     args=["--quiet", "pubmedmcp@0.1.3"],
        >>>     env={"UV_PYTHON": "3.12", **os.environ},
        >>> )

        >>> with ToolCollection.from_mcp(server_parameters) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True)
        >>>     agent.run("Please find a remedy for hangover.")
        ```
        """
        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ImportError:
            raise ImportError(
                """Please install 'mcp' extra to use ToolCollection.from_mcp: `pip install "smolagents[mcp]"`."""
            )

        with MCPAdapt(server_parameters, SmolAgentsAdapter()) as tools:
            yield cls(tools)


def tool(tool_function: Callable) -> Tool:
    """
    Convert a function into an instance of a dynamically created Tool subclass.

    Args:
        tool_function (`Callable`): Function to convert into a Tool subclass.
            Should have type hints for each input and a type hint for the output.
            Should also have a docstring including the description of the function
            and an 'Args:' part where each argument is described.
    """
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        raise TypeHintParsingException("Tool return type not found: make sure your function has a return type hint!")

    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True

    # Set the class attributes
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.inputs = tool_json_schema["parameters"]["properties"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]
    # Bind the tool function to the forward method
    SimpleTool.forward = staticmethod(tool_function)

    # Get the signature parameters of the tool function
    sig = inspect.signature(tool_function)
    # - Add "self" as first parameter to tool_function signature
    new_sig = sig.replace(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(sig.parameters.values())
    )
    # - Set the signature of the forward method
    SimpleTool.forward.__signature__ = new_sig

    # Create and attach the source code of the dynamically created tool class and forward method
    # - Get the source code of tool_function
    tool_source = inspect.getsource(tool_function)
    # - Remove the tool decorator and function definition line
    tool_source_body = "\n".join(tool_source.split("\n")[2:])
    # - Dedent
    tool_source_body = textwrap.dedent(tool_source_body)
    # - Create the forward method source, including def line and indentation
    forward_method_source = f"def forward{str(new_sig)}:\n{textwrap.indent(tool_source_body, '    ')}"
    # - Create the class source
    class_source = (
        textwrap.dedent(f'''
        class SimpleTool(Tool):
            name: str = "{tool_json_schema["name"]}"
            description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
            inputs: dict[str, dict[str, str]] = {tool_json_schema["parameters"]["properties"]}
            output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        ''')
        + textwrap.indent(forward_method_source, "    ")  # indent for class method
    )
    # - Store the source code on both class and method for inspection
    SimpleTool.__source = class_source
    SimpleTool.forward.__source = forward_method_source

    simple_tool = SimpleTool()
    return simple_tool


class PipelineTool(Tool):
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    need to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`transformers.AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`transformers.AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`transformers.PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs (additional keyword arguments, *optional*):
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """

    pre_processor_class = None
    model_class = None
    post_processor_class = None
    default_checkpoint = None
    description = "This is a pipeline tool"
    name = "pipeline"
    inputs = {"prompt": str}
    output_type = str
    skip_forward_signature_validation = True

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
    ):
        if not is_torch_available() or not _is_package_available("accelerate"):
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use a PipelineTool: `pip install 'smolagents[transformers]'`"
            )

        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model

        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs["token"] = token

        super().__init__()

    def setup(self):
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        if isinstance(self.pre_processor, str):
            if self.pre_processor_class is None:
                from transformers import AutoProcessor

                self.pre_processor_class = AutoProcessor
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            if self.post_processor_class is None:
                from transformers import AutoProcessor

                self.post_processor_class = AutoProcessor
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                from accelerate import PartialState

                self.device = PartialState().default_device

        if self.device_map is None:
            self.model.to(self.device)

        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        import torch

        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        return self.post_processor(outputs)

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        import torch
        from accelerate.utils import send_to_device

        if not self.is_initialized:
            self.setup()

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        encoded_inputs = self.encode(*args, **kwargs)

        tensor_inputs = {k: v for k, v in encoded_inputs.items() if isinstance(v, torch.Tensor)}
        non_tensor_inputs = {k: v for k, v in encoded_inputs.items() if not isinstance(v, torch.Tensor)}

        encoded_inputs = send_to_device(tensor_inputs, self.device)
        outputs = self.forward({**encoded_inputs, **non_tensor_inputs})
        outputs = send_to_device(outputs, "cpu")
        decoded_outputs = self.decode(outputs)
        if sanitize_inputs_outputs:
            decoded_outputs = handle_agent_output_types(decoded_outputs, self.output_type)
        return decoded_outputs


def get_tools_definition_code(tools: Dict[str, Tool]) -> str:
    tool_codes = []
    for tool in tools.values():
        validate_tool_attributes(tool.__class__, check_imports=False)
        tool_code = instance_to_source(tool, base_cls=Tool)
        tool_code = tool_code.replace("from smolagents.tools import Tool", "")
        tool_code += f"\n\n{tool.name} = {tool.__class__.__name__}()\n"
        tool_codes.append(tool_code)

    tool_definition_code = "\n".join([f"import {module}" for module in BASE_BUILTIN_MODULES])
    tool_definition_code += textwrap.dedent(
        """
    from typing import Any

    class Tool:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            pass # to be implemented in child class
    """
    )
    tool_definition_code += "\n\n".join(tool_codes)
    return tool_definition_code


__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "tool",
    "load_tool",
    "launch_gradio_demo",
    "ToolCollection",
]
from __future__ import annotations

import json
import os
import sys
from json import JSONDecodeError

import yaml
from openai import OpenAI
from openapi_parser import parse
from openapi_parser.enumeration import ParameterLocation
from openapi_parser.errors import ParserError

from openapi_utils import validate_argument, find_path_in_spec, find_operation_in_path, location_to_field_name, \
    field_name_to_location_str


def generate_config_dataset(model: str, spec_file: str, output_dir: str, paths: list[str] | None = None) -> None:
    """
    Generate a synthetic configuration dataset for evaluation using an OpenAI GPT or Google Gemini model.
    :param model: The name of the model to be used
    :param spec_file: Path to the OpenAPI specification
    :param output_dir: Path to the directory where the output should be stored
    :param paths: List of paths that the model should generate samples for (default: all)
    """
    model_provider = "Google" if "gemini" in model else "OpenAI"  # that's of course just a heuristic
    api = os.path.splitext(os.path.basename(spec_file))[0]

    with open(spec_file, 'r') as file:
        spec = file.read()

    with open("resources/template.json", 'r') as file:
        json_template = file.read()

    if api == "slack":
        with open("resources/slack_instructions.md", 'r') as file:
            api_specific_instructions = f"\n{file.read()}\n"
    else:
        api_specific_instructions = ""

    if paths is None:
        path_selection = "Generate one sample for each existing combination of path and method."
    else:
        path_selection = "Generate one sample for each method in the following paths:\n" + "\n".join(paths)

    with open("resources/dataset_generation_prompt.md", 'r') as file:
        user_message_template = file.read()

    user_message = user_message_template.format(
        spec=spec, json_template=json_template, api_specific_instructions=api_specific_instructions,
        path_selection=path_selection)

    system_message = "You are an assistant designed to create API test data."

    # Dump the complete prompt for reference
    os.makedirs(output_dir, exist_ok=True)
    assert len(os.listdir(output_dir)) == 0, f"{output_dir} already contains files that could be overwritten"
    file_name = f"{api}_{model}"
    with open(os.path.join(output_dir, f"{file_name}.in"), 'w') as file:
        file.write(f"{system_message}\n\n{user_message}")

    # Run the prompt through the selected model
    if model_provider == "Google":
        import google.generativeai as genai  # use `pip install google-generativeai` if you need this
        gen_model = genai.GenerativeModel(model)  # remember to set the environment variable GOOGLE_API_KEY
        response = gen_model.generate_content(user_message)
        print(f"Received {response}")
        output = response.text
        output = output.removeprefix("```json\n").removesuffix("\n```")

    else:
        client = OpenAI()  # remember to set the environment variable OPENAI_API_KEY
        completion = client.chat.completions.create(model=model, response_format={'type': "json_object"}, messages=[
            {'role': "system", 'content': system_message},
            {'role': "user", 'content': user_message}
        ])
        print(f"Received {completion=}")
        output = completion.choices[0].message.content

    # Store the response
    with open(os.path.join(output_dir, f"{file_name}.out"), 'w') as file:
        file.write(output)

    test_data = json.loads(output)
    test_data['api'] = api
    test_data['model'] = model
    with open(os.path.join(output_dir, "test_data.json"), 'w') as file:
        try:
            json.dump(test_data, file, indent=2)
        except JSONDecodeError:
            print("Unable to store output in JSON format", file=sys.stderr)

    print(f"{len(test_data['samples'])} samples were generated")


def generate_config_dataset_in_chunks(model: str, spec_file: str, output_dir: str) -> None:
    """
    Generate a synthetic configuration dataset for evaluation using an OpenAI GPT or Google Gemini model.
    This method asks the model only for a subset of paths at a time to allow processing very lage specifications.
    :param model: The name of the model to be used
    :param spec_file: Path to the OpenAPI specification
    :param output_dir: Path to the directory where the output should be stored
    :return:
    """
    with open(spec_file, 'r') as file:
        spec = yaml.safe_load(file)

    chunk = []
    chunk_size = 0
    chunk_id = 0
    for (path, path_items) in spec['paths'].items():
        chunk.append(path)
        chunk_size += len(path_items) - 1 if 'parameters' in path_items else len(path_items)
        if chunk_size > 30:  # the larger the threshold, the higher the risk of incomplete outputs
            print(f"Chunk {chunk_id}")
            generate_config_dataset(model, spec_file, os.path.join(output_dir, str(chunk_id)), paths=chunk)
            chunk = []
            chunk_id += 1
            chunk_size = 0

    if chunk:
        print(f"Chunk {chunk_id}")
        generate_config_dataset(model, spec_file, os.path.join(output_dir, str(chunk_id)), paths=chunk)

    all_test_data = {'samples': [], 'api': os.path.splitext(os.path.basename(spec_file))[0], 'model': model}
    for i in range(chunk_id + 1):
        with open(os.path.join(output_dir, str(i), "test_data.json"), 'r') as file:
            samples = json.load(file)['samples']
            all_test_data['samples'].extend(samples)

    with open(os.path.join(output_dir, "test_data.json"), 'w') as file:
        json.dump(all_test_data, file, indent=2)

    print(f"{len(all_test_data['samples'])} samples were collected")


def cross_check_configs(output_dir: str, spec_file: str) -> None:
    """
    Compare the generated configs to the API specification to check for inconsistencies.
    Results are stored next to the test data.
    :param output_dir: Path to the directory where the test data is stored
    :param spec_file: Path to the OpenAPI specification
    """
    test_data_file = os.path.join(output_dir, "test_data.json")
    if not os.path.isfile(test_data_file):
        print(f"{test_data_file} does not exist; skipping the checks", file=sys.stderr)
        return

    with open(test_data_file, 'r') as file:
        test_data = json.load(file)['samples']

    with open(spec_file, 'r') as file:
        spec_string = file.read()

    try:
        spec = parse(spec_string=spec_string, strict_enum=False)
    except ParserError as e:
        # Our parser cannot handle specs with recursive definitions
        if "Recursion reached limit" in e.args[0]:
            print(f"{spec_file} cannot be parsed; skipping the checks", file=sys.stderr)
            with open(os.path.join(output_dir, "cross_checking_results.json"), 'w') as file:
                json.dump({}, file)
            return
        # For some reason some specs cannot be parsed from a string, so we have to read the file a second time
        spec = parse(uri=spec_file, strict_enum=False)

    results = []
    for sample in test_data:
        config = sample['config']
        url = config['url']
        method = config['method']

        # Does the expected URL contain query parameters?
        if "?" in url:
            results.append(f"URL '{url}' seems to contain query parameters (which should be placed in `params`).\n")
            continue

        # Does the expected URL match any path?
        paths = find_path_in_spec(url, spec)
        if not paths:
            results.append(f"URL '{url}' could not be matched to an existing path in the specification.\n")
            continue
        path = paths[0]

        # Is the expected method defined for this path?
        operation = find_operation_in_path(method, path)
        if operation is None:
            results.append(f"Method '{method}' does not exist for path '{url}'.\n")
            continue

        protocol = ""

        # Are all arguments in config defined in the spec?
        for field_name in ['headers', 'params', 'data']:
            if field_name not in config:
                continue
            for (key, value) in config[field_name].items():
                arg_exists = validate_argument(key, field_name, method, path, spec.security_schemas)
                if not arg_exists:
                    protocol += (f"Argument '{key}' is not defined for location "
                                 f"'{field_name_to_location_str(field_name)}', method '{method}', and URL '{url}'.\n")

        # Are all required header/query arguments in the spec present in the config?
        for param in path.parameters + operation.parameters:
            if param.required and param.location is not ParameterLocation.PATH:
                field_name = location_to_field_name(param.location)
                if field_name not in config or param.name not in config[field_name]:
                    protocol += f"Required argument '{param.name}' is missing in location '{param.location.value}'.\n'"

        # Are all required body arguments in the spec present in the config?
        if operation.request_body:
            # noinspection PyUnresolvedReferences
            required_args = operation.request_body.content[0].schema.required  # many assumptions here on the structure
            for arg_name in required_args:
                if arg_name not in config['data']:
                    protocol += f"Required argument '{arg_name}' is missing in location 'request body'.\n"

        if protocol:
            results.append(protocol)
        else:
            results.append("OK")

    # Store the results
    with open(os.path.join(output_dir, "cross_checking_results.json"), 'w') as file:
        json.dump(results, file, indent=2)


def _generate_dataset() -> None:
    """Entry point for generating the dataset."""

    input("You are about to use an AI model that costs money. Press any key to continue.")
    print("OK, please wait ...")

    model = "gemini-1.5-pro"
    apis = ["asana", "gmail_v3", "google_calendar_v3", "google_sheet_v4", "hubspot_association", "hubspot_companies",
            "hubspot_contact", "hubspot_crm_objects", "hubspot_deal", "hubspot_exports", "hubspot_imports",
            "hubspot_lineitems", "hubspot_object_meetings", "hubspot_object_notes", "hubspot_objects_email",
            "hubspot_object_tasks", "hubspot_pipelines", "hubspot_products", "hubspot_properties", "hubspot_quotes",
            "hubspot_tickets", "salesforce_crm", "service_now", "slack"]
    for api in apis:
        spec_file = f"openapi/real_world_specs/{api}.yaml"
        output_dir = f"data/synthetic/{api}/"  # make sure not to overwrite existing data

        print(f"Generating dataset for {api} API ...")
        if api in ["asana", "gmail_v3", "slack"]:
            generate_config_dataset_in_chunks(model, spec_file, output_dir)
        else:
            generate_config_dataset(model, spec_file, output_dir)

        print(f"Checking the generated {api} dataset ...")
        cross_check_configs(output_dir, spec_file)


if __name__ == "__main__":
    os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
    sys.path.append(os.getcwd())

    _generate_dataset()

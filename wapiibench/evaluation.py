from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from enum import auto

import regex as re
from openapi_parser.enumeration import ParameterLocation
from openapi_parser.specification import Path, Specification
from strenum import StrEnum
from tqdm.auto import tqdm
from transformers import GenerationConfig

from logits_processor import OpenApiDecoder
from model_utils import ModelWrapper
from openapi_utils import find_path_in_spec, find_operation_in_path, validate_argument, AxiosSyntax, parse_spec
from rag_utils import Retriever

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS = [
    "bigcode/starcoder2-3b",
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-15b",
    "bigcode/starcoderbase",
    "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-ai/deepseek-coder-7b-base-v1.5",
    "deepseek-ai/deepseek-coder-33b-base",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "google/gemini-pro-1.5",
    "meta-llama/CodeLlama-7b-hf",
    "meta-llama/CodeLlama-13b-hf",
    "meta-llama/CodeLlama-70b-hf",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "Qwen/Qwen2.5-Coder-0.5B",
    "Qwen/Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-3B",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-14B",
    "Qwen/Qwen2.5-Coder-32B",
    "Salesforce/codet5p-16b",
    "Salesforce/instructcodet5p-16b",
]

APIS = {
    'asana': "Asana",
    'google_calendar_v3': "Google Calendar",
    'google_sheet_v4': "Google Sheets",
    'slack': "Slack Web",
}

SETUPS = ['import', 'invocation', 'endpoint']
SETTINGS = ['vanilla', 'rag', 'constrained', 'constrained-rag']

STARTER_CODES = {
    'import': """\
// {task}
const axios = require('axios');
""",

    'invocation': """\
// {task}
const axios = require('axios');

axios.\
""",

    'endpoint': """\
// {task}
const axios = require('axios');

axios.{method}('{url}',\
"""
}

FIELD_KEYS = ['headers', 'params', 'path_params', 'data']
SPECIAL_KEYS = ['Accept', 'Content-Type']


class Verdict(StrEnum):
    TIMEOUT = auto()
    UNSATISFIABLE_CONSTRAINTS = auto()
    ABSENT_REQUEST = auto()
    INCOMPLETE_REQUEST = auto()
    EXECUTION_ERROR = auto()
    NONEXISTENT_ENDPOINT = auto()
    WRONG_ENDPOINT = auto()
    ILLEGAL_KEY = auto()
    UNNECESSARY_KEY = auto()
    MISSING_KEY = auto()
    INCORRECT_VALUE = auto()
    CORRECT = auto()


def generate(model_name: str, api_name: str, spec_file: str, test_data_file: str, output_dir: str,
             starter_code_template: str, setting: str, num_outputs: int = 1, openai_batch: bool = False,
             **kwargs) -> None:
    """
    Run the previously generated prompts with the given decoding setting.
    :param model_name: The name of the model to be used
    :param api_name: The name of the API
    :param spec_file: Path to the OpenAPI specification
    :param test_data_file: Path to the JSON file that contains the task descriptions
    :param output_dir: Path to the directory where the output should be stored
    :param starter_code_template: Code given as part of the prompt to steer the generation in a certain direction
    :param setting: The decoding setting to use
    :param num_outputs: How many outputs to generate per task
    :param openai_batch: Whether to use OpenAI's Batch API to generate the code asynchronously
    :param kwargs: Additional arguments for the generation config
    """
    kwargs.setdefault('max_new_tokens', 250)
    kwargs.setdefault('do_sample', num_outputs > 1)
    kwargs.setdefault('num_beams', 1)
    if num_outputs > 1:
        kwargs.setdefault('temperature', 0.2)
    kwargs.setdefault('top_k', 50)
    kwargs.setdefault('top_p', 1.0)

    use_cd = 'constrained' in setting
    use_rag = 'rag' in setting

    with open("resources/code_generation_prompt.md", 'r') as file:
        prompt_template = file.read()
    syntax = "axios.method(url[, config])"  # the alternative syntax would be "axios.request(config)"
    prompt = prompt_template.format(
        syntax=syntax, api=api_name,
        extra_instructions="without any explanations" if "-instruct" in model_name.lower() else "")

    # Dump the prompt for later reference
    if num_outputs == 1:
        os.makedirs(output_dir, exist_ok=True)
    else:
        for i in range(num_outputs):
            os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
    with open(os.path.join(output_dir, "model.in"), 'w') as file:
        file.write(prompt + starter_code_template)

    # Load the task descriptions
    with open(test_data_file, 'r') as file:
        test_data = json.load(file)['samples']

    # Load the model
    model = ModelWrapper(model_name)
    generation_config = GenerationConfig(
        stop_strings="\n```\n", num_return_sequences=num_outputs,
        pad_token_id=model.tokenizer.eos_token_id if model.tokenizer is not None else None, **kwargs)

    if model.provider == 'OpenAI' and openai_batch:
        can_continue = model.batch_init(output_dir)
        if not can_continue:
            logger.warning("Batch results not yet available - exiting generation method")
            return

    logits_processor = OpenApiDecoder(generation_config, model.tokenizer, spec_file,
                                      AxiosSyntax.METHOD_AS_FUNCTION) if use_cd else None

    retriever = Retriever(spec_file) if use_rag else None

    # Run decoding in the given setting
    for index, task in enumerate(tqdm(test_data, leave=False)):

        # Construct the full prompt
        rag_context = retriever.retrieve_spec_for_task(
            task['task'], num_chunks=5, truncation_threshold=30000) + "\n\n\n" if use_rag else ""
        starter_code = starter_code_template.format(
            task=task['task'], method=task['config']['method'], url=task['config']['url'])
        input_text = rag_context + prompt + starter_code

        if num_outputs == 1 and os.path.isfile(os.path.join(output_dir, f"{index:04d}_code.js")):
            logger.warning(f"Sample {index} already exists - skipping")
            continue  # avoid recomputation (only for greedy decoding)

        # Dump the retrieved information for later reference
        if use_rag:
            with open(os.path.join(output_dir, f"{index:04d}_retrieved.yaml"), 'w') as file:
                file.write(rag_context)

        # Reset the ConstrainedDecoder's state
        if use_cd:
            logits_processor.reset(completion_prefix=starter_code)

        # Run generation
        output_texts = model.run(
            input_text, generation_config=generation_config, logits_processor=logits_processor, batch=openai_batch,
            batch_file_dir=output_dir, sample_id=index)

        if model.provider == 'OpenAI' and openai_batch and output_texts is None:
            continue  # results are not available yet

        # Postprocess and store each generated sample
        for i, output_text in enumerate(output_texts):

            if use_cd:
                if logits_processor.timed_out(i):
                    output_text += "//" + Verdict.TIMEOUT
                elif logits_processor.constraints_unsatisfiable(i):
                    output_text += "//" + Verdict.UNSATISFIABLE_CONSTRAINTS

            if model.is_chat_model:
                # Chat models create very different outputs that we need to convert to a unified format.
                # The output may include speaking role markers, natural language text, and/or parts of the starter code.
                if model.provider == 'OpenAI':
                    try:
                        generated_code = \
                            starter_code[:starter_code.rindex("\n") + 1] + output_text[output_text.index("axios."):]
                    except ValueError:
                        logger.error(f"Please manually check and correct sample {index} ({i})")
                        generated_code = starter_code + output_text.removeprefix("```javascript\n")
                else:
                    generated_code = output_text[output_text.index(starter_code):]
                    last_line = starter_code[starter_code.rindex("\n") + 1:]
                    first_index = generated_code.index(last_line)
                    last_index = generated_code.rindex(last_line)
                    if first_index != last_index:
                        # Cut out everything between the end and the continuation of the starter code
                        generated_code = generated_code[:first_index] + generated_code[last_index:]
                    else:
                        # Cut out the role marker and surrounding whitespace
                        generated_code = re.sub(r"\s*(?:#+ )?(?:Assistant|Response):\s*", "", generated_code, count=1)
            else:
                generated_code = output_text[output_text.rindex(starter_code):]

            if generated_code.endswith("```\n"):
                generated_code = generated_code[:-1]  # remove trailing blank line

            with open(os.path.join(output_dir, str(i) if num_outputs > 1 else "", f"{index:04d}_code.js"), 'w') as file:
                file.write(generated_code)

    if model.provider == 'OpenAI' and openai_batch:
        model.submit_batch(output_dir)


def execute(code_dir: str, node: str) -> None:
    """
    Execute the generated codes and store the resulting request configs in JSON files.
    :param code_dir: Path to the generated codes; configs are stored there as well
    :param node: The path to the node binary to execute the JS code in the shell
    """
    with open("wapiibench/mock.js", 'r') as file:
        mock_code_template = file.read()

    with (open(os.path.join(code_dir, "execution.out"), 'w') as out_file,
          open(os.path.join(code_dir, "execution.err"), 'w') as err_file):

        files = os.listdir(code_dir)
        files.sort()
        for file_name in files:
            file_path = os.path.join(code_dir, file_name)
            root, ext = os.path.splitext(file_name)
            if not os.path.isfile(file_path) or ext != ".js" or not root.endswith("_code"):
                continue

            index, _ = root.split(sep="_", maxsplit=2)

            separator_msg = f"\n### Now processing {file_name} ###\n"
            out_file.write(separator_msg)
            out_file.flush()
            err_file.write(separator_msg)
            err_file.flush()

            # Read the generated code and prepare it for execution
            config_log_file = os.path.join(code_dir, f"{index}_config.json")
            with open(file_path, 'r') as file:
                generated_code = file.read()

            axios_call, verdict, message = _extract_axios_call(generated_code)
            out_file.write(message)

            if axios_call is None:  # could the axios call be extracted successfully?
                with open(config_log_file, 'w') as file:
                    json.dump({'ERROR': verdict}, file, indent=2)
                continue

            mock_code = mock_code_template % config_log_file
            executable_code = f"{mock_code}\n{axios_call}"

            # Run the code and log the request config as a side effect
            proc = subprocess.run([node, "-"], stdout=out_file, stderr=err_file, input=executable_code, text=True)
            out_file.write(f"\nExit code: {proc.returncode}\n")

            # In case of an error, create an empty placeholder file
            if proc.returncode != 0:
                with open(config_log_file, 'w') as file:
                    json.dump({'ERROR': Verdict.EXECUTION_ERROR}, file, indent=2)


def _extract_axios_call(code: str) -> tuple[str | None, Verdict, str]:
    """
    Cuts out the first axios call, i.e., removes import, comment, and the irrelevant and potentially incomplete suffix.
    :param code: The code from which the axios call should be extracted
    :return: The extracted axios call or ``None``, a verdict, and a log message
    """
    if code.endswith(Verdict.TIMEOUT):
        return None, Verdict.TIMEOUT, f"Incomplete code due to timeout:\n{code}\n"

    if code.endswith(Verdict.UNSATISFIABLE_CONSTRAINTS):
        return None, Verdict.UNSATISFIABLE_CONSTRAINTS, f"Incomplete code due to unsatisfiable constraints:\n{code}\n"

    start_index = code.find("axios.")
    if start_index == -1:
        return None, Verdict.ABSENT_REQUEST, f"Code does not contain an axios call:\n{code}\n"

    # Prioritize cutoff points that are less likely to break the code
    offset = start_index + len("axios.")
    for cutoff_point in [".then", ".catch", ".finally", "});", "}\n);", ");", ";", "\naxios.", "```"]:
        cutoff_index = code.find(cutoff_point, offset)
        if cutoff_index != -1:
            break

    if cutoff_index == -1:
        return None, Verdict.INCOMPLETE_REQUEST, f"Unable to extract axios call from code:\n{code}\n"

    if cutoff_point in [".then", ".catch", ".finally"]:
        replacement = ";\n"
    elif cutoff_point in ["});", "}\n);", ");", ";"]:
        replacement = cutoff_point + "\n"
    else:
        replacement = ""

    axios_call = code[start_index:cutoff_index] + replacement
    return axios_call, Verdict.CORRECT, f"Axios call extracted successfully:\n{axios_call}"


def compare(test_data_file: str, spec_file: str, code_dir: str) -> None:
    """
    Compare the logged configurations against the expected ones and write all keys, values, and comparison results into
    a JSON file (separate for each setting).
    :param test_data_file: File that contains the expected configs
    :param spec_file: OpenAPI specification of the tested API
    :param code_dir: Directory that contains the logged configs
    """
    test_results = {}
    spec = parse_spec(spec_file)

    with open(test_data_file, 'r') as file:
        test_data = json.load(file)['samples']

    # Go through all logged configs (and ignore all other files)
    for file_name in os.listdir(code_dir):
        file_path = os.path.join(code_dir, file_name)
        root, ext = os.path.splitext(file_name)
        if not os.path.isfile(file_path) or ext != ".json" or not root.endswith("_config"):
            continue

        index, _ = root.split(sep="_", maxsplit=2)

        expected_config = test_data[int(index)]['config']

        with open(file_path, 'r') as file:
            logged_config = json.load(file)

        expected_paths = _add_path_params(expected_config, spec)
        logged_paths = _add_path_params(logged_config, spec)
        assert expected_paths  # the logged paths may be empty

        test_results[index] = _compare_configs(expected_config, logged_config, expected_paths[0], logged_paths, spec)

    test_result_file = os.path.join(code_dir, "results.json")
    with open(test_result_file, 'w') as file:
        json.dump(test_results, file, indent=2, sort_keys=True)


def _add_path_params(config: dict[str, any], spec: Specification) -> list[Path]:
    """
    Add an entry ``path_params`` to the given config which contains keys and values for all path parameters in the URL.
    If there are multiple candidate paths, the most likely one is used for this.
    :param config: The configuration that is modified in place
    :param spec: Specification of the tested API
    :return: Potentially empty list of Path objects that correspond to ``config``
    """
    if not config or 'ERROR' in config:
        return []

    paths = find_path_in_spec(config['url'], spec)
    if not paths:
        return paths
    path = paths[0]

    path_param_names = \
        [param.name for op in path.operations for param in op.parameters if param.location is ParameterLocation.PATH]
    pattern = path.url
    for name in path_param_names:
        pattern = pattern.replace(f"{{{name}}}", fr"(?P<{name}>[^/]+)")

    match = re.search(pattern, config['url'])
    path_params = match.groupdict()
    if path_params:
        config['path_params'] = path_params

    return paths


def _compare_configs(expected: dict[str, any], actual: dict[str, any], expected_path: Path, actual_paths: list[Path],
                     spec: Specification) -> dict[str, any]:
    """
    Perform a deep comparison between the expected and actual configuration objects.
    :param expected: The expected config
    :param actual: The actual config
    :param expected_path: Path object of the expected endpoint
    :param actual_paths: Path objects of the actual endpoint
    :param spec: Complete specification of the API
    :return: A dict structured like the given configs containing the two values and the comparison result for each key
    """
    assert actual, f"actual config shouldn't be empty at this point"
    if 'ERROR' in actual:
        # No data due to an execution error, but for the statistics, we still need to know the argument count
        expected_arguments = {field_key: len(expected[field_key]) for field_key in FIELD_KEYS if field_key in expected}
        if 'headers' in expected:
            for special_key in SPECIAL_KEYS:
                if special_key in expected['headers']:
                    expected_arguments['headers'] -= 1
        return {'ERROR': actual['ERROR'], 'expected_arguments': expected_arguments}

    results = {}

    # Are all expected parameters actually present?
    for (field_key, field_value) in expected.items():

        if field_key == 'url':
            if expected_path in actual_paths:
                verdict = Verdict.CORRECT
            elif not actual_paths:
                verdict = Verdict.NONEXISTENT_ENDPOINT
            else:
                verdict = Verdict.WRONG_ENDPOINT
            results['url'] = {'expected': field_value, 'actual': actual['url'], 'verdict': verdict}

        elif field_key == 'method':
            if field_value == actual['method']:
                verdict = Verdict.CORRECT
            elif (not actual_paths or
                  all(find_operation_in_path(actual['method'], path) is None for path in actual_paths)):
                verdict = Verdict.NONEXISTENT_ENDPOINT
            else:
                verdict = Verdict.WRONG_ENDPOINT
            results['method'] = {'expected': field_value, 'actual': actual['method'], 'verdict': verdict}

        else:
            assert field_key in FIELD_KEYS
            field_results = {}

            for (expected_key, expected_value) in field_value.items():
                field_results[expected_key] = {'expected': expected_value}

                if field_key in actual and actual[field_key] is not None and expected_key in actual[field_key]:
                    actual_value = actual[field_key][expected_key]
                    field_results[expected_key]['actual'] = actual_value
                    field_results[expected_key]['verdict'] = \
                        Verdict.CORRECT if actual_value == expected_value else Verdict.INCORRECT_VALUE

                else:
                    field_results[expected_key]['actual'] = None
                    field_results[expected_key]['verdict'] = Verdict.MISSING_KEY

            results[field_key] = field_results

    # Are there any parameters present that are not expected?
    for (field_key, field_value) in actual.items():
        if field_value is None:
            # Sometimes the model generates a field and sets its value to null
            continue

        if field_key not in FIELD_KEYS:
            # The logged configs do actually contain several fields that are not included in the expected configs,
            # but they are all irrelevant. We are only interested in real request arguments.
            continue

        if field_key not in expected:
            assert field_key not in results
            expected[field_key] = {}
            results[field_key] = {}

        for (actual_key, actual_value) in field_value.items():
            if actual_key not in expected[field_key]:
                arg_exists = validate_argument(
                    actual_key, field_key, expected['method'], expected_path, spec.security_schemas)
                results[field_key][actual_key] = {
                    'expected': None, 'actual': actual_value,
                    'verdict': Verdict.UNNECESSARY_KEY if arg_exists else Verdict.ILLEGAL_KEY}

    return results


def evaluate(output_dir: str, keep_comparison: bool = True) -> None:
    """
    Calculate some statistics about the test results and add them to the file.
    :param output_dir: Directory where the results are stored
    :param keep_comparison: Whether to store also the comparison results or only the statistics
    """
    # sample counts
    samples_total = 0
    samples_executable = 0
    samples_correct = 0
    samples_wrong = 0
    samples_illegal = 0
    samples_nonexecutable = 0

    # error counts
    errors_total = 0
    errors_timeout = 0
    errors_unsatisfiable = 0
    errors_no_request = 0
    errors_incomplete_request = 0
    errors_runtime_error = 0

    # URL counts
    urls_correct = 0
    urls_wrong = 0
    urls_illegal = 0

    # method counts
    methods_correct = 0
    methods_wrong = 0
    methods_illegal = 0

    # endpoint counts
    endpoints_correct = 0
    endpoints_wrong = 0
    endpoints_illegal = 0

    # argument counts
    arguments_all_total = Counter()
    arguments_all_executable = Counter()
    arguments_expected_total = Counter()
    arguments_expected_executable = Counter()
    arguments_unexpected_executable = Counter()
    arguments_correct_name_executable = Counter()
    arguments_correct_value_executable = Counter()
    arguments_missing_total = Counter()
    arguments_missing_executable = Counter()
    arguments_unnecessary_executable = Counter()
    arguments_illegal_executable = Counter()

    # argument metric aggregates
    argument_name_precision_aggregate = Counter()
    argument_name_recall_aggregate = Counter()
    argument_name_jaccard_aggregate = Counter()
    argument_value_accuracy_wrt_all_aggregate = Counter()
    argument_value_accuracy_wrt_expected_aggregate = Counter()
    argument_value_accuracy_wrt_correct_name_aggregate = Counter()

    # argument location counts
    arguments_all_locations_total = Counter()
    arguments_all_locations_executable = Counter()
    arguments_expected_locations_total = Counter()
    arguments_expected_locations_executable = Counter()
    arguments_retrieved_locations_executable = Counter()
    arguments_correct_name_locations_executable = Counter()

    test_results_file = os.path.join(output_dir, "results.json")
    with open(test_results_file, 'r') as file:
        test_results = json.load(file)
    test_results.pop('statistics', None)  # in case we re-evaluate a file, temporarily remove the statistics field

    for (index, sample) in test_results.items():
        stats = _evaluate_sample(sample)
        test_results[index]['statistics'] = stats

        samples_total += 1
        if stats['sample_verdict'] == 'correct':
            samples_correct += 1
        elif stats['sample_verdict'] == 'wrong':
            samples_wrong += 1
        elif stats['sample_verdict'] == 'illegal':
            samples_illegal += 1
        elif stats['sample_verdict'] == 'nonexecutable':
            samples_nonexecutable += 1
        else:
            raise AssertionError(f"Unexpected verdict {stats['sample_verdict']}")

        if stats['error_verdict'] is not None:
            errors_total += 1
            if stats['error_verdict'] == 'timeout':
                errors_timeout += 1
            elif stats['error_verdict'] == 'unsatisfiable':
                errors_unsatisfiable += 1
            elif stats['error_verdict'] == 'no_request':
                errors_no_request += 1
            elif stats['error_verdict'] == 'incomplete_request':
                errors_incomplete_request += 1
            elif stats['error_verdict'] == 'runtime_error':
                errors_runtime_error += 1
            else:
                raise AssertionError(f"Unexpected verdict {stats['error_verdict']}")

        else:
            samples_executable += 1

            if stats['url_verdict'] == 'correct':
                urls_correct += 1
            elif stats['url_verdict'] == 'wrong':
                urls_wrong += 1
            elif stats['url_verdict'] == 'illegal':
                urls_illegal += 1
            else:
                raise AssertionError(f"Unexpected verdict {stats['url_verdict']}")

            if stats['method_verdict'] == 'correct':
                methods_correct += 1
            elif stats['method_verdict'] == 'wrong':
                methods_wrong += 1
            elif stats['method_verdict'] == 'illegal':
                methods_illegal += 1
            else:
                raise AssertionError(f"Unexpected verdict {stats['method_verdict']}")

            if stats['endpoint_verdict'] == 'correct':
                endpoints_correct += 1
            elif stats['endpoint_verdict'] == 'wrong':
                endpoints_wrong += 1
            elif stats['endpoint_verdict'] == 'illegal':
                endpoints_illegal += 1
            else:
                raise AssertionError(f"Unexpected verdict {stats['endpoint_verdict']}")

            arguments_all_executable += stats['arguments_all']
            arguments_expected_executable += stats['arguments_expected']
            arguments_unexpected_executable += stats['arguments_unexpected']
            arguments_correct_name_executable += stats['arguments_correct_name']
            arguments_correct_value_executable += stats['arguments_correct_value']
            arguments_missing_executable += stats['arguments_missing']
            arguments_unnecessary_executable += stats['arguments_unnecessary']
            arguments_illegal_executable += stats['arguments_illegal']

            arguments_all_locations_executable += Counter(stats['arguments_all'].keys())
            arguments_expected_locations_executable += Counter(stats['arguments_expected'].keys())
            arguments_correct_name_locations_executable += Counter(stats['arguments_correct_name'].keys())
            arguments_retrieved_locations_executable += Counter(
                stats['arguments_unexpected'].keys() | stats['arguments_correct_name'].keys())

        arguments_all_total += stats['arguments_all']
        arguments_expected_total += stats['arguments_expected']
        arguments_missing_total += stats['arguments_missing']

        argument_name_precision_aggregate += stats['argument_name_precision']
        argument_name_recall_aggregate += stats['argument_name_recall']
        argument_name_jaccard_aggregate += stats['argument_name_jaccard']
        argument_value_accuracy_wrt_all_aggregate += stats['argument_value_accuracy_wrt_all']
        argument_value_accuracy_wrt_expected_aggregate += stats['argument_value_accuracy_wrt_expected']
        argument_value_accuracy_wrt_correct_name_aggregate += stats['argument_value_accuracy_wrt_correct_name']

        arguments_all_locations_total += Counter(stats['arguments_all'].keys())
        arguments_expected_locations_total += Counter(stats['arguments_expected'].keys())

    assert samples_total == samples_executable + samples_nonexecutable == len(test_results)
    assert errors_total == errors_timeout + errors_unsatisfiable + errors_no_request + errors_incomplete_request + errors_runtime_error
    assert arguments_all_total == arguments_expected_total + arguments_unexpected_executable
    assert arguments_all_executable == arguments_expected_executable + arguments_unexpected_executable
    assert arguments_expected_total == arguments_correct_name_executable + arguments_missing_total
    assert arguments_expected_executable == arguments_correct_name_executable + arguments_missing_executable
    assert arguments_unexpected_executable == arguments_unnecessary_executable + arguments_illegal_executable

    arguments_retrieved_total = arguments_correct_name_executable + arguments_unexpected_executable

    samples_executable_or_inf = samples_executable if samples_executable > 0 else float('inf')

    # Calculate statistics and add them to the test results
    test_results['statistics'] = {
        # sample counts
        'samples_total': samples_total,
        'samples_executable': samples_executable,
        'samples_correct': samples_correct,
        'samples_wrong': samples_wrong,
        'samples_illegal': samples_illegal,
        'samples_nonexecutable': samples_nonexecutable,  # redundant with errors_total
        # sample ratios
        'samples_executable_wrt_total': samples_executable / samples_total,
        'samples_correct_wrt_total': samples_correct / samples_total,
        'samples_correct_wrt_executable': samples_correct / samples_executable_or_inf,
        'samples_wrong_wrt_total': samples_wrong / samples_total,
        'samples_wrong_wrt_executable': samples_wrong / samples_executable_or_inf,
        'samples_illegal_wrt_total': samples_illegal / samples_total,
        'samples_illegal_wrt_executable': samples_illegal / samples_executable_or_inf,

        # error counts
        'errors_total': errors_total,
        'errors_timeout': errors_timeout,
        'errors_unsatisfiable': errors_unsatisfiable,
        'errors_no_request': errors_no_request,
        'errors_incomplete_request': errors_incomplete_request,
        'errors_runtime_error': errors_runtime_error,
        # error ratios
        'errors_total_wrt_samples': errors_total / samples_total,
        'errors_timeout_wrt_samples': errors_timeout / samples_total,
        'errors_timeout_wrt_errors': 0 if errors_total == 0 else errors_timeout / errors_total,
        'errors_unsatisfiable_wrt_samples': errors_unsatisfiable / samples_total,
        'errors_unsatisfiable_wrt_errors': 0 if errors_total == 0 else errors_unsatisfiable / errors_total,
        'errors_no_request_wrt_samples': errors_no_request / samples_total,
        'errors_no_request_wrt_errors': 0 if errors_total == 0 else errors_no_request / errors_total,
        'errors_incomplete_request_wrt_samples': errors_incomplete_request / samples_total,
        'errors_incomplete_request_wrt_errors': 0 if errors_total == 0 else errors_incomplete_request / errors_total,
        'errors_runtime_error_wrt_samples': errors_runtime_error / samples_total,
        'errors_runtime_error_wrt_errors': 0 if errors_total == 0 else errors_runtime_error / errors_total,

        # URL counts
        'urls_correct': urls_correct,
        'urls_wrong': urls_wrong,
        'urls_illegal': urls_illegal,
        # URL ratios
        'urls_correct_wrt_total': urls_correct / samples_total,
        'urls_correct_wrt_executable': urls_correct / samples_executable_or_inf,
        'urls_wrong_wrt_total': urls_wrong / samples_total,
        'urls_wrong_wrt_executable': urls_wrong / samples_executable_or_inf,
        'urls_illegal_wrt_total': urls_illegal / samples_total,
        'urls_illegal_wrt_executable': urls_illegal / samples_executable_or_inf,

        # method counts
        'methods_correct': methods_correct,
        'methods_wrong': methods_wrong,
        'methods_illegal': methods_illegal,
        # method ratios
        'methods_correct_wrt_total': methods_correct / samples_total,
        'methods_correct_wrt_executable': methods_correct / samples_executable_or_inf,
        'methods_wrong_wrt_total': methods_wrong / samples_total,
        'methods_wrong_wrt_executable': methods_wrong / samples_executable_or_inf,
        'methods_illegal_wrt_total': methods_illegal / samples_total,
        'methods_illegal_wrt_executable': methods_illegal / samples_executable_or_inf,

        # endpoint counts
        'endpoints_correct': endpoints_correct,
        'endpoints_wrong': endpoints_wrong,
        'endpoints_illegal': endpoints_illegal,
        # endpoint ratios
        'endpoints_correct_wrt_total': endpoints_correct / samples_total,
        'endpoints_correct_wrt_executable': endpoints_correct / samples_executable_or_inf,
        'endpoints_wrong_wrt_total': endpoints_wrong / samples_total,
        'endpoints_wrong_wrt_executable': endpoints_wrong / samples_executable_or_inf,
        'endpoints_illegal_wrt_total': endpoints_illegal / samples_total,
        'endpoints_illegal_wrt_executable': endpoints_illegal / samples_executable_or_inf,

        # argument counts
        'arguments_all_total': arguments_all_total,
        'arguments_all_executable': arguments_all_executable,
        'arguments_expected_total': arguments_expected_total,
        'arguments_expected_executable': arguments_expected_executable,
        'arguments_unexpected': arguments_unexpected_executable,
        'arguments_correct_name': arguments_correct_name_executable,
        'arguments_correct_value': arguments_correct_value_executable,
        'arguments_missing_total': arguments_missing_total,
        'arguments_missing_executable': arguments_missing_executable,
        'arguments_unnecessary': arguments_unnecessary_executable,
        'arguments_illegal': arguments_illegal_executable,
        # argument ratios
        'arguments_unexpected_wrt_all_total':
            _divide_counters(arguments_unexpected_executable, arguments_all_total),
        'arguments_unexpected_wrt_all_executable':
            _divide_counters(arguments_unexpected_executable, arguments_all_executable),
        'arguments_correct_name_wrt_expected_total':
            _divide_counters(arguments_correct_name_executable, arguments_expected_total),
        'arguments_correct_name_wrt_expected_executable':
            _divide_counters(arguments_correct_name_executable, arguments_expected_executable),
        'arguments_correct_value_wrt_expected_total':
            _divide_counters(arguments_correct_value_executable, arguments_expected_total),
        'arguments_correct_value_wrt_expected_executable':
            _divide_counters(arguments_correct_value_executable, arguments_expected_executable),
        'arguments_unnecessary_wrt_all_total':
            _divide_counters(arguments_unnecessary_executable, arguments_all_total),
        'arguments_unnecessary_wrt_all_executable':
            _divide_counters(arguments_unnecessary_executable, arguments_all_executable),
        'arguments_missing_wrt_expected_total':
            _divide_counters(arguments_missing_total, arguments_expected_total),
        'arguments_missing_wrt_expected_executable':  # redundant with arguments_missing_wrt_expected_total (?)
            _divide_counters(arguments_missing_executable, arguments_expected_executable),
        'arguments_illegal_wrt_all_total':
            _divide_counters(arguments_illegal_executable, arguments_all_total),
        'arguments_illegal_wrt_all_executable':
            _divide_counters(arguments_illegal_executable, arguments_all_executable),

        # argument metrics
        'argument_name_sum_precision_wrt_total':
            _divide_counters(arguments_correct_name_executable, arguments_retrieved_total),
        'argument_name_sum_precision_wrt_executable':  # redundant with argument_name_sum_precision_wrt_total
            _divide_counters(arguments_correct_name_executable, arguments_retrieved_total),
        'argument_name_sum_recall_wrt_total':
            _divide_counters(arguments_correct_name_executable, arguments_expected_total),
        'argument_name_sum_recall_wrt_executable':
            _divide_counters(arguments_correct_name_executable, arguments_expected_executable),
        'argument_name_sum_jaccard_wrt_total':
            _divide_counters(arguments_correct_name_executable, arguments_all_total),
        'argument_name_sum_jaccard_wrt_executable':
            _divide_counters(arguments_correct_name_executable, arguments_all_executable),
        'argument_value_sum_accuracy_wrt_all_total':
            _divide_counters(arguments_correct_value_executable, arguments_all_total),
        'argument_value_sum_accuracy_wrt_all_executable':
            _divide_counters(arguments_correct_value_executable, arguments_all_executable),
        'argument_value_sum_accuracy_wrt_expected_total':
            _divide_counters(arguments_correct_value_executable, arguments_expected_total),
        'argument_value_sum_accuracy_wrt_expected_executable':
            _divide_counters(arguments_correct_value_executable, arguments_expected_executable),
        'argument_value_sum_accuracy_wrt_correct_total':
            _divide_counters(arguments_correct_value_executable, arguments_correct_name_executable),
        'argument_value_sum_accuracy_wrt_correct_executable':  # redundant with argument_value_sum_accuracy_wrt_correct_total
            _divide_counters(arguments_correct_value_executable, arguments_correct_name_executable),
        'argument_name_mean_precision_wrt_total':
            _divide_counters(argument_name_precision_aggregate, arguments_retrieved_locations_executable),
        'argument_name_mean_precision_wrt_executable':  # redundant with argument_name_mean_precision_wrt_total
            _divide_counters(argument_name_precision_aggregate, arguments_retrieved_locations_executable),
        'argument_name_mean_recall_wrt_total':
            _divide_counters(argument_name_recall_aggregate, arguments_expected_locations_total),
        'argument_name_mean_recall_wrt_executable':
            _divide_counters(argument_name_recall_aggregate, arguments_expected_locations_executable),
        'argument_name_mean_jaccard_wrt_total':
            _divide_counters(argument_name_jaccard_aggregate, arguments_all_locations_total),
        'argument_name_mean_jaccard_wrt_executable':
            _divide_counters(argument_name_jaccard_aggregate, arguments_all_locations_executable),
        'argument_value_mean_accuracy_wrt_all_total':
            _divide_counters(argument_value_accuracy_wrt_all_aggregate, arguments_all_locations_total),
        'argument_value_mean_accuracy_wrt_all_executable':
            _divide_counters(argument_value_accuracy_wrt_all_aggregate, arguments_all_locations_executable),
        'argument_value_mean_accuracy_wrt_expected_total':
            _divide_counters(argument_value_accuracy_wrt_expected_aggregate, arguments_expected_locations_total),
        'argument_value_mean_accuracy_wrt_expected_executable':
            _divide_counters(argument_value_accuracy_wrt_expected_aggregate, arguments_expected_locations_executable),
        'argument_value_mean_accuracy_wrt_correct_total':
            _divide_counters(
                argument_value_accuracy_wrt_correct_name_aggregate, arguments_correct_name_locations_executable),
        'argument_value_mean_accuracy_wrt_correct_executable':  # redundant with argument_value_mean_accuracy_wrt_correct_total
            _divide_counters(
                argument_value_accuracy_wrt_correct_name_aggregate, arguments_correct_name_locations_executable),
    }

    # Write the results back into the file
    with open(test_results_file, 'w') as file:
        json.dump(test_results if keep_comparison else {'statistics': test_results['statistics']}, file,
                  allow_nan=False, indent=2, sort_keys=True)


def _evaluate_sample(sample: dict[str, any]) -> dict[str, any]:
    """
    Evaluate a single sample and return the resulting statistics about it.
    :param sample: The sample to evaluate
    :return: A dict of statistical values
    """
    if 'ERROR' in sample:
        arguments_expected = sample['expected_arguments'].copy()
        if arguments_expected:
            arguments_expected['all'] = sum(arguments_expected.values())

        if sample['ERROR'] == Verdict.TIMEOUT:
            error_verdict = 'timeout'
        elif sample['ERROR'] == Verdict.UNSATISFIABLE_CONSTRAINTS:
            error_verdict = 'unsatisfiable'
        elif sample['ERROR'] == Verdict.ABSENT_REQUEST:
            error_verdict = 'no_request'
        elif sample['ERROR'] == Verdict.INCOMPLETE_REQUEST:
            error_verdict = 'incomplete_request'
        elif sample['ERROR'] == Verdict.EXECUTION_ERROR:
            error_verdict = 'runtime_error'
        else:
            raise AssertionError(f"Unexpected verdict {sample['ERROR']}")

        return {
            # sample verdicts
            'sample_verdict': 'nonexecutable',
            'error_verdict': error_verdict,
            'url_verdict': None,
            'method_verdict': None,
            'endpoint_verdict': None,
            # argument counts
            'arguments_all': Counter(arguments_expected),
            'arguments_expected': Counter(arguments_expected),
            'arguments_unexpected': Counter(),
            'arguments_correct_name': Counter(),
            'arguments_correct_value': Counter(),
            'arguments_missing': Counter(arguments_expected),
            'arguments_unnecessary': Counter(),
            'arguments_illegal': Counter(),
            # argument metrics
            'argument_name_precision': Counter(),  # mathematically it would be NaN, but 0 makes more sense here
            'argument_name_recall': Counter(),
            'argument_name_jaccard': Counter(),
            'argument_value_accuracy_wrt_all': Counter(),
            'argument_value_accuracy_wrt_expected': Counter(),
            'argument_value_accuracy_wrt_correct_name': Counter(),  # again, mathematically it would be NaN
        }

    arguments_all = Counter()
    arguments_expected = Counter()
    arguments_unexpected = Counter()
    arguments_correct_name = Counter()
    arguments_correct_value = Counter()
    arguments_missing = Counter()
    arguments_unnecessary = Counter()
    arguments_illegal = Counter()

    if sample['url']['verdict'] == Verdict.CORRECT:
        url_verdict = 'correct'
    elif sample['url']['verdict'] == Verdict.WRONG_ENDPOINT:
        url_verdict = 'wrong'
    elif sample['url']['verdict'] == Verdict.NONEXISTENT_ENDPOINT:
        url_verdict = 'illegal'
    else:
        raise AssertionError(f"Unexpected verdict {sample['method']['verdict']}")

    if sample['method']['verdict'] == Verdict.CORRECT:
        method_verdict = 'correct'
    elif sample['method']['verdict'] == Verdict.WRONG_ENDPOINT:
        method_verdict = 'wrong'
    elif sample['method']['verdict'] == Verdict.NONEXISTENT_ENDPOINT:
        method_verdict = 'illegal'
    else:
        raise AssertionError(f"Unexpected verdict {sample['method']['verdict']}")

    if url_verdict == 'correct' and method_verdict == 'correct':
        endpoint_verdict = 'correct'
    elif url_verdict == 'illegal' or method_verdict == 'illegal':
        endpoint_verdict = 'illegal'
    else:
        endpoint_verdict = 'wrong'

    for (field_key, field_value) in sample.items():
        if field_key in FIELD_KEYS:

            for (key, value) in field_value.items():
                if field_key == 'headers' and key in SPECIAL_KEYS:
                    continue  # ignore these header args as they are always present and correct

                arguments_all[field_key] += 1
                if value['verdict'] == Verdict.CORRECT:
                    arguments_expected[field_key] += 1
                    arguments_correct_name[field_key] += 1
                    arguments_correct_value[field_key] += 1
                elif value['verdict'] == Verdict.INCORRECT_VALUE:
                    arguments_expected[field_key] += 1
                    arguments_correct_name[field_key] += 1
                elif value['verdict'] == Verdict.MISSING_KEY:
                    arguments_expected[field_key] += 1
                    arguments_missing[field_key] += 1
                elif value['verdict'] == Verdict.UNNECESSARY_KEY:
                    arguments_unexpected[field_key] += 1
                    arguments_unnecessary[field_key] += 1
                elif value['verdict'] == Verdict.ILLEGAL_KEY:
                    arguments_unexpected[field_key] += 1
                    arguments_illegal[field_key] += 1
                else:
                    raise AssertionError(f"Unexpected verdict {value['verdict']}")

    assert arguments_all == arguments_expected + arguments_unexpected
    assert arguments_expected == arguments_correct_name + arguments_missing
    assert arguments_unexpected == arguments_unnecessary + arguments_illegal

    for counter in [arguments_all, arguments_expected, arguments_unexpected, arguments_correct_name,
                    arguments_correct_value, arguments_missing, arguments_unnecessary, arguments_illegal]:
        if counter:  # skip empty counters
            counter['all'] = counter.total()

    arguments_retrieved = arguments_correct_name + arguments_unexpected

    if endpoint_verdict == 'correct' and arguments_correct_value.get('all', 0) == arguments_all.get('all', 0):
        sample_verdict = 'correct'
    elif endpoint_verdict == 'illegal' or arguments_illegal.get('all', 0) > 0:
        sample_verdict = 'illegal'
    else:
        sample_verdict = 'wrong'

    return {
        # sample verdicts
        'sample_verdict': sample_verdict,
        'error_verdict': None,
        'url_verdict': url_verdict,
        'method_verdict': method_verdict,
        'endpoint_verdict': endpoint_verdict,
        # argument counts
        'arguments_all': arguments_all,
        'arguments_expected': arguments_expected,
        'arguments_unexpected': arguments_unexpected,
        'arguments_correct_name': arguments_correct_name,
        'arguments_correct_value': arguments_correct_value,
        'arguments_missing': arguments_missing,
        'arguments_unnecessary': arguments_unnecessary,
        'arguments_illegal': arguments_illegal,
        # argument metrics
        'argument_name_precision': _divide_counters(arguments_correct_name, arguments_retrieved),
        'argument_name_recall': _divide_counters(arguments_correct_name, arguments_expected),
        'argument_name_jaccard': _divide_counters(arguments_correct_name, arguments_all),
        'argument_value_accuracy_wrt_all': _divide_counters(arguments_correct_value, arguments_all),
        'argument_value_accuracy_wrt_expected': _divide_counters(arguments_correct_value, arguments_expected),
        'argument_value_accuracy_wrt_correct_name': _divide_counters(arguments_correct_value, arguments_correct_name),
    }


def _divide_counters(numerator: dict[any, float], denominator: dict[any, float]) -> dict[any, float]:
    """
    Element-wise division of two Counters or dicts. Missing elements are treated as having zero counts.
    :param numerator: Counter in the numerator
    :param denominator: Counter in the denominator
    :return: The result of the division as a dict
    """
    try:
        return {key: (numerator[key] / denominator[key]) for key in numerator.keys() | denominator.keys()}
    except ZeroDivisionError:
        logger.error(f"Tried to divide {numerator} by {denominator}")
        raise


def evaluate_all(data_root: str, apis: list[str], setup: str, setting: str) -> None:
    """
    Merge all test results into one file (stored in the folder "all") and evaluate them collectively.
    :param data_root: The place where the folders with the results for each API are located
    :param apis: All APIs to merge
    :param setup: The setup, i.e., starter code variation
    :param setting: The decoding setting
    """
    all_test_results = {}
    for api in apis:
        if not os.path.isdir(os.path.join(data_root, api, setup, setting)):
            logger.warning(f"Results for API '{api}' not available - skipping")
            continue
        with open(os.path.join(data_root, api, setup, setting, "results.json"), 'r') as file:
            api_test_results = json.load(file)
            api_test_results.pop('statistics', None)
            all_test_results |= {f"{api}_{index}": value for index, value in api_test_results.items()}

    output_dir = os.path.join(data_root, "all", setup, setting)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), 'w') as file:
        json.dump(all_test_results, file, indent=2)

    evaluate(output_dir, keep_comparison=False)


def _run_evaluation() -> None:
    """Entry point for running the evaluation. Controlled via command line arguments."""

    default_node = os.path.join(
        os.environ.get("NVM_SYMLINK", os.path.expanduser("~/.nvm/versions/node/v22.20.0/bin")), "node")

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=MODELS, nargs='+', help="the models to evaluate")
    parser.add_argument("--skip-models", default=[], nargs='+', help="the models to skip")
    parser.add_argument("--apis", default=APIS.keys(), nargs='+', help="the APIs to evaluate")
    parser.add_argument("--skip-apis", default=[], nargs='+', help="the APIs to skip")
    parser.add_argument("--setups", default=SETUPS, nargs='+', help="the setups to evaluate")
    parser.add_argument("--skip-setups", default=[], nargs='+', help="the setups to skip")
    parser.add_argument("--settings", default=SETTINGS, nargs='+', help="the settings to evaluate")
    parser.add_argument("--skip-settings", default=[], nargs='+', help="the settings to skip")
    parser.add_argument("--node", default=default_node, type=str, help="the node version to use")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", action="store_true", help="only generate the code, then stop")
    group.add_argument("--evaluate-only", action="store_true", help="skip code generation, just execute and evaluate")
    args = parser.parse_args()

    models = [model for model in args.models if model not in args.skip_models]
    apis = [api for api in args.apis if api not in args.skip_apis]
    setups = [setup for setup in args.setups if setup not in args.skip_setups]
    settings = [setting for setting in args.settings if setting not in args.skip_settings]
    node = args.node
    generate_only = args.generate_only
    evaluate_only = args.evaluate_only

    assert set(models).issubset(MODELS), f"Unsupported model(s): {set(models).difference(MODELS)}"
    assert set(apis).issubset(APIS), f"Unsupported API(s): {set(apis).difference(APIS)}"
    assert set(setups).issubset(SETUPS), f"Unsupported setup(s): {set(setups).difference(SETUPS)}"
    assert set(settings).issubset(SETTINGS), f"Unsupported setting(s): {set(settings).difference(SETTINGS)}"

    for model in models:

        data_root = f"data/generated/{model.split('/', 1)[1]}/"

        for api in apis:

            api_name = APIS.get(api, None)
            if api_name is None:
                logger.error(f"Unknown API '{api}'")
                continue

            spec_file = f"openapi/real_world_specs/{api}.yaml"
            test_data_file = f"data/synthetic/{api}/test_data_corrected.json"
            if not os.path.isfile(test_data_file):
                logger.warning(f"{api}'s test_data_corrected.json does not exists, falling back to test_data.json")
                test_data_file = f"data/synthetic/{api}/test_data.json"

            for setup in setups:

                starter_code = STARTER_CODES.get(setup, None)
                if starter_code is None:
                    logger.error(f"Unknown setup '{setup}'")
                    continue

                for setting in settings:

                    if setting not in SETTINGS:
                        logger.error(f"Unknown setting '{setting}'")
                        continue

                    logger.info(f"Running evaluation pipeline with {model = }, {api = }, {setup = }, {setting = } ...")
                    output_dir = os.path.join(data_root, api, setup, setting)

                    if not evaluate_only:
                        generate(model, api_name, spec_file, test_data_file, output_dir, starter_code, setting,
                                 openai_batch=False)

                    if not generate_only:
                        execute(output_dir, node=node)
                        compare(test_data_file, spec_file, output_dir)
                        evaluate(output_dir)

        if not generate_only:
            for setup in setups:
                for setting in settings:
                    logger.info(
                        f"Running evaluation pipeline with {model = }, api = 'all', {setup = }, {setting = } ...")
                    evaluate_all(data_root, apis, setup, setting)


if __name__ == "__main__":
    os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
    sys.path.append(os.getcwd())

    _run_evaluation()

from types import SimpleNamespace


def full_preset():
    args = SimpleNamespace()
    args.outputs = [
        "png",
        "pdf",
        "tex",
    ]
    args.models = [
        "codet5p-16b",
        "instructcodet5p-16b",
        "starcoderbase",
        "starcoder2-3b",
        "starcoder2-7b",
        "starcoder2-15b",
        "deepseek-coder-1.3b-base",
        "deepseek-coder-6.7b-base",
        "deepseek-coder-7b-base-v1.5",
        "deepseek-coder-33b-base",
        "DeepSeek-Coder-V2-Lite-Base",
        "Qwen2.5-Coder-0.5B",
        "Qwen2.5-Coder-1.5B",
        "Qwen2.5-Coder-3B",
        "Qwen2.5-Coder-7B",
        "Qwen2.5-Coder-14B",
        "Qwen2.5-Coder-32B",
        "Llama-3.1-8B",
        "Llama-3.1-70B",
        "CodeLlama-7b-hf",
        "CodeLlama-13b-hf",
        "CodeLlama-70b-hf",
        "gpt-4o-mini",
        "gpt-4o",
        "gemini-pro-1.5",
    ]
    args.apis = [
        "all",
        "asana",
        "google_calendar_v3",
        "google_sheet_v4",
        "slack",
    ]
    args.setups = [
        # "import",
        "invocation",
        "endpoint",
    ]
    args.settings = [
        "vanilla",
        "rag",
        "constrained",
        "constrained-rag",
    ]
    args.metrics = {
        "argument_name_mean_jaccard_wrt_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_mean_jaccard_wrt_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_mean_precision_wrt_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_mean_precision_wrt_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_mean_recall_wrt_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_mean_recall_wrt_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_sum_jaccard_wrt_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_sum_jaccard_wrt_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_sum_precision_wrt_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_sum_precision_wrt_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_sum_recall_wrt_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_name_sum_recall_wrt_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_mean_accuracy_wrt_all_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_mean_accuracy_wrt_all_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_mean_accuracy_wrt_correct_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_mean_accuracy_wrt_correct_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_mean_accuracy_wrt_expected_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_mean_accuracy_wrt_expected_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_sum_accuracy_wrt_all_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_sum_accuracy_wrt_all_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_sum_accuracy_wrt_correct_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_sum_accuracy_wrt_correct_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_sum_accuracy_wrt_expected_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "argument_value_sum_accuracy_wrt_expected_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_all_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_all_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_correct_name": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_correct_name_wrt_expected_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_correct_name_wrt_expected_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_correct_value": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_correct_value_wrt_expected_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_correct_value_wrt_expected_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_expected_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_expected_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_illegal": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_illegal_wrt_all_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_illegal_wrt_all_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_missing_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_missing_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_missing_wrt_expected_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_missing_wrt_expected_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_unexpected": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_unexpected_wrt_all_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_unexpected_wrt_all_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_unnecessary": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_unnecessary_wrt_all_executable": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "arguments_unnecessary_wrt_all_total": [
            "all",
            "data",
            "headers",
            "params",
            "path_params",
        ],
        "endpoints_correct": [],
        "endpoints_correct_wrt_executable": [],
        "endpoints_correct_wrt_total": [],
        "endpoints_illegal": [],
        "endpoints_illegal_wrt_executable": [],
        "endpoints_illegal_wrt_total": [],
        "endpoints_wrong": [],
        "endpoints_wrong_wrt_executable": [],
        "endpoints_wrong_wrt_total": [],
        "errors_incomplete_request": [],
        "errors_incomplete_request_wrt_errors": [],
        "errors_incomplete_request_wrt_samples": [],
        "errors_no_request": [],
        "errors_no_request_wrt_errors": [],
        "errors_no_request_wrt_samples": [],
        "errors_runtime_error": [],
        "errors_runtime_error_wrt_errors": [],
        "errors_runtime_error_wrt_samples": [],
        "errors_timeout": [],
        "errors_timeout_wrt_errors": [],
        "errors_timeout_wrt_samples": [],
        "errors_total": [],
        "errors_total_wrt_samples": [],
        "errors_unsatisfiable": [],
        "errors_unsatisfiable_wrt_errors": [],
        "errors_unsatisfiable_wrt_samples": [],
        "methods_correct": [],
        "methods_correct_wrt_executable": [],
        "methods_correct_wrt_total": [],
        "methods_illegal": [],
        "methods_illegal_wrt_executable": [],
        "methods_illegal_wrt_total": [],
        "methods_wrong": [],
        "methods_wrong_wrt_executable": [],
        "methods_wrong_wrt_total": [],
        "samples_correct": [],
        "samples_correct_wrt_executable": [],
        "samples_correct_wrt_total": [],
        "samples_executable": [],
        "samples_executable_wrt_total": [],
        "samples_illegal": [],
        "samples_illegal_wrt_executable": [],
        "samples_illegal_wrt_total": [],
        "samples_nonexecutable": [],
        "samples_total": [],
        "samples_wrong": [],
        "samples_wrong_wrt_executable": [],
        "samples_wrong_wrt_total": [],
        "urls_correct": [],
        "urls_correct_wrt_executable": [],
        "urls_correct_wrt_total": [],
        "urls_illegal": [],
        "urls_illegal_wrt_executable": [],
        "urls_illegal_wrt_total": [],
        "urls_wrong": [],
        "urls_wrong_wrt_executable": [],
        "urls_wrong_wrt_total": [],
    }
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _default_api_renaming(args)
    args.filename = "full"
    args.delete_t_and_e = False
    args.show_legend = True
    return args


def single_model_plot_preset(model):
    args = SimpleNamespace()
    args.outputs = ["pdf"]
    args.models = [model]
    args.apis = ["all"]
    args.setups = [
        "invocation",
        "endpoint",
    ]
    args.settings = [
        "vanilla",
        "constrained",
    ]
    args.metrics = {
        "samples_executable_wrt_total": [],
        "samples_correct_wrt_executable": [],
        "samples_wrong_wrt_executable": [],
        "samples_illegal_wrt_executable": [],
        "methods_correct_wrt_executable": [],
        "methods_wrong_wrt_executable": [],
        "methods_illegal_wrt_executable": [],
        "urls_correct_wrt_executable": [],
        "urls_wrong_wrt_executable": [],
        "urls_illegal_wrt_executable": [],
        "arguments_correct_name_wrt_expected_executable": ["all"],
        "arguments_correct_value_wrt_expected_executable": ["all"],
        "arguments_missing_wrt_expected_executable": ["all"],
        "arguments_unnecessary_wrt_all_executable": ["all"],
        "arguments_illegal_wrt_all_executable": ["all"],
        "argument_name_mean_precision_wrt_executable": ["all"],
        "argument_name_mean_recall_wrt_executable": ["all"],
        "argument_name_mean_jaccard_wrt_executable": ["all"],
        "argument_value_mean_accuracy_wrt_correct_executable": ["all"],
    }
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _setup_setting_renaming(args)
    args.filename = f"{'-'.join(args.models)}"
    args.delete_t_and_e = True
    args.show_legend = True
    return args


def multi_model_plot_preset(setup, setting):
    args = SimpleNamespace()
    args.outputs = ["pdf"]
    args.models = [
        "codet5p-16b",
        "starcoderbase",
        "starcoder2-15b",
        "deepseek-coder-6.7b-base",
        "DeepSeek-Coder-V2-Lite-Base",
        "Qwen2.5-Coder-14B",
        "Llama-3.1-8B",
        "CodeLlama-13b-hf",
        "CodeLlama-70b-hf",
        "gpt-4o",
    ]
    args.apis = ["all"]
    args.setups = [setup]
    args.settings = [setting]
    args.metrics = {
        "samples_correct_wrt_total": [],
        "samples_illegal_wrt_executable": [],
        "urls_correct_wrt_executable": [],
        "urls_illegal_wrt_executable": [],
        "methods_correct_wrt_executable": [],
        "methods_illegal_wrt_executable": [],
        "argument_name_mean_precision_wrt_executable": ["all"],
        "argument_name_mean_recall_wrt_executable": ["all"],
        "arguments_illegal_wrt_all_executable": ["all"],
        "argument_value_mean_accuracy_wrt_correct_executable": ["all"],
    }
    if setup == "endpoint":
        del args.metrics["methods_correct_wrt_executable"]
        del args.metrics["urls_correct_wrt_executable"]
        del args.metrics["methods_illegal_wrt_executable"]
        del args.metrics["urls_illegal_wrt_executable"]
    else:
        del args.metrics["arguments_illegal_wrt_all_executable"]
        del args.metrics["samples_illegal_wrt_executable"]
    if "constrained" in setting:
        args.models = [model for model in args.models if not (model.startswith("gpt-") or model.startswith("gemini-"))]
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _default_model_renaming(args)
    args.filename = f"multi_{'-'.join(args.setups)}_{'-'.join(args.settings)}"
    args.delete_t_and_e = False
    args.show_legend = True
    return args


def single_model_table_preset(model, setup, setting):
    args = SimpleNamespace()
    args.outputs = ["tex"]
    args.models = [model]
    args.apis = [
        "all",
        "asana",
        "google_calendar_v3",
        "google_sheet_v4",
        "slack",
    ]
    args.setups = [setup]
    args.settings = [setting]
    args.metrics = {
        "samples_executable_wrt_total": [],
        "samples_correct_wrt_total": [],
        "samples_correct_wrt_executable": [],
        "samples_illegal_wrt_total": [],
        "samples_illegal_wrt_executable": [],
        "urls_correct_wrt_total": [],
        "urls_correct_wrt_executable": [],
        "urls_illegal_wrt_total": [],
        "urls_illegal_wrt_executable": [],
        "methods_correct_wrt_total": [],
        "methods_correct_wrt_executable": [],
        "methods_illegal_wrt_total": [],
        "methods_illegal_wrt_executable": [],
        "arguments_correct_name_wrt_expected_total": ["all"],
        "arguments_correct_name_wrt_expected_executable": ["all"],
        "arguments_correct_value_wrt_expected_total": ["all"],
        "arguments_correct_value_wrt_expected_executable": ["all"],
        "arguments_missing_wrt_expected_total": ["all"],
        "arguments_missing_wrt_expected_executable": ["all"],
        "arguments_unexpected_wrt_all_total": ["all"],
        "arguments_unexpected_wrt_all_executable": ["all"],
        "arguments_unnecessary_wrt_all_total": ["all"],
        "arguments_unnecessary_wrt_all_executable": ["all"],
        "arguments_illegal_wrt_all_total": ["all"],
        "arguments_illegal_wrt_all_executable": ["all"],
        "argument_name_mean_precision_wrt_total": ["all"],
        "argument_name_mean_precision_wrt_executable": ["all"],
        "argument_name_mean_recall_wrt_total": ["all"],
        "argument_name_mean_recall_wrt_executable": ["all"],
        "argument_name_mean_jaccard_wrt_total": ["all"],
        "argument_name_mean_jaccard_wrt_executable": ["all"],
        "argument_value_mean_accuracy_wrt_correct_total": ["all"],
        "argument_value_mean_accuracy_wrt_correct_executable": ["all"],
        "errors_total": [],
        "errors_incomplete_request": [],
        "errors_runtime_error": [],
        "errors_timeout": [],
        "errors_unsatisfiable": [],
    }
    if setup == "endpoint":
        del args.metrics["urls_correct_wrt_total"]
        del args.metrics["urls_correct_wrt_executable"]
        del args.metrics["urls_illegal_wrt_total"]
        del args.metrics["urls_illegal_wrt_executable"]
        del args.metrics["methods_correct_wrt_total"]
        del args.metrics["methods_correct_wrt_executable"]
        del args.metrics["methods_illegal_wrt_total"]
        del args.metrics["methods_illegal_wrt_executable"]
        del args.metrics["arguments_unexpected_wrt_all_total"]
        del args.metrics["arguments_unexpected_wrt_all_executable"]
    else:
        del args.metrics["samples_illegal_wrt_total"]
        del args.metrics["samples_illegal_wrt_executable"]
        del args.metrics["arguments_unnecessary_wrt_all_total"]
        del args.metrics["arguments_unnecessary_wrt_all_executable"]
        del args.metrics["arguments_illegal_wrt_all_total"]
        del args.metrics["arguments_illegal_wrt_all_executable"]
    if "constrained" not in setting:
        del args.metrics["errors_timeout"]
        del args.metrics["errors_unsatisfiable"]
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _default_api_renaming(args)
    args.filename = f"{'-'.join(args.models)}_{'-'.join(args.setups)}_{'-'.join(args.settings)}"
    args.delete_t_and_e = False
    return args


def multi_model_table_preset(setup, setting):
    args = SimpleNamespace()
    args.outputs = ["tex"]
    args.models = [
        "codet5p-16b",
        "starcoderbase",
        "starcoder2-3b",
        "starcoder2-7b",
        "starcoder2-15b",
        "deepseek-coder-1.3b-base",
        "deepseek-coder-6.7b-base",
        "deepseek-coder-7b-base-v1.5",
        "deepseek-coder-33b-base",
        "DeepSeek-Coder-V2-Lite-Base",
        "Qwen2.5-Coder-0.5B",
        "Qwen2.5-Coder-1.5B",
        "Qwen2.5-Coder-3B",
        "Qwen2.5-Coder-7B",
        "Qwen2.5-Coder-14B",
        "Qwen2.5-Coder-32B",
        "Llama-3.1-8B",
        "Llama-3.1-70B",
        "CodeLlama-7b-hf",
        "CodeLlama-13b-hf",
        "CodeLlama-70b-hf",
        "gemini-pro-1.5",
        "gpt-4o-mini",
        "gpt-4o",
    ]
    args.apis = ["all"]
    args.setups = [setup]
    args.settings = [setting]
    args.metrics = {
        "samples_executable_wrt_total": [],
        "samples_correct_wrt_total": [],
        "samples_correct_wrt_executable": [],
        "samples_illegal_wrt_total": [],
        "samples_illegal_wrt_executable": [],
        "urls_correct_wrt_total": [],
        "urls_correct_wrt_executable": [],
        "urls_illegal_wrt_total": [],
        "urls_illegal_wrt_executable": [],
        "methods_correct_wrt_total": [],
        "methods_correct_wrt_executable": [],
        "methods_illegal_wrt_total": [],
        "methods_illegal_wrt_executable": [],
        "arguments_correct_name_wrt_expected_total": ["all"],
        "arguments_correct_name_wrt_expected_executable": ["all"],
        "arguments_correct_value_wrt_expected_total": ["all"],
        "arguments_correct_value_wrt_expected_executable": ["all"],
        "arguments_missing_wrt_expected_total": ["all"],
        "arguments_missing_wrt_expected_executable": ["all"],
        "arguments_unexpected_wrt_all_total": ["all"],
        "arguments_unexpected_wrt_all_executable": ["all"],
        "arguments_unnecessary_wrt_all_total": ["all"],
        "arguments_unnecessary_wrt_all_executable": ["all"],
        "arguments_illegal_wrt_all_total": ["all"],
        "arguments_illegal_wrt_all_executable": ["all"],
        "argument_name_mean_precision_wrt_total": ["all"],
        "argument_name_mean_precision_wrt_executable": ["all"],
        "argument_name_mean_recall_wrt_total": ["all"],
        "argument_name_mean_recall_wrt_executable": ["all"],
        "argument_name_mean_jaccard_wrt_total": ["all"],
        "argument_name_mean_jaccard_wrt_executable": ["all"],
        "argument_value_mean_accuracy_wrt_correct_total": ["all"],
        "argument_value_mean_accuracy_wrt_correct_executable": ["all"],
        "errors_total": [],
        "errors_incomplete_request": [],
        "errors_runtime_error": [],
        "errors_timeout": [],
        "errors_unsatisfiable": [],
    }
    if setup == "endpoint":
        del args.metrics["urls_correct_wrt_total"]
        del args.metrics["urls_correct_wrt_executable"]
        del args.metrics["urls_illegal_wrt_total"]
        del args.metrics["urls_illegal_wrt_executable"]
        del args.metrics["methods_correct_wrt_total"]
        del args.metrics["methods_correct_wrt_executable"]
        del args.metrics["methods_illegal_wrt_total"]
        del args.metrics["methods_illegal_wrt_executable"]
        del args.metrics["arguments_unexpected_wrt_all_total"]
        del args.metrics["arguments_unexpected_wrt_all_executable"]
    else:
        del args.metrics["samples_illegal_wrt_total"]
        del args.metrics["samples_illegal_wrt_executable"]
        del args.metrics["arguments_unnecessary_wrt_all_total"]
        del args.metrics["arguments_unnecessary_wrt_all_executable"]
        del args.metrics["arguments_illegal_wrt_all_total"]
        del args.metrics["arguments_illegal_wrt_all_executable"]
    if "constrained" in setting:
        args.models = [model for model in args.models if not (model.startswith("gpt-") or model.startswith("gemini-"))]
    else:
        del args.metrics["errors_timeout"]
        del args.metrics["errors_unsatisfiable"]
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _default_model_renaming(args)
    args.filename = f"multi_{'-'.join(args.setups)}_{'-'.join(args.settings)}"
    args.delete_t_and_e = False
    return args


def multi_model_submetric_table_preset(setup, setting):
    args = SimpleNamespace()
    args.outputs = ["tex"]
    args.models = [
        "codet5p-16b",
        "starcoderbase",
        "starcoder2-3b",
        "starcoder2-7b",
        "starcoder2-15b",
        "deepseek-coder-1.3b-base",
        "deepseek-coder-6.7b-base",
        "deepseek-coder-7b-base-v1.5",
        "deepseek-coder-33b-base",
        "DeepSeek-Coder-V2-Lite-Base",
        "Qwen2.5-Coder-0.5B",
        "Qwen2.5-Coder-1.5B",
        "Qwen2.5-Coder-3B",
        "Qwen2.5-Coder-7B",
        "Qwen2.5-Coder-14B",
        "Qwen2.5-Coder-32B",
        "Llama-3.1-8B",
        "Llama-3.1-70B",
        "CodeLlama-7b-hf",
        "CodeLlama-13b-hf",
        "CodeLlama-70b-hf",
        "gemini-pro-1.5",
        "gpt-4o-mini",
        "gpt-4o",
    ]
    args.apis = ["all"]
    args.setups = [setup]
    args.settings = [setting]
    submetrics = ["all", "path_params", "params", "headers", "data"]
    args.metrics = {
        "arguments_correct_name_wrt_expected_total": submetrics,
        "arguments_correct_name_wrt_expected_executable": submetrics,
        "arguments_correct_value_wrt_expected_total": submetrics,
        "arguments_correct_value_wrt_expected_executable": submetrics,
        "arguments_missing_wrt_expected_total": submetrics,
        "arguments_missing_wrt_expected_executable": submetrics,
        "arguments_unexpected_wrt_all_total": submetrics,
        "arguments_unexpected_wrt_all_executable": submetrics,
        "arguments_unnecessary_wrt_all_total": submetrics,
        "arguments_unnecessary_wrt_all_executable": submetrics,
        "arguments_illegal_wrt_all_total": submetrics,
        "arguments_illegal_wrt_all_executable": submetrics,
        "argument_name_mean_precision_wrt_total": submetrics,
        "argument_name_mean_precision_wrt_executable": submetrics,
        "argument_name_mean_recall_wrt_total": submetrics,
        "argument_name_mean_recall_wrt_executable": submetrics,
        "argument_name_mean_jaccard_wrt_total": submetrics,
        "argument_name_mean_jaccard_wrt_executable": submetrics,
        "argument_value_mean_accuracy_wrt_correct_total": submetrics,
        "argument_value_mean_accuracy_wrt_correct_executable": submetrics,
    }
    if setup == "endpoint":
        del args.metrics["arguments_unexpected_wrt_all_total"]
        del args.metrics["arguments_unexpected_wrt_all_executable"]
    else:
        del args.metrics["arguments_unnecessary_wrt_all_total"]
        del args.metrics["arguments_unnecessary_wrt_all_executable"]
        del args.metrics["arguments_illegal_wrt_all_total"]
        del args.metrics["arguments_illegal_wrt_all_executable"]
    if "constrained" in setting:
        args.models = [model for model in args.models if not (model.startswith("gpt-") or model.startswith("gemini-"))]
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _default_model_renaming(args)
    args.filename = f"multi_submetric_{'-'.join(args.setups)}_{'-'.join(args.settings)}"
    args.delete_t_and_e = False
    return args


def compact_table_preset(model):
    args = SimpleNamespace()
    args.outputs = ["tex"]
    args.models = [model]
    args.apis = ["all"]
    args.setups = [
        "invocation",
        "endpoint",
    ]
    args.settings = [
        "vanilla",
        "constrained",
    ]
    args.metrics = {
        "samples_correct_wrt_executable": [],
        "samples_executable_wrt_total": [],
        "samples_illegal_wrt_executable": [],
        "urls_correct_wrt_executable": [],
        "urls_illegal_wrt_executable": [],
        "methods_correct_wrt_executable": [],
        "methods_illegal_wrt_executable": [],
        "arguments_correct_name_wrt_expected_executable": ["all"],
        "arguments_correct_value_wrt_expected_executable": ["all"],
        "arguments_illegal_wrt_all_executable": ["all"],
        "argument_name_mean_precision_wrt_executable": ["all"],
        "argument_name_mean_recall_wrt_executable": ["all"],
    }
    args.row_renaming = _default_metric_renaming(args)
    args.column_renaming = _setup_setting_renaming(args)
    args.filename = f"{'-'.join(args.models)}_compact"
    args.delete_t_and_e = False
    return args


def ud_cd_comparison_preset(metric, setup):
    args = SimpleNamespace()
    args.outputs = ["pdf"]
    args.models = [
        "codet5p-16b",
        "starcoderbase",
        "starcoder2-7b",
        "starcoder2-15b",
        "deepseek-coder-6.7b-base",
        "deepseek-coder-33b-base",
        "DeepSeek-Coder-V2-Lite-Base",
        "Qwen2.5-Coder-7B",
        "Qwen2.5-Coder-14B",
        "Qwen2.5-Coder-32B",
        "Llama-3.1-8B",
        "Llama-3.1-70B",
        "CodeLlama-7b-hf",
        "CodeLlama-13b-hf",
        "CodeLlama-70b-hf",
    ]
    args.apis = ["all"]
    args.setups = [setup]
    args.settings = None
    args.metrics = [metric]
    args.row_renaming = MODEL_MAP
    args.column_renaming = None
    args.filename = "ud-vs-cd"
    args.delete_t_and_e = False
    args.show_legend = True
    return args


def settings_comparison_preset(metric, setup):
    args = SimpleNamespace()
    args.outputs = ["pdf"]
    args.models = [
        "codet5p-16b",
        "starcoderbase",
        "starcoder2-15b",
        "deepseek-coder-6.7b-base",
        "DeepSeek-Coder-V2-Lite-Base",
        "Qwen2.5-Coder-14B",
        "Llama-3.1-8B",
        "CodeLlama-13b-hf",
        "CodeLlama-70b-hf",
        "gpt-4o",
    ]
    args.apis = ["all"]
    args.setups = [setup]
    args.settings = [
        "vanilla",
        "rag",
        "constrained",
        "constrained-rag",
    ]
    args.metrics = [metric]
    args.row_renaming = MODEL_MAP
    args.column_renaming = SETTING_MAP
    args.filename = "comparison"
    args.delete_t_and_e = False
    args.show_legend = True
    return args


SETTING_MAP = {
    "vanilla": "Vanilla",
    "rag": "RAG",
    "constrained": "Constrained",
    "constrained-rag": "Constrained + RAG",
}

MODEL_MAP = {
    "codet5p-16b": "CodeT5+ (16B)",
    "instructcodet5p-16b": "InstructCodeT5+ (16B)",
    "starcoderbase": "StarCoder (15.5B)",
    "starcoder2-3b": "StarCoder2 (3B)",
    "starcoder2-7b": "StarCoder2 (7B)",
    "starcoder2-15b": "StarCoder2 (15B)",
    "deepseek-coder-1.3b-base": "DeepSeek-Coder (1.3B)",
    "deepseek-coder-6.7b-base": "DeepSeek-Coder (6.7B)",
    "deepseek-coder-7b-base-v1.5": "DeepSeek-Coder (7B)",
    "deepseek-coder-33b-base": "DeepSeek-Coder (33B)",
    "DeepSeek-Coder-V2-Lite-Base": "DeepSeek-Coder-V2 (16B)",
    "Qwen2.5-Coder-0.5B": "Qwen2.5-Coder (0.5B)",
    "Qwen2.5-Coder-1.5B": "Qwen2.5-Coder (1.5B)",
    "Qwen2.5-Coder-3B": "Qwen2.5-Coder (3B)",
    "Qwen2.5-Coder-7B": "Qwen2.5-Coder (7B)",
    "Qwen2.5-Coder-14B": "Qwen2.5-Coder (14B)",
    "Qwen2.5-Coder-32B": "Qwen2.5-Coder (32B)",
    "CodeLlama-7b-hf": "Code Llama (7B)",
    "CodeLlama-13b-hf": "Code Llama (13B)",
    "CodeLlama-70b-hf": "Code Llama (70B)",
    "Llama-3.1-8B": "Llama 3.1 (8B)",
    "Llama-3.1-70B": "Llama 3.1 (70B)",
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4o": "GPT-4o",
    "gemini-pro-1.5": "Gemini 1.5 Pro",
}

API_MAP = {
    "asana": "Asana",
    "google_calendar_v3": "Google Calendar",
    "google_sheet_v4": "Google Sheets",
    "slack": "Slack",
    "all": "Overall",
}

SUBMETRIC_MAP = {
    "all": "",
    "data": " body",
    "headers": " header",
    "params": " query",
    "path_params": " path",
}

METRIC_MAP = {
    **{f"{submetric}_{metric}": name % SUBMETRIC_MAP[submetric] for submetric in
       ["all", "data", "headers", "params", "path_params"] for metric, name in {
           "argument_name_mean_jaccard_wrt_executable": "Mean%s argument Jaccard index (e)",
           "argument_name_mean_jaccard_wrt_total": "Mean%s argument Jaccard index (t)",
           "argument_name_mean_precision_wrt_executable": "Mean%s argument precision (e)",
           "argument_name_mean_precision_wrt_total": "Mean%s argument precision (t)",
           "argument_name_mean_recall_wrt_executable": "Mean%s argument recall (e)",
           "argument_name_mean_recall_wrt_total": "Mean%s argument recall (t)",
           "argument_name_sum_jaccard_wrt_executable": "Aggregated%s argument Jaccard index (e)",
           "argument_name_sum_jaccard_wrt_total": "Aggregated%s argument Jaccard index (t)",
           "argument_name_sum_precision_wrt_executable": "Aggregated%s argument precision (e)",
           "argument_name_sum_precision_wrt_total": "Aggregated%s argument precision (t)",
           "argument_name_sum_recall_wrt_executable": "Aggregated %s argument recall (e)",
           "argument_name_sum_recall_wrt_total": "Aggregated %s argument recall (t)",
           "argument_value_mean_accuracy_wrt_all_executable": "Mean%s argument value absolute accuracy (e)",
           "argument_value_mean_accuracy_wrt_all_total": "Mean%s argument value absolute accuracy (t)",
           "argument_value_mean_accuracy_wrt_correct_executable": "Mean%s argument value conditional accuracy (e)",
           "argument_value_mean_accuracy_wrt_correct_total": "Mean%s argument value conditional accuracy (t)",
           "argument_value_mean_accuracy_wrt_expected_executable": "Mean%s argument value relative accuracy (e)",
           "argument_value_mean_accuracy_wrt_expected_total": "Mean%s argument value relative accuracy (t)",
           "argument_value_sum_accuracy_wrt_all_executable": "Aggregated%s argument value absolute accuracy (e)",
           "argument_value_sum_accuracy_wrt_all_total": "Aggregated%s argument value absolute accuracy (t)",
           "argument_value_sum_accuracy_wrt_correct_executable": "Aggregated%s argument value conditional accuracy (e)",
           "argument_value_sum_accuracy_wrt_correct_total": "Aggregated%s argument value conditional accuracy (t)",
           "argument_value_sum_accuracy_wrt_expected_executable": "Aggregated%s argument value relative accuracy (e)",
           "argument_value_sum_accuracy_wrt_expected_total": "Aggregated%s argument value relative accuracy (t)",
           "arguments_all_executable": "Encountered%s arguments in executable code",
           "arguments_all_total": "Encountered%s arguments in total",
           "arguments_correct_name": "Correct%s argument names",
           "arguments_correct_name_wrt_expected_executable": "Correct%s argument names (e)",
           "arguments_correct_name_wrt_expected_total": "Correct%s argument names (t)",
           "arguments_correct_value": "Correct%s argument values",
           "arguments_correct_value_wrt_expected_executable": "Correct%s argument values (e)",
           "arguments_correct_value_wrt_expected_total": "Correct%s argument values (t)",
           "arguments_expected_executable": "Expected%s arguments in executable code",
           "arguments_expected_total": "Expected%s arguments in total",
           "arguments_illegal": "Illegal%s arguments",
           "arguments_illegal_wrt_all_executable": "Illegal%s arguments (e)",
           "arguments_illegal_wrt_all_total": "Illegal%s arguments (t)",
           "arguments_missing_executable": "Missing%s arguments in executable code",
           "arguments_missing_total": "Missing%s arguments in total",
           "arguments_missing_wrt_expected_executable": "Missing%s arguments (e)",
           "arguments_missing_wrt_expected_total": "Missing%s arguments (t)",
           "arguments_unexpected": "Unexpected%s arguments",
           "arguments_unexpected_wrt_all_executable": "Unexpected%s arguments (e)",
           "arguments_unexpected_wrt_all_total": "Unexpected%s arguments (t)",
           "arguments_unnecessary": "Unnecessary%s arguments",
           "arguments_unnecessary_wrt_all_executable": "Unnecessary%s arguments (e)",
           "arguments_unnecessary_wrt_all_total": "Unnecessary%s arguments (t)",
       }.items()},
    "endpoints_correct": "Correct endpoints",
    "endpoints_correct_wrt_executable": "Correct endpoints (e)",
    "endpoints_correct_wrt_total": "Correct endpoints (t)",
    "endpoints_illegal": "Illegal endpoints",
    "endpoints_illegal_wrt_executable": "Illegal endpoints (e)",
    "endpoints_illegal_wrt_total": "Illegal endpoints (t)",
    "endpoints_wrong": "Wrong endpoints",
    "endpoints_wrong_wrt_executable": "Wrong endpoints (e)",
    "endpoints_wrong_wrt_total": "Wrong endpoints (t)",
    "errors_incomplete_request": "Incomplete implementations",
    "errors_incomplete_request_wrt_errors": "Incomplete implementations (s)",
    "errors_incomplete_request_wrt_samples": "Incomplete implementations (t)",
    "errors_no_request": "No requests",
    "errors_no_request_wrt_errors": "No requests (s)",
    "errors_no_request_wrt_samples": "No requests (t)",
    "errors_runtime_error": "Runtime errors",
    "errors_runtime_error_wrt_errors": "Runtime errors (s)",
    "errors_runtime_error_wrt_samples": "Runtime errors (t)",
    "errors_timeout": "Timeouts",
    "errors_timeout_wrt_errors": "Timeouts (s)",
    "errors_timeout_wrt_samples": "Timeouts (t)",
    "errors_total": "Total errors",
    "errors_total_wrt_samples": "Total errors (t)",
    "errors_unsatisfiable": "Unsatisfiable constraints",
    "errors_unsatisfiable_wrt_errors": "Unsatisfiable constraints (s)",
    "errors_unsatisfiable_wrt_samples": "Unsatisfiable constraints (t)",
    "methods_correct": "Correct methods",
    "methods_correct_wrt_executable": "Correct methods (e)",
    "methods_correct_wrt_total": "Correct methods (t)",
    "methods_illegal": "Illegal methods",
    "methods_illegal_wrt_executable": "Illegal methods (e)",
    "methods_illegal_wrt_total": "Illegal methods (t)",
    "methods_wrong": "Wrong methods",
    "methods_wrong_wrt_executable": "Wrong methods (e)",
    "methods_wrong_wrt_total": "Wrong methods (t)",
    "samples_correct": "Correct implementations",
    "samples_correct_wrt_executable": "Correct implementations (e)",
    "samples_correct_wrt_total": "Correct implementations (t)",
    "samples_executable": "Executable implementations",
    "samples_executable_wrt_total": "Executable implementations (t)",
    "samples_illegal": "Illegal implementations",
    "samples_illegal_wrt_executable": "Illegal implementations (e)",
    "samples_illegal_wrt_total": "Illegal implementations (t)",
    "samples_nonexecutable": "Nonexecutable implementations",
    "samples_total": "Total implementations",
    "samples_wrong": "Wrong implementations",
    "samples_wrong_wrt_executable": "Wrong implementations (e)",
    "samples_wrong_wrt_total": "Wrong implementations (t)",
    "urls_correct": "Correct URLs",
    "urls_correct_wrt_executable": "Correct URLs (e)",
    "urls_correct_wrt_total": "Correct URLs (t)",
    "urls_illegal": "Illegal URLs",
    "urls_illegal_wrt_executable": "Illegal URLs (e)",
    "urls_illegal_wrt_total": "Illegal URLs (t)",
    "urls_wrong": "Wrong URLs",
    "urls_wrong_wrt_executable": "Wrong URLs (e)",
    "urls_wrong_wrt_total": "Wrong URLs (t)",
}

INVERSE_METRIC_MAP = {v: k for k, v in METRIC_MAP.items()}

LOWER_IS_BETTER = {
    "arguments_illegal",
    "arguments_illegal_wrt_all_executable",
    "arguments_illegal_wrt_all_total",
    "arguments_missing_executable",
    "arguments_missing_total",
    "arguments_missing_wrt_expected_executable",
    "arguments_missing_wrt_expected_total",
    "arguments_missing_wrt_total",
    "arguments_unexpected",
    "arguments_unexpected_wrt_all_executable",
    "arguments_unexpected_wrt_all_total",
    "arguments_unnecessary",
    "arguments_unnecessary_wrt_all_executable",
    "arguments_unnecessary_wrt_all_total",
    *{f"{submetric}_{metric}" for submetric in ["all", "data", "headers", "params", "path_params"] for metric in [
        "arguments_illegal",
        "arguments_illegal_wrt_all_executable",
        "arguments_illegal_wrt_all_total",
        "arguments_missing_executable",
        "arguments_missing_total",
        "arguments_missing_wrt_expected_executable",
        "arguments_missing_wrt_expected_total",
        "arguments_missing_wrt_total",
        "arguments_unexpected",
        "arguments_unexpected_wrt_all_executable",
        "arguments_unexpected_wrt_all_total",
        "arguments_unnecessary",
        "arguments_unnecessary_wrt_all_executable",
        "arguments_unnecessary_wrt_all_total",
    ]},
    "endpoints_illegal",
    "endpoints_illegal_wrt_executable",
    "endpoints_illegal_wrt_total",
    "endpoints_wrong",
    "endpoints_wrong_wrt_executable",
    "endpoints_wrong_wrt_total",
    "errors_incomplete_request",
    "errors_incomplete_request_wrt_errors",
    "errors_incomplete_request_wrt_samples",
    "errors_no_request",
    "errors_no_request_wrt_errors",
    "errors_no_request_wrt_samples",
    "errors_runtime_error",
    "errors_runtime_error_wrt_errors",
    "errors_runtime_error_wrt_samples",
    "errors_timeout",
    "errors_timeout_wrt_errors",
    "errors_timeout_wrt_samples",
    "errors_total",
    "errors_total_wrt_samples",
    "errors_unsatisfiable",
    "errors_unsatisfiable_wrt_errors",
    "errors_unsatisfiable_wrt_samples",
    "methods_illegal",
    "methods_illegal_wrt_executable",
    "methods_illegal_wrt_total",
    "methods_wrong",
    "methods_wrong_wrt_executable",
    "methods_wrong_wrt_total",
    "samples_illegal",
    "samples_illegal_wrt_executable",
    "samples_illegal_wrt_total",
    "samples_nonexecutable",
    "samples_wrong",
    "samples_wrong_wrt_executable",
    "samples_wrong_wrt_total",
    "urls_illegal",
    "urls_illegal_wrt_executable",
    "urls_illegal_wrt_total",
    "urls_wrong",
    "urls_wrong_wrt_executable",
    "urls_wrong_wrt_total"
}

NA_list = [
    ("samples_illegal_wrt_total", "invocation"),
    ("samples_illegal_wrt_executable", "invocation"),
    ("urls_correct_wrt_total", "endpoint"),
    ("urls_correct_wrt_executable", "endpoint"),
    ("urls_illegal_wrt_total", "endpoint"),
    ("urls_illegal_wrt_executable", "endpoint"),
    ("methods_correct_wrt_total", "endpoint"),
    ("methods_correct_wrt_executable", "endpoint"),
    ("methods_illegal_wrt_total", "endpoint"),
    ("methods_illegal_wrt_executable", "endpoint"),
    ("arguments_illegal_wrt_all_total", "invocation"),
    ("arguments_illegal_wrt_all_executable", "invocation"),
    ("errors_timeout", "vanilla"),
    ("errors_unsatisfiable", "vanilla")
]

# All options: ["o", "v", "^", "<", ">", "s", "D", "X", "P", "*", "d", "p", "h", "H", "8"]
MARKER_MAP = {
    "codet5p-16b": "P",
    "instructcodet5p-16b": "P",
    "starcoderbase": "*",
    "starcoder2-3b": "*",
    "starcoder2-7b": "*",
    "starcoder2-15b": "*",
    "deepseek-coder-1.3b-base": "^",
    "deepseek-coder-6.7b-base": "^",
    "deepseek-coder-7b-base-v1.5": "^",
    "deepseek-coder-33b-base": "^",
    "DeepSeek-Coder-V2-Lite-Base": "^",
    "Qwen2.5-Coder-0.5B": "v",
    "Qwen2.5-Coder-1.5B": "v",
    "Qwen2.5-Coder-3B": "v",
    "Qwen2.5-Coder-7B": "v",
    "Qwen2.5-Coder-14B": "v",
    "Qwen2.5-Coder-32B": "v",
    "CodeLlama-7b-hf": "s",
    "CodeLlama-13b-hf": "s",
    "CodeLlama-70b-hf": "s",
    "Llama-3.1-8B": "D",
    "Llama-3.1-70B": "D",
    "gpt-4o-mini": "o",
    "gpt-4o": "o",
    "gemini-pro-1.5": "p",
}


def _default_model_renaming(args):
    return {f"{model}_{api}_{setup}_{setting}": MODEL_MAP[model]
            for model in args.models for api in args.apis for setup in args.setups for setting in args.settings}


def _default_api_renaming(args):
    return {f"{model}_{api}_{setup}_{setting}": API_MAP[api]
            for model in args.models for api in args.apis for setup in args.setups for setting in args.settings}


def _default_metric_renaming(args):
    return {f"{submetric}_{metric}": METRIC_MAP[f"{submetric}_{metric}"]
            for metric, submetrics in args.metrics.items() for submetric in submetrics} \
        | {metric: METRIC_MAP[metric] for metric, submetrics in args.metrics.items() if not submetrics}


def _setup_setting_renaming(args):
    return {f"{model}_{api}_{setup}_{setting}": f"{setup.capitalize()} {SETTING_MAP[setting]}"
            for model in args.models for api in args.apis for setup in args.setups for setting in args.settings}

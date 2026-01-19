# WAPIIBench

A benchmark for web API integration code generation. For more information, check out our AIware 2025
paper [Benchmarking Web API Integration Code Generation](https://arxiv.org/abs/2509.20172v5). An appendix to the paper
and all evaluation results are provided in our [artifact](https://doi.org/10.5281/zenodo.17607587).

## Project Overview

### Benchmark

Evaluation pipeline: [wapiibench/evaluation.py](wapiibench/evaluation.py)

Dataset creation: [wapiibench/dataset_generation.py](wapiibench/dataset_generation.py)

Full dataset: [data/synthetic/all/test_data_final.json](data/synthetic/all/test_data_final.json)

API-specific subsets: `data/synthetic/{api}/test_data_corrected.json`

Codes, configs, logs, results: `data/generated/{model}/{api}/{setup}/{setting}/`

Aggregated results: `data/generated/{model}/all/{setup}/{setting}/results.json`

Data visualization: [wapiibench/export_results.py](wapiibench/export_results.py)

Running the evaluation pipeline:

```commandline
python evaluation.py --models MODELS --apis APIS --setups SETUPS --settings SETTINGS
```

Example:

```commandline
python evaluation.py --models 'bigcode/starcoder2-15b' --apis 'asana' 'google_calendar_v3' 'google_sheet_v4' 'slack' --setups 'invocation' 'endpoint' --settings 'unconstrained'
```

(Further options exist; see `python evaluation.py --help`)

### Constrained Decoding

Translation of OpenAPI specs to regex constraints: [wapiibench/openapi_utils.py](wapiibench/openapi_utils.py)

Constrained decoding implementation: [wapiibench/logits_processor.py](wapiibench/logits_processor.py)

### Retrieval-Augmented Generation (work in progress)

Retrieval of endpoint documentation from OpenAPI specs: [wapiibench/rag_utils.py](wapiibench/rag_utils.py)

## Setup

The minimum Python version is 3.9. We recommend using a virtual environment.

Install/upgrade basic dependencies:

```commandline
pip install --upgrade torch transformers accelerate numpy openapi3-parser pyyaml regex strenum tqdm
```

Additional optional dependencies:

- `openai` for running API-based models
- `scikit-learn` for retrieval-augmented generation
- `pandas matplotlib` for plotting results
- `argilla` for data curation

<details>
<summary>Special dependencies for certain models</summary>

- `transformers<4.50.0` for codet5p-*b, instructcodet5p-16b, codegen-*B-multi, codegen2-*B_P
- `transformers<4.41.0 flash-attn` for DeepSeek-Coder-V2-Lite-Base, DeepSeek-Coder-V2-Lite-Instruct (to install
  `flash-attn` with `pip`, use the flags `--use-pep517` `--no-build-isolation`)

</details>

Dump dependencies to make setup reproducible:

```commandline
pip freeze > requirements.txt
```

Install/update `node` using `nvm`:

```commandline
nvm install --lts
```

Install JS dependencies (requires `node`):

```commandline
npm install axios axios-mock-adapter
```

Upgrade JS dependencies:

```commandline
npm update
```

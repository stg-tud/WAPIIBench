from __future__ import annotations

import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, \
    LogitsProcessor, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizer, T5ForConditionalGeneration

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelWrapper:
    """
    Wraps a HuggingFace or OpenAI/OpenRouter model.
    :param model_name: Name of the model
    :param kwargs: Additional arguments to pass to ``from_pretrained`` or ``OpenAI()``, respectively
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        if model_name.startswith("openai/") or model_name.startswith("google/"):
            logger.info(f"Setting up OpenAI model '{model_name}' ...")
            from openai import OpenAI
            self.provider = 'OpenAI'
            self.model_name = model_name.removeprefix("openai/")
            self.is_chat_model = True
            self.model = None
            self.tokenizer = None
            if not model_name.startswith("openai/"):
                kwargs.setdefault('base_url', "https://openrouter.ai/api/v1")
                kwargs.setdefault('api_key', os.getenv('OPENROUTER_API_KEY'))
            self.client = OpenAI(**kwargs)

        else:
            logger.info(f"Setting up HuggingFace model '{model_name}' ...")
            self.provider = 'HuggingFace'
            self.model_name = model_name
            self.is_chat_model = "-instruct" in model_name.lower()
            self.model, self.tokenizer = self._load_hf_model(**kwargs)
            self.client = None

    def _load_hf_model(self, **kwargs) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Utility function to load a model and its tokenizer from HuggingFace.
        :param kwargs: Additional arguments to pass to ``from_pretrained``
        :return: The model and tokenizer, and whether inputs should also be passed to the decoder
        """
        kwargs.setdefault('token', os.getenv('HF_TOKEN'))  # provides access to gated models on HF
        kwargs.setdefault('trust_remote_code', True)

        model_name = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

        kwargs.setdefault('low_cpu_mem_usage', True)
        kwargs.setdefault('device_map', 'auto')

        if model_name.startswith("Salesforce/codet5-"):
            model_class = T5ForConditionalGeneration
            del kwargs['trust_remote_code']
        elif model_name.startswith("Salesforce/codet5p-") or model_name.startswith("Salesforce/instructcodet5p-"):
            model_class = AutoModelForSeq2SeqLM
            kwargs.setdefault('torch_dtype', torch.float16)
        else:
            model_class = AutoModelForCausalLM
            if model_name.startswith("meta-llama/CodeLlama-"):
                kwargs.setdefault('dtype', torch.float16)
            elif (model_name in ["bigcode/starcoderbase", "bigcode/starcoder", "bigcode/starcoderplus"] or
                  model_name.startswith("bigcode/starcoder2-") or
                  model_name.startswith("deepseek-ai/deepseek-coder-") or
                  model_name.startswith("Qwen/Qwen2.5-Coder-") or
                  model_name.startswith("meta-llama/Llama-3.1-")):
                kwargs.setdefault('dtype', torch.bfloat16)
            elif model_name.startswith("deepseek-ai/DeepSeek-Coder-V2-"):
                kwargs.setdefault('torch_dtype', torch.bfloat16)
            elif model_name.startswith("Salesforce/codegen2-"):
                kwargs.setdefault('revision', "main")
            else:
                # Models not explicitly addressed above haven't been tried yet, but they may still work
                kwargs.setdefault('dtype', "auto")

        model = model_class.from_pretrained(model_name, **kwargs)
        logger.info(f"Loaded model '{type(model).__name__}' with tokenizer '{type(tokenizer).__name__}'")

        return model, tokenizer

    def run(self, input_text: str, generation_config: GenerationConfig | None = None,
            logits_processor: LogitsProcessorList | LogitsProcessor | None = None, batch: bool = False,
            batch_file_dir: str | None = None, sample_id: int | None = None, **kwargs) -> list[str] | None:
        """
        Run the model on a given input.
        :param input_text: The prompt
        :param generation_config: Object that holds further parameters (only for HF models)
        :param logits_processor: One or many logits processors (only for HF models)
        :param batch: Whether to use OpenAI's Batch API (only for OpenAI models)
        :param batch_file_dir: Location of the batch file (only if ``batch`` is True)
        :param sample_id: ID to be used to identify this sample in the batch (only if ``batch`` is True)
        :param kwargs: Additional arguments for the model
        :return: The text completion or None if the batch isn't completed yet
        """
        if self.provider == 'OpenAI':
            if logits_processor is not None:
                logger.warning("OpenAI models do not support LogitsProcessors")
            if generation_config is not None:
                kwargs.setdefault('max_completion_tokens', generation_config.max_new_tokens)
                kwargs.setdefault('n', generation_config.num_return_sequences)
                kwargs.setdefault('stop', generation_config.stop_strings)
                kwargs.setdefault(
                    'temperature', 0 if not generation_config.do_sample else generation_config.temperature)
                kwargs.setdefault('top_p', generation_config.top_p)
                if generation_config.num_beams > 1:
                    logger.warning("OpenAI models do not support beam search")
                if generation_config.top_k != 50:  # 50 is the default; any other number means the user explicitly set k
                    logger.warning("OpenAI models do not support top-k sampling")

            return self._run_openai_model(
                input_text, batch=batch, batch_file_dir=batch_file_dir, sample_id=sample_id, **kwargs)

        else:
            return self._run_hf_model(input_text, generation_config, logits_processor, **kwargs)

    def _run_hf_model(self, input_text: str, generation_config: GenerationConfig | None = None,
                      logits_processor: LogitsProcessorList | LogitsProcessor | None = None, **kwargs) -> list[str]:
        """
        Utility function to run a HuggingFace model on a given input.
        :param input_text: The prompt
        :param generation_config: Object that holds further parameters
        :param logits_processor: One or many logits processors
        :param kwargs: Additional arguments for model and tokenizer
        :return: The completed text (input + output)
        """
        if self.is_chat_model:
            inputs = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': input_text}],
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
        else:
            inputs = self.tokenizer(input_text, return_tensors='pt')
        if "codet5p" in self.model_name:
            inputs['decoder_input_ids'] = inputs['input_ids'].clone()
        inputs.to(self.model.device)

        if isinstance(logits_processor, LogitsProcessor):
            logits_processor = LogitsProcessorList([logits_processor])

        generated_ids = self.model.generate(
            **inputs, generation_config=generation_config, logits_processor=logits_processor, tokenizer=self.tokenizer,
            **kwargs)

        generated_texts = \
            [self.tokenizer.decode(token_ids, skip_special_tokens=True, **kwargs) for token_ids in generated_ids]

        if "<extra_id_0>" in input_text:
            generated_texts = [input_text.replace("<extra_id_0>", generated_text) for generated_text in generated_texts]

        return generated_texts

    def _run_openai_model(self, input_text: str, batch: bool = False, batch_file_dir: str | None = None,
                          sample_id: int | None = None, **kwargs) -> list[str] | None:
        """
        Utility function to run an OpenAI model on a given input.
        :param input_text: The prompt
        :param batch: Whether to use OpenAI's Batch API
        :param batch_file_dir: Location of the batch file if ``batch`` is True
        :param sample_id: ID to be used to identify this sample in the batch if ``batch`` is True
        :param kwargs: Additional arguments for the model
        :return: The text completion or None if the batch isn't completed yet
        """
        system_message, user_message = input_text.split(". ", 1)
        system_message += "."
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]

        if batch:
            if batch_file_dir is None or sample_id is None:
                raise ValueError("For batch processing, `batch_file_dir` and `sample_id` must be provided")

            batch_output_file = os.path.join(batch_file_dir, "batch_output.jsonl")
            if os.path.isfile(batch_output_file):
                # Batch results have arrived - serve them
                with open(batch_output_file, 'r') as file:
                    for line in file:
                        if f'"custom_id": "{sample_id}"' in line:
                            return [choice['message']['content']
                                    for choice in json.loads(line)['response']['body']['choices']]

                raise AssertionError(f"Sample {sample_id} not found in batch")

            else:
                # Add this sample to the batch input file
                task = {
                    'custom_id': str(sample_id),
                    'method': 'POST',
                    'url': "/v1/chat/completions",
                    'body': {
                        'model': self.model_name,
                        'messages': messages,
                        **kwargs,
                    }
                }

                with open(os.path.join(batch_file_dir, "batch_input.jsonl"), 'a') as file:
                    file.write(json.dumps(task) + "\n")

                return None

        else:
            completion = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
            logger.debug(f"Received {completion=}")
            return [choice.message.content for choice in completion.choices]

    def batch_init(self, batch_file_dir: str) -> bool:
        """
        Initialize batch processing.
        :param batch_file_dir: Location of the batch files
        :return: Whether we are **not** currently waiting for a batch to complete
        """
        batch_id_file = os.path.join(batch_file_dir, "batch_id.txt")
        if os.path.isfile(batch_id_file):
            # There is a batch currently being processed - check if it is completed
            with open(batch_id_file, 'r') as file:
                batch_id = file.read()
            can_continue = self._retrieve_batch(batch_id, batch_file_dir)
            if can_continue:
                os.remove(batch_id_file)
            return can_continue

        elif os.path.isfile(os.path.join(batch_file_dir, "batch_output.jsonl")):
            # Avoid deleting the batch input file
            logger.warning("Batch has already been retrieved")
            return True

        else:
            # Create an empty batch input file
            open(os.path.join(batch_file_dir, "batch_input.jsonl"), 'w').close()
            return True

    def _retrieve_batch(self, batch_id: str, batch_file_dir: str) -> bool:
        """
        Try to retrieve a batch from OpenAI.
        :param batch_id: ID of the batch
        :param batch_file_dir: Location of the batch files
        :return: Whether we are **not** currently waiting for a batch to complete
        """
        batch_job = self.client.batches.retrieve(batch_id)
        logger.debug(f"Retrieved {batch_job=}")
        status = batch_job.status

        if status == 'completed':
            logger.info(f"Batch is completed - retrieving results")
            batch_result = self.client.files.content(batch_job.output_file_id)
            with open(os.path.join(batch_file_dir, "batch_output.jsonl"), 'wb') as file:
                file.write(batch_result.content)
            return True

        else:
            logger.warning(f"Batch status is '{status}'")
            return False

    def submit_batch(self, batch_file_dir: str) -> None:
        """
        Submit a batch to OpenAI if there is not already one being processed (and if the results aren't already retrieved).
        :param batch_file_dir: Location of the batch files
        """
        batch_input_file = os.path.join(batch_file_dir, "batch_input.jsonl")
        batch_output_file = os.path.join(batch_file_dir, "batch_output.jsonl")
        batch_id_file = os.path.join(batch_file_dir, "batch_id.txt")
        if not os.path.isfile(batch_input_file) or os.path.isfile(batch_output_file) or os.path.isfile(batch_id_file):
            return

        batch_file = self.client.files.create(
            file=open(batch_input_file, 'rb'),
            purpose='batch',
        )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        with open(batch_id_file, 'w') as file:
            file.write(batch_job.id)

        logger.info(f"Batch submitted with ID '{batch_job.id}'")

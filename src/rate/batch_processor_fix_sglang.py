"""
Improved batch processing with better error handling and validation.

Vendored from YalaLab/rate@79b23df src/core/batch_processor.py (ECL 2.0). Changes:
- replace OpenAI batch-file workflow (upload → poll → download) with direct
  sglang chat-completions calls over a long-lived httpx client
- add `create_openai_client_with_authost` + `read_autohost_config` to discover
  the local sglang server via the user's `~/llmhost/` config files
- switch from stdlib logging to loguru; switch validator to ResultValidatorWithReasoning

# fields accepted by sglang
https://huggingface.co/tuandunghcmut/vlm_clone_2/blob/74f589365d1364e44cb5874a664f0142ae726c38/sglang/python/sglang/srt/openai_api/protocol.py#L148

# extra fields accepted by qwen
https://qwen.readthedocs.io/en/latest/deployment/sglang.html
# OpenAI Python SDK: extra_body={"chat_template_kwargs": {"enable_thinking": False}}

# testing fields
extra_body={"repetition_penalty": "wrong_type"} -> Error 400 wrong type
--log_level debug  for sglang server
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from packg.strings.formatters import dict_to_str_comma_equals
from packg.tqdmext import tqdm_max_ncols

from .exceptions import BatchProcessingError
from .validators_with_reasoning import ResultValidatorWithReasoning


def create_openai_client_with_authost(server_config: Dict) -> OpenAI:
    if server_config["autohost"]:
        host, port = read_autohost_config(server_config["autohost"])
        server_config["base_url"] = host
        server_config["port"] = port
    base_url = f"{server_config['base_url']}:{server_config['port']}/v1"
    http_client = httpx.Client(
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
    )
    return OpenAI(base_url=base_url, api_key="None", http_client=http_client)


def read_autohost_config(model_name: str) -> Tuple[str, int]:
    """Read host and port from the latest llmhost config file."""
    llmhost_dir = Path.home() / "llmhost"
    if not llmhost_dir.is_dir():
        raise FileNotFoundError(f"Directory {llmhost_dir} does not exist")
    pattern = f"{model_name}_hostname_*"
    matching_files = sorted(llmhost_dir.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(
            f"No hostname files found matching pattern '{pattern}' in {llmhost_dir}"
        )
    # Get the latest file (last in sorted list)
    latest_file = matching_files[-1]
    print(f"Reading host config from: {latest_file}")
    # Read host and port from file
    try:
        with open(latest_file, "r") as f:
            lines = f.read().strip().split("\n")
        if len(lines) < 2:
            raise ValueError(
                f"Config file {latest_file} must contain at least 2 lines (host and port)"
            )
        host = lines[0].strip()
        port = int(lines[1].strip())
        print(f"Found host: {host}, port: {port}")
        return host, port
    except (IOError, ValueError) as e:
        raise ValueError(f"Failed to read config from {latest_file}: {e}") from e


class BatchProcessorFixSglang:
    """Handles batch processing with improved error handling and validation."""

    def __init__(
        self, config: Dict, validator: ResultValidatorWithReasoning, verbose: bool = False
    ):
        self.config = config
        self.validator = validator
        # with these values the processing will retry after 4, 8, 16, ... seconds
        # so in total up to 1023 seconds (~17 minutes) before finally giving up
        self.max_retries = config.get("processing", {}).get("max_retries", 999)  # 7: 512s max
        self.retry_delay = config.get("processing", {}).get("retry_delay", 4)
        self.num_workers = int(config["processing"]["num-workers"])

        self.client = create_openai_client_with_authost(self.config["server"])
        self.verbose = verbose
        print(
            f"Using server at {self.client.base_url} with model config "
            f"{dict_to_str_comma_equals(self.config['model'])}"
        )

    def process_batch(
        self, requests: List[Tuple[str, str]], stage_name: str
    ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
        """Process a batch of requests with improved error handling and validation."""
        if not requests:
            return {}, {}

        logger.info(f"Processing batch of {len(requests)} requests for {stage_name}")

        for attempt in range(self.max_retries + 1):
            try:
                results = self._execute_requests_directly(requests, attempt)

                # Validate results
                validated_results, invalidated_results = self.validator.validate_batch_results(
                    results, requests, stage_name
                )

                logger.info(
                    f"Batch done: {len(validated_results)}/{len(requests)} successful, "
                    f"{len(invalidated_results)} invalidated"
                )
                return validated_results, invalidated_results

            except BatchProcessingError as e:
                logger.warning(f"Batch processing attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries:
                    # Exponential backoff: wait retry_delay * 2^attempt seconds
                    backoff_time = self.retry_delay * 2**attempt
                    logger.info(f"Retrying in {backoff_time} seconds (exponential backoff)...")
                    time.sleep(backoff_time)
                    if self.config["server"]["autohost"]:
                        # check if llm is running under a new autohost server now
                        logger.error("Autohost server failure detected, trying to reconnect")
                        self.client = create_openai_client_with_authost(self.config["server"])
                        logger.info(f"Using server at {self.client.base_url}")
                        time.sleep(1)
                    continue
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for batch")
                    raise e
        raise BatchProcessingError("Batch processing failed after all retries")

    def _execute_requests_directly(
        self, requests: List[Tuple[str, str]], attempt: int
    ) -> Dict[str, Tuple[str, str]]:
        """
        Execute each request directly against the chat endpoint. This avoids the batch
        file upload flow which is unsupported in sglang.
        """
        if not requests:
            return {}
        unique_requests_count = len(set(req_id for req_id, _ in requests))

        worker_count = max(1, min(self.num_workers, len(requests)))
        logger.debug(f"Direct execution attempt {attempt + 1} using {worker_count} worker(s)")

        results: Dict[str, Tuple[str, str]] = {}
        failures = 0

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_request = {
                executor.submit(self._execute_single_request, prompt): request_id
                for request_id, prompt in requests
            }

            for future in tqdm_max_ncols(
                as_completed(future_to_request),
                total=len(future_to_request),
                desc=f"Direct batch attempt {attempt + 1}",
                unit="req",
                bar_format="{desc}: {n_fmt}/{total_fmt} ({elapsed})",
                smoothing=0,
            ):
                request_id = future_to_request[future]
                try:
                    content, reasoning_content = future.result(timeout=60)
                    results[request_id] = content, reasoning_content
                except TimeoutError:
                    failures += 1
                    logger.warning(f"Request {request_id} timed out (60s) on attempt {attempt + 1}")
                    results[request_id] = "", ""
                except Exception as e:
                    failures += 1
                    logger.warning(f"Request {request_id} failed on attempt {attempt + 1}: {e}")
                    results[request_id] = "", ""

        if failures == len(requests):
            raise BatchProcessingError(
                f"All requests failed on attempt {attempt + 1}. Last error logged above."
            )

        non_empty_results = sum(
            1 for content, reasoning_content in results.values() if content.strip()
        )
        empty_results = len(results) - non_empty_results
        logger.info(
            f"Direct batch attempt {attempt + 1} finished: {len(requests)} input requests, "
            f"({unique_requests_count} unique), "
            f"{len(future_to_request)} futures, {len(results)} total results, "
            f"with {non_empty_results} non-empty, {empty_results} empty responses"
        )

        return results

    def _execute_single_request(self, prompt: str):
        """Send a single prompt to the chat endpoint and return normalized text."""
        response = self.client.chat.completions.create(
            model=self.config["model"]["name"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please output in plain text without any formatting.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config["model"]["temperature"],
            top_p=self.config["model"]["top_p"],
            max_tokens=self.config["model"]["max_tokens"],
            # extra_headers=  # header
            # extra_body={"repetition_penalty": "wrong_type"}
            # extra_query=  # get args
        )
        content, reasoning_content = self._extract_content_from_chat_response(response)
        return content, reasoning_content

    def _extract_content_from_chat_response(self, response: ChatCompletion):
        """Normalize chat completion responses from OpenAI or sglang-compatible servers."""
        choices: list[Choice] = response.choices
        assert len(choices) > 0, f"{len(choices)=} in LLM reponse."
        message: ChatCompletionMessage = choices[0].message
        content: str = str(message.content) if message.content is not None else ""
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning_content = str(reasoning_content) if reasoning_content is not None else ""
        if "</think>" in content:
            content = content.split("</think>")[-1]
            think_content, content = content.rsplit("</think>", 1)
            print(f"Removed thinking trace: '{think_content}'")

        return content.strip(), reasoning_content.strip()

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }

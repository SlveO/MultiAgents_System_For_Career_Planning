from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from project.core.brain_client import (
    BrainAuthError,
    BrainConfigError,
    BrainRateLimitError,
    BrainServerError,
    BrainTimeoutError,
    DeepSeekBrainClient,
)


class FakeHttpClient:
    def __init__(self, *, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.request_json = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, _url, *, headers, json):
        self.request_json = json
        if self.error:
            raise self.error
        return self.response


def build_client(api_key: str = "test-key") -> DeepSeekBrainClient:
    client = DeepSeekBrainClient.__new__(DeepSeekBrainClient)
    client.api_key = api_key
    client.base_url = "https://api.deepseek.com"
    client.default_model = "deepseek-v4-flash"
    client.timeout = 1.0
    return client


class TestDeepSeekBrainClient(unittest.TestCase):
    def test_plan_sends_frozen_model_and_non_thinking_payload(self) -> None:
        response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "plan"}}]},
        )
        fake_http = FakeHttpClient(response=response)

        with patch("project.core.brain_client.httpx.Client", return_value=fake_http):
            result = build_client().plan("career prompt")

        self.assertEqual(result, "plan")
        self.assertEqual(fake_http.request_json["model"], "deepseek-v4-flash")
        self.assertEqual(fake_http.request_json["thinking"], {"type": "disabled"})

    def test_missing_key_uses_typed_configuration_error(self) -> None:
        with self.assertRaises(BrainConfigError):
            build_client(api_key="").plan("career prompt")

    def test_status_codes_map_to_typed_errors(self) -> None:
        mappings = [
            (401, BrainAuthError),
            (429, BrainRateLimitError),
            (500, BrainServerError),
        ]
        for status_code, error_type in mappings:
            with self.subTest(status_code=status_code):
                with self.assertRaises(error_type):
                    DeepSeekBrainClient._raise_for_status(httpx.Response(status_code))

    def test_timeout_is_retryable_and_does_not_expose_request_data(self) -> None:
        request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
        fake_http = FakeHttpClient(error=httpx.ReadTimeout("timeout", request=request))

        with patch("project.core.brain_client.httpx.Client", return_value=fake_http):
            with self.assertRaises(BrainTimeoutError) as captured:
                build_client().plan("private prompt")

        self.assertTrue(captured.exception.retryable)
        self.assertNotIn("private prompt", str(captured.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

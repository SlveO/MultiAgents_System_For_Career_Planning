from __future__ import annotations

import unittest
import os
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
    def test_https_proxy_wins_over_invalid_all_proxy_without_mutation(self) -> None:
        env = {"HTTPS_PROXY": "http://proxy.example:8080", "ALL_PROXY": "socks://unused:1080"}
        with patch.dict(os.environ, env, clear=True):
            before = dict(os.environ)
            with patch("project.core.brain_client.httpx.Client") as constructor:
                build_client()._http_client()
            self.assertEqual(constructor.call_args.kwargs["proxy"], env["HTTPS_PROXY"])
            self.assertFalse(constructor.call_args.kwargs["trust_env"])
            self.assertIs(constructor.call_args.kwargs["verify"], True)
            self.assertEqual(dict(os.environ), before)

    def test_no_proxy_bypasses_invalid_all_proxy(self) -> None:
        with patch.dict(os.environ, {"ALL_PROXY": "socks://unused:1080", "NO_PROXY": "api.deepseek.com"}, clear=True):
            with patch("project.core.brain_client.httpx.Client") as constructor:
                build_client()._http_client()
            self.assertIsNone(constructor.call_args.kwargs["proxy"])

    def test_invalid_effective_proxy_is_typed_and_redacted(self) -> None:
        with patch.dict(os.environ, {"ALL_PROXY": "socks://private:secret@host:1080"}, clear=True):
            with self.assertRaises(BrainConfigError) as captured:
                build_client()._http_client()
            self.assertNotIn("secret", str(captured.exception))
            self.assertFalse(captured.exception.retryable)

    def test_lowercase_proxy_and_direct_connection(self) -> None:
        for env, expected in [({}, None), ({"https_proxy": "http://lower:8080", "HTTPS_PROXY": "http://upper:8080"}, "http://lower:8080")]:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with patch("project.core.brain_client.httpx.Client") as constructor:
                    build_client()._http_client()
                self.assertEqual(constructor.call_args.kwargs["proxy"], expected)

    def test_ca_setting_is_preserved(self) -> None:
        with patch.dict(os.environ, {"SSL_CERT_FILE": "test-ca.pem"}, clear=True):
            with patch("project.core.brain_client.ssl.create_default_context") as context:
                with patch("project.core.brain_client.httpx.Client") as constructor:
                    build_client()._http_client()
                context.assert_called_once_with(cafile="test-ca.pem")
                self.assertIs(constructor.call_args.kwargs["verify"], context.return_value)

    def test_stream_uses_same_proxy_client(self) -> None:
        with patch.object(DeepSeekBrainClient, "_http_client", side_effect=BrainConfigError("configuration")) as factory:
            with self.assertRaises(BrainConfigError):
                list(build_client().plan_stream("prompt"))
            factory.assert_called_once()

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

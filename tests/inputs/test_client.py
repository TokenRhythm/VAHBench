"""Keep GLM's reasoning from silently consuming the checklist output budget."""
import io
import json
import unittest
from unittest.mock import patch

import stage1_inputs.client as llm


class GLMClientTests(unittest.TestCase):
    def test_request_budget_and_audit(self):
        reply = {"model": "z-ai/glm-5.3", "choices": [
            {"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]}
        log = llm.CallLog()
        with patch.object(llm, "resolve_provider", return_value=(
                "openrouter", "https://example.invalid/chat", "test-key")), \
                patch.object(llm.urllib.request, "urlopen", return_value=io.BytesIO(
                    json.dumps(reply).encode())) as send:
            call = llm.make_call_fn("z-ai/glm-5.3", log=log)
            self.assertEqual(call("extract checklist"), '{"ok":true}')
        request = json.loads(send.call_args.args[0].data)
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["max_tokens"], 32768)
        self.assertEqual(log.reasoning_effort, "low")
        self.assertEqual(log.model_served, "z-ai/glm-5.3")
        self.assertEqual(log.errors, [])

    def test_other_models_keep_their_provider_defaults(self):
        reply = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        with patch.object(llm, "resolve_provider", return_value=(
                "openrouter", "https://example.invalid/chat", "test-key")), \
                patch.object(llm.urllib.request, "urlopen", return_value=io.BytesIO(
                    json.dumps(reply).encode())) as send:
            llm.make_call_fn("another/model")("extract checklist")
        self.assertNotIn("reasoning", json.loads(send.call_args.args[0].data))

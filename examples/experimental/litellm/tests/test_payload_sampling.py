from switchyard_litellm.client import _payload


def test_sampling_temperature_and_top_p_forwarded() -> None:
    request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "sampling": {"temperature": 0.5, "top_p": 0.9},
    }
    payload = _payload(request, "strong")
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9

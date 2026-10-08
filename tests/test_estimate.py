import base64

from budget_guard.estimate import BLOB_TOKENS, estimate_usage

MSG = [{"role": "user", "content": "hello world " * 25}]  # 300 characters


def test_input_from_text_and_output_from_max_tokens():
    usage = estimate_usage({"model": "m", "messages": MSG, "max_tokens": 500}, default_output_tokens=1024)
    assert usage.input_tokens == (4 + 300) // 3 + 1  # "user" + the content, 3 characters a token, rounded up
    assert usage.output_tokens == 500


def test_each_max_tokens_spelling_and_the_default():
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        assert estimate_usage({key: 77}, default_output_tokens=1024).output_tokens == 77
    assert estimate_usage({}, default_output_tokens=1024).output_tokens == 1024
    assert estimate_usage({"max_tokens": 10, "n": 3}, default_output_tokens=1024).output_tokens == 30


def test_inline_images_count_as_a_fixed_size_not_their_base64_length():
    data = base64.b64encode(b"\x89PNG" * 100_000).decode()
    message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        ],
    }
    usage = estimate_usage({"messages": [message]}, default_output_tokens=0)
    assert BLOB_TOKENS * 2 <= usage.input_tokens < BLOB_TOKENS * 2 + 50


def test_sdk_objects_and_classes_are_handled():
    class Model:
        def model_dump(self):
            return {"content": "abc"}

    usage = estimate_usage({"messages": [Model()], "response_format": Model}, default_output_tokens=0)
    assert usage.input_tokens == 1

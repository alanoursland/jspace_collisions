from jspace.models.wrapper import ModelWrapper


class FakeChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "Question?"}]
        assert tokenize is False
        assert add_generation_prompt is True
        return "<user>Question?</user><assistant>"


def test_raw_prompt_format_is_identity():
    wrapper = object.__new__(ModelWrapper)
    wrapper.prompt_format = "raw"
    assert wrapper.format_prompt("Question?") == "Question?"


def test_chat_prompt_format_uses_native_template():
    wrapper = object.__new__(ModelWrapper)
    wrapper.prompt_format = "chat"
    wrapper.tokenizer = FakeChatTokenizer()
    assert (
        wrapper.format_prompt("Question?")
        == "<user>Question?</user><assistant>"
    )

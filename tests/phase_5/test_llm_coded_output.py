from pathlib import Path

import pytest

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.llm import FewShotLlmBaseline, _chat_template_options


def _coded_model() -> FewShotLlmBaseline:
    model = FewShotLlmBaseline(
        cache_folder=Path("models/huggingface"), label_encoding="codes"
    )
    for field in ANNOTATION_FIELDS:
        model._vocabulary[field] = ['["not_stated"]', '["explicit"]']
        model._codebook[field] = {
            '["not_stated"]': "c0",
            '["explicit"]': "c1",
        }
    return model


def test_coded_output_parser_maps_short_codes_to_signatures() -> None:
    model = _coded_model()
    output = "{" + ",".join(f'\"{field}\":\"c1\"' for field in ANNOTATION_FIELDS) + "}"

    parsed = model._parse(output)

    assert all(parsed[field] == ("explicit",) for field in ANNOTATION_FIELDS)
    assert model.invalid_field_count == 0


def test_coded_output_parser_accepts_numeric_code_and_falls_back() -> None:
    model = _coded_model()
    fields = list(ANNOTATION_FIELDS)
    output = "{" + f'\"{fields[0]}\":1' + "}"

    parsed = model._parse(output)

    assert parsed[fields[0]] == ("explicit",)
    assert all(parsed[field] == ("not_stated",) for field in fields[1:])
    assert model.invalid_field_count == len(fields) - 1


def test_unknown_label_encoding_is_rejected() -> None:
    with pytest.raises(ValueError, match="label_encoding"):
        FewShotLlmBaseline(
            cache_folder=Path("models/huggingface"), label_encoding="unknown"
        )


def test_qwen3_uses_documented_non_thinking_mode() -> None:
    assert _chat_template_options("Qwen/Qwen3-0.6B") == {
        "enable_thinking": False
    }
    assert _chat_template_options("Qwen/Qwen3-1.7B") == {
        "enable_thinking": False
    }


def test_other_model_families_keep_native_chat_template() -> None:
    assert _chat_template_options("Qwen/Qwen2.5-1.5B-Instruct") == {}
    assert _chat_template_options("utter-project/EuroLLM-1.7B-Instruct") == {}

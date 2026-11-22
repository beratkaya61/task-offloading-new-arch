from pathlib import Path
from typing import Any, cast

import pytest

from task_offloading.semantic.finetune import FineTunedEncoderBaseline


def test_gradual_strategy_configuration_is_accepted() -> None:
    FineTunedEncoderBaseline(
        cache_folder=Path("models/huggingface"), training_strategy="gradual"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"training_strategy": "unknown"}, "training_strategy"),
        (
            {"training_strategy": "gradual", "epochs": 1, "warmup_epochs": 1},
            "warmup_epochs",
        ),
        ({"unfreeze_last_layers": 0}, "unfreeze_last_layers"),
    ],
)
def test_invalid_improvement_configuration_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FineTunedEncoderBaseline(
            cache_folder=Path("models/huggingface"), **cast(Any, kwargs)
        )

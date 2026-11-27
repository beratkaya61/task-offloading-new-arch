# ruff: noqa: E501, RUF001
from __future__ import annotations

from typing import Any

from task_offloading.semantic.natural_rewrite import clean_output, rewrite_issues


def _label(**explicit: dict[str, Any]) -> dict[str, Any]:
    fields = ("domain", "latency", "privacy", "execution_policy", "reliability", "accuracy", "energy_priority", "divisibility")
    label: dict[str, Any] = {f: {"status": "not_stated"} for f in fields}
    label["domain"] = {"status": "explicit", "value": "agriculture"}
    for name, value in explicit.items():
        label[name] = {"status": "explicit", **value}
    return label


SOURCE = "Akıllı sulama vanası, tarımsal sensör verisi üzerinde su akışını planlama görevini yürütür. Sonuç en geç 50 ms içinde hazır olmalıdır."


def test_faithful_rewrite_is_accepted() -> None:
    rewrite = "Akıllı sulama vanası, tarımsal sensörlerden gelen verilere göre su akışını kontrol eder. İşlem sonucu 50 ms içinde hazır hale gelmelidir."
    assert rewrite_issues(SOURCE, rewrite, _label(latency={"class": "real_time", "max_ms": 50.0})) == []


def test_missing_number_is_rejected() -> None:
    rewrite = "Akıllı sulama vanası tarımsal verilerle su akışını planlar ve sonucu çok hızlı üretir."
    issues = rewrite_issues(SOURCE, rewrite, _label(latency={"class": "real_time", "max_ms": 50.0}))
    assert any(i.startswith("numbers_missing") for i in issues)


def test_added_constraint_is_rejected() -> None:
    rewrite = "Akıllı sulama vanası, tarımsal verilere göre su akışını kontrol eder. İşlem sonucu 50 ms içinde hazır olmalıdır. Güvenilir ve doğruluğu yüksek bir sistemdir."
    issues = rewrite_issues(SOURCE, rewrite, _label(latency={"class": "real_time", "max_ms": 50.0}))
    assert "added_accuracy" in issues
    assert "added_reliability" in issues


def test_device_name_words_are_not_counted_as_added_constraints() -> None:
    source = "Fabrika enerji sayacı, üretim telemetrisi üzerinde tüketim anomalisini bulma görevini yürütür."
    rewrite = "Fabrikadaki enerji sayacı, üretim telemetrisini inceleyerek olağandışı tüketimi tespit eder."
    assert rewrite_issues(source, rewrite, _label()) == []


def test_privacy_meaning_shift_is_rejected() -> None:
    source = "Akıllı sulama vanası, tarımsal sensör verisi üzerinde su akışını planlama görevini yürütür. İşlenen ham kayıt kısıtlı veridir."
    rewrite = "Akıllı sulama vanası, tarımsal verilerle su akışını optimize eder. İşlenen veri, sınırlı miktarda ham kayıt içerir."
    assert "privacy_marker" in rewrite_issues(source, rewrite, _label(privacy={"data_class": "restricted"}))


def test_clean_output_strips_wrappers() -> None:
    assert clean_output('Yeniden yazılmış metin: "Merhaba  dünya."') == "Merhaba dünya."

"""Tests for the EdgeTTS provider tool."""

import os

import pytest


def test_edge_tts_contract():
    from tools.audio.edge_tts import EdgeTTS

    tool = EdgeTTS()
    assert tool.name == "edge_tts"
    assert tool.capability == "tts"
    assert tool.provider == "edge"
    assert tool.runtime.value == "api"
    assert tool.estimate_cost({"text": "hi"}) == 0.0
    assert (
        tool.input_schema["properties"]["voice"]["default"]
        == "zh-CN-XiaoxiaoNeural"
    )
    info = tool.get_info()
    assert info["setup_offer"]["kind"] == "pip_install"
    assert info["dependencies"] == ["python:edge_tts"]
    # edge_tts is installed (added to requirements.txt), so status is available.
    assert tool.get_status().value == "available"


def test_edge_tts_registry_discovery():
    from tools.tool_registry import registry

    registry.discover()
    assert "edge_tts" in registry.list_all()
    assert any(
        t.provider == "edge"
        for t in registry.get_by_capability("tts")
        if t.name == "edge_tts"
    )

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


class _FakeCommunicate:
    """Stand-in for edge_tts.Communicate; writes a fake mp3 body."""

    def __init__(self, text, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    async def save(self, path):
        from pathlib import Path

        Path(path).write_bytes(b"ID3\x03\x00\x00\x00fake-mp3-body")


def test_edge_tts_generate_mocked(monkeypatch, tmp_path):
    import edge_tts
    import tools.analysis.audio_probe as ap
    from tools.audio.edge_tts import EdgeTTS

    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    monkeypatch.setattr(ap, "probe_duration", lambda p: 1.23)

    tool = EdgeTTS()
    out = tmp_path / "out.mp3"
    result = tool.execute(
        {
            "text": "你好，世界",
            "voice": "zh-CN-XiaoxiaoNeural",
            "speed": 1.2,  # selector-style numeric -> "+20%"
            "pitch": 5,  # selector-style numeric -> "+5Hz"
            "output_path": str(out),
        }
    )
    assert result.success
    assert result.artifacts == [str(out)]
    assert result.data["format"] == "mp3"
    assert result.data["voice"] == "zh-CN-XiaoxiaoNeural"
    assert result.data["rate"] == "+20%"
    assert result.data["volume"] == "+0%"
    assert result.data["pitch"] == "+5Hz"
    assert result.data["audio_duration_seconds"] == 1.23
    assert result.data["text_length"] == len("你好，世界")
    assert result.cost_usd == 0.0
    assert result.model == "zh-CN-XiaoxiaoNeural"


def test_edge_tts_string_params_pass_through(monkeypatch, tmp_path):
    import edge_tts
    import tools.analysis.audio_probe as ap
    from tools.audio.edge_tts import EdgeTTS

    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    monkeypatch.setattr(ap, "probe_duration", lambda p: 0.5)

    tool = EdgeTTS()
    out = tmp_path / "str.mp3"
    result = tool.execute(
        {
            "text": "hello",
            "rate": "-10%",
            "volume": "-20%",
            "pitch": "+5Hz",
            "output_path": str(out),
        }
    )
    assert result.success
    assert result.data["rate"] == "-10%"
    assert result.data["volume"] == "-20%"
    assert result.data["pitch"] == "+5Hz"


def test_edge_tts_empty_text():
    from tools.audio.edge_tts import EdgeTTS

    tool = EdgeTTS()
    result = tool.execute({"text": "   "})
    assert not result.success
    assert "empty" in result.error.lower()


def test_paid_tts_fallbacks_prefer_edge_before_piper():
    from tools.audio.dashscope_tts import DashscopeTTS
    from tools.audio.doubao_tts import DoubaoTTS
    from tools.audio.elevenlabs_tts import ElevenLabsTTS
    from tools.audio.google_tts import GoogleTTS
    from tools.audio.openai_tts import OpenAITTS

    for cls in (OpenAITTS, ElevenLabsTTS, GoogleTTS, DashscopeTTS, DoubaoTTS):
        ft = cls.fallback_tools
        assert "edge_tts" in ft, f"{cls.name} missing edge_tts fallback"
        assert "piper_tts" in ft, f"{cls.name} missing piper_tts fallback"
        assert ft.index("edge_tts") < ft.index("piper_tts"), (
            f"{cls.name}: edge_tts must come before piper_tts"
        )


@pytest.mark.skipif(
    not os.environ.get("OPENMONTAGE_NETWORK_TESTS"),
    reason="set OPENMONTAGE_NETWORK_TESTS=1 to run network TTS test",
)
def test_edge_tts_network_generate(tmp_path):
    from tools.audio.edge_tts import EdgeTTS

    tool = EdgeTTS()
    out = tmp_path / "net.mp3"
    result = tool.execute({"text": "你好，世界", "output_path": str(out)})
    assert result.success, result.error
    assert out.exists()
    assert (
        result.data["audio_duration_seconds"]
        and result.data["audio_duration_seconds"] > 0
    )

import json
import logging
from rag.logging_config import JsonFormatter


def test_json_formatter_includes_event():
    record = logging.LogRecord(
    name="rag.test",
    level=logging.INFO,
    pathname="",
    lineno=0,
    msg="Test message",
    args=(),
    exc_info=None,
    )
    record.event = "test_event"
    formatter = JsonFormatter()
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["event"] == "test_event"
    assert data["message"] == "Test message"
    assert data["level"] == "INFO"
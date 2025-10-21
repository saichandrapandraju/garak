from garak.attempt import Attempt, Message
import garak._plugins
import garak.detectors.notes
import pytest

@pytest.fixture
def detector():
    """Fixture to load the MatchSuccessBool detector"""
    return garak._plugins.load_plugin("detectors.notes.MatchSuccessBool")

def test_notes_match_success_bool_load(detector):
    """Test that the detector can be loaded"""
    
    assert isinstance(detector, garak.detectors.notes.MatchSuccessBool)

def test_notes_match_success_bool_true(detector):
    """Test that detector returns 1.0 when success is True in notes"""
    
    attempt = Attempt(prompt="test")
    attempt.notes = {"success": True}
    attempt.outputs = [Message("some output")]
    
    results = detector.detect(attempt)
    assert results == [1.0]


def test_notes_match_success_bool_false(detector):
    """Test that detector returns 0.0 when success is False in notes"""
    
    attempt = Attempt(prompt="test")
    attempt.notes = {"success": False}
    attempt.outputs = [Message("some output")]
    
    results = detector.detect(attempt)
    assert results == [0.0]


def test_notes_match_success_bool_missing(detector):
    """Test that detector returns 0.0 when success is not present in notes"""
    
    attempt = Attempt(prompt="test")
    attempt.notes = {"other_field": "value"}
    attempt.outputs = [Message("some output")]
    
    results = detector.detect(attempt)
    assert results == [0.0]


def test_notes_match_success_bool_none(detector):
    """Test that detector returns 0.0 when success is None in notes"""
    
    attempt = Attempt(prompt="test")
    attempt.notes = {"success": None}
    attempt.outputs = [Message("some output")]
    
    results = detector.detect(attempt)
    assert results == [0.0]


def test_notes_match_success_bool_empty_notes(detector):
    """Test that detector returns 0.0 when notes is empty"""
    
    attempt = Attempt(prompt="test")
    attempt.notes = {}
    attempt.outputs = [Message("some output")]
    
    results = detector.detect(attempt)
    assert results == [0.0]


def test_notes_match_success_bool_no_notes(detector):
    """Test that detector returns 0.0 when notes is None"""
    
    attempt = Attempt(prompt="test")
    attempt.notes = None
    attempt.outputs = [Message("some output")]
    
    results = detector.detect(attempt)
    assert results == [0.0]


def test_notes_match_success_bool_truthy_values(detector):
    """Test that detector only returns 1.0 for actual boolean True, not truthy values"""
    
    
    # Test various truthy values that should NOT trigger 1.0
    truthy_values = ["yes", 1, "true", [1, 2, 3], {"key": "value"}]
    
    for value in truthy_values:
        attempt = Attempt(prompt="test")
        attempt.notes = {"success": value}
        attempt.outputs = [Message("some output")]
        
        results = detector.detect(attempt)
        assert results == [0.0], f"Should return 0.0 for truthy value: {value}"
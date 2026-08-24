from __future__ import annotations

import os

from train import ProfileStepNotifier


def test_profile_step_notifier_streams_bounded_steps_and_syncs_boundary(
    monkeypatch,
) -> None:
    step_read, step_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    monkeypatch.setenv("TRAINVM_PROFILE_STEP_FD", str(step_write))
    monkeypatch.setenv("TRAINVM_PROFILE_ACK_FD", str(acknowledgement_read))
    monkeypatch.setenv("TRAINVM_PROFILE_SYNC_COUNTS", "2")
    monkeypatch.setenv("TRAINVM_PROFILE_TOTAL_COUNT", "2")
    notifier = ProfileStepNotifier()

    notifier.step(41)
    os.write(acknowledgement_write, b"1")
    notifier.step(42)
    notifier.step(43)

    os.close(step_write)
    assert os.read(step_read, 1024) == b"41\n42\n"
    for descriptor in (
        step_read,
        acknowledgement_read,
        acknowledgement_write,
    ):
        os.close(descriptor)


def test_profile_step_notifier_is_free_when_bridge_is_absent(monkeypatch) -> None:
    for name in (
        "TRAINVM_PROFILE_STEP_FD",
        "TRAINVM_PROFILE_ACK_FD",
        "TRAINVM_PROFILE_SYNC_COUNTS",
        "TRAINVM_PROFILE_TOTAL_COUNT",
    ):
        monkeypatch.delenv(name, raising=False)
    ProfileStepNotifier().step(1)

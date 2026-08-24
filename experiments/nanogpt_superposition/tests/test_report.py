from report import _recovery_cost


def test_recovery_cost_starts_at_ordinary_recovery_phase() -> None:
    run = {
        "representation": "fixed",
        "rows": [
            {
                "phase": "mixed",
                "mode": "ordinary",
                "elapsed_seconds": 10.0,
                "source_bytes": 100,
                "validation_bits_per_byte": 2.0,
            },
            {
                "phase": "ordinary",
                "mode": "ordinary",
                "elapsed_seconds": 20.0,
                "source_bytes": 200,
                "validation_bits_per_byte": 1.8,
            },
            {
                "phase": "ordinary",
                "mode": "ordinary",
                "elapsed_seconds": 30.0,
                "source_bytes": 300,
                "validation_bits_per_byte": 1.4,
            },
        ],
    }

    assert _recovery_cost(run, 1.5) == {
        "status": "recovered",
        "seconds": 10.0,
        "source_bytes": 100,
        "target_bits_per_byte": 1.5,
    }


def test_recovery_cost_distinguishes_completed_phase_from_missing_phase() -> None:
    run = {
        "representation": "fixed",
        "rows": [
            {
                "phase": "ordinary",
                "mode": "ordinary",
                "elapsed_seconds": 20.0,
                "source_bytes": 200,
                "validation_bits_per_byte": 1.8,
            },
            {
                "phase": "ordinary",
                "mode": "ordinary",
                "elapsed_seconds": 30.0,
                "source_bytes": 300,
                "validation_bits_per_byte": 1.7,
            },
        ],
    }

    assert _recovery_cost(run, 1.5) == {
        "status": "phase_complete_target_not_reached",
        "seconds": 10.0,
        "source_bytes": 100,
        "target_bits_per_byte": 1.5,
    }

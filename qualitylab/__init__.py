"""NEXORA voice quality lab.

Builds a reproducible test corpus, runs the nexora-audio pipeline over it, measures
noise, echo, cutting, turn-taking and packet-loss behaviour, and checks the results
against quality gates (``gates.toml``).

Run from the ``nexora-audio`` directory::

    python -m qualitylab fetch                 # download pinned public test audio
    python -m qualitylab run --profile fast    # CI gate set (a few minutes)
    python -m qualitylab run --profile full    # nightly / release set

The lab is a development tool and is not part of the ``nexora_audio`` wheel.
"""

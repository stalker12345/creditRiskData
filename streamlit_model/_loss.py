"""Compatibility shim for serialized scikit-learn loss objects.

Older skops artifacts can reference the internal module name ``_loss``.
Re-export the current scikit-learn implementation so model loading keeps
working across environments.
"""

from sklearn._loss._loss import *  # noqa: F403

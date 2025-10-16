"""
SPMA Environment Package

This package contains environment implementations for SPMA ML training:
- Stub environment for development and testing
- ns-3 bridge environment for real network simulation (future integration)

The stub environment provides a realistic simulation of SPMA dynamics
for DQN training without requiring ns-3 integration.
"""

from .spma_stub_env import SPMASubEnvironment
from .ns3_bridge_env import NS3BridgeEnvironment

__all__ = ['SPMASubEnvironment', 'NS3BridgeEnvironment']

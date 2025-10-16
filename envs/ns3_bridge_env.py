"""
ns-3 Bridge Environment for SPMA ML integration.

This module provides the interface for connecting SPMA ML models to ns-3
network simulator. It handles communication, state synchronization, and
action execution between the ML models and ns-3 simulation.

TODO: Implement actual ns-3 integration when ns-3 simulator is available.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, Tuple, Optional
import socket
import json
import time
import logging


class NS3BridgeEnvironment(gym.Env):
    """
    ns-3 Bridge Environment for real network simulation.
    
    This environment connects to an ns-3 simulation running the SPMA protocol
    and provides real-time state observations and action execution capabilities.
    
    Architecture:
    - TCP socket communication with ns-3 simulator
    - JSON-based message protocol for state/action exchange
    - Real-time synchronization with simulation timesteps
    - Fallback to stub environment if ns-3 is unavailable
    
    State space: Real COS data from ns-3 simulation
    Action space: Discrete(5) - backoff steps {-2, -1, 0, +1, +2}
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: float = 5.0,
        cos_dim: int = 14,
        state_dim: int = 10,
        max_episode_steps: int = 1000,
        fallback_to_stub: bool = True
    ):
        super().__init__()
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.cos_dim = cos_dim
        self.state_dim = state_dim
        self.max_episode_steps = max_episode_steps
        self.fallback_to_stub = fallback_to_stub
        
        # Action space: backoff steps {-2, -1, 0, +1, +2}
        self.action_space = spaces.Discrete(5)
        
        # State space: continuous values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        
        # Connection state
        self.socket = None
        self.connected = False
        self.current_step = 0
        self.episode_id = None
        
        # Fallback environment
        self.stub_env = None
        if self.fallback_to_stub:
            from .spma_stub_env import SPMASubEnvironment
            self.stub_env = SPMASubEnvironment(
                cos_dim=cos_dim,
                state_dim=state_dim,
                max_episode_steps=max_episode_steps
            )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _connect_to_ns3(self) -> bool:
        """
        Establish connection to ns-3 simulator.
        
        Returns:
            success: Whether connection was successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info(f"Connected to ns-3 simulator at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to connect to ns-3: {e}")
            self.connected = False
            return False
    
    def _disconnect_from_ns3(self):
        """Disconnect from ns-3 simulator."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def _send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send message to ns-3 simulator.
        
        Args:
            message: Message dictionary to send
        
        Returns:
            success: Whether message was sent successfully
        """
        if not self.connected:
            return False
        
        try:
            message_json = json.dumps(message) + '\n'
            self.socket.sendall(message_json.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self.connected = False
            return False
    
    def _receive_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive message from ns-3 simulator.
        
        Returns:
            message: Received message dictionary, or None if failed
        """
        if not self.connected:
            return None
        
        try:
            data = self.socket.recv(4096).decode('utf-8')
            if not data:
                self.connected = False
                return None
            
            # Parse JSON message
            lines = data.strip().split('\n')
            for line in lines:
                if line:
                    return json.loads(line)
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            self.connected = False
            return None
    
    def reset(self) -> np.ndarray:
        """Reset environment and start new episode."""
        self.current_step = 0
        
        # Try to connect to ns-3
        if not self.connected:
            if not self._connect_to_ns3():
                if self.fallback_to_stub:
                    self.logger.info("Using stub environment as fallback")
                    return self.stub_env.reset()
                else:
                    raise RuntimeError("Cannot connect to ns-3 and fallback disabled")
        
        # Send reset message to ns-3
        reset_message = {
            'type': 'reset',
            'episode_id': int(time.time()),
            'config': {
                'cos_dim': self.cos_dim,
                'max_episode_steps': self.max_episode_steps
            }
        }
        
        if not self._send_message(reset_message):
            if self.fallback_to_stub:
                self.logger.info("ns-3 communication failed, using stub environment")
                return self.stub_env.reset()
            else:
                raise RuntimeError("Failed to communicate with ns-3")
        
        # Wait for initial state
        initial_state = self._receive_message()
        if initial_state is None:
            if self.fallback_to_stub:
                return self.stub_env.reset()
            else:
                raise RuntimeError("Failed to receive initial state from ns-3")
        
        self.episode_id = initial_state.get('episode_id')
        return self._parse_observation(initial_state)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Backoff action (0-4 mapping to {-2, -1, 0, +1, +2})
        
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Convert action to backoff step
        backoff_step = action - 2  # Map {0,1,2,3,4} to {-2,-1,0,+1,+2}
        
        # Send action to ns-3
        action_message = {
            'type': 'action',
            'episode_id': self.episode_id,
            'step': self.current_step,
            'action': backoff_step
        }
        
        if not self.connected or not self._send_message(action_message):
            if self.fallback_to_stub:
                return self.stub_env.step(action)
            else:
                raise RuntimeError("Failed to send action to ns-3")
        
        # Receive next state from ns-3
        next_state = self._receive_message()
        if next_state is None:
            if self.fallback_to_stub:
                return self.stub_env.step(action)
            else:
                raise RuntimeError("Failed to receive next state from ns-3")
        
        # Parse response
        observation = self._parse_observation(next_state)
        reward = next_state.get('reward', 0.0)
        done = next_state.get('done', False) or self.current_step >= self.max_episode_steps
        info = next_state.get('info', {})
        
        return observation, reward, done, info
    
    def _parse_observation(self, state_message: Dict[str, Any]) -> np.ndarray:
        """
        Parse observation from ns-3 state message.
        
        Args:
            state_message: State message from ns-3
        
        Returns:
            observation: Parsed state observation
        """
        # Extract COS data
        cos_data = state_message.get('cos', np.zeros(self.cos_dim))
        if len(cos_data) != self.cos_dim:
            cos_data = np.zeros(self.cos_dim)
        
        # Extract additional state components
        capacity = state_message.get('capacity', 0.0)
        priority = state_message.get('priority', 0.5)
        recent_reward = state_message.get('recent_reward', 0.0)
        
        # Combine into observation vector
        obs_components = []
        
        # COS features (first 6 channels)
        obs_components.extend(cos_data[:6])
        
        # Capacity usage
        obs_components.append(capacity)
        
        # Recent reward
        obs_components.append(recent_reward)
        
        # Priority level
        obs_components.append(priority)
        
        # Step information (normalized)
        obs_components.append(self.current_step / self.max_episode_steps)
        
        # Ensure we have exactly state_dim components
        while len(obs_components) < self.state_dim:
            obs_components.append(0.0)
        
        obs = np.array(obs_components[:self.state_dim], dtype=np.float32)
        
        # Ensure observation is in valid range
        return np.clip(obs, 0.0, 1.0)
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if self.connected:
            print(f"Connected to ns-3 simulator")
            print(f"Episode ID: {self.episode_id}")
            print(f"Step: {self.current_step}")
        else:
            print("Using stub environment (ns-3 not connected)")
        
        if self.fallback_to_stub and self.stub_env:
            self.stub_env.render(mode)
    
    def close(self):
        """Clean up environment resources."""
        if self.connected:
            # Send close message
            close_message = {
                'type': 'close',
                'episode_id': self.episode_id
            }
            self._send_message(close_message)
        
        self._disconnect_from_ns3()
        
        if self.stub_env:
            self.stub_env.close()
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if self.stub_env:
            self.stub_env.seed(seed)
        return [seed]


def create_ns3_bridge_env(config: Dict[str, Any]) -> NS3BridgeEnvironment:
    """Create ns-3 bridge environment from configuration."""
    env_config = config.get('environment', {})
    ns3_config = env_config.get('ns3_bridge', {})
    
    return NS3BridgeEnvironment(
        host=ns3_config.get('host', 'localhost'),
        port=ns3_config.get('port', 8080),
        timeout=ns3_config.get('timeout', 5.0),
        cos_dim=config.get('tcn', {}).get('input_dim', 14) - config.get('tcn', {}).get('n_prio', 2),
        state_dim=config.get('dqn', {}).get('state_dim', 10),
        max_episode_steps=env_config.get('max_episode_steps', 1000),
        fallback_to_stub=ns3_config.get('fallback_to_stub', True)
    )


# TODO: Implement ns-3 integration protocol
"""
ns-3 Integration Protocol:

1. Connection Setup:
   - TCP socket connection to ns-3 simulator
   - Handshake with simulation parameters
   - Episode initialization

2. Message Format:
   {
       "type": "reset|action|close",
       "episode_id": 12345,
       "step": 100,
       "action": -1,  // backoff step
       "cos": [0.1, 0.2, ...],  // channel occupancy
       "capacity": 0.75,
       "reward": 0.8,
       "done": false,
       "info": {...}
   }

3. State Synchronization:
   - Real-time COS data from ns-3
   - Capacity utilization metrics
   - Reward computation in ns-3
   - Episode termination conditions

4. Action Execution:
   - Send backoff actions to ns-3
   - Wait for simulation step completion
   - Receive updated state and rewards

5. Error Handling:
   - Connection failure fallback to stub
   - Timeout handling
   - Message parsing errors
   - Simulation crash recovery
"""

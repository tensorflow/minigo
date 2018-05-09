# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Go player interfaces."""

from abc import ABC, abstractmethod


class PlayerInterface(ABC):
    """Interface for a basic Go player."""

    @abstractmethod
    def get_position(self):
        """Get the current position.

        Returns:
          A go.Position instance.
        """
        pass

    @abstractmethod
    def get_result_string(self):
        """Get the result as a string.

        Returns:
          The result as a string, e.g. B+R, W+1.5.
        """
        pass

    @abstractmethod
    def initialize_game(self, position=None):
        """Initializes a new game.

        Args:
          position: the board position to copy for the initial game state. If
                    None, an empty board state is used for the initial position.
        """
        pass

    @abstractmethod
    def suggest_move(self, position):
        """Suggests a move to play from the given position.

        Args:
          position: the current board position.

        Returns:
          The players's best guess as the best move to play.
        """
        pass

    @abstractmethod
    def play_move(self, c):
        """Play the given move.

        Args:
          c: the move to play as a Minigo coordinate (see coords.py).

        Returns:
          True if the move was successfully played.
          False if the requested move is illegal.
        """
        pass

    @abstractmethod
    def should_resign(self):
        """Should the current player resign?

        Returns:
          True if the player thinks the current player is doing so badly they
          had better just give up.
        """
        pass

    @abstractmethod
    def to_sgf(self, use_comments=True):
        """Format the game history as SGF.

        Args:
          use_comments: True to add debug info as a comment to the move nodes.

        Returns:
          A formatted SGF string
        """
        pass

    @abstractmethod
    def set_result(self, winner, was_resign):
        """Sets the game result.

        Args:
          winner: +1 for a black win, -1 a white win.
          was_resign: True if the win was by resignation.
        """
        pass


class MCTSPlayerInterface(PlayerInterface):
    """Interface for a MCTS-based Go player."""

    @abstractmethod
    def get_root(self):
        """Get the current root node.

        Returns:
          The current MCTSNode root of the search tree.
        """
        pass

    @abstractmethod
    def tree_search(self, parallel_readouts=None):
        """Performs one tree search step.

        Each tree search step may potentially expand multiple leaves of the
        game tree, depending on parallel_readouts.

        Args:
          parallel_readouts: number of leaves to expand in parallel. If None,
          the number of parallel readouts, the player is free to choose a
          sensible default.

        Returns:
          A list of the newly expanded leaves.
        """
        pass

    @abstractmethod
    def get_num_readouts(self):
        """Get the number of readouts.

        Returns:
          The number of readouts.
        """
        pass

    @abstractmethod
    def set_num_readouts(self, readouts):
        """Set the number of readouts.

        Args:
          readouts: the number of readouts.
        """
        pass

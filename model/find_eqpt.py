"""Abstract base class and tools for finding equilibrium operating points.

Provides ``OperatingPoint``, a structured, python-control compatible ABC that
allows computing the steady-state equilibrium state x* and input u* such that
dx/dt = 0 for any plant model.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import control as ct
import numpy as np


class OperatingPoint(ABC):
    """Abstract base class for finding steady-state equilibrium operating points.

    Provides a uniform interface to search for x* and u* such that dx/dt = 0
    under given output constraints and fixed input disturbances.
    """

    def __init__(self, plant: ct.NonlinearIOSystem) -> None:
        self.plant = plant

    @abstractmethod
    def get_initial_guess(self) -> np.ndarray:
        """Return the initial guess for the state vector (x_guess)."""
        pass

    @abstractmethod
    def get_nominal_inputs(self) -> np.ndarray:
        """Return the nominal input vector (u_steady)."""
        pass

    @abstractmethod
    def get_target_outputs(self) -> np.ndarray:
        """Return the target steady output vector (y_steady)."""
        pass

    @abstractmethod
    def get_input_indices(self) -> list[int]:
        """Return indices of inputs that are held fixed during search."""
        pass

    @abstractmethod
    def get_output_indices(self) -> list[int]:
        """Return indices of outputs that are constrained during search."""
        pass

    def find(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Find the equilibrium operating point using ct.find_eqpt."""
        x_guess = self.get_initial_guess()
        u_steady = self.get_nominal_inputs()
        y_steady = self.get_target_outputs()
        input_indices = self.get_input_indices()
        output_indices = self.get_output_indices()

        try:
            result = ct.find_eqpt(
                self.plant,
                x_guess,  # type: ignore
                u_steady,
                y_steady,
                input_indices=input_indices,
                output_indices=output_indices,
                return_result=True,
            )

            x_eqpt, u_eqpt, res = result

            if x_eqpt is None:
                return None

            return x_eqpt, u_eqpt

        except Exception:
            return None


def find_optimal_operating_point(
    plant: ct.NonlinearIOSystem,
    u_steady: np.ndarray,
    y_steady: np.ndarray,
    x_guess: np.ndarray,
    input_indices: list[int],
    output_indices: list[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """General wrapper function for finding equilibrium operating points.

    Dynamically constructs a subclass of OperatingPoint to run the search.
    """
    class CustomFinder(OperatingPoint):
        def get_initial_guess(self) -> np.ndarray:
            return x_guess
        def get_nominal_inputs(self) -> np.ndarray:
            return u_steady
        def get_target_outputs(self) -> np.ndarray:
            return y_steady
        def get_input_indices(self) -> list[int]:
            return input_indices
        def get_output_indices(self) -> list[int]:
            return output_indices

    finder = CustomFinder(plant)
    return finder.find()

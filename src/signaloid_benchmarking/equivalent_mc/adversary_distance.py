#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

import queue

import numpy as np

from signaloid.distributional_distance.wasserstein import wasserstein_p_distance


def _adversary_distance_loop(
    adversary_size_array: list[int] | np.ndarray,
    adversary_array: np.ndarray,
    ground_truth_positions: np.ndarray,
    progress_queue: "queue.Queue[int]",
    ground_truth_masses: np.ndarray | None,
    p: int,
) -> list[tuple[int, list[float]]]:
    """
    Subsample the adversary array at each requested size and score it.

    Args:
        adversary_size_array: Sample sizes to draw and score, in order.
        adversary_array: Pool of adversary samples to subsample from.
        ground_truth_positions: Ground-truth distribution positions.
        progress_queue: Queue notified as each size is scored, for the
            progress bar.
        ground_truth_masses: Ground-truth masses, or ``None`` for unweighted
            samples.
        p: Order of the Wasserstein distance, 1 or 2.

    Returns:
        A list of ``(size, [distance])`` tuples, one per requested size.
    """
    output_list = []

    # Generate a new seed for each calculation to achieve independence
    seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed=seed)

    for adversary_sample_size in adversary_size_array:
        # Subsample with replacement (requested size may exceed the array)
        sample_indices = rng.choice(
            adversary_array.size, size=adversary_sample_size, replace=True
        )
        distance = wasserstein_p_distance(
            adversary_array[sample_indices],
            None,
            ground_truth_positions,
            ground_truth_masses,
            p=p,
        )
        output_list.append((adversary_sample_size, [distance]))
        progress_queue.put(1)

    return output_list

"""
Integrated Gradient Auto-Correlation (IGaC) utilities.

.. note::
    Welford's online algorithm :cite:`WelfordNoteMethodCalculating1962`
    is used for the computation of mean, standard deviation, and correlation
    statistics (see `wikipedia`_).

.. _wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

import numpy as np
import torch
from tqdm import tqdm

from .base import DataManager
from .igc import IntegratedGradients


class IntGradAutoCorr(IntegratedGradients):
    """
    Integrated Gradient Auto-Correlation (IGaC).

    See the original paper :cite:`LelievreIntegratedGradientCorrelation2024` for
    more information.
    """

    def compute(  # pylint: disable=W0221,W0237
        self,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=None,
        x_0_seed=100,
        n_x=None,
        check_error=True,
    ):
        """
        Compute Integrated Gradient Auto-Correlation (IGaC).

        Parameters
        ----------
        x_0 : None | int | float | ArrayLike | tuple(ArrayLike)
        y_idx : None | int | ArrayLike
        n_steps : int
        batch_size : int | tuple(int)
        x_seed : None | int
        x_0_seed : None | int
        n_x : None | int
        check_error : bool

        Returns
        -------
        ArrayLike | tuple(ArrayLike)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data(
            n_x, x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed
        )
        # Prepare interpolation coefficients w
        w = tuple(
            torch.linspace(
                0.0, 1.0, n_steps, dtype=self.dtype, device=self.device
            )[(...,) + (None,) * (1 + len(sz_i))]
            for sz_i in self.embedding_size
        )
        # Prepare outputs
        ig_error = 0.0
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_var = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        igac = self._prepare_output((dtmg.n_y_idx,), self.ig_post_size)
        igac_mean = self._prepare_output((dtmg.n_y_idx,), self.ig_post_size)
        # Iterate over x
        postfix = None
        if check_error:
            postfix = "ig err: ?"
        tqdm_iterator = tqdm(
            dtmg.x_dtld, total=dtmg.x_nb, desc="igac", postfix=postfix
        )
        for i, (x_i, _) in enumerate(tqdm_iterator):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Record original x
            x_i_np = (x_i_j.cpu().numpy() for x_i_j in x_i)
            # Prepare x
            with torch.no_grad():
                # Send x to the device
                x_i = tuple(x_i_j.to(self.device) for x_i_j in x_i)
                # Embed discrete inputs
                x_i = self._emb(x_i)
                # Repeat x along batch dimension
                x_i = tuple(
                    x_i_j.repeat(
                        *(
                            (dtmg.x_0_bsz * dtmg.y_idx_bsz,)
                            + (1,) * (x_i_j.dim() - 1)
                        )
                    )
                    for x_i_j in x_i
                )
            # Update x_0_dtld seed
            dtmg.update_x_0_dtld_seed()
            # Compute integrated gradients
            y_0_i, y_r_i, ig_i = self._int_grad_per_x(dtmg, x_i, n_steps, w)
            # Update y_r_mean and y_r_std
            y_r_delta = y_r_i - y_r_mean
            y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
            y_r_delta_2 = y_r_i - y_r_mean
            y_r_var += np.sum(y_r_delta * y_r_delta_2, axis=0)
            # Apply IG post-function
            ig_i = self._ig_post(ig_i, x_i_np)
            # Update IGaC
            for j, (ig_i_j, sz_j) in enumerate(zip(ig_i, self.ig_post_size)):
                igac_delta = ig_i_j - igac_mean[j]
                igac_mean[j][...] += np.sum(igac_delta, axis=0) / n_x_count
                igac[j][...] += np.sum(
                    igac_delta * y_r_delta_2[(...,) + (None,) * len(sz_j)],
                    axis=0,
                )
            # Check IG error and display incremental value in tqdm
            if check_error:
                ig_sum_i = np.zeros(
                    (dtmg.x_bsz, dtmg.n_y_idx), dtype=self.dtype_np
                )
                for ig_i_j in ig_i:
                    ig_sum_i += np.sum(
                        ig_i_j.reshape((dtmg.x_bsz, dtmg.n_y_idx, -1)), axis=2
                    )
                ig_error_i = np.mean(np.abs(ig_sum_i - y_r_i + y_0_i), axis=1)
                ig_error += np.sum(ig_error_i - ig_error) / n_x_count
                tqdm_iterator.set_postfix_str(
                    f"ig err: {ig_error:>9.6f}", refresh=False
                )
        # Finalize y_r_var
        y_r_var /= dtmg.n_x
        # Finalize IGaC
        for igac_i in igac:
            igac_i /= dtmg.n_x
            igac_i /= y_r_var[(...,) + (None,) * (igac_i.ndim - 1)]
        # Check IGaC error
        if check_error:
            igac_sum = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
            for igac_i in igac:
                igac_sum += np.sum(
                    np.reshape(igac_i, (dtmg.n_y_idx, -1)), axis=1
                )
            print(f"igac err: {np.mean(np.abs(igac_sum - 1.0)):>9.6f}")
        # Return results
        if self.multi_x:
            return igac
        return igac[0]

    def error(self, igac):
        """
        Compute IGaC error.

        Parameters
        ----------
        igac : ArrayLike | tuple(ArrayLike)

        Returns
        -------
        ArrayLike
        """
        # Multi x
        if not isinstance(igac, tuple):
            igac = (igac,)
        # Check IGaC error
        n_y = igac[0].shape[0]
        igac_sum = np.zeros(n_y, dtype=igac[0].dtype)
        for igac_i in igac:
            igac_sum += np.sum(np.reshape(igac_i, (n_y, -1)), axis=1)
        error = np.abs(igac_sum - 1.0)
        print(f"igac err: {np.mean(error):>9.6f}")
        return error

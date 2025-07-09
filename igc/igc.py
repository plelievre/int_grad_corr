"""
Integrated Gradient Correlation (IGC) utilities.

.. note::
    Welford's online algorithm :cite:`WelfordNoteMethodCalculating1962`
    is used for the computation of mean, standard deviation, and correlation
    statistics (see `wikipedia`_).

.. _wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

import numpy as np
import torch
from tqdm import tqdm

from .base import AbstractAttributionMethod, DataManager


class Gradients(AbstractAttributionMethod):
    """
    Gradients.
    """

    def compute(  # pylint: disable=W0221
        self, x, y_idx=None, batch_size=None, x_seed=None
    ):
        """
        Compute gradients.

        Parameters
        ----------
        x : None | int | ArrayLike | tuple(ArrayLike)
        y_idx : None | int | ArrayLike
        batch_size : int | tuple(int)
        x_seed : None | int

        Returns
        -------
        tuple(ArrayLike)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self, y_required=False)
        y_idx = dtmg.add_data_iter_x_y_idx(x, y_idx, batch_size, x_seed)
        # Prepare outputs
        x_np = self._prepare_output((dtmg.n_x,), self.x_size)
        y_np = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        grad = self._prepare_output(
            (dtmg.n_x, dtmg.n_y_idx), self.embedding_size
        )
        # Iterate over x #######################################################
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="grad")
        ):
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Current x slice
            x_slc = slice(i * dtmg.x_bsz, (i + 1) * dtmg.x_bsz)
            # Record x, y
            for k, x_i_k in enumerate(x_i):
                x_np[k][x_slc] += x_i_k.cpu().numpy()
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            if y_i_np is not None:
                y_np[x_slc] += y_i_np
            # Prepare x
            with torch.no_grad():
                # Send x to the device
                x_i = tuple(x_i_k.to(self.device) for x_i_k in x_i)
                # Embed discrete inputs
                x_i = self._emb(x_i)
                # Repeat x along batch dimension
                x_i = tuple(
                    x_i_k.repeat(
                        *((dtmg.y_idx_bsz,) + (1,) * (x_i_k.dim() - 1))
                    )
                    for x_i_k in x_i
                )
            # Forward pass
            y_f = self._fwd(x_i)
            # Iterate over output features y_idx ###############################
            for j, y_idx_j in enumerate(dtmg.y_idx_dtld):
                # Current slice and batchsize
                y_slc = slice(
                    j * dtmg.y_idx_bsz,
                    min(dtmg.n_y_idx, (j + 1) * dtmg.y_idx_bsz),
                )
                batch_size = y_slc.stop - y_slc.start
                # Compute predictions and gradients
                y_r_i_j, grad_i_j = self._bwd(x_i, y_f, y_idx_j)
                y_r_i_j = y_r_i_j.reshape((dtmg.y_idx_bsz, dtmg.x_bsz))[
                    :batch_size
                ]
                grad_i_j = tuple(
                    grad_i_j_k.reshape(
                        (dtmg.y_idx_bsz, dtmg.x_bsz) + grad_i_j_k.shape[1:]
                    )[:batch_size]
                    for grad_i_j_k in grad_i_j
                )
                # Record results
                y_r[x_slc, y_slc] += y_r_i_j.T
                for k, grad_i_j_k in enumerate(grad_i_j):
                    grad[k][x_slc, y_slc] += grad_i_j_k.swapaxes(0, 1)
        # Return results
        if self.multi_x:
            return x_np, y_np, y_r, grad
        return x_np[0], y_np, y_r, grad[0]


class IntegratedGradients(AbstractAttributionMethod):
    """
    Integrated Gradients (IG).

    See the original paper :cite:`SundararajanAxiomaticAttributionDeep2017` for
    more information.
    """

    def __init__(
        self,
        module,
        dataset=None,
        dtld_kwargs=None,
        forward_method_name=None,
        forward_method_kwargs=None,
        dtype=torch.float32,
        dtype_cat=torch.int32,
    ):
        super().__init__(
            module,
            dataset,
            dtld_kwargs,
            forward_method_name,
            forward_method_kwargs,
            dtype,
            dtype_cat,
        )
        # Init other parameters
        self.ig_post_func = None
        self.ig_post_func_kwargs = {}
        self.ig_post_size = self.embedding_size

    def add_embedding_method(
        self,
        embedding_method_name,
        embedding_method_kwargs=None,
        embedding_n_cat=None,
    ):
        super().add_embedding_method(
            embedding_method_name,
            embedding_method_kwargs,
            embedding_n_cat,
        )
        if self.ig_post_func is None:
            self.ig_post_size = self.embedding_size
        return self

    @torch.no_grad()
    def _get_ig_post_size_from_dtld(self):
        x, _ = self.dataset[0]
        if self.multi_x:
            x = tuple(x_i.unsqueeze(dim=0).to(self.device) for x_i in x)
        else:
            x = (x.unsqueeze(dim=0).to(self.device),)
        x_emb = self._emb(x)
        ig_ = tuple(x_emb_i.unsqueeze(dim=1).numpy() for x_emb_i in x_emb)
        x_ = tuple(x_i.numpy() for x_i in x)
        ig_p = self._ig_post(ig_, x_)
        if isinstance(ig_p, (tuple, list)):
            return tuple(ig_p_i.shape[2:] for ig_p_i in ig_p)
        return (ig_p.shape[2:],)

    def add_ig_post_function(self, ig_post_func, ig_post_func_kwargs=None):
        """
        Add a function to postprocess individual IG attributions.

        .. warning::
            The IG postprocessing function must have the following signature:

            .. code-block:: python

                def ig_post(ig_x_1, ..., ig_x_n, x_1, ..., x_n, **kwargs)
                    # Do something on IG data
                    return ig_x_1, ..., ig_x_n

        Parameters
        ----------
        ig_post_func : function
            Function to postprocess individual IG attributions.
        ig_post_func_kwargs : dict
            Additional keyword arguments to the IG postprocessing function.

        Returns
        -------
        self
        """
        self.ig_post_func = ig_post_func
        self.ig_post_func_kwargs = self._check_kwargs(ig_post_func_kwargs)
        # Check ig_post_size
        ig_post_size = self._get_ig_post_size_from_dtld()
        self.ig_post_size = self._check_x_size(ig_post_size)
        return self

    def _ig_post(self, ig_x, x):
        if self.ig_post_func is None:
            return ig_x
        # pylint: disable=E1102
        return self.ig_post_func(*ig_x, *x, **self.ig_post_func_kwargs)

    def _int_grad_per_x_per_x_0(self, dtmg, x, x_0, n_steps, w):
        # Prepare outputs
        y_0 = np.zeros(
            (dtmg.n_y_idx, dtmg.x_0_bsz, dtmg.x_bsz),
            dtype=self.dtype_np,
        )
        y_r = np.zeros(
            (dtmg.n_y_idx, dtmg.x_0_bsz, dtmg.x_bsz),
            dtype=self.dtype_np,
        )
        int_grad = self._prepare_output(
            (dtmg.n_y_idx, dtmg.x_0_bsz, dtmg.x_bsz),
            self.embedding_size,
        )
        # Generate inputs along a linear path between x_0 and x
        with torch.no_grad():
            x_s = tuple()
            for x_0_i, x_i, w_i in zip(x_0, x, w):
                x_s_i = (1.0 - w_i) * x_0_i.unsqueeze(
                    dim=0
                ) + w_i * x_i.unsqueeze(dim=0)
                x_s += (x_s_i.flatten(0, 1),)
        # Compute input/baseline differences
        x_diff = tuple(
            (x_i - x_0_i)
            .cpu()
            .numpy()
            .reshape((dtmg.y_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz) + sz_i)
            for x_i, x_0_i, sz_i in zip(x, x_0, self.embedding_size)
        )
        # Forward pass
        y_f = self._fwd(x_s)
        # Iterate over output features y_idx
        for i, y_idx_i in enumerate(dtmg.y_idx_dtld):
            # Current slice and batchsize
            y_slc = slice(
                i * dtmg.y_idx_bsz, min(dtmg.n_y_idx, (i + 1) * dtmg.y_idx_bsz)
            )
            batch_size = y_slc.stop - y_slc.start
            # Compute predictions and gradients
            y_r_i, grad_i = self._bwd(x_s, y_f, y_idx_i)
            y_r_i = y_r_i.reshape(
                (n_steps, dtmg.y_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz)
            )[:, :batch_size]
            grad_i = tuple(
                grad_i_j.reshape(
                    (n_steps, dtmg.y_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz) + sz_j
                )[:, :batch_size]
                for grad_i_j, sz_j in zip(grad_i, self.embedding_size)
            )
            # Record y_0 and y_r
            y_0[y_slc] += y_r_i[0]
            y_r[y_slc] += y_r_i[-1]
            # Compute integrated gradients (Riemann sums, trapezoidal rule)
            for j, (grad_i_j, x_diff_j) in enumerate(zip(grad_i, x_diff)):
                int_grad_i_j = grad_i_j[:-1] + grad_i_j[1:]
                int_grad_i_j = 0.5 * np.mean(int_grad_i_j, axis=0)
                int_grad_i_j *= x_diff_j[:batch_size]
                int_grad[j][y_slc] += int_grad_i_j
        return y_0, y_r, int_grad

    def _int_grad_per_x(self, dtmg, x, n_steps, w):
        # Prepare outputs
        y_0 = np.zeros((dtmg.x_bsz, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.x_bsz, dtmg.n_y_idx), dtype=self.dtype_np)
        int_grad = self._prepare_output(
            (dtmg.x_bsz, dtmg.n_y_idx), self.embedding_size
        )
        # Iterate over baselines
        for i, (x_0_i, _) in enumerate(dtmg.x_0_dtld):
            # Break when x_0_nb is reached
            if i == dtmg.x_0_nb:
                break
            # Multi x
            if not self.multi_x:
                x_0_i = (x_0_i,)
            # Prepare x_0
            with torch.no_grad():
                # Send x_0 to the device
                x_0_i = tuple(x_0_i_j.to(self.device) for x_0_i_j in x_0_i)
                # Embed discrete inputs
                x_0_i = self._emb(x_0_i)
                # Repeat x_0 along batch dimension
                x_0_i = tuple(
                    x_0_i_j.repeat(
                        *((dtmg.y_idx_bsz,) + (1,) * (x_0_i_j.dim() - 1))
                    )
                    for x_0_i_j in x_0_i
                )
            # Compute integrated gradients
            y_0_i, y_r_i, int_grad_i = self._int_grad_per_x_per_x_0(
                dtmg, x, x_0_i, n_steps, w
            )
            # Record y_0 and y_r
            y_0 += np.sum(y_0_i, axis=1).T
            y_r += np.sum(y_r_i, axis=1).T
            # Record integrated gradients
            for j, int_grad_i_j in enumerate(int_grad_i):
                int_grad[j][...] += np.sum(int_grad_i_j, axis=1).swapaxes(0, 1)
        # Average across baselines
        y_0 /= dtmg.n_x_0
        y_r /= dtmg.n_x_0
        for int_grad_i in int_grad:
            int_grad_i /= dtmg.n_x_0
        return y_0, y_r, int_grad

    def compute(  # pylint: disable=W0221
        self,
        x,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=None,
        x_0_seed=100,
        check_error=True,
    ):
        """
        Compute Integrated Gradients.

        Parameters
        ----------
        x : None | int | ArrayLike | tuple(ArrayLike)
        x_0 : None | int | float | ArrayLike | tuple(ArrayLike)
        y_idx : None | int | ArrayLike
        n_steps : int
        batch_size : int | tuple(int)
        x_seed : None | int
        x_0_seed : None | int
        check_error : bool

        Returns
        -------
        tuple(ArrayLike)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self, y_required=False)
        y_idx = dtmg.add_data(
            x, x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed
        )
        # Prepare interpolation coefficients w
        w = tuple(
            torch.linspace(
                0.0, 1.0, n_steps, dtype=self.dtype, device=self.device
            )[(...,) + (None,) * (1 + len(sz_i))]
            for sz_i in self.embedding_size
        )
        # Prepare outputs
        x_np = self._prepare_output((dtmg.n_x,), self.x_size)
        y_np = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_0 = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        int_grad = self._prepare_output(
            (dtmg.n_x, dtmg.n_y_idx), self.ig_post_size
        )
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="ig")
        ):
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Current slice
            slc = slice(i * dtmg.x_bsz, (i + 1) * dtmg.x_bsz)
            # Record x, y
            for j, x_i_j in enumerate(x_i):
                x_np[j][slc] += x_i_j.cpu().numpy()
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            if y_i_np is not None:
                y_np[slc] += y_i_np
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
            # Record y_0 and y_r
            y_0[slc] += y_0_i
            y_r[slc] += y_r_i
            # Apply integrated gradients post-function
            ig_i = self._ig_post(ig_i, tuple(x_np_i[slc] for x_np_i in x_np))
            # Record integrated gradients
            for j, ig_i_j in enumerate(ig_i):
                int_grad[j][slc] += ig_i_j
        # Check error
        if check_error:
            int_grad_sum = np.zeros(
                (dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np
            )
            for ig_i in int_grad:
                int_grad_sum += np.sum(
                    ig_i.reshape((dtmg.n_x, dtmg.n_y_idx, -1)), axis=2
                )
            print(f"ig err: {np.mean(np.abs(int_grad_sum - y_r + y_0)):>9.6f}")
        # Return results
        if self.multi_x:
            return x_np, y_np, y_0, y_r, int_grad
        return x_np[0], y_np, y_0, y_r, int_grad[0]


class IntGradCorr(IntegratedGradients):
    """
    Integrated Gradient Correlation (IGC).

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
        Compute Integrated Gradient Correlation (IGC).

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
        y_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        corr = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        igc = self._prepare_output((dtmg.n_y_idx,), self.ig_post_size)
        igc_mean = self._prepare_output((dtmg.n_y_idx,), self.ig_post_size)
        # Iterate over x
        postfix = None
        if check_error:
            postfix = "ig err: ?"
        tqdm_iterator = tqdm(
            dtmg.x_dtld, total=dtmg.x_nb, desc="igc", postfix=postfix
        )
        for i, (x_i, y_i) in enumerate(tqdm_iterator):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Update y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
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
            y_r_std += np.sum(y_r_delta * (y_r_i - y_r_mean), axis=0)
            # Update correlation
            corr += np.sum(y_r_delta * y_delta_2, axis=0)
            # Apply IG post-function
            ig_i = self._ig_post(ig_i, x_i_np)
            # Update IGC
            for j, (ig_i_j, sz_j) in enumerate(zip(ig_i, self.ig_post_size)):
                igc_delta = ig_i_j - igc_mean[j]
                igc_mean[j][...] += np.sum(igc_delta, axis=0) / n_x_count
                igc[j][...] += np.sum(
                    igc_delta * y_delta_2[(...,) + (None,) * len(sz_j)], axis=0
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
        # Finalize y_std and y_r_std
        y_std /= dtmg.n_x
        y_r_std /= dtmg.n_x
        y_y_r_std = np.sqrt(y_std * y_r_std)
        # Finalize IGC
        for igc_i in igc:
            igc_i /= dtmg.n_x
            igc_i /= y_y_r_std[(...,) + (None,) * (igc_i.ndim - 1)]
        # Check IGC error
        if check_error:
            igc_sum = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
            for igc_i in igc:
                igc_sum += np.sum(np.reshape(igc_i, (dtmg.n_y_idx, -1)), axis=1)
            corr /= dtmg.n_x
            corr /= y_y_r_std
            print(f"igc err: {np.mean(np.abs(igc_sum - corr)):>9.6f}")
        # Return results
        if self.multi_x:
            return igc
        return igc[0]

    @torch.no_grad()
    def error(self, igc, y_idx=None, batch_size=None, x_seed=None, n_x=None):
        """
        Compute IGC error.

        Parameters
        ----------
        igc : ArrayLike | tuple(ArrayLike)
        y_idx : None | int | ArrayLike
        batch_size : int
        x_seed : None | int
        n_x : None | int

        Returns
        -------
        ArrayLike
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data_iter_x(n_x, y_idx, batch_size, x_seed)
        # Prepare outputs
        y_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        corr = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="igc err")
        ):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Update y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
            # Send inputs to the device
            x_i = tuple(x_i_j.to(self.device) for x_i_j in x_i)
            y_i = y_i.to(self.device)
            # Embed discrete inputs
            x_i = self._emb(x_i)
            # Compute predictions
            y_r_i = self._fwd_no_grad(x_i)
            # Update y_r_mean and y_r_std
            y_r_i_np = self._record_y(y_r_i, y_idx, dtmg.x_bsz)
            y_r_delta = y_r_i_np - y_r_mean
            y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
            y_r_std += np.sum(y_r_delta * (y_r_i_np - y_r_mean), axis=0)
            # Update correlation
            corr += np.sum(y_r_delta * y_delta_2, axis=0)
        # Finalize y_std and y_r_std
        y_std /= dtmg.n_x
        y_r_std /= dtmg.n_x
        y_y_r_std = np.sqrt(y_std * y_r_std)
        # Finalize correlation
        corr /= dtmg.n_x
        corr /= y_y_r_std
        # Multi x
        if not self.multi_x:
            igc = (igc,)
        # Check IGC error
        igc_sum = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        for igc_i in igc:
            igc_sum += np.sum(np.reshape(igc_i, (dtmg.n_y_idx, -1)), axis=1)
        error = np.abs(igc_sum - corr)
        print(f"igc err: {np.mean(error):>9.6f}")
        return error

import torch

class PseudoOpenLoop:
    def __init__(self, interaction_matrix, dm, delay, n_modes, device=None):
        """
        Pseudo-open-loop (POL) slopes reconstruction for a SAOS LightPath with one DM.

        Timing in SAOS: at iteration i, LightPath.get_wavefront_error() returns the slopes measured at i-d,
        and the DM is updated after the propagation. Hence, the slopes measured at i-d contain the DM shape
        produced by the commands computed up to iteration i-d-1. If the DM has a dynamic model, that shape is
        the output of the DM state-space, not the last command (e.g. the ASM model has D=0: a command computed
        at i is first seen at i+2).

        The DM dynamics (same SISO state-space for every actuator) are linear, so they are replicated in the
        modal space to track the modal shape actually applied by the DM:
            s_pol(i-d) = s_res(i-d) - IM @ m(i-d)

        Limitations: the DM saturation is not modelled, and the IM is a linear model of the WFS response.

        Parameters
        ----------
        interaction_matrix : np.ndarray or torch.Tensor
            Interaction matrix [nSlopes x nModes], slopes per unit of modal command.
        dm : DeformableMirror
            SAOS DM commanded with modal_basis @ modal_cmd. Its dynamic model (dyn_A, dyn_B, dyn_C, dyn_D) is used if defined.
        delay : int
            LightPath delay in samples.
        n_modes : int
            Number of controlled modes.
        device : str, optional
            Torch device. Defaults to cuda if available.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.delay = int(delay)
        self.im = torch.as_tensor(interaction_matrix, dtype=torch.float64, device=self.device)

        if self.im.shape[1] != n_modes:
            raise ValueError(f'PseudoOpenLoop::__init__ - The IM has {self.im.shape[1]} modes but {n_modes} modes are controlled.')

        self.has_dynamics = getattr(dm, 'dyn_A', None) is not None
        if self.has_dynamics:
            self.A = dm.dyn_A.to(self.device, dtype=torch.float64)
            self.B = dm.dyn_B.to(self.device, dtype=torch.float64)
            self.C = dm.dyn_C.to(self.device, dtype=torch.float64)
            self.D = dm.dyn_D.to(self.device, dtype=torch.float64)
            self.state = torch.zeros((n_modes, self.A.shape[0]), dtype=torch.float64, device=self.device)

        # applied_history[k] is the modal shape on the DM during the propagation of iteration i-d+k (oldest first)
        self.applied_history = [torch.zeros((n_modes, 1), dtype=torch.float64, device=self.device) for _ in range(self.delay + 1)]

    def reconstruct(self, res_slopes):
        """
        Compute the POL slopes from the delayed residual slopes. Call it before update() in each iteration.

        Parameters
        ----------
        res_slopes : torch.Tensor
            Residual slopes returned by LightPath.get_wavefront_error() [nSlopes x 1].

        Returns
        -------
        torch.Tensor
            POL slopes [nSlopes x 1], estimate of the open-loop slopes measured at i-d.
        """
        return res_slopes - self.im @ self.applied_history[0]

    def update(self, modal_cmd):
        """
        Register the modal command sent to the DM in this iteration. Call it once per iteration, after
        DeformableMirror.updateDMShape(modal_basis @ modal_cmd).

        Parameters
        ----------
        modal_cmd : torch.Tensor
            Modal command [nModes x 1].
        """
        modal_cmd = modal_cmd.to(self.device, dtype=torch.float64)
        if self.has_dynamics:
            # Same order as DeformableMirror.applyDynamics: output with the current state, then advance the state
            applied = self.state @ self.C.T + modal_cmd @ self.D.T
            self.state = self.state @ self.A.T + modal_cmd @ self.B.T
        else:
            applied = modal_cmd

        self.applied_history.pop(0)
        self.applied_history.append(applied.clone())

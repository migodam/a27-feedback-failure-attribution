"""GPU execution adapter for the frozen A24 full-state equations.

No Green, constitutive law, reference, receiver or linear solver is redesigned.
This is the same L=I-diag(chi)G LU/RHS operation as Physics2D, using Torch
complex128 for matched resident latency and explicit real mass derivatives.
"""
from dataclasses import dataclass
import torch
from a24.contracts import NumericalFailure
from a25.torch_physics import mass_pack, mass_unpack, pack_data, unpack_data


@dataclass
class ExactState:
    mass: torch.Tensor
    chi: torch.Tensor
    current: torch.Tensor
    field: torch.Tensor
    w: torch.Tensor
    data: torch.Tensor
    matrix: torch.Tensor
    lu: torch.Tensor
    pivots: torch.Tensor
    residual: torch.Tensor
    owner: int


class ExactMaxwell:
    kind = 'F'

    def __init__(self, reference):
        self.reference = reference

    @property
    def book(self):
        return self.reference.book

    def _match(self, mass, state):
        if state is None:
            return self(mass)
        if state.owner != id(self) or not torch.equal(mass, state.mass):
            raise ValueError('A27_FOREIGN_OR_STALE_FULL_STATE')
        return state

    def _solve(self, state, rhs, *, adjoint=False, role='full'):
        ref = self.reference
        with self.book.span('a27_' + role + ('_adjoint' if adjoint else '_forward') + '_solve',
                            linear_solve_rhs=len(rhs) * rhs.shape[-1],
                            **{role + '_RHS': len(rhs) * rhs.shape[-1]}):
            try:
                value = torch.linalg.lu_solve(state.lu, state.pivots, rhs, adjoint=adjoint)
            except RuntimeError as exc:
                raise NumericalFailure('A27_FULL_LU_SOLVE_FAILED') from exc
            residual = (state.matrix.mH if adjoint else state.matrix) @ value - rhs
            denom = torch.linalg.vector_norm(rhs, dim=(-2, -1))
            numer = torch.linalg.vector_norm(residual, dim=(-2, -1))
            scaled = torch.where(denom > 0, numer / denom.clamp_min(torch.finfo(torch.float64).tiny), numer)
            if (not torch.isfinite(value).all() or not torch.isfinite(scaled).all()
                    or bool((scaled > ref.config.get('backward_residual_tolerance', 1e-10)).any())):
                raise NumericalFailure('A27_FULL_BACKWARD_RESIDUAL')
            self.book.counts['a27_full_backward_residual_rhs'] += len(rhs) * rhs.shape[-1]
        return value, scaled

    @torch.no_grad()
    def __call__(self, mass, *, recover_current=False):
        ref = self.reference
        if mass.ndim == 1:
            mass = mass[None]
        if mass.shape[1:] != (2 * ref.n,) or mass.dtype != torch.float64:
            raise ValueError('A27_FULL_REAL_MASS_REQUIRED')
        chi = mass_unpack(mass, ref.volume)
        tol = ref.config.get('feasibility_tolerance', 1e-10)
        if (not torch.isfinite(chi).all()
                or bool((chi.real < ref.config.get('real_lower', -.5) - tol).any())
                or bool((chi.real > ref.config.get('real_upper', 4.) + tol).any())
                or bool((chi.imag < -tol).any())
                or bool((chi.imag > ref.config.get('imag_upper', .5) + tol).any())):
            raise NumericalFailure('A27_FULL_MATERIAL_DOMAIN')
        with self.book.span('a27_full_matrix_assembly', full_matrix_assemblies=len(mass)):
            matrix = torch.eye(ref.n, dtype=torch.complex128, device=mass.device) - chi[..., None] * ref.G
        with self.book.span('a27_full_LU_factor', LU_factorizations=len(mass), full_LU_factorizations=len(mass)):
            lu, pivots, info = torch.linalg.lu_factor_ex(matrix, check_errors=False)
            if bool((info != 0).any()) or not torch.isfinite(lu).all():
                raise NumericalFailure('A27_SINGULAR_FULL_LU')
        state = ExactState(mass.clone(), chi, None, None, None, None, matrix, lu, pivots, None, id(self))
        state.current, state.residual = self._solve(state, chi[..., None] * ref.incident)
        state.field = ref.incident + ref.green(state.current)
        state.w = (chi - ref.chi_b)[..., None] * state.field
        state.data = ref.observe(state.current)
        return state

    @torch.no_grad()
    def jvp(self, mass, direction, state=None):
        state = self._match(mass if mass.ndim == 2 else mass[None], state)
        if direction.ndim == 1:
            direction = direction[None]
        dchi = mass_unpack(direction, self.reference.volume)
        dc, _ = self._solve(state, dchi[..., None] * state.field, role='full_tangent')
        return pack_data(self.reference.observe(dc))

    @torch.no_grad()
    def vjp_complex(self, mass, load, state=None):
        state = self._match(mass if mass.ndim == 2 else mass[None], state)
        if load.ndim == 2:
            load = load[None]
        rhs = self.reference.observe(load, adjoint=True)
        z, _ = self._solve(state, rhs, adjoint=True, role='full_data')
        return mass_pack((state.field.conj() * z).sum(-1), 1.) / self.reference.sqrt_volume

    def adjoint(self, mass, real_load, state=None):
        return self.vjp_complex(mass, unpack_data(real_load, self.reference.q, self.reference.s), state)

    def data_gradient(self, mass, observation, state=None):
        state = self._match(mass if mass.ndim == 2 else mass[None], state)
        return self.vjp_complex(mass, (state.data - observation) / self.reference.yb_norm2, state)

    @torch.no_grad()
    def material_jacobian(self, mass, state=None):
        """OFFLINE spectrum diagnostic; never an encoder/corrector feature."""
        state = self._match(mass if mass.ndim == 2 else mass[None], state)
        ref = self.reference
        if len(state.mass) != 1:
            raise ValueError('A27_SPECTRAL_DIAGNOSTIC_SINGLE_STATE')
        # T=S L^-1 from q adjoint RHS, reused for all illuminations.
        z, _ = self._solve(state, ref.S.mH[None], adjoint=True, role='offline_receiver_adjoint')
        transfer = z[0].mH
        b = torch.einsum('qn,ns->sqn', transfer, state.field[0]) / ref.sqrt_volume
        re = torch.cat((b.real, b.imag), dim=1).reshape(2 * ref.q * ref.s, ref.n)
        im = torch.cat((-b.imag, b.real), dim=1).reshape(2 * ref.q * ref.s, ref.n)
        self.book.counts['offline_full_material_J_builds'] += 1
        return torch.cat((re, im), dim=1)

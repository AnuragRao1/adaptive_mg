import firedrake
import ufl
import finat.ufl
import weakref
from enum import IntEnum
from firedrake.petsc import PETSc
from firedrake import *
from firedrake.mg.embedded import TransferManager, get_embedding_element
from firedrake.embedding import get_embedding_dg_element


__all__ = ("AdaptiveTransferManager", )


native_families = frozenset(["Lagrange", "Discontinuous Lagrange", "Real", "Q", "DQ", "BrokenElement"])
alfeld_families = frozenset(["Hsieh-Clough-Tocher", "Reduced-Hsieh-Clough-Tocher", "Johnson-Mercier",
                             "Alfeld-Sorokina", "Arnold-Qin", "Reduced-Arnold-Qin", "Christiansen-Hu",
                             "Guzman-Neilan", "Guzman-Neilan Bubble"])
non_native_variants = frozenset(["integral", "fdm", "alfeld"])



class Op(IntEnum):
    PROLONG = 0
    RESTRICT = 1
    INJECT = 2


class AdaptiveTransferManager(TransferManager):
    def __init__(self, *, native_transfers=None, use_averaging=True):  
        super().__init__(native_transfers=native_transfers, use_averaging=use_averaging)
        self.tm = TransferManager()

    def generic_transfer(self, source, target, transfer_op, amh):
        # determine which meshes to iterate over
        for l, mesh in enumerate(amh.meshes):
            if source.function_space().mesh() == mesh:
                source_level = l
            if target.function_space().mesh() == mesh:
                target_level = l
            
        # decide order of iteration depending on coarse -> fine or fine -> coarse
        order = 1
        if target_level < source_level: order = -1

        curr_source = source
        if source_level == target_level: 
            target.assign(source)
            return

        for level in range(source_level, target_level, order):
            
            
            if level  + order == target_level:
                target_v = target
            else:
                target_mesh = amh.meshes[level + order]
                curr_space = curr_source.function_space()
                target_space = curr_space.reconstruct(mesh=target_mesh)
                target_v = Function(target_space)

            if order == 1:
                source_function_splits = amh.split_function(curr_source, child=False)
                target_function_splits = amh.split_function(target_v, child=True)
            else:
                source_function_splits = amh.split_function(curr_source, child=True)
                target_function_splits = amh.split_function(target_v, child=False)

            for split_label, _ in source_function_splits.items():
                transfer_op(source_function_splits[split_label], target_function_splits[split_label]) 


            amh.recombine(target_function_splits, target_v, child=order+1)
            curr_source = target_v
            
            print(f"Level {level} finished")


    def prolong(self, uc, uf, amh):
        self.generic_transfer(uc, uf, transfer_op=self.tm.prolong, amh=amh)

    def inject(self, uf, uc, amh):
        self.generic_transfer(uf, uc, transfer_op=self.tm.inject, amh=amh)

    def restrict(self, source, target, amh):
        """Restrict a dual function.

        :arg source: The source (fine grid) :class:`.Cofunction`.
        :arg target: The target (coarse grid) :class:`.Cofunction`.
        """
        Vs_star = source.function_space()
        Vt_star = target.function_space()
        source_element = Vs_star.ufl_element()
        target_element = Vt_star.ufl_element()
        if not self.requires_transfer(Vs_star, Op.RESTRICT, source, target):
            return

        if all(self.is_native(e, Op.RESTRICT) for e in (source_element, target_element)):
            self._native_transfer(source_element, Op.RESTRICT)(source, target)
        elif type(source_element) is finat.ufl.MixedElement:
            assert type(target_element) is finat.ufl.MixedElement
            for source_, target_ in zip(source.subfunctions, target.subfunctions):
                self.restrict(source_, target_)
        else:
            Vs = Vs_star.dual()
            Vt = Vt_star.dual()
            # Get some work vectors
            dgsource = self.DG_work(Vs_star)
            dgtarget = self.DG_work(Vt_star)
            VDGs = dgsource.function_space().dual()
            VDGt = dgtarget.function_space().dual()
            work = self.work_vec(Vs)
            dgwork = self.work_vec(VDGt)

            # g \in Vs^* -> g \in VDGs^*
            with source.dat.vec_ro as sv, dgsource.dat.vec_wo as dgv:
                if self.use_averaging:
                    work.pointwiseDivide(sv, self.V_dof_weights(Vs))
                    self.V_approx_inv_mass(Vs, VDGs).multTranspose(work, dgv)
                else:
                    self.V_inv_mass_ksp(Vs).solve(sv, work)
                    self.V_DG_mass(Vs, VDGs).mult(work, dgv)

            # g \in VDGs^* -> g \in VDGt^*
            self.restrict(dgsource, dgtarget)

            # g \in VDGt^* -> g \in Vt^*
            with dgtarget.dat.vec_ro as dgv, target.dat.vec_wo as t:
                self.DG_inv_mass(VDGt).mult(dgv, dgwork)
                self.V_DG_mass(Vt, VDGt).multTranspose(dgwork, t)
        self.cache_dat_versions(Vs_star, Op.RESTRICT, source, target)

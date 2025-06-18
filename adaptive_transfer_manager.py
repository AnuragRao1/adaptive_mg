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
        self.generic_transfer(source, target, transfer_op=self.tm.restrict, amh=amh)
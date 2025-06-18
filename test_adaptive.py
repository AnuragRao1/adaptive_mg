import pytest
from firedrake import *
from netgen.occ import *
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager

@pytest.fixture
def amh():
    import random
    random.seed(1234)
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 1
    ngmesh = geo.GenerateMesh(maxh=maxh)
    base = Mesh(ngmesh)
    amh = AdaptiveMeshHierarchy([base])
    for i in range(2):
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 0
            # if random.random() < 0.5:
            #     el.refine = 1        
        el.refine = 1
        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        amh.add_mesh(mesh)
    return amh

@pytest.fixture
def atm():
    return AdaptiveTransferManager()

def test_prolong_DG0(amh, atm):

    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    x, y = SpatialCoordinate(V_coarse.mesh())
    step = conditional(ge(x, 0), 1, 0)
    u_coarse.interpolate(step)

    assert errornorm(step, u_coarse) <= 1e-12

    atm.prolong(u_coarse, u_fine, amh)

    x, y = SpatialCoordinate(V_fine.mesh())
    step = conditional(ge(x, 0), 1, 0)
    
    assert errornorm(step, u_fine) <= 1e-12


def test_prolong_CG1(amh, atm):

    V_coarse = FunctionSpace(amh[0], "CG", 1)
    V_fine = FunctionSpace(amh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    x, y = SpatialCoordinate(V_coarse.mesh())
    u_coarse.interpolate(x)

    atm.prolong(u_coarse, u_fine, amh)

    x, y = SpatialCoordinate(V_fine.mesh())
    assert errornorm(x, u_fine) <= 1e-12

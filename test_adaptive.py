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


@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_DG0(amh, atm, operator):

    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, yc = SpatialCoordinate(V_coarse.mesh())
    stepc = conditional(ge(xc, 0), 1, 0)
    xf, yf = SpatialCoordinate(V_fine.mesh())
    stepf = conditional(ge(xf, 0), 1, 0)

    if operator == "prolong":
        u_coarse.interpolate(stepc)
        assert errornorm(stepc, u_coarse) <= 1e-12

        atm.prolong(u_coarse, u_fine, amh)
        assert errornorm(stepf, u_fine) <= 1e-12
    
    if operator == "inject":
        u_fine.interpolate(stepf)
        assert errornorm(stepf, u_fine) <= 1e-12

        atm.inject(u_fine, u_coarse, amh)
        assert errornorm(stepc, u_coarse) <= 1e-12


@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_CG1(amh, atm, operator):

    V_coarse = FunctionSpace(amh[0], "CG", 1)
    V_fine = FunctionSpace(amh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, yc = SpatialCoordinate(V_coarse.mesh())
    xf, yf = SpatialCoordinate(V_fine.mesh())


    if operator == "prolong":
        u_coarse.interpolate(xc)
        assert errornorm(xc, u_coarse) <= 1e-12

        atm.prolong(u_coarse, u_fine, amh)
        assert errornorm(xf, u_fine) <= 1e-12
    
    if operator == "inject":
        u_fine.interpolate(xf)
        assert errornorm(xf, u_fine) <= 1e-12

        atm.inject(u_fine, u_coarse, amh)
        assert errornorm(xc, u_coarse) <= 1e-12

def test_restrict_DG0(amh, atm):
    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, yc = SpatialCoordinate(V_coarse.mesh())
    xf, yf = SpatialCoordinate(V_fine.mesh())

    u_coarse.interpolate(xc)
    atm.prolong(u_coarse, u_fine, amh)

    rf = Cofunction(V_fine.dual()).assign(1)
    rc = Cofunction(V_coarse.dual())
    atm.restrict(rf, rc, amh)
    
    print(assemble(action(rc, u_coarse)) - assemble(action(rf, u_fine)))
    assert assemble(action(rc, u_coarse)) == assemble(action(rf, u_fine))






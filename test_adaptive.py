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
def mh_res():
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 1
    ngmesh = geo.GenerateMesh(maxh=maxh)
    base = Mesh(ngmesh)
    mesh2 = Mesh(ngmesh)
    amh = AdaptiveMeshHierarchy([base])
    for i in range(2):
        refs = np.ones(len(ngmesh.Elements2D()))
        amh.refine(refs)
    
    mh = MeshHierarchy(mesh2, 2)

    return amh, mh

@pytest.fixture
def atm():
    return AdaptiveTransferManager()

@pytest.fixture
def tm():
    return TransferManager()


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

def test_restrict_DG0(mh_res, atm, tm):
    amh = mh_res[0]
    mh = mh_res[1]

    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, _ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    atm.prolong(u_coarse, u_fine, amh)

    rf = assemble(TestFunction(V_fine)*dx)
    rc = Cofunction(V_coarse.dual())
    atm.restrict(rf, rc, amh)
    
    # compare with mesh_hierarchy
    xcoarse, _ = SpatialCoordinate(mh[0])
    Vcoarse = FunctionSpace(mh[0], "DG", 0)
    Vfine = FunctionSpace(mh[-1], "DG", 0)
    
    mhuc  = Function(Vcoarse)
    mhuc.interpolate(xcoarse)
    mhuf = Function(Vfine)
    tm.prolong(mhuc, mhuf)

    mhrf = assemble(TestFunction(Vfine) * dx)
    mhrc = Cofunction(Vcoarse.dual())
    
    tm.restrict(mhrf, mhrc)

    assert (assemble(action(mhrc, mhuc)) - assemble(action(mhrf, mhuf))) / assemble(action(mhrf, mhuf)) <= 1e-12
    assert (assemble(action(rc, u_coarse)) - assemble(action(mhrc, mhuc))) / assemble(action(mhrc, mhuc)) <= 1e-12

def test_restrict_CG1(mh_res, atm, tm):
    amh = mh_res[0]
    mh = mh_res[1]

    V_coarse = FunctionSpace(amh[0], "CG", 1)
    V_fine = FunctionSpace(amh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, _ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    atm.prolong(u_coarse, u_fine, amh)

    rf = assemble(TestFunction(V_fine)*dx)
    rc = Cofunction(V_coarse.dual())
    atm.restrict(rf, rc, amh)
    
    # compare with mesh_hierarchy
    xcoarse, _ = SpatialCoordinate(mh[0])
    Vcoarse = FunctionSpace(mh[0], "CG", 1)
    Vfine = FunctionSpace(mh[-1], "CG", 1)
    
    mhuc  = Function(Vcoarse)
    mhuc.interpolate(xcoarse)
    mhuf = Function(Vfine)
    tm.prolong(mhuc, mhuf)

    mhrf = assemble(TestFunction(Vfine) * dx)
    mhrc = Cofunction(Vcoarse.dual())
    
    tm.restrict(mhrf, mhrc)

    assert (assemble(action(mhrc, mhuc)) - assemble(action(mhrf, mhuf))) / assemble(action(mhrf, mhuf)) <= 1e-12
    assert (assemble(action(rc, u_coarse)) - assemble(action(mhrc, mhuc))) / assemble(action(mhrc, mhuc)) <= 1e-12









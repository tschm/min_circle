import mosek.fusion as mf
from .utils.circle import Circle


def min_circle_mosek(points, **kwargs):
    with mf.Model() as M:
        r = M.variable("Radius", 1)
        x = M.variable("Midpoint", [1, points.shape[1]])

        k = points.shape[0]

        # repeat the quantities
        R0 = mf.Var.repeat(r, k)
        X0 = mf.Var.repeat(x, k)

        # https://github.com/MOSEK/Tutorials/blob/master/minimum-ellipsoid/minimum-ellipsoid.ipynb
        M.constraint(mf.Expr.hstack(R0, mf.Expr.sub(X0, points)), mf.Domain.inQCone())

        M.objective("obj", mf.ObjectiveSense.Minimize, r)
        M.solve(**kwargs)
        return Circle(radius=r.level(), center=x.level())

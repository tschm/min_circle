import mosek.fusion as mf

from .utils.circle import Circle


def min_circle_mosek(points, **kwargs):
    with mf.Model() as model:
        r = model.variable("Radius", 1)
        x = model.variable("Midpoint", [1, points.shape[1]])

        k = points.shape[0]

        # repeat the quantities
        r0 = mf.Var.repeat(r, k)
        x0 = mf.Var.repeat(x, k)

        # https://github.com/MOSEK/Tutorials/blob/master/minimum-ellipsoid/minimum-ellipsoid.ipynb
        model.constraint(mf.Expr.hstack(r0, mf.Expr.sub(x0, points)), mf.Domain.inQCone())

        model.objective("obj", mf.ObjectiveSense.Minimize, r)
        model.solve(**kwargs)
        return Circle(radius=r.level(), center=x.level())

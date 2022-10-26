import taichi as ti
from typing import Tuple
from common.formulae import *
from common.constants import *
import common.mixing as mixing
import common.theta as theta

ti.init(arch=ti.cpu, debug=True, print_ir=True)

# Constants
# ========================
r_c0 = 5e-4  # Auto-conversion threshold
r_eps = 2e-5  # Absolute tolerance
# Kessler auto-conversion (eq. 5a in Grabowski & Smolarkiewicz 1996)
k_acnv = 1e-3
# Number of iterations in Newton-Raphson saturation adjustment
nwtrph_iters = 3

# Resolution
# ========================
nx = 2  # Number of grid points in x direction
nz = 2  # Number of grid points in z direction
nt = 5  # Number of total time steps
spinup = 0  # Number of time steps to spin up
relax_th_rv = False


@ti.func
def copysign(x: ti.template(), y: ti.template()):
    sign = 1
    if y < 0:
        sign = -1
    return sign * ti.abs(x)


@ti.data_oriented
class SingleMoment:
    """
    th, thp: potential temperature and its derivative
    rv, rvp: water vapor mixing ratio and its derivative
    rc, rcp: cloud water mixing ratio and its derivative
    rr, rrp: rain water mixing ratio and its derivative
    """

    def __init__(self, size):
        self.rhod = ti.field(dtype=ti.f32)
        self.p = ti.field(dtype=ti.f32)
        self.th = ti.field(dtype=ti.f32)
        self.rv = ti.field(dtype=ti.f32)
        self.rc = ti.field(dtype=ti.f32)
        self.rr = ti.field(dtype=ti.f32)
        self.thp = ti.field(dtype=ti.f32)
        self.rvp = ti.field(dtype=ti.f32)
        self.rcp = ti.field(dtype=ti.f32)
        self.rrp = ti.field(dtype=ti.f32)
        self.f_rhod = ti.field(dtype=ti.f32)
        self.f_p = ti.field(dtype=ti.f32)
        self.f_r = ti.field(dtype=ti.f32)
        self.f_rs = ti.field(dtype=ti.f32)
        self.f_T = ti.field(dtype=ti.f32)
        print(size)
        ti.root.dense(ti.j, size[1]).dense(ti.i, size[0]).place(
            self.rhod, self.p, self.th, self.rv, self.rc, self.rr, self.thp, self.rvp,
            self.rcp, self.rrp, self.f_rhod, self.f_p, self.f_r, self.f_rs, self.f_T,
        )

        self.init()

    @ti.kernel
    def init(self):
        for i in ti.grouped(self.th):
            self.rhod[i] = 1.0
            self.th[i] = 300.0
            self.p[i] = p_0
            self.rc[i] = 0.01

    @ti.func
    def init_F(self, i, const_p):
        self.f_rhod[i] = self.rhod[i]
        self.f_p[i] = self.p[i]
        self.update_F(i, self.th[i], self.rv[i], const_p)

    @ti.func
    def update_F(self, i, th, rv, const_p):
        # self.f_r[i] = rv
        if const_p == 0:
            self.f_T[i] = theta.T(th, self.f_rhod[i])
            self.f_p[i] = theta.p(self.f_rhod[i], rv, self.f_T[i])
        else:
            self.f_T[i] = th * theta.exner(self.f_p[i])
        self.f_rs[i] = mixing.r_vs(self.f_T[i], self.f_p[i])


    @ti.kernel
    def adj_cellwise_hlpr(self, dt: ti.template(), const_p: ti.template()):
        for i in ti.grouped(self.p):
            self.init_F(i, const_p)
            vapor_excess = 0.0

            while True:
            # for _ in range(200):
                vapor_excess = self.rv[i] - self.f_rs[i]
                # vapor_excess = ti.random(ti.f32)

                # FIXME: need to call the following statement
                # otherwise f_r will be reset to 0.0
                # Only Taichi v1.2.0 has this issue. v1.1.3 is OK.
                # assert(self.f_r[i] == self.rv[i])
                print(self.f_r[i])

                if vapor_excess >= -r_eps:
                    break

                drv = -1.0 * copysign(min(0.5 * r_eps, 0.5 *
                                vapor_excess), vapor_excess)
                self.f_r[i] = self.rv[i] + drv
                self.rv[i] += drv

                assert(self.f_r[i] == self.rv[i])
            print(self.f_r[i], self.rv[i])
            assert(self.f_r[i] == self.rv[i])



def main():
    cld = SingleMoment((nz, nx))
    for t in range(nt):
        print('t = ', t)
        cld.adj_cellwise_hlpr(1, 0)


if __name__ == '__main__':
    main()
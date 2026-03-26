from sympy import symbols, diff, sqrt, exp, simplify, MatrixSymbol

# view direction in SH
xp, yp, zp, xc, yc, zc, n = symbols("xp yp zp xt yt zt n")
dx, dy, dz = symbols("dx dy dz")
n, n2, x, y, z = symbols("n n2 x y z")
dx, dy, dz = xp - xc, yp - yc, zp - zc
n = sqrt(dx**2 + dy**2 + dz**2)
x, y, z = dx / n, dy / n, dz / n

subs_map = {
    xp - xc: dx,
    yp - yc: dy,
    zp - zc: dz,
    (xp - xc) ** 2 + (yp - yc) ** 2 + (zp - zc) ** 2: n2,
}

for i, e in [
    ("-SH_C1_y", y),
    ("SH_C1_z", z),
    ("-SH_C1_x", x),
    ("SH_C2_xy", x * y),
    ("SH_C2_yz", y * z),
    ("SH_C2_zz", 3 * z**2 - 1),
    ("SH_C2_xz", x * z),
    ("SH_C2_xx_yy", x**2 - y**2),
    ("SH_C3_yxx_yyy", y * (x**2 - y**2)),
    ("SH_C3_xyz", x * y * z),
    ("SH_C3_yzz_yxx_yyy", y * (4 * z**2 - x**2 - y**2)),
    ("SH_C3_zzz_zxx_zyy", z * (4 * z**2 - 3 * x**2 - 3 * y**2)),
    ("SH_C3_xzz_xxx_xyy", x * (4 * z**2 - x**2 - y**2)),
    ("SH_C3_zxx_zyy", z * (x**2 - y**2)),
    ("SH_C3_xxx_xyy", x * (x**2 - 3 * y**2)),
]:
    print(i)
    print("grad_xp:", simplify(diff(e, xp)).xreplace(subs_map))
    print("grad_yp:", simplify(diff(e, yp)).xreplace(subs_map))
    print("grad_zp:", simplify(diff(e, zp)).xreplace(subs_map))
    print()

# # alpha blending
# u, v, x, y, inv_cov_xx, inv_cov_yy, inv_cov_xy = symbols(
#     "u v x y inv_cov_xx inv_cov_yy inv_cov_xy"
# )
# opacity, transmittance, color = symbols("opacity transmittance color")
# du = u - x
# dv = v - y
# exponent = inv_cov_xx * du**2 + inv_cov_yy * dv**2 + 2 * inv_cov_xy * du * dv
# alpha = exp(-0.5 * exponent) * opacity
# weight = alpha * transmittance
# out = weight * color
# grad_x = diff(out, x)
# print("grad_x:", simplify(grad_x))
# grad_inv_cov_xx = diff(out, inv_cov_xx)
# print("grad_inv_cov_xx:", simplify(grad_inv_cov_xx))
# grad_inv_cov_xy = diff(out, inv_cov_xy)
# print("grad_inv_cov_xy:", simplify(grad_inv_cov_xy))

# # project points
# w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12 = symbols(
#     "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12"
# )
# xw, yw, zw, xc, yc, zc = symbols("xw yw zw xc yc zc")
# fx, fy, cx, cy = symbols("fx fy cx cy")
# du, dv, a = symbols("du dv a")
# xc = w0 * xw + w1 * yw + w2 * zw + w3
# yc = w4 * xw + w5 * yw + w6 * zw + w7
# zc = w8 * xw + w9 * yw + w10 * zw + w11
# u = fx * xc / zc + cx
# v = fy * yc / zc + cy
# a = du * u + dv * v
# grad_xw = diff(a, xw)
# print("grad_xw:", simplify(grad_xw))
# grad_yw = diff(a, yw)
# print("grad_yw:", simplify(grad_yw))
# grad_zw = diff(a, zw)
# print("grad_zw:", simplify(grad_zw))

# # project covariance
# cov_00, cov_01, cov_11 = symbols("a b c")
# inv_cov_00, inv_cov_01, inv_cov_11 = symbols("inv_a inv_b inv_c")
# det = cov_00 * cov_11 - cov_01 * cov_01
# inv_cov_00 = cov_11 / det
# inv_cov_11 = cov_00 / det
# inv_cov_01 = -cov_01 / det
# print("diff(inv_cov_00, cov_00):", simplify(diff(inv_cov_00, cov_00)))
# print("diff(inv_cov_01, cov_00):", simplify(diff(inv_cov_01, cov_00)))
# print("diff(inv_cov_11, cov_00):", simplify(diff(inv_cov_11, cov_00)))
# print("diff(inv_cov_00, cov_01):", simplify(diff(inv_cov_00, cov_01)))
# print("diff(inv_cov_01, cov_01):", simplify(diff(inv_cov_01, cov_01)))
# print("diff(inv_cov_11, cov_11):", simplify(diff(inv_cov_11, cov_11)))
# print("diff(inv_cov_00, cov_11):", simplify(diff(inv_cov_00, cov_11)))
# print("diff(inv_cov_01, cov_11):", simplify(diff(inv_cov_01, cov_11)))
# print("diff(inv_cov_11, cov_11):", simplify(diff(inv_cov_11, cov_11)))

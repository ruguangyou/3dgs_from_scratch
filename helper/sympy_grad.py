from sympy import symbols, diff, sqrt, exp, simplify, MatrixSymbol

# # view direction in SH
# xp, yp, zp, xc, yc, zc, n = symbols("xp yp zp xc yc zc n")
# n = sqrt((xp - xc) ** 2 + (yp - yc) ** 2 + (zp - zc) ** 2)
# f = (xp - xc) / n
# grad_xp = diff(f, xp)
# print("grad_xp:", simplify(grad_xp))
# grad_xc = diff(f, xc)
# print("grad_xc:", simplify(grad_xc))

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

# project covariance
cov_00, cov_01, cov_11 = symbols("cov_00 cov_01 cov_11")
inv_cov_00, inv_cov_01, inv_cov_11 = symbols("inv_cov_00 inv_cov_01 inv_cov_11")
det = cov_00 * cov_11 - cov_01 * cov_01
inv_cov_00 = cov_11 / det
inv_cov_11 = cov_00 / det
inv_cov_01 = -cov_01 / det
inv_cov = inv_cov_00 + inv_cov_11 + 2 * inv_cov_01
grad_cov_00 = diff(inv_cov, cov_00)
print("grad_cov_00:", simplify(grad_cov_00))
grad_cov_01 = diff(inv_cov, cov_01)
print("grad_cov_01:", simplify(grad_cov_01))
grad_cov_11 = diff(inv_cov, cov_11)
print("grad_cov_11:", simplify(grad_cov_11))

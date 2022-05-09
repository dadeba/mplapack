/*
 * Copyright (c) 2021-2022
 *      Nakata, Maho
 *      All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#include <mpblas.h>
#include <mplapack.h>

#include <fem.hpp> // Fortran EMulation library of fable module
using namespace fem::major_types;
using fem::common;

#include <mplapack_matgen.h>
#include <mplapack_eig.h>

#include <mplapack_debug.h>
#include <lapacke.h>

void Ccsdts(INTEGER const m, INTEGER const p, INTEGER const q, COMPLEX *x, COMPLEX *xf, INTEGER const ldx, COMPLEX *u1, INTEGER const ldu1, COMPLEX *u2, INTEGER const ldu2, COMPLEX *v1t, INTEGER const ldv1t, COMPLEX *v2t, INTEGER const ldv2t, REAL *theta, INTEGER *iwork, COMPLEX *work, INTEGER const lwork, REAL *rwork, REAL *result) {
    //
    INTEGER ldxf = ldx;
    REAL ulp = Rlamch("Precision");
    const REAL realone = 1.0;
    REAL ulpinv = realone / ulp;
    //
    //     The first half of the routine checks the 2-by-2 CSD
    //
    const COMPLEX zero = COMPLEX(0.0, 0.0);
    const COMPLEX one = COMPLEX(1.0, 0.0);
    Claset("Full", m, m, zero, one, work, ldx);
    Cherk("Upper", "Conjugate transpose", m, m, -realone, x, ldx, realone, work, ldx);
    REAL eps2 = 0.0;
    if (m > 0) {
        eps2 = max(ulp, REAL(Clange("1", m, m, work, ldx, rwork) / castREAL(m)));
    } else {
        eps2 = ulp;
    }
    INTEGER r = min({p, m - p, q, m - q});
    //
    //     Copy the matrix X to the array XF.
    //
    Clacpy("Full", m, m, x, ldx, xf, ldx);
    //
    //     Compute the CSD
    //
    INTEGER info = 0;
    Cuncsd("Y", "Y", "Y", "Y", "N", "D", m, p, q, &xf[(1 - 1)], ldx, &xf[((q + 1) - 1) * ldxf], ldx, &xf[((p + 1) - 1)], ldx, &xf[((p + 1) - 1) + ((q + 1) - 1) * ldxf], ldx, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, v2t, ldv2t, work, lwork, rwork, 17 * (r + 2), iwork, info);
    //
    //     Compute XF := diag(U1,U2)'*X*diag(V1,V2) - [D11 D12; D21 D22]
    //
    Clacpy("Full", m, m, x, ldx, xf, ldx);
    //
    Cgemm("No transpose", "Conjugate transpose", p, q, q, one, xf, ldx, v1t, ldv1t, zero, work, ldx);
    //
    Cgemm("Conjugate transpose", "No transpose", p, q, p, one, u1, ldu1, work, ldx, zero, xf, ldx);
    //
    INTEGER i = 0;
    for (i = 1; i <= min(p, q) - r; i = i + 1) {
        xf[(i - 1) + (i - 1) * ldxf] = xf[(i - 1) + (i - 1) * ldxf] - one;
    }
    for (i = 1; i <= r; i = i + 1) {
        xf[(min(p, q) - r + i - 1) + (min(p, q) - r + i - 1) * ldx] = xf[(min(p, q) - r + i - 1) + (min(p, q) - r + i - 1) * ldx] - COMPLEX(cos(theta[i - 1]), 0.0);
    }
    //
    Cgemm("No transpose", "Conjugate transpose", p, m - q, m - q, one, &xf[((q + 1) - 1) * ldxf], ldx, v2t, ldv2t, zero, work, ldx);
    //
    Cgemm("Conjugate transpose", "No transpose", p, m - q, p, one, u1, ldu1, work, ldx, zero, &xf[((q + 1) - 1) * ldxf], ldx);
    //
    for (i = 1; i <= min(p, m - q) - r; i = i + 1) {
        xf[((p - i + 1) - 1) + ((m - i + 1) - 1) * ldxf] += one;
    }
    for (i = 1; i <= r; i = i + 1) {
        xf[(p - (min(p, m - q) - r) + 1 - i - 1) + (m - (min(p, m - q) - r) + 1 - i - 1) * ldx] += COMPLEX(sin(theta[(r - i + 1) - 1]), 0.0);
    }
    //
    Cgemm("No transpose", "Conjugate transpose", m - p, q, q, one, &xf[((p + 1) - 1)], ldx, v1t, ldv1t, zero, work, ldx);
    //
    Cgemm("Conjugate transpose", "No transpose", m - p, q, m - p, one, u2, ldu2, work, ldx, zero, &xf[((p + 1) - 1)], ldx);
    //
    for (i = 1; i <= min(m - p, q) - r; i = i + 1) {
        xf[((m - i + 1) - 1) + ((q - i + 1) - 1) * ldxf] = xf[((m - i + 1) - 1) + ((q - i + 1) - 1) * ldxf] - one;
    }
    for (i = 1; i <= r; i = i + 1) {
        xf[(m - (min(m - p, q) - r) + 1 - i - 1) + (q - (min(m - p, q) - r) + 1 - i - 1) * ldx] = xf[(m - (min(m - p, q) - r) + 1 - i - 1) + (q - (min(m - p, q) - r) + 1 - i - 1) * ldx] - COMPLEX(sin(theta[(r - i + 1) - 1]), 0.0);
    }
    //
    Cgemm("No transpose", "Conjugate transpose", m - p, m - q, m - q, one, &xf[((p + 1) - 1) + ((q + 1) - 1) * ldxf], ldx, v2t, ldv2t, zero, work, ldx);
    //
    Cgemm("Conjugate transpose", "No transpose", m - p, m - q, m - p, one, u2, ldu2, work, ldx, zero, &xf[((p + 1) - 1) + ((q + 1) - 1) * ldxf], ldx);
    //
    for (i = 1; i <= min(m - p, m - q) - r; i = i + 1) {
        xf[((p + i) - 1) + ((q + i) - 1) * ldxf] = xf[((p + i) - 1) + ((q + i) - 1) * ldxf] - one;
    }
    for (i = 1; i <= r; i = i + 1) {
        xf[(p + (min(m - p, m - q) - r) + i - 1) + (q + (min(m - p, m - q) - r) + i - 1) * ldx] = xf[(p + (min(m - p, m - q) - r) + i - 1) + (q + (min(m - p, m - q) - r) + i - 1) * ldx] - COMPLEX(cos(theta[i - 1]), 0.0);
    }
    //
    //     Compute norm( U1'*X11*V1 - D11 ) / ( MAX(1,P,Q)*EPS2 ) .
    //
    REAL resid = Clange("1", p, q, xf, ldx, rwork);
    result[1 - 1] = (resid / castREAL(max({(INTEGER)1, p, q}))) / eps2;
    //
    //     Compute norm( U1'*X12*V2 - D12 ) / ( MAX(1,P,M-Q)*EPS2 ) .
    //
    resid = Clange("1", p, m - q, &xf[((q + 1) - 1) * ldxf], ldx, rwork);
    result[2 - 1] = (resid / castREAL(max({(INTEGER)1, p, m - q}))) / eps2;
    //
    //     Compute norm( U2'*X21*V1 - D21 ) / ( MAX(1,M-P,Q)*EPS2 ) .
    //
    resid = Clange("1", m - p, q, &xf[((p + 1) - 1)], ldx, rwork);
    result[3 - 1] = (resid / castREAL(max({(INTEGER)1, m - p, q}))) / eps2;
    //
    //     Compute norm( U2'*X22*V2 - D22 ) / ( MAX(1,M-P,M-Q)*EPS2 ) .
    //
    resid = Clange("1", m - p, m - q, &xf[((p + 1) - 1) + ((q + 1) - 1) * ldxf], ldx, rwork);
    result[4 - 1] = (resid / castREAL(max({(INTEGER)1, m - p, m - q}))) / eps2;
    //
    //     Compute I - U1'*U1
    //
    Claset("Full", p, p, zero, one, work, ldu1);
    Cherk("Upper", "Conjugate transpose", p, p, -realone, u1, ldu1, realone, work, ldu1);
    //
    //     Compute norm( I - U'*U ) / ( MAX(1,P) * ULP ) .
    //
    resid = Clanhe("1", "Upper", p, work, ldu1, rwork);
    result[5 - 1] = (resid / castREAL(max((INTEGER)1, p))) / ulp;
    //
    //     Compute I - U2'*U2
    //
    Claset("Full", m - p, m - p, zero, one, work, ldu2);
    Cherk("Upper", "Conjugate transpose", m - p, m - p, -realone, u2, ldu2, realone, work, ldu2);
    //
    //     Compute norm( I - U2'*U2 ) / ( MAX(1,M-P) * ULP ) .
    //
    resid = Clanhe("1", "Upper", m - p, work, ldu2, rwork);
    result[6 - 1] = (resid / castREAL(max((INTEGER)1, (m - p)))) / ulp;
    //
    //     Compute I - V1T*V1T'
    //
    Claset("Full", q, q, zero, one, work, ldv1t);
    Cherk("Upper", "No transpose", q, q, -realone, v1t, ldv1t, realone, work, ldv1t);
    //
    //     Compute norm( I - V1T*V1T' ) / ( MAX(1,Q) * ULP ) .
    //
    resid = Clanhe("1", "Upper", q, work, ldv1t, rwork);
    result[7 - 1] = (resid / castREAL(max((INTEGER)1, q))) / ulp;
    //
    //     Compute I - V2T*V2T'
    //
    Claset("Full", m - q, m - q, zero, one, work, ldv2t);
    Cherk("Upper", "No transpose", m - q, m - q, -realone, v2t, ldv2t, realone, work, ldv2t);
    //
    //     Compute norm( I - V2T*V2T' ) / ( MAX(1,M-Q) * ULP ) .
    //
    resid = Clanhe("1", "Upper", m - q, work, ldv2t, rwork);
    result[8 - 1] = (resid / castREAL(max((INTEGER)1, m - q))) / ulp;
    //
    //     Check sorting
    //
    const REAL realzero = 0.0;
    result[9 - 1] = realzero;
    REAL dummy;
    const REAL piover2 = pi(dummy) / 2.0;
    for (i = 1; i <= r; i = i + 1) {
        if (theta[i - 1] < realzero || theta[i - 1] > piover2) {
            result[9 - 1] = ulpinv;
        }
        if (i > 1) {
            if (theta[i - 1] < theta[(i - 1) - 1]) {
                result[9 - 1] = ulpinv;
            }
        }
    }
    //
    //     The second half of the routine checks the 2-by-1 CSD
    //
    Claset("Full", q, q, zero, one, work, ldx);
    Cherk("Upper", "Conjugate transpose", q, m, -realone, x, ldx, realone, work, ldx);
    if (m > 0) {
        eps2 = max(ulp, REAL(Clange("1", q, q, work, ldx, rwork) / castREAL(m)));
    } else {
        eps2 = ulp;
    }
    r = min({p, m - p, q, m - q});
    //
    //     Copy the matrix X to the array XF.
    //
    Clacpy("Full", m, m, x, ldx, xf, ldx);
    //
    //     Compute the CSD
    //
    Cuncsd2by1("Y", "Y", "Y", m, p, q, &xf[(1 - 1)], ldx, &xf[((p + 1) - 1)], ldx, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, work, lwork, rwork, INTEGER(17 * (r + 2)), iwork, info);
    {
        if (m > 0 && p > 0 && q > 0 && m - p > 0) {
            printf(" m: %d, p: %d, q: %d\n", (int)m, (int)p, (int)q);
            __complex__ double *xf_d = new __complex__ double[max(m * m, (INTEGER)1)];
            __complex__ double *u1_d = new __complex__ double[max(p * p, (INTEGER)1)];
            __complex__ double *u2_d = new __complex__ double[max((m - p) * (m - p), (INTEGER)1)];
            __complex__ double *v1t_d = new __complex__ double[max(q * q, (INTEGER)1)];
            double *theta_d = new double[max(r, (INTEGER)1)];
            int ldx_d = (int)m;
            int ldu1_d = (int)p;
            int ldu2_d = (int)(m - p);
            int ldv1t_d = (int)q;
            double dtmp_r, dtmp_i, dtmp;
            for (int pp = 0; pp < m; pp++) {
                for (int qq = 0; qq < m; qq++) {
                    dtmp_r = cast2double(x[pp + qq * ldx].real());
                    dtmp_i = cast2double(x[pp + qq * ldx].imag());
                    __real__ xf_d[pp + qq * ldx_d] = dtmp_r;
                    __imag__ xf_d[pp + qq * ldx_d] = dtmp_i;
                }
            }
            //            printf("x_org="); printmat(m, m, x, ldx); printf("\n");
            //            printf("x_org_d="); printmat(m, m, xf_d, ldx_d); printf("\n");
            double norm = 0.0;
            for (int pp = 0; pp < m; pp++) {
                for (int qq = 0; qq < m; qq++) {
                    dtmp_r = cast2double(x[pp + qq * ldx].real());
                    dtmp_i = cast2double(x[pp + qq * ldx].imag());
                    norm = norm + fabs(__real__ xf_d[pp + qq * ldx_d] - dtmp_r) + fabs(__imag__ xf_d[pp + qq * ldx_d] - dtmp_i);
                }
            }
            // printf("norm_orig = %lf\n", norm);
            LAPACKE_zuncsd2by1(LAPACK_COL_MAJOR, 'Y', 'Y', 'Y', (int)m, (int)p, (int)q, &xf_d[(1 - 1)], ldx_d, &xf_d[((p + 1) - 1)], ldx_d, theta_d, u1_d, ldu1_d, u2_d, ldu2_d, v1t_d, ldv1t_d);
            //            printf("xf="); printmat(m, m, xf, ldx); printf("\n");
            //            printf("xf_d="); printmat(m, m, xf_d, ldx_d); printf("\n");
            norm = 0.0;
            for (int pp = 0; pp < m; pp++) {
                for (int qq = 0; qq < m; qq++) {
                    dtmp_r = cast2double(xf[pp + qq * ldx].real());
                    dtmp_i = cast2double(xf[pp + qq * ldx].imag());
                    norm = norm + fabs(__real__ xf_d[pp + qq * ldx_d] - dtmp_r) + fabs(__imag__ xf_d[pp + qq * ldx_d] - dtmp_i);
                }
            }
            // printf("norm_result = %lf\n", norm);
            printf("ccsdts.cpp l.260\n");
            printf("theta="); printvec(theta, r); printf("\n");
            printf("theta_d="); printvec(theta_d, r); printf("\n");
            norm = 0.0;
            for (int pp = 0; pp < r; pp++) {
                dtmp = cast2double(theta[pp]);
                norm = norm + fabs(cast2double(theta[pp]) - theta_d[pp]);
            }
            // printf("norm_theta = %lf\n", norm);

            delete[] theta_d;
            delete[] v1t_d;
            delete[] u2_d;
            delete[] u1_d;
            delete[] xf_d;
        }
    }
    //
    //     Compute [X11;X21] := diag(U1,U2)'*[X11;X21]*V1 - [D11;D21]
    //
    Cgemm("No transpose", "Conjugate transpose", p, q, q, one, x, ldx, v1t, ldv1t, zero, work, ldx);
    //
    Cgemm("Conjugate transpose", "No transpose", p, q, p, one, u1, ldu1, work, ldx, zero, x, ldx);
    //
    for (i = 1; i <= min(p, q) - r; i = i + 1) {
        x[(i - 1) + (i - 1) * ldx] = x[(i - 1) + (i - 1) * ldx] - one;
    }
    for (i = 1; i <= r; i = i + 1) {
        x[(min(p, q) - r + i - 1) + (min(p, q) - r + i - 1) * ldx] = x[(min(p, q) - r + i - 1) + (min(p, q) - r + i - 1) * ldx] - COMPLEX(cos(theta[i - 1]), 0.0);
    }
    //
    Cgemm("No transpose", "Conjugate transpose", m - p, q, q, one, &x[((p + 1) - 1)], ldx, v1t, ldv1t, zero, work, ldx);
    //
    Cgemm("Conjugate transpose", "No transpose", m - p, q, m - p, one, u2, ldu2, work, ldx, zero, &x[((p + 1) - 1)], ldx);
    //
    for (i = 1; i <= min(m - p, q) - r; i = i + 1) {
        x[((m - i + 1) - 1) + ((q - i + 1) - 1) * ldx] = x[((m - i + 1) - 1) + ((q - i + 1) - 1) * ldx] - one;
    }
    for (i = 1; i <= r; i = i + 1) {
        x[((m - (min(m - p, q) - r) + 1 - i) - 1) + ((q - (min(m - p, q) - r) + 1 - i) - 1) * ldx] = x[((m - (min(m - p, q) - r) + 1 - i) - 1) + ((q - (min(m - p, q) - r) + 1 - i) - 1) * ldx] - COMPLEX(sin(theta[(r - i + 1) - 1]), 0.0);
    }
    //
    //     Compute norm( U1'*X11*V1 - D11 ) / ( MAX(1,P,Q)*EPS2 ) .
    //
    resid = Clange("1", p, q, x, ldx, rwork);
    result[10 - 1] = (resid / castREAL(max({(INTEGER)1, p, q}))) / eps2;
    //
    //     Compute norm( U2'*X21*V1 - D21 ) / ( MAX(1,M-P,Q)*EPS2 ) .
    //
    resid = Clange("1", m - p, q, &x[((p + 1) - 1)], ldx, rwork);
    result[11 - 1] = (resid / castREAL(max({(INTEGER)1, m - p, q}))) / eps2;
    //
    //     Compute I - U1'*U1
    //
    Claset("Full", p, p, zero, one, work, ldu1);
    Cherk("Upper", "Conjugate transpose", p, p, -realone, u1, ldu1, realone, work, ldu1);
    //
    //     Compute norm( I - U'*U ) / ( MAX(1,P) * ULP ) .
    //
    resid = Clanhe("1", "Upper", p, work, ldu1, rwork);
    result[12 - 1] = (resid / castREAL(max((INTEGER)1, p))) / ulp;
    //
    //     Compute I - U2'*U2
    //
    Claset("Full", m - p, m - p, zero, one, work, ldu2);
    Cherk("Upper", "Conjugate transpose", m - p, m - p, -realone, u2, ldu2, realone, work, ldu2);
    //
    //     Compute norm( I - U2'*U2 ) / ( MAX(1,M-P) * ULP ) .
    //
    resid = Clanhe("1", "Upper", m - p, work, ldu2, rwork);
    result[13 - 1] = (resid / castREAL(max((INTEGER)1, (m - p)))) / ulp;
    //
    //     Compute I - V1T*V1T'
    //
    Claset("Full", q, q, zero, one, work, ldv1t);
    Cherk("Upper", "No transpose", q, q, -realone, v1t, ldv1t, realone, work, ldv1t);
    //
    //     Compute norm( I - V1T*V1T' ) / ( MAX(1,Q) * ULP ) .
    //
    resid = Clanhe("1", "Upper", q, work, ldv1t, rwork);
    result[14 - 1] = (resid / castREAL(max((INTEGER)1, q))) / ulp;
    //
    //     Check sorting
    //
    result[15 - 1] = realzero;
    for (i = 1; i <= r; i = i + 1) {
        if (theta[i - 1] < realzero || theta[i - 1] > piover2) {
            result[15 - 1] = ulpinv;
        }
        if (i > 1) {
            if (theta[i - 1] < theta[(i - 1) - 1]) {
                result[15 - 1] = ulpinv;
            }
        }
    }
    //
    //     End of Ccsdts
    //
}

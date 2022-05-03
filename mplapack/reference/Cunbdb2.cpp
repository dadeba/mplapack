/*
 * Copyright (c) 2008-2021
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

void Cunbdb2(INTEGER const m, INTEGER const p, INTEGER const q, COMPLEX *x11, INTEGER const ldx11, COMPLEX *x21, INTEGER const ldx21, REAL *theta, REAL *phi, COMPLEX *taup1, COMPLEX *taup2, COMPLEX *tauq1, COMPLEX *work, INTEGER const lwork, INTEGER &info) {
    //
    //     Test input arguments
    //
    info = 0;
    bool lquery = lwork == -1;
    //
    if (m < 0) {
        info = -1;
    } else if (p < 0 || p > m - p) {
        info = -2;
    } else if (q < 0 || q < p || m - q < p) {
        info = -3;
    } else if (ldx11 < max((INTEGER)1, p)) {
        info = -5;
    } else if (ldx21 < max((INTEGER)1, m - p)) {
        info = -7;
    }
    //
    //     Compute workspace
    //
    INTEGER ilarf = 0;
    INTEGER llarf = 0;
    INTEGER iorbdb5 = 0;
    INTEGER lorbdb5 = 0;
    INTEGER lworkopt = 0;
    INTEGER lworkmin = 0;
    if (info == 0) {
        ilarf = 2;
        llarf = max({p - 1, m - p, q - 1});
        iorbdb5 = 2;
        lorbdb5 = q - 1;
        lworkopt = max(ilarf + llarf - 1, iorbdb5 + lorbdb5 - 1);
        lworkmin = lworkopt;
        work[1 - 1] = lworkopt;
        if (lwork < lworkmin && !lquery) {
            info = -14;
        }
    }
    if (info != 0) {
        Mxerbla("Cunbdb2", -info);
        return;
    } else if (lquery) {
        return;
    }
    //
    //     Reduce rows 1, ..., P of X11 and X21
    //
    INTEGER i = 0;
    REAL c = 0.0;
    REAL s = 0.0;
    const COMPLEX one = COMPLEX(1.0, 0.0);
    INTEGER childinfo = 0;
    const COMPLEX negone = COMPLEX(-1.0, 0.0);
    for (i = 1; i <= p; i = i + 1) {
        //
        if (i > 1) {
            CRrot(q - i + 1, &x11[(i - 1) + (i - 1) * ldx11], ldx11, &x21[((i - 1) - 1) + (i - 1) * ldx21], ldx21, c, s);
        }
        Clacgv(q - i + 1, &x11[(i - 1) + (i - 1) * ldx11], ldx11);
        Clarfgp(q - i + 1, x11[(i - 1) + (i - 1) * ldx11], &x11[(i - 1) + ((i + 1) - 1) * ldx11], ldx11, tauq1[i - 1]);
        c = x11[(i - 1) + (i - 1) * ldx11].real();
        x11[(i - 1) + (i - 1) * ldx11] = one;
        Clarf("R", p - i, q - i + 1, &x11[(i - 1) + (i - 1) * ldx11], ldx11, tauq1[i - 1], &x11[((i + 1) - 1) + (i - 1) * ldx11], ldx11, &work[ilarf - 1]);
        Clarf("R", m - p - i + 1, q - i + 1, &x11[(i - 1) + (i - 1) * ldx11], ldx11, tauq1[i - 1], &x21[(i - 1) + (i - 1) * ldx21], ldx21, &work[ilarf - 1]);
        Clacgv(q - i + 1, &x11[(i - 1) + (i - 1) * ldx11], ldx11);
        s = sqrt(pow2(RCnrm2(p - i, &x11[((i + 1) - 1) + (i - 1) * ldx11], 1)) + pow2(RCnrm2(m - p - i + 1, &x21[(i - 1) + (i - 1) * ldx21], 1)));
        theta[i - 1] = atan2(s, c);
        //
        Cunbdb5(p - i, m - p - i + 1, q - i, &x11[((i + 1) - 1) + (i - 1) * ldx11], 1, &x21[(i - 1) + (i - 1) * ldx21], 1, &x11[((i + 1) - 1) + ((i + 1) - 1) * ldx11], ldx11, &x21[(i - 1) + ((i + 1) - 1) * ldx21], ldx21, &work[iorbdb5 - 1], lorbdb5, childinfo);
        Cscal(p - i, negone, &x11[((i + 1) - 1) + (i - 1) * ldx11], 1);
        Clarfgp(m - p - i + 1, x21[(i - 1) + (i - 1) * ldx21], &x21[((i + 1) - 1) + (i - 1) * ldx21], 1, taup2[i - 1]);
        if (i < p) {
            Clarfgp(p - i, x11[((i + 1) - 1) + (i - 1) * ldx11], &x11[((i + 2) - 1) + (i - 1) * ldx11], 1, taup1[i - 1]);
            phi[i - 1] = atan2(x11[((i + 1) - 1) + (i - 1) * ldx11].real(), x21[(i - 1) + (i - 1) * ldx21].real());
            c = cos(phi[i - 1]);
            s = sin(phi[i - 1]);
            x11[((i + 1) - 1) + (i - 1) * ldx11] = one;
            Clarf("L", p - i, q - i, &x11[((i + 1) - 1) + (i - 1) * ldx11], 1, conj(taup1[i - 1]), &x11[((i + 1) - 1) + ((i + 1) - 1) * ldx11], ldx11, &work[ilarf - 1]);
        }
        x21[(i - 1) + (i - 1) * ldx21] = one;
        Clarf("L", m - p - i + 1, q - i, &x21[(i - 1) + (i - 1) * ldx21], 1, conj(taup2[i - 1]), &x21[(i - 1) + ((i + 1) - 1) * ldx21], ldx21, &work[ilarf - 1]);
        //
    }
    //
    //     Reduce the bottom-right portion of X21 to the identity matrix
    //
    for (i = p + 1; i <= q; i = i + 1) {
        Clarfgp(m - p - i + 1, x21[(i - 1) + (i - 1) * ldx21], &x21[((i + 1) - 1) + (i - 1) * ldx21], 1, taup2[i - 1]);
        x21[(i - 1) + (i - 1) * ldx21] = one;
        Clarf("L", m - p - i + 1, q - i, &x21[(i - 1) + (i - 1) * ldx21], 1, conj(taup2[i - 1]), &x21[(i - 1) + ((i + 1) - 1) * ldx21], ldx21, &work[ilarf - 1]);
    }
    //
    //     End of Cunbdb2
    //
}

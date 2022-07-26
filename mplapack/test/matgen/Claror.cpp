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
#include <mplapack_matgen.h>

#if defined ___MPLAPACK_BUILD_WITH_DD___
#pragma GCC push_options
#pragma GCC optimize("O0")
#endif

void Claror(const char *side, const char *init, INTEGER const m, INTEGER const n, COMPLEX *a, INTEGER const lda, INTEGER *iseed, COMPLEX *x, INTEGER &info) {
    //
    info = 0;
    if (n == 0 || m == 0) {
        return;
    }
    //
    INTEGER itype = 0;
    if (Mlsame(side, "L")) {
        itype = 1;
    } else if (Mlsame(side, "R")) {
        itype = 2;
    } else if (Mlsame(side, "C")) {
        itype = 3;
    } else if (Mlsame(side, "T")) {
        itype = 4;
    }
    //
    //     Check for argument errors.
    //
    if (itype == 0) {
        info = -1;
    } else if (m < 0) {
        info = -3;
    } else if (n < 0 || (itype == 3 && n != m)) {
        info = -4;
    } else if (lda < m) {
        info = -6;
    }
    if (info != 0) {
        Mxerbla("Claror", -info);
        return;
    }
    //
    INTEGER nxfrm = 0;
    if (itype == 1) {
        nxfrm = m;
    } else {
        nxfrm = n;
    }
    //
    //     Initialize A to the identity matrix if desired
    //
    const COMPLEX czero = COMPLEX(0.0, 0.0);
    const COMPLEX cone = COMPLEX(1.0, 0.0);
    if (Mlsame(init, "I")) {
        Claset("Full", m, n, czero, cone, a, lda);
    }
    //
    //     If no rotation possible, still multiply by
    //     a random complex number from the circle |x| = 1
    //
    //      2)      Compute Rotation by computing Householder
    //              Transformations H(2), H(3), ..., H(n).  Note that the
    //              order in which they are computed is irrelevant.
    //
    INTEGER j = 0;
    for (j = 1; j <= nxfrm; j = j + 1) {
        x[j - 1] = czero;
    }
    //
    INTEGER ixfrm = 0;
    INTEGER kbeg = 0;
    REAL xnorm = 0.0;
    REAL xabs = 0.0;
    COMPLEX csign = 0.0;
    COMPLEX xnorms = 0.0;
    REAL factor = 0.0;
    const REAL toosml = 1.0e-20;
    const REAL one = 1.0;
    for (ixfrm = 2; ixfrm <= nxfrm; ixfrm = ixfrm + 1) {
        kbeg = nxfrm - ixfrm + 1;
        //
        //        Generate independent normal( 0, 1 ) random numbers
        //
        for (j = kbeg; j <= nxfrm; j = j + 1) {
            x[j - 1] = Clarnd(3, iseed);
        }
        //
        //        Generate a Householder transformation from the random vector X
        //
        xnorm = RCnrm2(ixfrm, &x[kbeg - 1], 1);
        xabs = abs(x[kbeg - 1]);
        if (xabs != czero) {
            csign = x[kbeg - 1] / xabs;
        } else {
            csign = cone;
        }
        xnorms = csign * xnorm;
        x[(nxfrm + kbeg) - 1] = -csign;
        factor = xnorm * (xnorm + xabs);
        if (abs(factor) < toosml) {
            info = 1;
            Mxerbla("Claror", -info);
            return;
        } else {
            factor = one / factor;
        }
        x[kbeg - 1] += xnorms; //this somehow doesn't work properly with GCC + libqd
        //
        //        Apply Householder transformation to A
        //
        if (itype == 1 || itype == 3 || itype == 4) {
            //
            //           Apply H(k) on the left of A
            //
            Cgemv("C", ixfrm, n, cone, &a[(kbeg - 1)], lda, &x[kbeg - 1], 1, czero, &x[(2 * nxfrm + 1) - 1], 1);
            Cgerc(ixfrm, n, -COMPLEX(factor), &x[kbeg - 1], 1, &x[(2 * nxfrm + 1) - 1], 1, &a[(kbeg - 1)], lda);
            //
        }
        //
        if (itype >= 2 && itype <= 4) {
            //
            //           Apply H(k)* (or H(k)') on the right of A
            //
            if (itype == 4) {
                Clacgv(ixfrm, &x[kbeg - 1], 1);
            }
            //
            Cgemv("N", m, ixfrm, cone, &a[(kbeg - 1) * lda], lda, &x[kbeg - 1], 1, czero, &x[(2 * nxfrm + 1) - 1], 1);
            Cgerc(m, ixfrm, -COMPLEX(factor), &x[(2 * nxfrm + 1) - 1], 1, &x[kbeg - 1], 1, &a[(kbeg - 1) * lda], lda);
            //
        }
    }
    //
    x[1 - 1] = Clarnd(3, iseed);
    xabs = abs(x[1 - 1]);
    const REAL zero = 0.0;
    if (xabs != zero) {
        csign = x[1 - 1] / xabs;
    } else {
        csign = cone;
    }
    x[(2 * nxfrm) - 1] = csign;
    //
    //     Scale the matrix A by D.
    //
    INTEGER irow = 0;
    if (itype == 1 || itype == 3 || itype == 4) {
        for (irow = 1; irow <= m; irow = irow + 1) {
            Cscal(n, conj(x[(nxfrm + irow) - 1]), &a[(irow - 1)], lda);
        }
    }
    //
    INTEGER jcol = 0;
    if (itype == 2 || itype == 3) {
        for (jcol = 1; jcol <= n; jcol = jcol + 1) {
            Cscal(m, x[(nxfrm + jcol) - 1], &a[(jcol - 1) * lda], 1);
        }
    }
    //
    if (itype == 4) {
        for (jcol = 1; jcol <= n; jcol = jcol + 1) {
            Cscal(m, conj(x[(nxfrm + jcol) - 1]), &a[(jcol - 1) * lda], 1);
        }
    }
    //
    //     End of Claror
    //
}

#if defined ___MPLAPACK_BUILD_WITH_DD___
#pragma GCC pop_options
#endif

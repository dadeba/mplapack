/*
 * Copyright (c) 2021
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
#include <mplapack_lin.h>

inline REAL abs1(COMPLEX zdum) { return abs(zdum.real()) + abs(zdum.imag()); }

void Cget07(const char *trans, INTEGER const n, INTEGER const nrhs, COMPLEX *a, INTEGER const lda, COMPLEX *b, INTEGER const ldb, COMPLEX *x, INTEGER const ldx, COMPLEX *xact, INTEGER const ldxact, REAL *ferr, bool const chkferr, REAL *berr, REAL *reslts) {
    COMPLEX zdum = 0.0;
    const REAL zero = 0.0;
    REAL eps = 0.0;
    REAL unfl = 0.0;
    const REAL one = 1.0;
    REAL ovfl = 0.0;
    bool notran = false;
    REAL errbnd = 0.0;
    INTEGER j = 0;
    INTEGER imax = 0;
    REAL xnorm = 0.0;
    REAL diff = 0.0;
    INTEGER i = 0;
    INTEGER k = 0;
    REAL tmp = 0.0;
    REAL axbi = 0.0;
    //
    //     Quick exit if N = 0 or NRHS = 0.
    //
    if (n <= 0 || nrhs <= 0) {
        reslts[1 - 1] = zero;
        reslts[2 - 1] = zero;
        return;
    }
    //
    eps = Rlamch("Epsilon");
    unfl = Rlamch("Safe minimum");
    ovfl = one / unfl;
    notran = Mlsame(trans, "N");
    //
    //     Test 1:  Compute the maximum of
    //        norm(X - XACT) / ( norm(X) * FERR )
    //     over all the vectors X and XACT using the infinity-norm.
    //
    errbnd = zero;
    if (chkferr) {
        for (j = 1; j <= nrhs; j = j + 1) {
            imax = iCamax(n, &x[(j - 1) * ldx], 1);
            xnorm = max(abs1(x[(imax - 1) + (j - 1) * ldx]), unfl);
            diff = zero;
            for (i = 1; i <= n; i = i + 1) {
                diff = max(diff, abs1(x[(i - 1) + (j - 1) * ldx] - xact[(i - 1) + (j - 1) * ldxact]));
            }
            //
            if (xnorm > one) {
                goto statement_20;
            } else if (diff <= ovfl * xnorm) {
                goto statement_20;
            } else {
                errbnd = one / eps;
                goto statement_30;
            }
        //
        statement_20:
            if (diff / xnorm <= ferr[j - 1]) {
                errbnd = max(errbnd, REAL((diff / xnorm) / ferr[j - 1]));
            } else {
                errbnd = one / eps;
            }
        statement_30:;
        }
    }
    reslts[1 - 1] = errbnd;
    //
    //     Test 2:  Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
    //     (*) = (n+1)*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
    //
    for (k = 1; k <= nrhs; k = k + 1) {
        for (i = 1; i <= n; i = i + 1) {
            tmp = abs1(b[(i - 1) + (k - 1) * ldb]);
            if (notran) {
                for (j = 1; j <= n; j = j + 1) {
                    tmp += abs1(a[(i - 1) + (j - 1) * lda]) * abs1(x[(j - 1) + (k - 1) * ldx]);
                }
            } else {
                for (j = 1; j <= n; j = j + 1) {
                    tmp += abs1(a[(j - 1) + (i - 1) * lda]) * abs1(x[(j - 1) + (k - 1) * ldx]);
                }
            }
            if (i == 1) {
                axbi = tmp;
            } else {
                axbi = min(axbi, tmp);
            }
        }
        tmp = berr[k - 1] / (castREAL(n + 1) * eps + castREAL(n + 1) * unfl / max(axbi, REAL(castREAL(n + 1) * unfl)));
        if (k == 1) {
            reslts[2 - 1] = tmp;
        } else {
            reslts[2 - 1] = max(reslts[2 - 1], tmp);
        }
    }
    //
    //     End of Cget07
    //
}

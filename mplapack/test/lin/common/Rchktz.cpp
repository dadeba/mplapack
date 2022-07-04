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

#include <mplapack_debug.h>

void Rchktz(bool *dotype, INTEGER const nm, INTEGER *mval, INTEGER const nn, INTEGER *nval, REAL const thresh, bool const tsterr, REAL *a, REAL *copya, REAL *s, REAL *tau, REAL *work, INTEGER const nout) {
    common cmn;
    common_write write(cmn);
    INTEGER iseedy[] = {1988, 1989, 1990, 1991};
    char path[4] = {};
    char buf[1024];
    INTEGER nrun = 0;
    INTEGER nfail = 0;
    INTEGER nerrs = 0;
    INTEGER i = 0;
    INTEGER iseed[4];
    REAL eps = 0.0;
    INTEGER im = 0;
    INTEGER m = 0;
    INTEGER lda = 0;
    INTEGER in = 0;
    INTEGER n = 0;
    INTEGER mnmin = 0;
    INTEGER lwork = 0;
    INTEGER imode = 0;
    const INTEGER ntypes = 3;
    INTEGER mode = 0;
    const REAL zero = 0.0;
    const REAL one = 1.0;
    INTEGER info = 0;
    const INTEGER ntests = 3;
    REAL result[ntests];
    INTEGER k = 0;
    //
    path[0] = 'R';
    path[1] = 'T';
    path[2] = 'Z';
    nrun = 0;
    nfail = 0;
    nerrs = 0;
    for (i = 1; i <= 4; i = i + 1) {
        iseed[i - 1] = iseedy[i - 1];
    }
    eps = Rlamch("Epsilon");
    //
    //     Test the error exits
    //
    if (tsterr) {
        Rerrtz(path, nout);
    }
    infot = 0;
    //
    for (im = 1; im <= nm; im = im + 1) {
        //
        //        Do for each value of M in MVAL.
        //
        m = mval[im - 1];
        lda = max((INTEGER)1, m);
        //
        for (in = 1; in <= nn; in = in + 1) {
            //
            //           Do for each value of N in NVAL for which M .LE. N.
            //
            n = nval[in - 1];
            mnmin = min(m, n);
            lwork = max({(INTEGER)1, n * n + 4 * m + n, m * n + 2 * mnmin + 4 * n});
            //
            if (m <= n) {
                for (imode = 1; imode <= ntypes; imode = imode + 1) {
                    if (!dotype[imode - 1]) {
                        goto statement_50;
                    }
                    //
                    //                 Do for each type of singular value distribution.
                    //                    0:  zero matrix
                    //                    1:  one small singular value
                    //                    2:  exponential distribution
                    //
                    mode = imode - 1;
                    //
                    //                 Test DTZRQF
                    //
                    //                 Generate test matrix of size m by n using
                    //                 singular value distribution indicated by `mode'.
                    //
                    if (mode == 0) {
                        Rlaset("Full", m, n, zero, zero, a, lda);
                        for (i = 1; i <= mnmin; i = i + 1) {
                            s[i - 1] = zero;
                        }
                    } else {
                        Rlatms(m, n, "Uniform", iseed, "Nonsymmetric", s, imode, one / eps, one, m, n, "No packing", a, lda, work, info);
                        Rgeqr2(m, n, a, lda, work, &work[(mnmin + 1) - 1], info);
                        Rlaset("Lower", m - 1, n, zero, zero, &a[2 - 1], lda);
                        Rlaord("Decreasing", mnmin, s, 1);
                    }
                    //
                    //                 Save A and its singular values
                    //
                    Rlacpy("All", m, n, a, lda, copya, lda);
                    //
                    //                 Call Rtzrzf to reduce the upper trapezoidal matrix to
                    //                 upper triangular form.
                    //
                    strncpy(srnamt, "Rtzrzf", srnamt_len);
                    Rtzrzf(m, n, a, lda, tau, work, lwork, info);
                    //
                    //                 Compute norm(svd(a) - svd(r))
                    //
                    result[1 - 1] = Rqrt12(m, m, a, lda, s, work, lwork);
                    //
                    //                 Compute norm( A - R*Q )
                    //
                    result[2 - 1] = Rrzt01(m, n, copya, a, lda, tau, work, lwork);
                    //
                    //                 Compute norm(Q'*Q - I).
                    //
                    result[3 - 1] = Rrzt02(m, n, a, lda, tau, work, lwork);
                    //
                    //                 Print information about the tests that did not pass
                    //                 the threshold.
                    //
                    for (k = 1; k <= ntests; k = k + 1) {
                        if (result[k - 1] >= thresh) {
                            if (nfail == 0 && nerrs == 0) {
                                Alahd(nout, path);
                            }
                            sprintnum_short(buf, result[k - 1]);
                            write(nout, "(' M =',i5,', N =',i5,', type ',i2,', test ',i2,', ratio =',"
                                        "a)"),
                                m, n, imode, k, buf;
                            nfail++;
                        }
                    }
                    nrun += 3;
                statement_50:;
                }
            }
        }
    }
    //
    //     Print a summary of the results.
    //
    Alasum(path, nout, nfail, nrun, nerrs);
    //
    //     End if Rchktz
    //
}

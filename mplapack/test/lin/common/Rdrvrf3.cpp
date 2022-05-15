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
#include <mplapack_lin.h>

#include <mplapack_debug.h>

void Rdrvrf3(INTEGER const nout, INTEGER const nn, INTEGER *nval, REAL const thresh, REAL *a, INTEGER const lda, REAL *arf, REAL *b1, REAL *b2, REAL *d_work_Rlange, REAL *d_work_Rgeqrf, REAL *tau) {
    //
    //     Initialize constants and the random number seed.
    //
    common cmn;
    common_write write(cmn);
    char transs[] = {'N', 'T'};
    char diags[] = {'N', 'U'};
    char forms[] = {'N', 'T'};
    char sides[] = {'L', 'R'};
    char uplos[] = {'U', 'L'};
    char buf[1024];
    INTEGER iseedy[] = {1988, 1989, 1990, 1991};
    INTEGER ldb1 = lda;
    INTEGER ldb2 = lda;
    INTEGER nrun = 0;
    INTEGER nfail = 0;
    INTEGER info = 0;
    INTEGER i = 0;
    INTEGER iseed[4];
    for (i = 1; i <= 4; i = i + 1) {
        iseed[i - 1] = iseedy[i - 1];
    }
    REAL eps = Rlamch("Precision");
    //
    INTEGER iim = 0;
    INTEGER m = 0;
    INTEGER iin = 0;
    INTEGER n = 0;
    INTEGER iform = 0;
    char cform;
    INTEGER iuplo = 0;
    char uplo;
    INTEGER iside = 0;
    char side;
    INTEGER itrans = 0;
    char trans;
    INTEGER idiag = 0;
    char diag;
    INTEGER ialpha = 0;
    const REAL zero = (0.0, 0.0);
    REAL alpha = 0.0;
    const REAL one = (1.0, 0.0);
    INTEGER na = 0;
    INTEGER j = 0;
    const INTEGER ntests = 1;
    REAL result[ntests];
    for (iim = 1; iim <= nn; iim = iim + 1) {
        //
        m = nval[iim - 1];
        //
        for (iin = 1; iin <= nn; iin = iin + 1) {
            //
            n = nval[iin - 1];
            //
            for (iform = 1; iform <= 2; iform = iform + 1) {
                //
                cform = forms[iform - 1];
                //
                for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                    //
                    uplo = uplos[iuplo - 1];
                    //
                    for (iside = 1; iside <= 2; iside = iside + 1) {
                        //
                        side = sides[iside - 1];
                        //
                        for (itrans = 1; itrans <= 2; itrans = itrans + 1) {
                            //
                            trans = transs[itrans - 1];
                            //
                            for (idiag = 1; idiag <= 2; idiag = idiag + 1) {
                                //
                                diag = diags[idiag - 1];
                                //
                                for (ialpha = 1; ialpha <= 3; ialpha = ialpha + 1) {
                                    //
                                    if (ialpha == 1) {
                                        alpha = zero;
                                    } else if (ialpha == 2) {
                                        alpha = one;
                                    } else {
                                        alpha = Rlarnd(2, iseed);
                                    }
                                    //
                                    //                             All the parameters are set:
                                    //                                CFORM, SIDE, UPLO, TRANS, DIAG, M, N,
                                    //                                and ALPHA
                                    //                             READY TO TEST!
                                    //
                                    nrun++;
                                    //
                                    if (iside == 1) {
                                        //
                                        //                                The case ISIDE.EQ.1 is when SIDE.EQ.'L'
                                        //                                -> A is M-by-M ( B is M-by-N )
                                        //
                                        na = m;
                                        //
                                    } else {
                                        //
                                        //                                The case ISIDE.EQ.2 is when SIDE.EQ.'R'
                                        //                                -> A is N-by-N ( B is M-by-N )
                                        //
                                        na = n;
                                        //
                                    }
                                    //
                                    //                             Generate A our NA--by--NA triangular
                                    //                             matrix.
                                    //                             Our test is based on forward error so we
                                    //                             do want A to be well conditioned! To get
                                    //                             a well-conditioned triangular matrix, we
                                    //                             take the R factor of the QR/LQ factorization
                                    //                             of a random matrix.
                                    //
                                    for (j = 1; j <= na; j = j + 1) {
                                        for (i = 1; i <= na; i = i + 1) {
                                            a[(i - 1) + (j - 1) * lda] = Rlarnd(2, iseed);
                                        }
                                    }
                                    //
                                    if (iuplo == 1) {
                                        //
                                        //                                The case IUPLO.EQ.1 is when SIDE.EQ.'U'
                                        //                                -> QR factorization.
                                        //
                                        Rgeqrf(na, na, a, lda, tau, d_work_Rgeqrf, lda, info);
                                    } else {
                                        //
                                        //                                The case IUPLO.EQ.2 is when SIDE.EQ.'L'
                                        //                                -> QL factorization.
                                        //
                                        Rgelqf(na, na, a, lda, tau, d_work_Rgeqrf, lda, info);
                                    }
                                    //
                                    //                             Store a copy of A in RFP format (in ARF).
                                    //
                                    Rtrttf(&cform, &uplo, na, a, lda, arf, info);
                                    //
                                    //                             Generate B1 our M--by--N right-hand side
                                    //                             and store a copy in B2.
                                    //
                                    for (j = 1; j <= n; j = j + 1) {
                                        for (i = 1; i <= m; i = i + 1) {
                                            b1[(i - 1) + (j - 1) * ldb1] = Rlarnd(2, iseed);
                                            b2[(i - 1) + (j - 1) * ldb2] = b1[(i - 1) + (j - 1) * ldb1];
                                        }
                                    }
                                    //
                                    //                             Solve op( A ) X = B or X op( A ) = B
                                    //                             with Rtrsm
                                    //
                                    Rtrsm(&side, &uplo, &trans, &diag, m, n, alpha, a, lda, b1, lda);
                                    //
                                    //                             Solve op( A ) X = B or X op( A ) = B
                                    //                             with Rtfsm
                                    //
                                    Rtfsm(&cform, &side, &uplo, &trans, &diag, m, n, alpha, arf, b2, lda);
                                    //
                                    //                             Check that the result agrees.
                                    //
                                    for (j = 1; j <= n; j = j + 1) {
                                        for (i = 1; i <= m; i = i + 1) {
                                            b1[(i - 1) + (j - 1) * ldb1] = b2[(i - 1) + (j - 1) * ldb2] - b1[(i - 1) + (j - 1) * ldb1];
                                        }
                                    }
                                    //
                                    result[1 - 1] = Rlange("I", m, n, b1, lda, d_work_Rlange);
                                    //
                                    result[1 - 1] = result[1 - 1] / sqrt(eps) / castREAL(max({max(m, n), (INTEGER)1}));
                                    //
                                    if (result[1 - 1] >= thresh) {
                                        if (nfail == 0) {
                                            write(nout, star);
                                            write(nout, "(1x,' *** Error(s) or Failure(s) while testing Rtfsm "
                                                        "        ***')");
                                        }
                                        sprintnum_short(buf, result[1 - 1]);
                                        write(nout, "(1x,'     Failure in ',a5,', CFORM=''',a1,''',',"
                                                    "' SIDE=''',a1,''',',' UPLO=''',a1,''',',' TRANS=''',a1,"
                                                    "''',',' DIAG=''',a1,''',',' M=',i3,', N =',i3,"
                                                    "', test=',a)"),
                                            "Rtfsm", &cform, &side, &uplo, &trans, &diag, m, n, buf;
                                        nfail++;
                                    }
                                    //
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //
    //     Print a summary of the results.
    //
    if (nfail == 0) {
        write(nout, "(1x,'All tests for ',a5,' auxiliary routine passed the ',"
                    "'threshold ( ',i5,' tests run)')"),
            "Rtfsm", nrun;
    } else {
        write(nout, "(1x,a6,' auxiliary routine: ',i5,' out of ',i5,"
                    "' tests failed to pass the threshold')"),
            "Rtfsm", nfail, nrun;
    }
    //
    //     End of Rdrvrf3
    //
}

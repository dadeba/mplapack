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

void Cdrvrf4(INTEGER const nout, INTEGER const nn, INTEGER *nval, REAL const thresh, COMPLEX *c1, COMPLEX *c2, INTEGER const ldc, COMPLEX *crf, COMPLEX *a, INTEGER const lda, REAL *d_work_Clange) {
    common cmn;
    common_write write(cmn);
    //
    //     Initialize constants and the random number seed.
    //
    INTEGER ldc1 = ldc;
    INTEGER ldc2 = ldc;
    char transs[] = {'N', 'C'};
    char uplos[] = {'U', 'L'};
    char forms[] = {'N', 'C'};
    char buf[1024];
    INTEGER iseedy[] = {1988, 1989, 1990, 1991};
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
    INTEGER iin = 0;
    INTEGER n = 0;
    INTEGER iik = 0;
    INTEGER k = 0;
    INTEGER iform = 0;
    char cform;
    INTEGER iuplo = 0;
    char uplo;
    INTEGER itrans = 0;
    char trans;
    INTEGER ialpha = 0;
    const REAL zero = 0.0;
    REAL alpha = 0.0;
    REAL beta = 0.0;
    const REAL one = 1.0;
    INTEGER j = 0;
    REAL norma = 0.0;
    REAL normc = 0.0;
    const INTEGER ntests = 1;
    REAL result[ntests];
    for (iin = 1; iin <= nn; iin = iin + 1) {
        //
        n = nval[iin - 1];
        //
        for (iik = 1; iik <= nn; iik = iik + 1) {
            //
            k = nval[iin - 1];
            //
            for (iform = 1; iform <= 2; iform = iform + 1) {
                //
                cform = forms[iform - 1];
                //
                for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                    //
                    uplo = uplos[iuplo - 1];
                    //
                    for (itrans = 1; itrans <= 2; itrans = itrans + 1) {
                        //
                        trans = transs[itrans - 1];
                        //
                        for (ialpha = 1; ialpha <= 4; ialpha = ialpha + 1) {
                            //
                            if (ialpha == 1) {
                                alpha = zero;
                                beta = zero;
                            } else if (ialpha == 2) {
                                alpha = one;
                                beta = zero;
                            } else if (ialpha == 3) {
                                alpha = zero;
                                beta = one;
                            } else {
                                alpha = Rlarnd(2, iseed);
                                beta = Rlarnd(2, iseed);
                            }
                            //
                            //                       All the parameters are set:
                            //                          CFORM, UPLO, TRANS, M, N,
                            //                          ALPHA, and BETA
                            //                       READY TO TEST!
                            //
                            nrun++;
                            //
                            if (itrans == 1) {
                                //
                                //                          In this case we are NOTRANS, so A is N-by-K
                                //
                                for (j = 1; j <= k; j = j + 1) {
                                    for (i = 1; i <= n; i = i + 1) {
                                        a[(i - 1) + (j - 1) * lda] = Clarnd(4, iseed);
                                    }
                                }
                                //
                                norma = Clange("I", n, k, a, lda, d_work_Clange);
                                //
                            } else {
                                //
                                //                          In this case we are TRANS, so A is K-by-N
                                //
                                for (j = 1; j <= n; j = j + 1) {
                                    for (i = 1; i <= k; i = i + 1) {
                                        a[(i - 1) + (j - 1) * lda] = Clarnd(4, iseed);
                                    }
                                }
                                //
                                norma = Clange("I", k, n, a, lda, d_work_Clange);
                                //
                            }
                            //
                            //                       Generate C1 our N--by--N Hermitian matrix.
                            //                       Make sure C2 has the same upper/lower part,
                            //                       (the one that we do not touch), so
                            //                       copy the initial C1 in C2 in it.
                            //
                            for (j = 1; j <= n; j = j + 1) {
                                for (i = 1; i <= n; i = i + 1) {
                                    c1[(i - 1) + (j - 1) * ldc1] = Clarnd(4, iseed);
                                    c2[(i - 1) + (j - 1) * ldc2] = c1[(i - 1) + (j - 1) * ldc1];
                                }
                            }
                            //
                            //                       (See comment later on for why we use Clange and
                            //                       not Clanhe for C1.)
                            //
                            normc = Clange("I", n, n, c1, ldc, d_work_Clange);
                            //
                            Ctrttf(&cform, &uplo, n, c1, ldc, crf, info);
                            //
                            //                       call Cherk the BLAS routine -> gives C1
                            //
                            Cherk(&uplo, &trans, n, k, alpha, a, lda, beta, c1, ldc);
                            //
                            //                       call Chfrk the RFP routine -> gives CRF
                            //
                            Chfrk(&cform, &uplo, &trans, n, k, alpha, a, lda, beta, crf);
                            //
                            //                       convert CRF in full format -> gives C2
                            //
                            Ctfttr(&cform, &uplo, n, crf, c2, ldc, info);
                            //
                            //                       compare C1 and C2
                            //
                            for (j = 1; j <= n; j = j + 1) {
                                for (i = 1; i <= n; i = i + 1) {
                                    c1[(i - 1) + (j - 1) * ldc1] = c1[(i - 1) + (j - 1) * ldc1] - c2[(i - 1) + (j - 1) * ldc2];
                                }
                            }
                            //
                            //                       Yes, C1 is Hermitian so we could call Clanhe,
                            //                       but we want to check the upper part that is
                            //                       supposed to be unchanged and the diagonal that
                            //                       is supposed to be real -> Clange
                            //
                            result[1 - 1] = Clange("I", n, n, c1, ldc, d_work_Clange);
                            result[1 - 1] = result[1 - 1] / max(REAL(abs(alpha) * norma * norma + abs(beta) * normc), one) / castREAL(max(n, (INTEGER)1)) / eps;
                            //
                            if (result[1 - 1] >= thresh) {
                                if (nfail == 0) {
                                    write(nout, star);
                                    write(nout, "(1x,' *** Error(s) or Failure(s) while testing Chfrk     "
                                                "    ***')");
                                }
                                sprintnum_short(buf, result[0]);
                                write(nout, "(1x,'     Failure in ',a5,', CFORM=''',a1,''',',' UPLO=''',"
                                            "a1,''',',' TRANS=''',a1,''',',' N=',i3,', K =',i3,"
                                            "', test=',a)"),
                                    "Chfrk", &cform, &uplo, &trans, n, k, buf;
                                nfail++;
                            }
                            //
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
                    "'threshold ( ',i6,' tests run)')"),
            "Chfrk", nrun;
    } else {
        write(nout, "(1x,a6,' auxiliary routine: ',i6,' out of ',i6,"
                    "' tests failed to pass the threshold')"),
            "Chfrk", nfail, nrun;
    }
    //
    //     End of Cdrvrf4
    //
}

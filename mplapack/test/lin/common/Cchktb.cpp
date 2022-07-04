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

void Cchktb(bool *dotype, INTEGER const nn, INTEGER *nval, INTEGER const nns, INTEGER *nsval, REAL const thresh, bool const tsterr, INTEGER const /* nmax */, COMPLEX *ab, COMPLEX *ainv, COMPLEX *b, COMPLEX *x, COMPLEX *xact, COMPLEX *work, REAL *rwork, INTEGER const nout) {
    common cmn;
    common_write write(cmn);
    //
    INTEGER iseedy[] = {1988, 1989, 1990, 1991};
    const INTEGER ntran = 3;
    char norm_trans_diag[4];
    char norm_uplo_diag[4];
    char uplo_trans_diag[4];
    char uplos[] = {'U', 'L'};
    char transs[] = {'N', 'T', 'C'};
    char path[4] = {};
    char buf[1024];
    INTEGER nrun = 0;
    INTEGER nfail = 0;
    INTEGER nerrs = 0;
    INTEGER i = 0;
    INTEGER iseed[4];
    INTEGER in = 0;
    INTEGER n = 0;
    INTEGER lda = 0;
    char xtype;
    const INTEGER ntype1 = 9;
    INTEGER nimat = 0;
    const INTEGER ntypes = 17;
    INTEGER nimat2 = 0;
    INTEGER nk = 0;
    INTEGER ik = 0;
    INTEGER kd = 0;
    INTEGER ldab = 0;
    INTEGER imat = 0;
    INTEGER iuplo = 0;
    char uplo[1];
    char diag[1];
    INTEGER info = 0;
    INTEGER idiag = 0;
    const REAL zero = 0.0;
    const REAL one = 1.0;
    INTEGER j = 0;
    REAL anorm = 0.0;
    REAL ainvnm = 0.0;
    REAL rcondo = 0.0;
    REAL rcondi = 0.0;
    INTEGER irhs = 0;
    INTEGER nrhs = 0;
    INTEGER itran = 0;
    char trans[1];
    char norm[1];
    REAL rcondc = 0.0;
    const INTEGER ntests = 8;
    REAL result[ntests];
    INTEGER k = 0;
    REAL rcond = 0.0;
    REAL scale = 0.0;
    static const char *format_9997 = "(1x,a,'( ''',a1,''', ''',a1,''', ''',a1,''', ''',a1,''',',i5,',',i5,"
                                     "', ...  ),  type ',i2,', test(',i1,')=',a)";
    //
    //     Initialize constants and the random number seed.
    //
    path[0] = 'C';
    path[1] = 'T';
    path[2] = 'B';
    nrun = 0;
    nfail = 0;
    nerrs = 0;
    for (i = 1; i <= 4; i = i + 1) {
        iseed[i - 1] = iseedy[i - 1];
    }
    //
    //     Test the error exits
    //
    if (tsterr) {
        Cerrtr(path, nout);
    }
    //
    for (in = 1; in <= nn; in = in + 1) {
        //
        //        Do for each value of N in NVAL
        //
        n = nval[in - 1];
        lda = max((INTEGER)1, n);
        xtype = 'N';
        nimat = ntype1;
        nimat2 = ntypes;
        if (n <= 0) {
            nimat = 1;
            nimat2 = ntype1 + 1;
        }
        //
        nk = min(n + 1, (INTEGER)4);
        for (ik = 1; ik <= nk; ik = ik + 1) {
            //
            //           Do for KD = 0, N, (3N-1)/4, and (N+1)/4. This order makes
            //           it easier to skip redundant values for small values of N.
            //
            if (ik == 1) {
                kd = 0;
            } else if (ik == 2) {
                kd = max(n, (INTEGER)0);
            } else if (ik == 3) {
                kd = (3 * n - 1) / 4;
            } else if (ik == 4) {
                kd = (n + 1) / 4;
            }
            ldab = kd + 1;
            //
            for (imat = 1; imat <= nimat; imat = imat + 1) {
                //
                //              Do the tests only if DOTYPE( IMAT ) is true.
                //
                if (!dotype[imat - 1]) {
                    goto statement_90;
                }
                //
                for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                    //
                    //                 Do first for UPLO = 'U', then for UPLO = 'L'
                    //
                    uplo[0] = uplos[iuplo - 1];
                    //
                    //                 Call Clattb to generate a triangular test matrix.
                    //
                    strncpy(srnamt, "Clattb", srnamt_len);
                    Clattb(imat, uplo, "No transpose", diag, iseed, n, kd, ab, ldab, x, work, rwork, info);
                    //
                    //                 Set IDIAG = 1 for non-unit matrices, 2 for unit.
                    //
                    if (Mlsame(diag, "N")) {
                        idiag = 1;
                    } else {
                        idiag = 2;
                    }
                    //
                    //                 Form the inverse of A so we can get a good estimate
                    //                 of RCONDC = 1/(norm(A) * norm(inv(A))).
                    //
                    Claset("Full", n, n, COMPLEX(zero), COMPLEX(one), ainv, lda);
                    if (Mlsame(uplo, "U")) {
                        for (j = 1; j <= n; j = j + 1) {
                            Ctbsv(uplo, "No transpose", diag, j, kd, ab, ldab, &ainv[((j - 1) * lda + 1) - 1], 1);
                        }
                    } else {
                        for (j = 1; j <= n; j = j + 1) {
                            Ctbsv(uplo, "No transpose", diag, n - j + 1, kd, &ab[((j - 1) * ldab + 1) - 1], ldab, &ainv[((j - 1) * lda + j) - 1], 1);
                        }
                    }
                    //
                    //                 Compute the 1-norm condition number of A.
                    //
                    anorm = Clantb("1", uplo, diag, n, kd, ab, ldab, rwork);
                    ainvnm = Clantr("1", uplo, diag, n, n, ainv, lda, rwork);
                    if (anorm <= zero || ainvnm <= zero) {
                        rcondo = one;
                    } else {
                        rcondo = (one / anorm) / ainvnm;
                    }
                    //
                    //                 Compute the infinity-norm condition number of A.
                    //
                    anorm = Clantb("I", uplo, diag, n, kd, ab, ldab, rwork);
                    ainvnm = Clantr("I", uplo, diag, n, n, ainv, lda, rwork);
                    if (anorm <= zero || ainvnm <= zero) {
                        rcondi = one;
                    } else {
                        rcondi = (one / anorm) / ainvnm;
                    }
                    //
                    for (irhs = 1; irhs <= nns; irhs = irhs + 1) {
                        nrhs = nsval[irhs - 1];
                        xtype = 'N';
                        //
                        for (itran = 1; itran <= ntran; itran = itran + 1) {
                            //
                            //                    Do for op(A) = A, A**T, or A**H.
                            //
                            trans[0] = transs[itran - 1];
                            if (itran == 1) {
                                norm[0] = 'O';
                                rcondc = rcondo;
                            } else {
                                norm[0] = 'I';
                                rcondc = rcondi;
                            }
                            //
                            //+    TEST 1
                            //                    Solve and compute residual for op(A)*x = b.
                            //
                            strncpy(srnamt, "Clarhs", srnamt_len);
                            Clarhs(path, &xtype, uplo, trans, n, n, kd, idiag, nrhs, ab, ldab, xact, lda, b, lda, iseed, info);
                            xtype = 'C';
                            Clacpy("Full", n, nrhs, b, lda, x, lda);
                            //
                            strncpy(srnamt, "Ctbtrs", srnamt_len);
                            Ctbtrs(uplo, trans, diag, n, kd, nrhs, ab, ldab, x, lda, info);
                            //
                            //                    Check error code from Ctbtrs.
                            //
                            if (info != 0) {
                                uplo_trans_diag[0] = uplo[0];
                                uplo_trans_diag[1] = trans[0];
                                uplo_trans_diag[2] = diag[0];
                                uplo_trans_diag[3] = '\0';

                                Alaerh(path, "Ctbtrs", info, 0, uplo_trans_diag, n, n, kd, kd, nrhs, imat, nfail, nerrs, nout);
                            }
                            //
                            Ctbt02(uplo, trans, diag, n, kd, nrhs, ab, ldab, x, lda, b, lda, work, rwork, result[1 - 1]);
                            //
                            //+    TEST 2
                            //                    Check solution from generated exact solution.
                            //
                            Cget04(n, nrhs, x, lda, xact, lda, rcondc, result[2 - 1]);
                            //
                            //+    TESTS 3, 4, and 5
                            //                    Use iterative refinement to improve the solution
                            //                    and compute error bounds.
                            //
                            strncpy(srnamt, "Ctbrfs", srnamt_len);
                            Ctbrfs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, lda, x, lda, rwork, &rwork[(nrhs + 1) - 1], work, &rwork[(2 * nrhs + 1) - 1], info);
                            //
                            //                    Check error code from Ctbrfs.
                            //
                            if (info != 0) {
                                uplo_trans_diag[0] = uplo[0];
                                uplo_trans_diag[1] = trans[0];
                                uplo_trans_diag[2] = diag[0];
                                uplo_trans_diag[3] = '\0';
                                Alaerh(path, "Ctbrfs", info, 0, uplo_trans_diag, n, n, kd, kd, nrhs, imat, nfail, nerrs, nout);
                            }
                            //
                            Cget04(n, nrhs, x, lda, xact, lda, rcondc, result[3 - 1]);
                            Ctbt05(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, lda, x, lda, xact, lda, rwork, &rwork[(nrhs + 1) - 1], &result[4 - 1]);
                            //
                            //                       Print information about the tests that did not
                            //                       pass the threshold.
                            //
                            for (k = 1; k <= 5; k = k + 1) {
                                if (result[k - 1] >= thresh) {
                                    if (nfail == 0 && nerrs == 0) {
                                        Alahd(nout, path);
                                    }
                                    sprintnum_short(buf, result[k - 1]);
                                    write(nout, "(' UPLO=''',a1,''', TRANS=''',a1,''',      DIAG=''',a1,"
                                                "''', N=',i5,', KD=',i5,', NRHS=',i5,', type ',i2,"
                                                "', test(',i2,')=',a)"),
                                        uplo, trans, diag, n, kd, nrhs, imat, k, buf;
                                    nfail++;
                                }
                            }
                            nrun += 5;
                        }
                    }
                    //
                    //+    TEST 6
                    //                    Get an estimate of RCOND = 1/CNDNUM.
                    //
                    for (itran = 1; itran <= 2; itran = itran + 1) {
                        if (itran == 1) {
                            norm[0] = 'O';
                            rcondc = rcondo;
                        } else {
                            norm[0] = 'I';
                            rcondc = rcondi;
                        }
                        strncpy(srnamt, "Ctbcon", srnamt_len);
                        Ctbcon(norm, uplo, diag, n, kd, ab, ldab, rcond, work, rwork, info);
                        //
                        //                    Check error code from Ctbcon.
                        //
                        if (info != 0) {
                            norm_uplo_diag[0] = norm[0];
                            norm_uplo_diag[1] = uplo[0];
                            norm_uplo_diag[2] = diag[0];
                            norm_uplo_diag[3] = '\0';
                            Alaerh(path, "Ctbcon", info, 0, norm_uplo_diag, n, n, kd, kd, -1, imat, nfail, nerrs, nout);
                        }
                        //
                        Ctbt06(rcond, rcondc, uplo, diag, n, kd, ab, ldab, rwork, result[6 - 1]);
                        //
                        //                    Print the test ratio if it is .GE. THRESH.
                        //
                        if (result[6 - 1] >= thresh) {
                            if (nfail == 0 && nerrs == 0) {
                                Alahd(nout, path);
                            }
                            sprintnum_short(buf, result[6 - 1]);
                            write(nout, "(1x,a,'( ''',a1,''', ''',a1,''', ''',a1,''',',i5,',',i5,"
                                        "',  ... ), type ',i2,', test(',i2,')=',a)"),
                                "Ctbcon", norm, uplo, diag, n, kd, imat, 6, buf;
                            nfail++;
                        }
                        nrun++;
                    }
                }
            statement_90:;
            }
            //
            //           Use pathological test matrices to test Clatbs.
            //
            for (imat = ntype1 + 1; imat <= nimat2; imat = imat + 1) {
                //
                //              Do the tests only if DOTYPE( IMAT ) is true.
                //
                if (!dotype[imat - 1]) {
                    goto statement_120;
                }
                //
                for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                    //
                    //                 Do first for UPLO = 'U', then for UPLO = 'L'
                    //
                    uplo[0] = uplos[iuplo - 1];
                    for (itran = 1; itran <= ntran; itran = itran + 1) {
                        //
                        //                    Do for op(A) = A, A**T, and A**H.
                        //
                        trans[0] = transs[itran - 1];
                        //
                        //                    Call Clattb to generate a triangular test matrix.
                        //
                        strncpy(srnamt, "Clattb", srnamt_len);
                        Clattb(imat, uplo, trans, diag, iseed, n, kd, ab, ldab, x, work, rwork, info);
                        //
                        //+    TEST 7
                        //                    Solve the system op(A)*x = b
                        //
                        strncpy(srnamt, "Clatbs", srnamt_len);
                        Ccopy(n, x, 1, b, 1);
                        Clatbs(uplo, trans, diag, "N", n, kd, ab, ldab, b, scale, rwork, info);
                        //
                        //                    Check error code from Clatbs.
                        //
                        if (info != 0) {
                            uplo_trans_diag[0] = uplo[0];
                            uplo_trans_diag[1] = trans[0];
                            uplo_trans_diag[2] = diag[0];
                            uplo_trans_diag[3] = 'N';
                            Alaerh(path, "Clatbs", info, 0, uplo_trans_diag, n, n, kd, kd, -1, imat, nfail, nerrs, nout);
                        }
                        //
                        Ctbt03(uplo, trans, diag, n, kd, 1, ab, ldab, scale, rwork, one, b, lda, x, lda, work, result[7 - 1]);
                        //
                        //+    TEST 8
                        //                    Solve op(A)*x = b again with NORMIN = 'Y'.
                        //
                        Ccopy(n, x, 1, b, 1);
                        Clatbs(uplo, trans, diag, "Y", n, kd, ab, ldab, b, scale, rwork, info);
                        //
                        //                    Check error code from Clatbs.
                        //
                        if (info != 0) {
                            uplo_trans_diag[0] = uplo[0];
                            uplo_trans_diag[1] = trans[0];
                            uplo_trans_diag[2] = diag[0];
                            uplo_trans_diag[3] = 'Y';
                            Alaerh(path, "Clatbs", info, 0, uplo_trans_diag, n, n, kd, kd, -1, imat, nfail, nerrs, nout);
                        }
                        //
                        Ctbt03(uplo, trans, diag, n, kd, 1, ab, ldab, scale, rwork, one, b, lda, x, lda, work, result[8 - 1]);
                        //
                        //                    Print information about the tests that did not pass
                        //                    the threshold.
                        //
                        if (result[7 - 1] >= thresh) {
                            if (nfail == 0 && nerrs == 0) {
                                Alahd(nout, path);
                            }
                            sprintnum_short(buf, result[7 - 1]);
                            write(nout, format_9997), "Clatbs", uplo, trans, diag, "N", n, kd, imat, 7, buf;
                            nfail++;
                        }
                        if (result[8 - 1] >= thresh) {
                            if (nfail == 0 && nerrs == 0) {
                                Alahd(nout, path);
                            }
                            sprintnum_short(buf, result[8 - 1]);
                            write(nout, format_9997), "Clatbs", uplo, trans, diag, "Y", n, kd, imat, 8, buf;
                            nfail++;
                        }
                        nrun += 2;
                    }
                }
            statement_120:;
            }
        }
    }
    //
    //     Print a summary of the results.
    //
    Alasum(path, nout, nfail, nrun, nerrs);
    //
    //     End of Cchktb
    //
}

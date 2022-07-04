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

void Cchktr(bool *dotype, INTEGER const nn, INTEGER *nval, INTEGER const nnb, INTEGER *nbval, INTEGER const nns, INTEGER *nsval, REAL const thresh, bool const tsterr, INTEGER const /* nmax */, COMPLEX *a, COMPLEX *ainv, COMPLEX *b, COMPLEX *x, COMPLEX *xact, COMPLEX *work, REAL *rwork, INTEGER const nout) {
    common cmn;
    common_write write(cmn);
    //
    const INTEGER ntran = 3;
    INTEGER iseedy[] = {1988, 1989, 1990, 1991};
    char uplos[] = {'U', 'L'};
    char transs[] = {'N', 'T', 'C'};
    char uplo_diag[3];
    char uplo_trans_diag[5];
    char norm_uplo_diag[4];
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
    INTEGER imat = 0;
    const INTEGER ntype1 = 10;
    INTEGER iuplo = 0;
    char uplo[1];
    char diag[1];
    INTEGER info = 0;
    INTEGER idiag = 0;
    INTEGER inb = 0;
    INTEGER nb = 0;
    REAL anorm = 0.0;
    REAL ainvnm = 0.0;
    const REAL zero = 0.0;
    const REAL one = 1.0;
    REAL rcondi = 0.0;
    REAL rcondo = 0.0;
    const INTEGER ntests = 9;
    REAL result[ntests];
    INTEGER irhs = 0;
    INTEGER nrhs = 0;
    INTEGER itran = 0;
    char trans[1];
    char norm[1];
    REAL rcondc = 0.0;
    REAL dummy = 0.0;
    INTEGER k = 0;
    REAL rcond = 0.0;
    const INTEGER ntypes = 18;
    REAL scale = 0.0;
    static const char *format_9996 = "(1x,a,'( ''',a1,''', ''',a1,''', ''',a1,''', ''',a1,''',',i5,"
                                     "', ... ), type ',i2,', test(',i2,')=',g12.5)";
    //
    //     Initialize constants and the random number seed.
    //
    path[0] = 'C';
    path[1] = 'T';
    path[2] = 'R';
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
    infot = 0;
    //
    for (in = 1; in <= nn; in = in + 1) {
        //
        //        Do for each value of N in NVAL
        //
        n = nval[in - 1];
        lda = max((INTEGER)1, n);
        xtype = 'N';
        //
        for (imat = 1; imat <= ntype1; imat = imat + 1) {
            //
            //           Do the tests only if DOTYPE( IMAT ) is true.
            //
            if (!dotype[imat - 1]) {
                goto statement_80;
            }
            //
            for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                //
                //              Do first for UPLO = 'U', then for UPLO = 'L'
                //
                uplo[0] = uplos[iuplo - 1];
                //
                //              Call Clattr to generate a triangular test matrix.
                //
                strncpy(srnamt, "Clattr", srnamt_len);
                Clattr(imat, uplo, "No transpose", diag, iseed, n, a, lda, x, work, rwork, info);
                //
                //              Set IDIAG = 1 for non-unit matrices, 2 for unit.
                //
                if (Mlsame(diag, "N")) {
                    idiag = 1;
                } else {
                    idiag = 2;
                }
                //
                for (inb = 1; inb <= nnb; inb = inb + 1) {
                    //
                    //                 Do for each blocksize in NBVAL
                    //
                    nb = nbval[inb - 1];
                    xlaenv(1, nb);
                    //
                    //+    TEST 1
                    //                 Form the inverse of A.
                    //
                    Clacpy(uplo, n, n, a, lda, ainv, lda);
                    strncpy(srnamt, "Ctrtri", srnamt_len);
                    Ctrtri(uplo, diag, n, ainv, lda, info);
                    //
                    //                 Check error code from Ctrtri.
                    //
                    if (info != 0) {
                        uplo_diag[0] = uplo[0];
                        uplo_diag[1] = diag[0];
                        uplo_diag[2] = '\0';
                        Alaerh(path, "Ctrtri", info, 0, uplo_diag, n, n, -1, -1, nb, imat, nfail, nerrs, nout);
                    }
                    //
                    //                 Compute the infinity-norm condition number of A.
                    //
                    anorm = Clantr("I", uplo, diag, n, n, a, lda, rwork);
                    ainvnm = Clantr("I", uplo, diag, n, n, ainv, lda, rwork);
                    if (anorm <= zero || ainvnm <= zero) {
                        rcondi = one;
                    } else {
                        rcondi = (one / anorm) / ainvnm;
                    }
                    //
                    //                 Compute the residual for the triangular matrix times
                    //                 its inverse.  Also compute the 1-norm condition number
                    //                 of A.
                    //
                    Ctrt01(uplo, diag, n, a, lda, ainv, lda, rcondo, rwork, result[1 - 1]);
                    //                 Print the test ratio if it is .GE. THRESH.
                    //
                    if (result[1 - 1] >= thresh) {
                        if (nfail == 0 && nerrs == 0) {
                            Alahd(nout, path);
                        }
                        sprintnum_short(buf, result[1 - 1]);
                        write(nout, "(' UPLO=''',a1,''', DIAG=''',a1,''', N=',i5,', NB=',i4,"
                                    "', type ',i2,', test(',i2,')= ',g12.5)"),
                            uplo, diag, n, nb, imat, 1, buf;
                        nfail++;
                    }
                    nrun++;
                    //
                    //                 Skip remaining tests if not the first block size.
                    //
                    if (inb != 1) {
                        goto statement_60;
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
                            //+    TEST 2
                            //                       Solve and compute residual for op(A)*x = b.
                            //
                            strncpy(srnamt, "Clarhs", srnamt_len);
                            Clarhs(path, &xtype, uplo, trans, n, n, 0, idiag, nrhs, a, lda, xact, lda, b, lda, iseed, info);
                            xtype = 'C';
                            Clacpy("Full", n, nrhs, b, lda, x, lda);
                            //
                            strncpy(srnamt, "Ctrtrs", srnamt_len);
                            Ctrtrs(uplo, trans, diag, n, nrhs, a, lda, x, lda, info);
                            //
                            //                       Check error code from Ctrtrs.
                            //
                            if (info != 0) {
                                uplo_trans_diag[0] = uplo[0];
                                uplo_trans_diag[1] = trans[0];
                                uplo_trans_diag[3] = diag[0];
                                uplo_trans_diag[4] = '\0';
                                Alaerh(path, "Ctrtrs", info, 0, uplo_trans_diag, n, n, -1, -1, nrhs, imat, nfail, nerrs, nout);
                            }
                            //
                            //                       This line is needed on a Sun SPARCstation.
                            //
                            if (n > 0) {
                                dummy = a[1 - 1].real();
                            }
                            //
                            Ctrt02(uplo, trans, diag, n, nrhs, a, lda, x, lda, b, lda, work, rwork, result[2 - 1]);
                            //
                            //+    TEST 3
                            //                       Check solution from generated exact solution.
                            //
                            Cget04(n, nrhs, x, lda, xact, lda, rcondc, result[3 - 1]);
                            //
                            //+    TESTS 4, 5, and 6
                            //                       Use iterative refinement to improve the solution
                            //                       and compute error bounds.
                            //
                            strncpy(srnamt, "Ctrrfs", srnamt_len);
                            Ctrrfs(uplo, trans, diag, n, nrhs, a, lda, b, lda, x, lda, rwork, &rwork[(nrhs + 1) - 1], work, &rwork[(2 * nrhs + 1) - 1], info);
                            //
                            //                       Check error code from Ctrrfs.
                            //
                            if (info != 0) {
                                uplo_trans_diag[0] = uplo[0];
                                uplo_trans_diag[1] = trans[0];
                                uplo_trans_diag[3] = diag[0];
                                uplo_trans_diag[4] = '\0';
                                Alaerh(path, "Ctrrfs", info, 0, uplo_trans_diag, n, n, -1, -1, nrhs, imat, nfail, nerrs, nout);
                            }
                            //
                            Cget04(n, nrhs, x, lda, xact, lda, rcondc, result[4 - 1]);
                            Ctrt05(uplo, trans, diag, n, nrhs, a, lda, b, lda, x, lda, xact, lda, rwork, &rwork[(nrhs + 1) - 1], &result[5 - 1]);
                            //
                            //                       Print information about the tests that did not
                            //                       pass the threshold.
                            //
                            for (k = 2; k <= 6; k = k + 1) {
                                if (result[k - 1] >= thresh) {
                                    if (nfail == 0 && nerrs == 0) {
                                        Alahd(nout, path);
                                    }
                                    sprintnum_short(buf, result[k - 1]);
                                    write(nout, "(' UPLO=''',a1,''', TRANS=''',a1,''', DIAG=''',a1,"
                                                "''', N=',i5,', NB=',i4,', type ',i2,',      test(',i2,"
                                                "')= ',a)"),
                                        uplo, trans, diag, n, nrhs, imat, k, buf;
                                    nfail++;
                                }
                            }
                            nrun += 5;
                        }
                    }
                    //
                    //+    TEST 7
                    //                       Get an estimate of RCOND = 1/CNDNUM.
                    //
                    for (itran = 1; itran <= 2; itran = itran + 1) {
                        if (itran == 1) {
                            norm[0] = 'O';
                            rcondc = rcondo;
                        } else {
                            norm[0] = 'I';
                            rcondc = rcondi;
                        }
                        strncpy(srnamt, "Ctrcon", srnamt_len);
                        Ctrcon(norm, uplo, diag, n, a, lda, rcond, work, rwork, info);
                        //
                        //                       Check error code from Ctrcon.
                        //
                        if (info != 0) {
                            norm_uplo_diag[0] = norm[0];
                            norm_uplo_diag[1] = uplo[0];
                            norm_uplo_diag[2] = diag[0];
                            norm_uplo_diag[3] = '\0';
                            Alaerh(path, "Ctrcon", info, 0, norm_uplo_diag, n, n, -1, -1, -1, imat, nfail, nerrs, nout);
                        }
                        //
                        Ctrt06(rcond, rcondc, uplo, diag, n, a, lda, rwork, result[7 - 1]);
                        //
                        //                    Print the test ratio if it is .GE. THRESH.
                        //
                        if (result[7 - 1] >= thresh) {
                            if (nfail == 0 && nerrs == 0) {
                                Alahd(nout, path);
                            }
                            sprintnum_short(buf, result[7 - 1]);
                            write(nout, "(' NORM=''',a1,''', UPLO =''',a1,''', N=',i5,',',11x,"
                                        "' type ',i2,', test(',i2,')=',a)"),
                                norm, uplo, n, imat, 7, buf;
                            nfail++;
                        }
                        nrun++;
                    }
                statement_60:;
                }
            }
        statement_80:;
        }
        //
        //        Use pathological test matrices to test Clatrs.
        //
        for (imat = ntype1 + 1; imat <= ntypes; imat = imat + 1) {
            //
            //           Do the tests only if DOTYPE( IMAT ) is true.
            //
            if (!dotype[imat - 1]) {
                goto statement_110;
            }
            //
            for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                //
                //              Do first for UPLO = 'U', then for UPLO = 'L'
                //
                uplo[0] = uplos[iuplo - 1];
                for (itran = 1; itran <= ntran; itran = itran + 1) {
                    //
                    //                 Do for op(A) = A, A**T, and A**H.
                    //
                    trans[0] = transs[itran - 1];
                    //
                    //                 Call Clattr to generate a triangular test matrix.
                    //
                    strncpy(srnamt, "Clattr", srnamt_len);
                    Clattr(imat, uplo, trans, diag, iseed, n, a, lda, x, work, rwork, info);
                    //
                    //+    TEST 8
                    //                 Solve the system op(A)*x = b.
                    //
                    strncpy(srnamt, "Clatrs", srnamt_len);
                    Ccopy(n, x, 1, b, 1);
                    Clatrs(uplo, trans, diag, "N", n, a, lda, b, scale, rwork, info);
                    //
                    //                 Check error code from Clatrs.
                    //
                    if (info != 0) {
                        uplo_trans_diag[0] = uplo[0];
                        uplo_trans_diag[1] = trans[0];
                        uplo_trans_diag[2] = diag[0];
                        uplo_trans_diag[3] = 'N';
                        Alaerh(path, "Clatrs", info, 0, uplo_trans_diag, n, n, -1, -1, -1, imat, nfail, nerrs, nout);
                    }
                    //
                    Ctrt03(uplo, trans, diag, n, 1, a, lda, scale, rwork, one, b, lda, x, lda, work, result[8 - 1]);
                    //
                    //+    TEST 9
                    //                 Solve op(A)*X = b again with NORMIN = 'Y'.
                    //
                    Ccopy(n, x, 1, &b[(n + 1) - 1], 1);
                    Clatrs(uplo, trans, diag, "Y", n, a, lda, &b[(n + 1) - 1], scale, rwork, info);
                    //
                    //                 Check error code from Clatrs.
                    //
                    if (info != 0) {
                        uplo_trans_diag[0] = uplo[0];
                        uplo_trans_diag[1] = trans[0];
                        uplo_trans_diag[2] = diag[0];
                        uplo_trans_diag[3] = 'Y';
                        Alaerh(path, "Clatrs", info, 0, uplo_trans_diag, n, n, -1, -1, -1, imat, nfail, nerrs, nout);
                    }
                    //
                    Ctrt03(uplo, trans, diag, n, 1, a, lda, scale, rwork, one, &b[(n + 1) - 1], lda, x, lda, work, result[9 - 1]);
                    //
                    //                 Print information about the tests that did not pass
                    //                 the threshold.
                    //
                    if (result[8 - 1] >= thresh) {
                        if (nfail == 0 && nerrs == 0) {
                            Alahd(nout, path);
                        }
                        sprintnum_short(buf, result[8 - 1]);
                        write(nout, format_9996), "Clatrs", uplo, trans, diag, "N", n, imat, 8, buf;
                        nfail++;
                    }
                    if (result[9 - 1] >= thresh) {
                        if (nfail == 0 && nerrs == 0) {
                            Alahd(nout, path);
                        }
                        sprintnum_short(buf, result[9 - 1]);
                        write(nout, format_9996), "Clatrs", uplo, trans, diag, "Y", n, imat, 9, buf;
                        nfail++;
                    }
                    nrun += 2;
                }
            }
        statement_110:;
        }
    }
    //
    //     Print a summary of the results.
    //
    Alasum(path, nout, nfail, nrun, nerrs);
    //
    //     End of Cchktr
    //
}

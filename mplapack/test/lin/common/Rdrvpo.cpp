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

void Rdrvpo(bool *dotype, INTEGER const nn, INTEGER *nval, INTEGER const nrhs, REAL const thresh, bool const tsterr, INTEGER const /* nmax */, REAL *a, REAL *afac, REAL *asav, REAL *b, REAL *bsav, REAL *x, REAL *xact, REAL *s, REAL *work, REAL *rwork, INTEGER *iwork, INTEGER const nout) {
    common cmn;
    common_write write(cmn);
    //
    INTEGER iseedy[] = {1988, 1989, 1990, 1991};
    char facts[] = {'F', 'N', 'E'};
    char equeds[] = {'N', 'Y'};
    char uplos[] = {'U', 'L'};
    char transs[] = {'N', 'T', 'C'};
    char fact_uplo[3];
    char path[4] = {};
    char matpath[4] = {};
    char buf[1024];
    INTEGER nrun = 0;
    INTEGER nfail = 0;
    INTEGER nerrs = 0;
    INTEGER i = 0;
    INTEGER iseed[4];
    INTEGER nb = 0;
    INTEGER nbmin = 0;
    INTEGER in = 0;
    INTEGER n = 0;
    INTEGER lda = 0;
    char xtype[1];
    const INTEGER ntypes = 9;
    INTEGER nimat = 0;
    INTEGER imat = 0;
    bool zerot = false;
    INTEGER iuplo = 0;
    char uplo[1];
    char type[1];
    INTEGER kl = 0;
    INTEGER ku = 0;
    REAL anorm = 0.0;
    INTEGER mode = 0;
    REAL cndnum = 0.0;
    char dist[1];
    INTEGER info = 0;
    INTEGER izero = 0;
    INTEGER ioff = 0;
    const REAL zero = 0.0;
    INTEGER iequed = 0;
    char equed[1];
    INTEGER nfact = 0;
    INTEGER ifact = 0;
    char fact[1];
    bool prefac = false;
    bool nofact = false;
    bool equil = false;
    REAL rcondc = 0.0;
    REAL scond = 0.0;
    REAL amax = 0.0;
    REAL roldc = 0.0;
    REAL ainvnm = 0.0;
    const REAL one = 1.0;
    const INTEGER ntests = 6;
    REAL result[ntests];
    INTEGER nt = 0;
    INTEGER k = 0;
    REAL rcond = 0.0;
    INTEGER k1 = 0;
    //
    //     Initialize constants and the random number seed.
    //
    path[0] = 'R';
    path[1] = 'P';
    path[2] = 'O';
    path[3] = '\0';
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
        Rerrvx(path, nout);
    }
    infot = 0;
    //
    //     Set the block size and minimum block size for testing.
    //
    nb = 1;
    nbmin = 2;
    xlaenv(1, nb);
    xlaenv(2, nbmin);
    //
    //     Do for each value of N in NVAL
    //
    for (in = 1; in <= nn; in = in + 1) {
        n = nval[in - 1];
        lda = max(n, (INTEGER)1);
        xtype[0] = 'N';
        nimat = ntypes;
        if (n <= 0) {
            nimat = 1;
        }
        //
        for (imat = 1; imat <= nimat; imat = imat + 1) {
            //
            //           Do the tests only if DOTYPE( IMAT ) is true.
            //
            if (!dotype[imat - 1]) {
                goto statement_120;
            }
            //
            //           Skip types 3, 4, or 5 if the matrix size is too small.
            //
            zerot = imat >= 3 && imat <= 5;
            if (zerot && n < imat - 2) {
                goto statement_120;
            }
            //
            //           Do first for UPLO = 'U', then for UPLO = 'L'
            //
            for (iuplo = 1; iuplo <= 2; iuplo = iuplo + 1) {
                uplo[0] = uplos[iuplo - 1];
                //
                //              Set up parameters with Rlatb4 and generate a test matrix
                //              with Rlatms.
                //
                Rlatb4(path, imat, n, n, type, kl, ku, anorm, mode, cndnum, dist);
                //
                strncpy(srnamt, "Rlatms", srnamt_len);
                Rlatms(n, n, dist, iseed, type, rwork, mode, cndnum, anorm, kl, ku, uplo, a, lda, work, info);
                //
                //              Check error code from Rlatms.
                //
                if (info != 0) {
                    Alaerh(path, "Rlatms", info, 0, uplo, n, n, -1, -1, -1, imat, nfail, nerrs, nout);
                    goto statement_110;
                }
                //
                //              For types 3-5, zero one row and column of the matrix to
                //              test that INFO is returned correctly.
                //
                if (zerot) {
                    if (imat == 3) {
                        izero = 1;
                    } else if (imat == 4) {
                        izero = n;
                    } else {
                        izero = n / 2 + 1;
                    }
                    ioff = (izero - 1) * lda;
                    //
                    //                 Set row and column IZERO of A to 0.
                    //
                    if (iuplo == 1) {
                        for (i = 1; i <= izero - 1; i = i + 1) {
                            a[(ioff + i) - 1] = zero;
                        }
                        ioff += izero;
                        for (i = izero; i <= n; i = i + 1) {
                            a[ioff - 1] = zero;
                            ioff += lda;
                        }
                    } else {
                        ioff = izero;
                        for (i = 1; i <= izero - 1; i = i + 1) {
                            a[ioff - 1] = zero;
                            ioff += lda;
                        }
                        ioff = ioff - izero;
                        for (i = izero; i <= n; i = i + 1) {
                            a[(ioff + i) - 1] = zero;
                        }
                    }
                } else {
                    izero = 0;
                }
                //
                //              Save a copy of the matrix A in ASAV.
                //
                Rlacpy(uplo, n, n, a, lda, asav, lda);
                //
                for (iequed = 1; iequed <= 2; iequed = iequed + 1) {
                    equed[0] = equeds[iequed - 1];
                    if (iequed == 1) {
                        nfact = 3;
                    } else {
                        nfact = 1;
                    }
                    //
                    for (ifact = 1; ifact <= nfact; ifact = ifact + 1) {
                        fact[0] = facts[ifact - 1];
                        prefac = Mlsame(fact, "F");
                        nofact = Mlsame(fact, "N");
                        equil = Mlsame(fact, "E");
                        //
                        if (zerot) {
                            if (prefac) {
                                goto statement_90;
                            }
                            rcondc = zero;
                            //
                        } else if (!Mlsame(fact, "N")) {
                            //
                            //                       Compute the condition number for comparison with
                            //                       the value returned by Rposvx (FACT = 'N' reuses
                            //                       the condition number from the previous iteration
                            //                       with FACT = 'F').
                            //
                            Rlacpy(uplo, n, n, asav, lda, afac, lda);
                            if (equil || iequed > 1) {
                                //
                                //                          Compute row and column scale factors to
                                //                          equilibrate the matrix A.
                                //
                                Rpoequ(n, afac, lda, s, scond, amax, info);
                                if (info == 0 && n > 0) {
                                    if (iequed > 1) {
                                        scond = zero;
                                    }
                                    //
                                    //                             Equilibrate the matrix.
                                    //
                                    Rlaqsy(uplo, n, afac, lda, s, scond, amax, equed);
                                }
                            }
                            //
                            //                       Save the condition number of the
                            //                       non-equilibrated system for use in Rget04.
                            //
                            if (equil) {
                                roldc = rcondc;
                            }
                            //
                            //                       Compute the 1-norm of A.
                            //
                            anorm = Rlansy("1", uplo, n, afac, lda, rwork);
                            //
                            //                       Factor the matrix A.
                            //
                            Rpotrf(uplo, n, afac, lda, info);
                            //
                            //                       Form the inverse of A.
                            //
                            Rlacpy(uplo, n, n, afac, lda, a, lda);
                            Rpotri(uplo, n, a, lda, info);
                            //
                            //                       Compute the 1-norm condition number of A.
                            //
                            ainvnm = Rlansy("1", uplo, n, a, lda, rwork);
                            if (anorm <= zero || ainvnm <= zero) {
                                rcondc = one;
                            } else {
                                rcondc = (one / anorm) / ainvnm;
                            }
                        }
                        //
                        //                    Restore the matrix A.
                        //
                        Rlacpy(uplo, n, n, asav, lda, a, lda);
                        //
                        //                    Form an exact solution and set the right hand side.
                        //
                        strncpy(srnamt, "Rlarhs", srnamt_len);
                        Rlarhs(path, xtype, uplo, " ", n, n, kl, ku, nrhs, a, lda, xact, lda, b, lda, iseed, info);
                        xtype[0] = 'C';
                        Rlacpy("Full", n, nrhs, b, lda, bsav, lda);
                        //
                        if (nofact) {
                            //
                            //                       --- Test Rposv  ---
                            //
                            //                       Compute the L*L' or U'*U factorization of the
                            //                       matrix and solve the system.
                            //
                            Rlacpy(uplo, n, n, a, lda, afac, lda);
                            Rlacpy("Full", n, nrhs, b, lda, x, lda);
                            //
                            strncpy(srnamt, "Rposv", srnamt_len);
                            Rposv(uplo, n, nrhs, afac, lda, x, lda, info);
                            //
                            //                       Check error code from Rposv .
                            //
                            if (info != izero) {
                                Alaerh(path, "Rposv ", info, izero, uplo, n, n, -1, -1, nrhs, imat, nfail, nerrs, nout);
                                goto statement_70;
                            } else if (info != 0) {
                                goto statement_70;
                            }
                            //
                            //                       Reconstruct matrix from factors and compute
                            //                       residual.
                            //
                            Rpot01(uplo, n, a, lda, afac, lda, rwork, result[1 - 1]);
                            //
                            //                       Compute residual of the computed solution.
                            //
                            Rlacpy("Full", n, nrhs, b, lda, work, lda);
                            Rpot02(uplo, n, nrhs, a, lda, x, lda, work, lda, rwork, result[2 - 1]);
                            //
                            //                       Check solution from generated exact solution.
                            //
                            Rget04(n, nrhs, x, lda, xact, lda, rcondc, result[3 - 1]);
                            nt = 3;
                            //
                            //                       Print information about the tests that did not
                            //                       pass the threshold.
                            //
                            for (k = 1; k <= nt; k = k + 1) {
                                if (result[k - 1] >= thresh) {
                                    if (nfail == 0 && nerrs == 0) {
                                        Aladhd(nout, path);
                                    }
                                    sprintnum_short(buf, result[k - 1]);
                                    write(nout, "(1x,a,', UPLO=''',a1,''', N =',i5,', type ',i1,', test(',"
                                                "i1,')=',a)"),
                                        "Rposv ", uplo, n, imat, k, buf;
                                    nfail++;
                                }
                            }
                            nrun += nt;
                        statement_70:;
                        }
                        //
                        //                    --- Test Rposvx ---
                        //
                        if (!prefac) {
                            Rlaset(uplo, n, n, zero, zero, afac, lda);
                        }
                        Rlaset("Full", n, nrhs, zero, zero, x, lda);
                        if (iequed > 1 && n > 0) {
                            //
                            //                       Equilibrate the matrix if FACT='F' and
                            //                       EQUED='Y'.
                            //
                            Rlaqsy(uplo, n, a, lda, s, scond, amax, equed);
                        }
                        //
                        //                    Solve the system and compute the condition number
                        //                    and error bounds using Rposvx.
                        //
                        strncpy(srnamt, "Rposvx", srnamt_len);
                        Rposvx(fact, uplo, n, nrhs, a, lda, afac, lda, equed, s, b, lda, x, lda, rcond, rwork, &rwork[(nrhs + 1) - 1], work, iwork, info);
                        //
                        //                    Check the error code from Rposvx.
                        //
                        if (info != izero) {
                            fact_uplo[0] = fact[0];
                            fact_uplo[1] = uplo[0];
                            fact_uplo[2] = '\0';
                            Alaerh(path, "Rposvx", info, izero, fact_uplo, n, n, -1, -1, nrhs, imat, nfail, nerrs, nout);
                            goto statement_90;
                        }
                        //
                        if (info == 0) {
                            if (!prefac) {
                                //
                                //                          Reconstruct matrix from factors and compute
                                //                          residual.
                                //
                                Rpot01(uplo, n, a, lda, afac, lda, &rwork[(2 * nrhs + 1) - 1], result[1 - 1]);
                                k1 = 1;
                            } else {
                                k1 = 2;
                            }
                            //
                            //                       Compute residual of the computed solution.
                            //
                            Rlacpy("Full", n, nrhs, bsav, lda, work, lda);
                            Rpot02(uplo, n, nrhs, asav, lda, x, lda, work, lda, &rwork[(2 * nrhs + 1) - 1], result[2 - 1]);
                            //
                            //                       Check solution from generated exact solution.
                            //
                            if (nofact || (prefac && Mlsame(equed, "N"))) {
                                Rget04(n, nrhs, x, lda, xact, lda, rcondc, result[3 - 1]);
                            } else {
                                Rget04(n, nrhs, x, lda, xact, lda, roldc, result[3 - 1]);
                            }
                            //
                            //                       Check the error bounds from iterative
                            //                       refinement.
                            //
                            Rpot05(uplo, n, nrhs, asav, lda, b, lda, x, lda, xact, lda, rwork, &rwork[(nrhs + 1) - 1], &result[4 - 1]);
                        } else {
                            k1 = 6;
                        }
                        //
                        //                    Compare RCOND from Rposvx with the computed value
                        //                    in RCONDC.
                        //
                        result[6 - 1] = Rget06(rcond, rcondc);
                        //
                        //                    Print information about the tests that did not pass
                        //                    the threshold.
                        //
                        for (k = k1; k <= 6; k = k + 1) {
                            if (result[k - 1] >= thresh) {
                                if (nfail == 0 && nerrs == 0) {
                                    Aladhd(nout, path);
                                }
                                if (prefac) {
                                    sprintnum_short(buf, result[k - 1]);
                                    write(nout, "(1x,a,', FACT=''',a1,''', UPLO=''',a1,''', N=',i5,"
                                                "', EQUED=''',a1,''', type ',i1,', test(',i1,') =',a)"),
                                        "Rposvx", fact, uplo, n, equed, imat, k, buf;
                                } else {
                                    sprintnum_short(buf, result[k - 1]);
                                    write(nout, "(1x,a,', FACT=''',a1,''', UPLO=''',a1,''', N=',i5,"
                                                "', type ',i1,', test(',i1,')=',a)"),
                                        "Rposvx", fact, uplo, n, imat, k, buf;
                                }
                                nfail++;
                            }
                        }
                        nrun += 7 - k1;
                    statement_90:;
                    }
                }
            statement_110:;
            }
        statement_120:;
        }
    }
    //
    //     Print a summary of the results.
    //
    Alasvm(path, nout, nfail, nrun, nerrs);
    //
    //     End of Rdrvpo
    //
}

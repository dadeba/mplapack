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
#include <mplapack_eig.h>

#include <mplapack_debug.h>

void Cckgqr(INTEGER const nm, INTEGER *mval, INTEGER const np, INTEGER *pval, INTEGER const nn, INTEGER *nval, INTEGER const nmats, INTEGER *iseed, REAL const thresh, INTEGER const nmax, COMPLEX *a, COMPLEX *af, COMPLEX *aq, COMPLEX *ar, COMPLEX *taua, COMPLEX *b, COMPLEX *bf, COMPLEX *bz, COMPLEX *bt, COMPLEX *bwk, COMPLEX *taub, COMPLEX *work, REAL *rwork, INTEGER const nin, INTEGER const nout, INTEGER &info) {
    common cmn;
    common_write write(cmn);
    char path[4];
    char buf[1024];
    INTEGER nrun = 0;
    INTEGER nfail = 0;
    bool firstt = false;
    const INTEGER ntypes = 8;
    bool dotype[ntypes];
    INTEGER lda = 0;
    INTEGER ldb = 0;
    INTEGER lwork = 0;
    INTEGER im = 0;
    INTEGER m = 0;
    INTEGER ip = 0;
    INTEGER p = 0;
    INTEGER in = 0;
    INTEGER n = 0;
    INTEGER imat = 0;
    char type;
    INTEGER kla = 0;
    INTEGER kua = 0;
    INTEGER klb = 0;
    INTEGER kub = 0;
    REAL anorm = 0.0;
    REAL bnorm = 0.0;
    INTEGER modea = 0;
    INTEGER modeb = 0;
    REAL cndnma = 0.0;
    REAL cndnmb = 0.0;
    char dista;
    char distb;
    INTEGER iinfo = 0;
    INTEGER nt = 0;
    const INTEGER ntests = 7;
    REAL result[ntests];
    INTEGER i = 0;
    static const char *format_9999 = "(' ZLATMS in Cckgqr:    INFO = ',i5)";
    //
    //  -- LAPACK test routine --
    //  -- LAPACK is a software package provided by Univ. of Tennessee,    --
    //  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //
    //     .. Scalar Arguments ..
    //     ..
    //     .. Array Arguments ..
    //     ..
    //
    //  =====================================================================
    //
    //     .. Parameters ..
    //     ..
    //     .. Local Scalars ..
    //     ..
    //     .. Local Arrays ..
    //     ..
    //     .. External Subroutines ..
    //     ..
    //     .. Intrinsic Functions ..
    //     ..
    //     .. Executable Statements ..
    //
    //     Initialize constants.
    //
    path[0] = 'G';
    path[1] = 'Q';
    path[2] = 'R';
    path[3] = '\0';
    info = 0;
    nrun = 0;
    nfail = 0;
    firstt = true;
    Alareq(path, nmats, dotype, ntypes, nin, nout);
    lda = nmax;
    ldb = nmax;
    lwork = nmax * nmax;
    //
    //     Do for each value of M in MVAL.
    //
    for (im = 1; im <= nm; im = im + 1) {
        m = mval[im - 1];
        //
        //        Do for each value of P in PVAL.
        //
        for (ip = 1; ip <= np; ip = ip + 1) {
            p = pval[ip - 1];
            //
            //           Do for each value of N in NVAL.
            //
            for (in = 1; in <= nn; in = in + 1) {
                n = nval[in - 1];
                //
                for (imat = 1; imat <= ntypes; imat = imat + 1) {
                    //
                    //                 Do the tests only if DOTYPE( IMAT ) is true.
                    //
                    if (!dotype[imat - 1]) {
                        goto statement_30;
                    }
                    //
                    //                 Test Cggrqf
                    //
                    //                 Set up parameters with Rlatb9 and generate test
                    //                 matrices A and B with ZLATMS.
                    //
                    Rlatb9("GRQ", imat, m, p, n, &type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, &dista, &distb);
                    //
                    Clatms(m, n, &dista, iseed, &type, rwork, modea, cndnma, anorm, kla, kua, "No packing", a, lda, work, iinfo);
                    if (iinfo != 0) {
                        write(nout, format_9999), iinfo;
                        info = abs(iinfo);
                        goto statement_30;
                    }
                    //
                    Clatms(p, n, &distb, iseed, &type, rwork, modeb, cndnmb, bnorm, klb, kub, "No packing", b, ldb, work, iinfo);
                    if (iinfo != 0) {
                        write(nout, format_9999), iinfo;
                        info = abs(iinfo);
                        goto statement_30;
                    }
                    //
                    nt = 4;
                    //
                    Cgrqts(m, p, n, a, af, aq, ar, lda, taua, b, bf, bz, bt, bwk, ldb, taub, work, lwork, rwork, result);
                    //
                    //                 Print information about the tests that did not
                    //                 pass the threshold.
                    //
                    for (i = 1; i <= nt; i = i + 1) {
                        if (result[i - 1] >= thresh) {
                            if (nfail == 0 && firstt) {
                                firstt = false;
                                Alahdg(nout, "GRQ");
                            }
                            sprintnum_short(buf, result[i - 1]);
                            write(nout, "(' M=',i4,' P=',i4,', N=',i4,', type ',i2,', test ',i2,"
                                        "', ratio=',a)"),
                                m, p, n, imat, i, buf;
                            nfail++;
                        }
                    }
                    nrun += nt;
                    //
                    //                 Test Cggqrf
                    //
                    //                 Set up parameters with Rlatb9 and generate test
                    //                 matrices A and B with CLATMS.
                    //
                    Rlatb9("GQR", imat, m, p, n, &type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, &dista, &distb);
                    //
                    Clatms(n, m, &dista, iseed, &type, rwork, modea, cndnma, anorm, kla, kua, "No packing", a, lda, work, iinfo);
                    if (iinfo != 0) {
                        write(nout, format_9999), iinfo;
                        info = abs(iinfo);
                        goto statement_30;
                    }
                    //
                    Clatms(n, p, &distb, iseed, &type, rwork, modea, cndnma, bnorm, klb, kub, "No packing", b, ldb, work, iinfo);
                    if (iinfo != 0) {
                        write(nout, format_9999), iinfo;
                        info = abs(iinfo);
                        goto statement_30;
                    }
                    //
                    nt = 4;
                    //
                    Cgqrts(n, m, p, a, af, aq, ar, lda, taua, b, bf, bz, bt, bwk, ldb, taub, work, lwork, rwork, result);
                    //
                    //                 Print information about the tests that did not
                    //                 pass the threshold.
                    //
                    for (i = 1; i <= nt; i = i + 1) {
                        if (result[i - 1] >= thresh) {
                            if (nfail == 0 && firstt) {
                                firstt = false;
                                Alahdg(nout, path);
                            }
                            sprintnum_short(buf, result[i - 1]);
                            write(nout, "(' N=',i4,' M=',i4,', P=',i4,', type ',i2,', test ',i2,"
                                        "', ratio=',a)"),
                                n, m, p, imat, i, buf;
                            nfail++;
                        }
                    }
                    nrun += nt;
                //
                statement_30:;
                }
            }
        }
    }
    //
    //     Print a summary of the results.
    //
    Alasum(path, nout, nfail, nrun, 0);
    //
    //     End of Cckgqr
    //
}

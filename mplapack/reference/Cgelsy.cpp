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

void Cgelsy(INTEGER const m, INTEGER const n, INTEGER const nrhs, COMPLEX *a, INTEGER const lda, COMPLEX *b, INTEGER const ldb, INTEGER *jpvt, REAL const rcond, INTEGER &rank, COMPLEX *work, INTEGER const lwork, REAL *rwork, INTEGER &info) {
    INTEGER mn = 0;
    INTEGER ismin = 0;
    INTEGER ismax = 0;
    INTEGER nb1 = 0;
    INTEGER nb2 = 0;
    INTEGER nb3 = 0;
    INTEGER nb4 = 0;
    INTEGER nb = 0;
    INTEGER lwkopt = 0;
    bool lquery = false;
    REAL smlnum = 0.0;
    const REAL one = 1.0;
    REAL bignum = 0.0;
    REAL anrm = 0.0;
    INTEGER iascl = 0;
    const REAL zero = 0.0;
    const COMPLEX czero = COMPLEX(0.0, 0.0);
    REAL bnrm = 0.0;
    INTEGER ibscl = 0;
    REAL wsize = 0.0;
    const COMPLEX cone = COMPLEX(1.0, 0.0);
    REAL smax = 0.0;
    REAL smin = 0.0;
    INTEGER i = 0;
    const INTEGER imin = 2;
    REAL sminpr = 0.0;
    COMPLEX s1 = 0.0;
    COMPLEX c1 = 0.0;
    const INTEGER imax = 1;
    REAL smaxpr = 0.0;
    COMPLEX s2 = 0.0;
    COMPLEX c2 = 0.0;
    INTEGER j = 0;
    //
    mn = min(m, n);
    ismin = mn + 1;
    ismax = 2 * mn + 1;
    //
    //     Test the input arguments.
    //
    info = 0;
    nb1 = iMlaenv(1, "Cgeqrf", " ", m, n, -1, -1);
    nb2 = iMlaenv(1, "Cgerqf", " ", m, n, -1, -1);
    nb3 = iMlaenv(1, "Cunmqr", " ", m, n, nrhs, -1);
    nb4 = iMlaenv(1, "Cunmrq", " ", m, n, nrhs, -1);
    nb = max({nb1, nb2, nb3, nb4});
    lwkopt = max({(INTEGER)1, mn + 2 * n + nb * (n + 1), 2 * mn + nb * nrhs});
    work[1 - 1] = COMPLEX(lwkopt);
    lquery = (lwork == -1);
    if (m < 0) {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (nrhs < 0) {
        info = -3;
    } else if (lda < max((INTEGER)1, m)) {
        info = -5;
    } else if (ldb < max({(INTEGER)1, m, n})) {
        info = -7;
    } else if (lwork < (mn + max({(INTEGER)2 * mn, n + 1, mn + nrhs})) && !lquery) {
        info = -12;
    }
    //
    if (info != 0) {
        Mxerbla("Cgelsy", -info);
        return;
    } else if (lquery) {
        return;
    }
    //
    //     Quick return if possible
    //
    if (min({m, n, nrhs}) == 0) {
        rank = 0;
        return;
    }
    //
    //     Get machine parameters
    //
    smlnum = Rlamch("S") / Rlamch("P");
    bignum = one / smlnum;
    //
    //     Scale A, B if max entries outside range [SMLNUM,BIGNUM]
    //
    anrm = Clange("M", m, n, a, lda, rwork);
    iascl = 0;
    if (anrm > zero && anrm < smlnum) {
        //
        //        Scale matrix norm up to SMLNUM
        //
        Clascl("G", 0, 0, anrm, smlnum, m, n, a, lda, info);
        iascl = 1;
    } else if (anrm > bignum) {
        //
        //        Scale matrix norm down to BIGNUM
        //
        Clascl("G", 0, 0, anrm, bignum, m, n, a, lda, info);
        iascl = 2;
    } else if (anrm == zero) {
        //
        //        Matrix all zero. Return zero solution.
        //
        Claset("F", max(m, n), nrhs, czero, czero, b, ldb);
        rank = 0;
        goto statement_70;
    }
    //
    bnrm = Clange("M", m, nrhs, b, ldb, rwork);
    ibscl = 0;
    if (bnrm > zero && bnrm < smlnum) {
        //
        //        Scale matrix norm up to SMLNUM
        //
        Clascl("G", 0, 0, bnrm, smlnum, m, nrhs, b, ldb, info);
        ibscl = 1;
    } else if (bnrm > bignum) {
        //
        //        Scale matrix norm down to BIGNUM
        //
        Clascl("G", 0, 0, bnrm, bignum, m, nrhs, b, ldb, info);
        ibscl = 2;
    }
    //
    //     Compute QR factorization with column pivoting of A:
    //        A * P = Q * R
    //
    Cgeqp3(m, n, a, lda, jpvt, &work[1 - 1], &work[(mn + 1) - 1], lwork - mn, rwork, info);
    wsize = mn + work[(mn + 1) - 1].real();
    //
    //     complex workspace: MN+NB*(N+1). real workspace 2*N.
    //     Details of Householder rotations stored in WORK(1:MN).
    //
    //     Determine RANK using incremental condition estimation
    //
    work[ismin - 1] = cone;
    work[ismax - 1] = cone;
    smax = abs(a[(1 - 1)]);
    smin = smax;
    if (abs(a[(1 - 1)]) == zero) {
        rank = 0;
        Claset("F", max(m, n), nrhs, czero, czero, b, ldb);
        goto statement_70;
    } else {
        rank = 1;
    }
//
statement_10:
    if (rank < mn) {
        i = rank + 1;
        Claic1(imin, rank, &work[ismin - 1], smin, &a[(i - 1) * lda], a[(i - 1) + (i - 1) * lda], sminpr, s1, c1);
        Claic1(imax, rank, &work[ismax - 1], smax, &a[(i - 1) * lda], a[(i - 1) + (i - 1) * lda], smaxpr, s2, c2);
        //
        if (smaxpr * rcond <= sminpr) {
            for (i = 1; i <= rank; i = i + 1) {
                work[(ismin + i - 1) - 1] = s1 * work[(ismin + i - 1) - 1];
                work[(ismax + i - 1) - 1] = s2 * work[(ismax + i - 1) - 1];
            }
            work[(ismin + rank) - 1] = c1;
            work[(ismax + rank) - 1] = c2;
            smin = sminpr;
            smax = smaxpr;
            rank++;
            goto statement_10;
        }
    }
    //
    //     complex workspace: 3*MN.
    //
    //     Logically partition R = [ R11 R12 ]
    //                             [  0  R22 ]
    //     where R11 = R(1:RANK,1:RANK)
    //
    //     [R11,R12] = [ T11, 0 ] * Y
    //
    if (rank < n) {
        Ctzrzf(rank, n, a, lda, &work[(mn + 1) - 1], &work[(2 * mn + 1) - 1], lwork - 2 * mn, info);
    }
    //
    //     complex workspace: 2*MN.
    //     Details of Householder rotations stored in WORK(MN+1:2*MN)
    //
    //     B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS)
    //
    Cunmqr("Left", "Conjugate transpose", m, nrhs, mn, a, lda, &work[1 - 1], b, ldb, &work[(2 * mn + 1) - 1], lwork - 2 * mn, info);
    wsize = max(wsize, REAL(2 * mn + work[(2 * mn + 1) - 1].real()));
    //
    //     complex workspace: 2*MN+NB*NRHS.
    //
    //     B(1:RANK,1:NRHS) := inv(T11) * B(1:RANK,1:NRHS)
    //
    Ctrsm("Left", "Upper", "No transpose", "Non-unit", rank, nrhs, cone, a, lda, b, ldb);
    //
    for (j = 1; j <= nrhs; j = j + 1) {
        for (i = rank + 1; i <= n; i = i + 1) {
            b[(i - 1) + (j - 1) * ldb] = czero;
        }
    }
    //
    //     B(1:N,1:NRHS) := Y**H * B(1:N,1:NRHS)
    //
    if (rank < n) {
        Cunmrz("Left", "Conjugate transpose", n, nrhs, rank, n - rank, a, lda, &work[(mn + 1) - 1], b, ldb, &work[(2 * mn + 1) - 1], lwork - 2 * mn, info);
    }
    //
    //     complex workspace: 2*MN+NRHS.
    //
    //     B(1:N,1:NRHS) := P * B(1:N,1:NRHS)
    //
    for (j = 1; j <= nrhs; j = j + 1) {
        for (i = 1; i <= n; i = i + 1) {
            work[jpvt[i - 1] - 1] = b[(i - 1) + (j - 1) * ldb];
        }
        Ccopy(n, &work[1 - 1], 1, &b[(j - 1) * ldb], 1);
    }
    //
    //     complex workspace: N.
    //
    //     Undo scaling
    //
    if (iascl == 1) {
        Clascl("G", 0, 0, anrm, smlnum, n, nrhs, b, ldb, info);
        Clascl("U", 0, 0, smlnum, anrm, rank, rank, a, lda, info);
    } else if (iascl == 2) {
        Clascl("G", 0, 0, anrm, bignum, n, nrhs, b, ldb, info);
        Clascl("U", 0, 0, bignum, anrm, rank, rank, a, lda, info);
    }
    if (ibscl == 1) {
        Clascl("G", 0, 0, smlnum, bnrm, n, nrhs, b, ldb, info);
    } else if (ibscl == 2) {
        Clascl("G", 0, 0, bignum, bnrm, n, nrhs, b, ldb, info);
    }
//
statement_70:
    work[1 - 1] = COMPLEX(lwkopt);
    //
    //     End of Cgelsy
    //
}

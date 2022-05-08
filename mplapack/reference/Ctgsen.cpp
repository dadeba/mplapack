/*
 * Copyright (c) 2008-2022
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

void Ctgsen(INTEGER const ijob, bool const wantq, bool const wantz, bool *select, INTEGER const n, COMPLEX *a, INTEGER const lda, COMPLEX *b, INTEGER const ldb, COMPLEX *alpha, COMPLEX *beta, COMPLEX *q, INTEGER const ldq, COMPLEX *z, INTEGER const ldz, INTEGER &m, REAL &pl, REAL &pr, REAL *dif, COMPLEX *work, INTEGER const lwork, INTEGER *iwork, INTEGER const liwork, INTEGER &info) {
    bool lquery = false;
    INTEGER ierr = 0;
    bool wantp = false;
    bool wantd1 = false;
    bool wantd2 = false;
    bool wantd = false;
    INTEGER k = 0;
    INTEGER lwmin = 0;
    INTEGER liwmin = 0;
    const REAL one = 1.0;
    const REAL zero = 0.0;
    REAL rscale = 0.0;
    REAL dsum = 0.0;
    INTEGER i = 0;
    REAL safmin = 0.0;
    INTEGER ks = 0;
    bool swap = false;
    INTEGER n1 = 0;
    INTEGER n2 = 0;
    INTEGER ijb = 0;
    REAL rRscal = 0.0;
    const INTEGER idifjb = 3;
    INTEGER kase = 0;
    INTEGER mn2 = 0;
    INTEGER isave[3];
    COMPLEX temp1 = 0.0;
    COMPLEX temp2 = 0.0;
    //
    //     Decode and test the input parameters
    //
    info = 0;
    lquery = (lwork == -1 || liwork == -1);
    //
    if (ijob < 0 || ijob > 5) {
        info = -1;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max((INTEGER)1, n)) {
        info = -7;
    } else if (ldb < max((INTEGER)1, n)) {
        info = -9;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        info = -13;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        info = -15;
    }
    //
    if (info != 0) {
        Mxerbla("Ctgsen", -info);
        return;
    }
    //
    ierr = 0;
    //
    wantp = ijob == 1 || ijob >= 4;
    wantd1 = ijob == 2 || ijob == 4;
    wantd2 = ijob == 3 || ijob == 5;
    wantd = wantd1 || wantd2;
    //
    //     subspaces.
    //
    m = 0;
    if (!lquery || ijob != 0) {
        for (k = 1; k <= n; k = k + 1) {
            alpha[k - 1] = a[(k - 1) + (k - 1) * lda];
            beta[k - 1] = b[(k - 1) + (k - 1) * ldb];
            if (k < n) {
                if (select[k - 1]) {
                    m++;
                }
            } else {
                if (select[n - 1]) {
                    m++;
                }
            }
        }
    }
    //
    if (ijob == 1 || ijob == 2 || ijob == 4) {
        lwmin = max((INTEGER)1, 2 * m * (n - m));
        liwmin = max((INTEGER)1, n + 2);
    } else if (ijob == 3 || ijob == 5) {
        lwmin = max((INTEGER)1, 4 * m * (n - m));
        liwmin = max({(INTEGER)1, 2 * m * (n - m), n + 2});
    } else {
        lwmin = 1;
        liwmin = 1;
    }
    //
    work[1 - 1] = lwmin;
    iwork[1 - 1] = liwmin;
    //
    if (lwork < lwmin && !lquery) {
        info = -21;
    } else if (liwork < liwmin && !lquery) {
        info = -23;
    }
    //
    if (info != 0) {
        Mxerbla("Ctgsen", -info);
        return;
    } else if (lquery) {
        return;
    }
    //
    //     Quick return if possible.
    //
    if (m == n || m == 0) {
        if (wantp) {
            pl = one;
            pr = one;
        }
        if (wantd) {
            rscale = zero;
            dsum = one;
            for (i = 1; i <= n; i = i + 1) {
                Classq(n, &a[(i - 1) * lda], 1, rscale, dsum);
                Classq(n, &b[(i - 1) * ldb], 1, rscale, dsum);
            }
            dif[1 - 1] = rscale * sqrt(dsum);
            dif[2 - 1] = dif[1 - 1];
        }
        goto statement_70;
    }
    //
    //     Get machine constant
    //
    safmin = Rlamch("S");
    //
    //     Collect the selected blocks at the top-left corner of (A, B).
    //
    ks = 0;
    for (k = 1; k <= n; k = k + 1) {
        swap = select[k - 1];
        if (swap) {
            ks++;
            //
            //           Swap the K-th block to position KS. Compute unitary Q
            //           and Z that will swap adjacent diagonal blocks in (A, B).
            //
            if (k != ks) {
                Ctgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, k, ks, ierr);
            }
            //
            if (ierr > 0) {
                //
                //              Swap is rejected: exit.
                //
                info = 1;
                if (wantp) {
                    pl = zero;
                    pr = zero;
                }
                if (wantd) {
                    dif[1 - 1] = zero;
                    dif[2 - 1] = zero;
                }
                goto statement_70;
            }
        }
    }
    if (wantp) {
        //
        //        Solve generalized Sylvester equation for R and L:
        //                   A11 * R - L * A22 = A12
        //                   B11 * R - L * B22 = B12
        //
        n1 = m;
        n2 = n - m;
        i = n1 + 1;
        Clacpy("Full", n1, n2, &a[(i - 1) * lda], lda, work, n1);
        Clacpy("Full", n1, n2, &b[(i - 1) * ldb], ldb, &work[(n1 * n2 + 1) - 1], n1);
        ijb = 0;
        Ctgsyl("N", ijb, n1, n2, a, lda, &a[(i - 1) + (i - 1) * lda], lda, work, n1, b, ldb, &b[(i - 1) + (i - 1) * ldb], ldb, &work[(n1 * n2 + 1) - 1], n1, rscale, dif[1 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
        //
        //        Estimate the reciprocal of norms of "projections" onto
        //        left and right eigenspaces
        //
        rRscal = zero;
        dsum = one;
        Classq(n1 * n2, work, 1, rRscal, dsum);
        pl = rRscal * sqrt(dsum);
        if (pl == zero) {
            pl = one;
        } else {
            pl = rscale / (sqrt(rscale * rscale / pl + pl) * sqrt(pl));
        }
        rRscal = zero;
        dsum = one;
        Classq(n1 * n2, &work[(n1 * n2 + 1) - 1], 1, rRscal, dsum);
        pr = rRscal * sqrt(dsum);
        if (pr == zero) {
            pr = one;
        } else {
            pr = rscale / (sqrt(rscale * rscale / pr + pr) * sqrt(pr));
        }
    }
    if (wantd) {
        //
        //        Compute estimates Difu and Difl.
        //
        if (wantd1) {
            n1 = m;
            n2 = n - m;
            i = n1 + 1;
            ijb = idifjb;
            //
            //           Frobenius norm-based Difu estimate.
            //
            Ctgsyl("N", ijb, n1, n2, a, lda, &a[(i - 1) + (i - 1) * lda], lda, work, n1, b, ldb, &b[(i - 1) + (i - 1) * ldb], ldb, &work[(n1 * n2 + 1) - 1], n1, rscale, dif[1 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
            //
            //           Frobenius norm-based Difl estimate.
            //
            Ctgsyl("N", ijb, n2, n1, &a[(i - 1) + (i - 1) * lda], lda, a, lda, work, n2, &b[(i - 1) + (i - 1) * ldb], ldb, b, ldb, &work[(n1 * n2 + 1) - 1], n2, rscale, dif[2 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
        } else {
            //
            //           Compute 1-norm-based estimates of Difu and Difl using
            //           reversed communication with Clacn2. In each step a
            //           generalized Sylvester equation or a transposed variant
            //           is solved.
            //
            kase = 0;
            n1 = m;
            n2 = n - m;
            i = n1 + 1;
            ijb = 0;
            mn2 = 2 * n1 * n2;
        //
        //           1-norm-based estimate of Difu.
        //
        statement_40:
            Clacn2(mn2, &work[(mn2 + 1) - 1], work, dif[1 - 1], kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    //
                    //                 Solve generalized Sylvester equation
                    //
                    Ctgsyl("N", ijb, n1, n2, a, lda, &a[(i - 1) + (i - 1) * lda], lda, work, n1, b, ldb, &b[(i - 1) + (i - 1) * ldb], ldb, &work[(n1 * n2 + 1) - 1], n1, rscale, dif[1 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
                } else {
                    //
                    //                 Solve the transposed variant.
                    //
                    Ctgsyl("C", ijb, n1, n2, a, lda, &a[(i - 1) + (i - 1) * lda], lda, work, n1, b, ldb, &b[(i - 1) + (i - 1) * ldb], ldb, &work[(n1 * n2 + 1) - 1], n1, rscale, dif[1 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
                }
                goto statement_40;
            }
            dif[1 - 1] = rscale / dif[1 - 1];
        //
        //           1-norm-based estimate of Difl.
        //
        statement_50:
            Clacn2(mn2, &work[(mn2 + 1) - 1], work, dif[2 - 1], kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    //
                    //                 Solve generalized Sylvester equation
                    //
                    Ctgsyl("N", ijb, n2, n1, &a[(i - 1) + (i - 1) * lda], lda, a, lda, work, n2, &b[(i - 1) + (i - 1) * ldb], ldb, b, ldb, &work[(n1 * n2 + 1) - 1], n2, rscale, dif[2 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
                } else {
                    //
                    //                 Solve the transposed variant.
                    //
                    Ctgsyl("C", ijb, n2, n1, &a[(i - 1) + (i - 1) * lda], lda, a, lda, work, n2, b, ldb, &b[(i - 1) + (i - 1) * ldb], ldb, &work[(n1 * n2 + 1) - 1], n2, rscale, dif[2 - 1], &work[(n1 * n2 * 2 + 1) - 1], lwork - 2 * n1 * n2, iwork, ierr);
                }
                goto statement_50;
            }
            dif[2 - 1] = rscale / dif[2 - 1];
        }
    }
    //
    //     If B(K,K) is complex, make it real and positive (normalization
    //     of the generalized Schur form) and Store the generalized
    //     eigenvalues of reordered pair (A, B)
    //
    for (k = 1; k <= n; k = k + 1) {
        rscale = abs(b[(k - 1) + (k - 1) * ldb]);
        if (rscale > safmin) {
            temp1 = conj(b[(k - 1) + (k - 1) * ldb] / rscale);
            temp2 = b[(k - 1) + (k - 1) * ldb] / rscale;
            b[(k - 1) + (k - 1) * ldb] = rscale;
            Cscal(n - k, temp1, &b[(k - 1) + ((k + 1) - 1) * ldb], ldb);
            Cscal(n - k + 1, temp1, &a[(k - 1) + (k - 1) * lda], lda);
            if (wantq) {
                Cscal(n, temp2, &q[(k - 1) * ldq], 1);
            }
        } else {
            b[(k - 1) + (k - 1) * ldb] = COMPLEX(zero, zero);
        }
        //
        alpha[k - 1] = a[(k - 1) + (k - 1) * lda];
        beta[k - 1] = b[(k - 1) + (k - 1) * ldb];
        //
    }
//
statement_70:
    //
    work[1 - 1] = lwmin;
    iwork[1 - 1] = liwmin;
    //
    //     End of Ctgsen
    //
}

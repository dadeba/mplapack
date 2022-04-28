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

void Cstein(INTEGER const n, REAL *d, REAL *e, INTEGER const m, REAL *w, INTEGER *iblock, INTEGER *isplit, COMPLEX *z, INTEGER const ldz, REAL *work, INTEGER *iwork, INTEGER *ifail, INTEGER &info) {
    INTEGER i = 0;
    INTEGER j = 0;
    const COMPLEX cone = COMPLEX(1.0, 0.0);
    REAL eps = 0.0;
    INTEGER iseed[4];
    INTEGER indrv1 = 0;
    INTEGER indrv2 = 0;
    INTEGER indrv3 = 0;
    INTEGER indrv4 = 0;
    INTEGER indrv5 = 0;
    INTEGER j1 = 0;
    INTEGER nblk = 0;
    INTEGER b1 = 0;
    INTEGER bn = 0;
    INTEGER blksiz = 0;
    INTEGER gpind = 0;
    REAL onenrm = 0.0;
    const REAL odm3 = 1.0e-3;
    REAL ortol = 0.0;
    const REAL odm1 = 1.0e-1;
    REAL dtpcrt = 0.0;
    INTEGER jblk = 0;
    REAL xj = 0.0;
    const REAL one = 1.0;
    REAL eps1 = 0.0;
    const REAL ten = 1.0e+1;
    REAL pertol = 0.0;
    REAL xjm = 0.0;
    REAL sep = 0.0;
    INTEGER its = 0;
    INTEGER nrmchk = 0;
    const REAL zero = 0.0;
    REAL tol = 0.0;
    INTEGER iinfo = 0;
    const INTEGER maxits = 5;
    INTEGER jmax = 0;
    REAL scl = 0.0;
    REAL ztr = 0.0;
    INTEGER jr = 0;
    REAL nrm = 0.0;
    const INTEGER extra = 2;
    const COMPLEX czero = COMPLEX(0.0, 0.0);
    //
    //     Test the input parameters.
    //
    info = 0;
    for (i = 1; i <= m; i = i + 1) {
        ifail[i - 1] = 0;
    }
    //
    if (n < 0) {
        info = -1;
    } else if (m < 0 || m > n) {
        info = -4;
    } else if (ldz < max((INTEGER)1, n)) {
        info = -9;
    } else {
        for (j = 2; j <= m; j = j + 1) {
            if (iblock[j - 1] < iblock[(j - 1) - 1]) {
                info = -6;
                goto statement_30;
            }
            if (iblock[j - 1] == iblock[(j - 1) - 1] && w[j - 1] < w[(j - 1) - 1]) {
                info = -5;
                goto statement_30;
            }
        }
    statement_30:;
    }
    //
    if (info != 0) {
        Mxerbla("Cstein", -info);
        return;
    }
    //
    //     Quick return if possible
    //
    if (n == 0 || m == 0) {
        return;
    } else if (n == 1) {
        z[(1 - 1)] = cone;
        return;
    }
    //
    //     Get machine constants.
    //
    eps = Rlamch("Precision");
    //
    //     Initialize seed for random number generator Rlarnv.
    //
    for (i = 1; i <= 4; i = i + 1) {
        iseed[i - 1] = 1;
    }
    //
    //     Initialize pointers.
    //
    indrv1 = 0;
    indrv2 = indrv1 + n;
    indrv3 = indrv2 + n;
    indrv4 = indrv3 + n;
    indrv5 = indrv4 + n;
    //
    //     Compute eigenvectors of matrix blocks.
    //
    j1 = 1;
    for (nblk = 1; nblk <= iblock[m - 1]; nblk = nblk + 1) {
        //
        //        Find starting and ending indices of block nblk.
        //
        if (nblk == 1) {
            b1 = 1;
        } else {
            b1 = isplit[(nblk - 1) - 1] + 1;
        }
        bn = isplit[nblk - 1];
        blksiz = bn - b1 + 1;
        if (blksiz == 1) {
            goto statement_60;
        }
        gpind = j1;
        //
        //        Compute reorthogonalization criterion and stopping criterion.
        //
        onenrm = abs(d[b1 - 1]) + abs(e[b1 - 1]);
        onenrm = max(onenrm, REAL(abs(d[bn - 1]) + abs(e[(bn - 1) - 1])));
        for (i = b1 + 1; i <= bn - 1; i = i + 1) {
            onenrm = max(onenrm, REAL(abs(d[i - 1]) + abs(e[(i - 1) - 1]) + abs(e[i - 1])));
        }
        ortol = odm3 * onenrm;
        //
        dtpcrt = sqrt(odm1 / blksiz);
    //
    //        Loop through eigenvalues of block nblk.
    //
    statement_60:
        jblk = 0;
        for (j = j1; j <= m; j = j + 1) {
            if (iblock[j - 1] != nblk) {
                j1 = j;
                goto statement_180;
            }
            jblk++;
            xj = w[j - 1];
            //
            //           Skip all the work if the block size is one.
            //
            if (blksiz == 1) {
                work[(indrv1 + 1) - 1] = one;
                goto statement_140;
            }
            //
            //           If eigenvalues j and j-1 are too close, add a relatively
            //           small perturbation.
            //
            if (jblk > 1) {
                eps1 = abs(eps * xj);
                pertol = ten * eps1;
                sep = xj - xjm;
                if (sep < pertol) {
                    xj = xjm + pertol;
                }
            }
            //
            its = 0;
            nrmchk = 0;
            //
            //           Get random starting vector.
            //
            Rlarnv(2, iseed, blksiz, &work[(indrv1 + 1) - 1]);
            //
            //           Copy the matrix T so it won't be destroyed in factorization.
            //
            Rcopy(blksiz, &d[b1 - 1], 1, &work[(indrv4 + 1) - 1], 1);
            Rcopy(blksiz - 1, &e[b1 - 1], 1, &work[(indrv2 + 2) - 1], 1);
            Rcopy(blksiz - 1, &e[b1 - 1], 1, &work[(indrv3 + 1) - 1], 1);
            //
            //           Compute LU factors with partial pivoting  ( PT = LU )
            //
            tol = zero;
            Rlagtf(blksiz, &work[(indrv4 + 1) - 1], xj, &work[(indrv2 + 2) - 1], &work[(indrv3 + 1) - 1], tol, &work[(indrv5 + 1) - 1], iwork, iinfo);
        //
        //           Update iteration count.
        //
        statement_70:
            its++;
            if (its > maxits) {
                goto statement_120;
            }
            //
            //           Normalize and scale the righthand side vector Pb.
            //
            jmax = iRamax(blksiz, &work[(indrv1 + 1) - 1], 1);
            scl = blksiz * onenrm * max(eps, REAL(abs(work[(indrv4 + blksiz) - 1]) / abs(work[(indrv1 + jmax) - 1])));
            Rscal(blksiz, scl, &work[(indrv1 + 1) - 1], 1);
            //
            //           Solve the system LU = Pb.
            //
            Rlagts(-1, blksiz, &work[(indrv4 + 1) - 1], &work[(indrv2 + 2) - 1], &work[(indrv3 + 1) - 1], &work[(indrv5 + 1) - 1], iwork, &work[(indrv1 + 1) - 1], tol, iinfo);
            //
            //           Reorthogonalize by modified Gram-Schmidt if eigenvalues are
            //           close enough.
            //
            if (jblk == 1) {
                goto statement_110;
            }
            if (abs(xj - xjm) > ortol) {
                gpind = j;
            }
            if (gpind != j) {
                for (i = gpind; i <= j - 1; i = i + 1) {
                    ztr = zero;
                    for (jr = 1; jr <= blksiz; jr = jr + 1) {
                        ztr += work[(indrv1 + jr) - 1] * z[((b1 - 1 + jr) - 1) + (i - 1) * ldz].real();
                    }
                    for (jr = 1; jr <= blksiz; jr = jr + 1) {
                        work[(indrv1 + jr) - 1] = work[(indrv1 + jr) - 1] - ztr * z[((b1 - 1 + jr) - 1) + (i - 1) * ldz].real();
                    }
                }
            }
        //
        //           Check the infinity norm of the iterate.
        //
        statement_110:
            jmax = iRamax(blksiz, &work[(indrv1 + 1) - 1], 1);
            nrm = abs(work[(indrv1 + jmax) - 1]);
            //
            //           Continue for additional iterations after norm reaches
            //           stopping criterion.
            //
            if (nrm < dtpcrt) {
                goto statement_70;
            }
            nrmchk++;
            if (nrmchk < extra + 1) {
                goto statement_70;
            }
            //
            goto statement_130;
        //
        //           If stopping criterion was not satisfied, update info and
        //           store eigenvector number in array ifail.
        //
        statement_120:
            info++;
            ifail[info - 1] = j;
        //
        //           Accept iterate as jth eigenvector.
        //
        statement_130:
            scl = one / Rnrm2(blksiz, &work[(indrv1 + 1) - 1], 1);
            jmax = iRamax(blksiz, &work[(indrv1 + 1) - 1], 1);
            if (work[(indrv1 + jmax) - 1] < zero) {
                scl = -scl;
            }
            Rscal(blksiz, scl, &work[(indrv1 + 1) - 1], 1);
        statement_140:
            for (i = 1; i <= n; i = i + 1) {
                z[(i - 1) + (j - 1) * ldz] = czero;
            }
            for (i = 1; i <= blksiz; i = i + 1) {
                z[((b1 + i - 1) - 1) + (j - 1) * ldz] = COMPLEX(work[(indrv1 + i) - 1], zero);
            }
            //
            //           Save the shift to check eigenvalue spacing at next
            //           iteration.
            //
            xjm = xj;
            //
        }
    statement_180:;
    }
    //
    //     End of Cstein
    //
}

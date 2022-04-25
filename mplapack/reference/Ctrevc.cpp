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

inline REAL cabs1(COMPLEX cdum) { return abs(cdum.real()) + abs(cdum.imag()); }

void Ctrevc(const char *side, const char *howmny, bool *select, INTEGER const n, COMPLEX *t, INTEGER const ldt, COMPLEX *vl, INTEGER const ldvl, COMPLEX *vr, INTEGER const ldvr, INTEGER const mm, INTEGER &m, COMPLEX *work, REAL *rwork, INTEGER &info) {
    COMPLEX cdum = 0.0;
    bool bothv = false;
    bool rightv = false;
    bool leftv = false;
    bool allv = false;
    bool over = false;
    bool somev = false;
    INTEGER j = 0;
    REAL unfl = 0.0;
    const REAL one = 1.0;
    REAL ovfl = 0.0;
    REAL ulp = 0.0;
    REAL smlnum = 0.0;
    INTEGER i = 0;
    const REAL zero = 0.0;
    INTEGER is = 0;
    INTEGER ki = 0;
    REAL smin = 0.0;
    const COMPLEX cmone = COMPLEX(1.0, 0.0);
    INTEGER k = 0;
    REAL scale = 0.0;
    INTEGER ii = 0;
    REAL remax = 0.0;
    const COMPLEX cmzero = COMPLEX(0.0, 0.0);
    //
    //     Decode and test the input parameters
    //
    bothv = Mlsame(side, "B");
    rightv = Mlsame(side, "R") || bothv;
    leftv = Mlsame(side, "L") || bothv;
    //
    allv = Mlsame(howmny, "A");
    over = Mlsame(howmny, "B");
    somev = Mlsame(howmny, "S");
    //
    //     Set M to the number of columns required to store the selected
    //     eigenvectors.
    //
    if (somev) {
        m = 0;
        for (j = 1; j <= n; j = j + 1) {
            if (select[j - 1]) {
                m++;
            }
        }
    } else {
        m = n;
    }
    //
    info = 0;
    if (!rightv && !leftv) {
        info = -1;
    } else if (!allv && !over && !somev) {
        info = -2;
    } else if (n < 0) {
        info = -4;
    } else if (ldt < max((INTEGER)1, n)) {
        info = -6;
    } else if (ldvl < 1 || (leftv && ldvl < n)) {
        info = -8;
    } else if (ldvr < 1 || (rightv && ldvr < n)) {
        info = -10;
    } else if (mm < m) {
        info = -11;
    }
    if (info != 0) {
        Mxerbla("Ctrevc", -info);
        return;
    }
    //
    //     Quick return if possible.
    //
    if (n == 0) {
        return;
    }
    //
    //     Set the constants to control overflow.
    //
    unfl = Rlamch("Safe minimum");
    ovfl = one / unfl;
    ulp = Rlamch("Precision");
    smlnum = unfl * (n / ulp);
    //
    //     Store the diagonal elements of T in working array WORK.
    //
    for (i = 1; i <= n; i = i + 1) {
        work[(i + n) - 1] = t[(i - 1) + (i - 1) * ldt];
    }
    //
    //     Compute 1-norm of each column of strictly upper triangular
    //     part of T to control overflow in triangular solver.
    //
    rwork[1 - 1] = zero;
    for (j = 2; j <= n; j = j + 1) {
        rwork[j - 1] = RCasum(j - 1, &t[(j - 1) * ldt], 1);
    }
    //
    if (rightv) {
        //
        //        Compute right eigenvectors.
        //
        is = m;
        for (ki = n; ki >= 1; ki = ki - 1) {
            //
            if (somev) {
                if (!select[ki - 1]) {
                    goto statement_80;
                }
            }
            smin = max(REAL(ulp * (cabs1(t[(ki - 1) + (ki - 1) * ldt]))), smlnum);
            //
            work[1 - 1] = cmone;
            //
            //           Form right-hand side.
            //
            for (k = 1; k <= ki - 1; k = k + 1) {
                work[k - 1] = -t[(k - 1) + (ki - 1) * ldt];
            }
            //
            //           Solve the triangular system:
            //              (T(1:KI-1,1:KI-1) - T(KI,KI))*X = SCALE*WORK.
            //
            for (k = 1; k <= ki - 1; k = k + 1) {
                t[(k - 1) + (k - 1) * ldt] = t[(k - 1) + (k - 1) * ldt] - t[(ki - 1) + (ki - 1) * ldt];
                if (cabs1(t[(k - 1) + (k - 1) * ldt]) < smin) {
                    t[(k - 1) + (k - 1) * ldt] = smin;
                }
            }
            //
            if (ki > 1) {
                Clatrs("Upper", "No transpose", "Non-unit", "Y", ki - 1, t, ldt, &work[1 - 1], scale, rwork, info);
                work[ki - 1] = scale;
            }
            //
            //           Copy the vector x or Q*x to VR and normalize.
            //
            if (!over) {
                Ccopy(ki, &work[1 - 1], 1, &vr[(is - 1) * ldvr], 1);
                //
                ii = iCamax(ki, &vr[(is - 1) * ldvr], 1);
                remax = one / abs1(vr[(ii - 1) + (is - 1) * ldvr]);
                CRscal(ki, remax, &vr[(is - 1) * ldvr], 1);
                //
                for (k = ki + 1; k <= n; k = k + 1) {
                    vr[(k - 1) + (is - 1) * ldvr] = cmzero;
                }
            } else {
                if (ki > 1) {
                    Cgemv("N", n, ki - 1, cmone, vr, ldvr, &work[1 - 1], 1, COMPLEX(scale), &vr[(ki - 1) * ldvr], 1);
                }
                //
                ii = iCamax(n, &vr[(ki - 1) * ldvr], 1);
                remax = one / abs1(vr[(ii - 1) + (ki - 1) * ldvr]);
                CRscal(n, remax, &vr[(ki - 1) * ldvr], 1);
            }
            //
            //           Set back the original diagonal elements of T.
            //
            for (k = 1; k <= ki - 1; k = k + 1) {
                t[(k - 1) + (k - 1) * ldt] = work[(k + n) - 1];
            }
            //
            is = is - 1;
        statement_80:;
        }
    }
    //
    if (leftv) {
        //
        //        Compute left eigenvectors.
        //
        is = 1;
        for (ki = 1; ki <= n; ki = ki + 1) {
            //
            if (somev) {
                if (!select[ki - 1]) {
                    goto statement_130;
                }
            }
            smin = max(REAL(ulp * (abs1(t[(ki - 1) + (ki - 1) * ldt]))), smlnum);
            //
            work[n - 1] = cmone;
            //
            //           Form right-hand side.
            //
            for (k = ki + 1; k <= n; k = k + 1) {
                work[k - 1] = -conj(t[(ki - 1) + (k - 1) * ldt]);
            }
            //
            //           Solve the triangular system:
            //              (T(KI+1:N,KI+1:N) - T(KI,KI))**H * X = SCALE*WORK.
            //
            for (k = ki + 1; k <= n; k = k + 1) {
                t[(k - 1) + (k - 1) * ldt] = t[(k - 1) + (k - 1) * ldt] - t[(ki - 1) + (ki - 1) * ldt];
                if (abs1(t[(k - 1) + (k - 1) * ldt]) < smin) {
                    t[(k - 1) + (k - 1) * ldt] = smin;
                }
            }
            //
            if (ki < n) {
                Clatrs("Upper", "Conjugate transpose", "Non-unit", "Y", n - ki, &t[((ki + 1) - 1) + ((ki + 1) - 1) * ldt], ldt, &work[(ki + 1) - 1], scale, rwork, info);
                work[ki - 1] = scale;
            }
            //
            //           Copy the vector x or Q*x to VL and normalize.
            //
            if (!over) {
                Ccopy(n - ki + 1, &work[ki - 1], 1, &vl[(ki - 1) + (is - 1) * ldvl], 1);
                //
                ii = iCamax(n - ki + 1, &vl[(ki - 1) + (is - 1) * ldvl], 1) + ki - 1;
                remax = one / abs1(vl[(ii - 1) + (is - 1) * ldvl]);
                CRscal(n - ki + 1, remax, &vl[(ki - 1) + (is - 1) * ldvl], 1);
                //
                for (k = 1; k <= ki - 1; k = k + 1) {
                    vl[(k - 1) + (is - 1) * ldvl] = cmzero;
                }
            } else {
                if (ki < n) {
                    Cgemv("N", n, n - ki, cmone, &vl[((ki + 1) - 1) * ldvl], ldvl, &work[(ki + 1) - 1], 1, COMPLEX(scale), &vl[(ki - 1) * ldvl], 1);
                }
                //
                ii = iCamax(n, &vl[(ki - 1) * ldvl], 1);
                remax = one / abs1(vl[(ii - 1) + (ki - 1) * ldvl]);
                CRscal(n, remax, &vl[(ki - 1) * ldvl], 1);
            }
            //
            //           Set back the original diagonal elements of T.
            //
            for (k = ki + 1; k <= n; k = k + 1) {
                t[(k - 1) + (k - 1) * ldt] = work[(k + n) - 1];
            }
            //
            is++;
        statement_130:;
        }
    }
    //
    //     End of Ctrevc
    //
}

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

void Rchkeq(REAL const thresh, INTEGER const nout) {
    common cmn;
    common_write write(cmn);
    //
    //  -- LAPACK test routine --
    //  -- LAPACK is a software package provided by Univ. of Tennessee,    --
    //  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //
    //     .. Scalar Arguments ..
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
    //     .. External Functions ..
    //     ..
    //     .. External Subroutines ..
    //     ..
    //     .. Intrinsic Functions ..
    //     ..
    //     .. Executable Statements ..
    //
    char path[3];
    char buf1[1024];
    char buf2[1024];
    path[0] = 'R';
    path[1] = 'E';
    path[2] = 'Q';
    //
    REAL eps = Rlamch("P");
    INTEGER i = 0;
    const REAL zero = 0.0;
    REAL reslts[5];
    for (i = 1; i <= 5; i = i + 1) {
        reslts[i - 1] = zero;
    }
    const INTEGER nsz = 5;
    const INTEGER npow = 2 * nsz + 1;
    const REAL ten = 10.0;
    REAL mpow[npow];
    const REAL one = 1.0;
    REAL rpow[npow];
    for (i = 1; i <= npow; i = i + 1) {
        mpow[i - 1] = (REAL)pow(cast2double(ten), double(i - 1));
        rpow[i - 1] = one / mpow[i - 1];
    }
    //
    //     Test Rgeequ
    //
    INTEGER n = 0;
    INTEGER m = 0;
    INTEGER j = 0;
    REAL a[nsz * nsz];
    INTEGER lda = nsz;
    REAL r[nsz];
    REAL c[nsz];
    REAL rcond = 0.0;
    REAL ccond = 0.0;
    REAL norm = 0.0;
    INTEGER info = 0;
    for (n = 0; n <= nsz; n = n + 1) {
        for (m = 0; m <= nsz; m = m + 1) {
            //
            for (j = 1; j <= nsz; j = j + 1) {
                for (i = 1; i <= nsz; i = i + 1) {
                    if (i <= m && j <= n) {
                        a[(i - 1) + (j - 1) * lda] = mpow[(i + j + 1) - 1] * pow((-1), (i + j));
                    } else {
                        a[(i - 1) + (j - 1) * lda] = zero;
                    }
                }
            }
            //
            Rgeequ(m, n, a, nsz, r, c, rcond, ccond, norm, info);
            //
            if (info != 0) {
                reslts[1 - 1] = one;
            } else {
                if (n != 0 && m != 0) {
                    reslts[1 - 1] = max(reslts[1 - 1], REAL(abs((rcond - rpow[m - 1]) / rpow[m - 1])));
                    reslts[1 - 1] = max(reslts[1 - 1], REAL(abs((ccond - rpow[n - 1]) / rpow[n - 1])));
                    reslts[1 - 1] = max(reslts[1 - 1], REAL(abs((norm - mpow[(n + m + 1) - 1]) / mpow[(n + m + 1) - 1])));
                    for (i = 1; i <= m; i = i + 1) {
                        reslts[1 - 1] = max(reslts[1 - 1], REAL(abs((r[i - 1] - rpow[(i + n + 1) - 1]) / rpow[(i + n + 1) - 1])));
                    }
                    for (j = 1; j <= n; j = j + 1) {
                        reslts[1 - 1] = max(reslts[1 - 1], REAL(abs((c[j - 1] - mpow[(n - j + 1) - 1]) / mpow[(n - j + 1) - 1])));
                    }
                }
            }
            //
        }
    }
    //
    //     Test with zero rows and columns
    //
    for (j = 1; j <= nsz; j = j + 1) {
        a[(max(nsz - 1, (INTEGER)1) - 1) + (j - 1) * lda] = zero;
    }
    Rgeequ(nsz, nsz, a, nsz, r, c, rcond, ccond, norm, info);
    if (info != max(nsz - 1, (INTEGER)1)) {
        reslts[1 - 1] = one;
    }
    //
    for (j = 1; j <= nsz; j = j + 1) {
        a[(max(nsz - 1, (INTEGER)1) - 1) + (j - 1) * lda] = one;
    }
    for (i = 1; i <= nsz; i = i + 1) {
        a[(i - 1) + (max(nsz - 1, (INTEGER)1) - 1) * lda] = zero;
    }
    Rgeequ(nsz, nsz, a, nsz, r, c, rcond, ccond, norm, info);
    if (info != nsz + max(nsz - 1, (INTEGER)1)) {
        reslts[1 - 1] = one;
    }
    reslts[1 - 1] = reslts[1 - 1] / eps;
    //
    //     Test Rgbequ
    //
    INTEGER kl = 0;
    INTEGER ku = 0;
    const INTEGER nszb = 3 * nsz - 2;
    REAL ab[nszb * nsz];
    INTEGER ldab = nszb;
    REAL rcmin = 0.0;
    REAL rcmax = 0.0;
    REAL ratio = 0.0;
    for (n = 0; n <= nsz; n = n + 1) {
        for (m = 0; m <= nsz; m = m + 1) {
            for (kl = 0; kl <= max(m - 1, (INTEGER)0); kl = kl + 1) {
                for (ku = 0; ku <= max(n - 1, (INTEGER)0); ku = ku + 1) {
                    //
                    for (j = 1; j <= nsz; j = j + 1) {
                        for (i = 1; i <= nszb; i = i + 1) {
                            ab[(i - 1) + (j - 1) * ldab] = zero;
                        }
                    }
                    for (j = 1; j <= n; j = j + 1) {
                        for (i = 1; i <= m; i = i + 1) {
                            if (i <= min(m, j + kl) && i >= max((INTEGER)1, j - ku) && j <= n) {
                                ab[((ku + 1 + i - j) - 1) + (j - 1) * ldab] = mpow[(i + j + 1) - 1] * std::pow((-1), (i + j));
                            }
                        }
                    }
                    //
                    Rgbequ(m, n, kl, ku, ab, nszb, r, c, rcond, ccond, norm, info);
                    //
                    if (info != 0) {
                        if (!((n + kl < m && info == n + kl + 1) || (m + ku < n && info == 2 * m + ku + 1))) {
                            reslts[2 - 1] = one;
                        }
                    } else {
                        if (n != 0 && m != 0) {
                            //
                            rcmin = r[1 - 1];
                            rcmax = r[1 - 1];
                            for (i = 1; i <= m; i = i + 1) {
                                rcmin = min(rcmin, r[i - 1]);
                                rcmax = max(rcmax, r[i - 1]);
                            }
                            ratio = rcmin / rcmax;
                            reslts[2 - 1] = max(reslts[2 - 1], REAL(abs((rcond - ratio) / ratio)));
                            //
                            rcmin = c[1 - 1];
                            rcmax = c[1 - 1];
                            for (j = 1; j <= n; j = j + 1) {
                                rcmin = min(rcmin, c[j - 1]);
                                rcmax = max(rcmax, c[j - 1]);
                            }
                            ratio = rcmin / rcmax;
                            reslts[2 - 1] = max(reslts[2 - 1], REAL(abs((ccond - ratio) / ratio)));
                            //
                            reslts[2 - 1] = max(reslts[2 - 1], REAL(abs((norm - mpow[(n + m + 1) - 1]) / mpow[(n + m + 1) - 1])));
                            for (i = 1; i <= m; i = i + 1) {
                                rcmax = zero;
                                for (j = 1; j <= n; j = j + 1) {
                                    if (i <= j + kl && i >= j - ku) {
                                        ratio = abs(r[i - 1] * mpow[(i + j + 1) - 1] * c[j - 1]);
                                        rcmax = max(rcmax, ratio);
                                    }
                                }
                                reslts[2 - 1] = max(reslts[2 - 1], REAL(abs(one - rcmax)));
                            }
                            //
                            for (j = 1; j <= n; j = j + 1) {
                                rcmax = zero;
                                for (i = 1; i <= m; i = i + 1) {
                                    if (i <= j + kl && i >= j - ku) {
                                        ratio = abs(r[i - 1] * mpow[(i + j + 1) - 1] * c[j - 1]);
                                        rcmax = max(rcmax, ratio);
                                    }
                                }
                                reslts[2 - 1] = max(reslts[2 - 1], REAL(abs(one - rcmax)));
                            }
                        }
                    }
                    //
                }
            }
        }
    }
    reslts[2 - 1] = reslts[2 - 1] / eps;
    //
    //     Test Rpoequ
    //
    for (n = 0; n <= nsz; n = n + 1) {
        //
        for (i = 1; i <= nsz; i = i + 1) {
            for (j = 1; j <= nsz; j = j + 1) {
                if (i <= n && j == i) {
                    a[(i - 1) + (j - 1) * lda] = mpow[(i + j + 1) - 1] * std::pow((-1), (i + j));
                } else {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
            }
        }
        //
        Rpoequ(n, a, nsz, r, rcond, norm, info);
        //
        if (info != 0) {
            reslts[3 - 1] = one;
        } else {
            if (n != 0) {
                reslts[3 - 1] = max(reslts[3 - 1], REAL(abs((rcond - rpow[n - 1]) / rpow[n - 1])));
                reslts[3 - 1] = max(reslts[3 - 1], REAL(abs((norm - mpow[(2 * n + 1) - 1]) / mpow[(2 * n + 1) - 1])));
                for (i = 1; i <= n; i = i + 1) {
                    reslts[3 - 1] = max(reslts[3 - 1], REAL(abs((r[i - 1] - rpow[(i + 1) - 1]) / rpow[(i + 1) - 1])));
                }
            }
        }
    }
    a[max(nsz - 1, (INTEGER)1) - 1 + (max(nsz - 1, (INTEGER)1) - 1) * lda] = -one;
    Rpoequ(nsz, a, nsz, r, rcond, norm, info);
    if (info != max(nsz - 1, (INTEGER)1)) {
        reslts[3 - 1] = one;
    }
    reslts[3 - 1] = reslts[3 - 1] / eps;
    //
    //     Test Rppequ
    //
    const INTEGER nszp = (nsz * (nsz + 1)) / 2;
    REAL ap[nszp];
    for (n = 0; n <= nsz; n = n + 1) {
        //
        //        Upper triangular packed storage
        //
        for (i = 1; i <= (n * (n + 1)) / 2; i = i + 1) {
            ap[i - 1] = zero;
        }
        for (i = 1; i <= n; i = i + 1) {
            ap[((i * (i + 1)) / 2) - 1] = mpow[(2 * i + 1) - 1];
        }
        //
        Rppequ("U", n, ap, r, rcond, norm, info);
        //
        if (info != 0) {
            reslts[4 - 1] = one;
        } else {
            if (n != 0) {
                reslts[4 - 1] = max(reslts[4 - 1], REAL(abs((rcond - rpow[n - 1]) / rpow[n - 1])));
                reslts[4 - 1] = max(reslts[4 - 1], REAL(abs((norm - mpow[(2 * n + 1) - 1]) / mpow[(2 * n + 1) - 1])));
                for (i = 1; i <= n; i = i + 1) {
                    reslts[4 - 1] = max(reslts[4 - 1], REAL(abs((r[i - 1] - rpow[(i + 1) - 1]) / rpow[(i + 1) - 1])));
                }
            }
        }
        //
        //        Lower triangular packed storage
        //
        for (i = 1; i <= (n * (n + 1)) / 2; i = i + 1) {
            ap[i - 1] = zero;
        }
        j = 1;
        for (i = 1; i <= n; i = i + 1) {
            ap[j - 1] = mpow[(2 * i + 1) - 1];
            j += (n - i + 1);
        }
        //
        Rppequ("L", n, ap, r, rcond, norm, info);
        //
        if (info != 0) {
            reslts[4 - 1] = one;
        } else {
            if (n != 0) {
                reslts[4 - 1] = max(reslts[4 - 1], REAL(abs((rcond - rpow[n - 1]) / rpow[n - 1])));
                reslts[4 - 1] = max(reslts[4 - 1], REAL(abs((norm - mpow[(2 * n + 1) - 1]) / mpow[(2 * n + 1) - 1])));
                for (i = 1; i <= n; i = i + 1) {
                    reslts[4 - 1] = max(reslts[4 - 1], REAL(abs((r[i - 1] - rpow[(i + 1) - 1]) / rpow[(i + 1) - 1])));
                }
            }
        }
        //
    }
    i = (nsz * (nsz + 1)) / 2 - 2;
    ap[i - 1] = -one;
    Rppequ("L", nsz, ap, r, rcond, norm, info);
    if (info != max(nsz - 1, (INTEGER)1)) {
        reslts[4 - 1] = one;
    }
    reslts[4 - 1] = reslts[4 - 1] / eps;
    //
    //     Test Rpbequ
    //
    for (n = 0; n <= nsz; n = n + 1) {
        for (kl = 0; kl <= max(n - 1, (INTEGER)0); kl = kl + 1) {
            //
            //           Test upper triangular storage
            //
            for (j = 1; j <= nsz; j = j + 1) {
                for (i = 1; i <= nszb; i = i + 1) {
                    ab[(i - 1) + (j - 1) * ldab] = zero;
                }
            }
            for (j = 1; j <= n; j = j + 1) {
                ab[((kl + 1) - 1) + (j - 1) * ldab] = mpow[(2 * j + 1) - 1];
            }
            //
            Rpbequ("U", n, kl, ab, nszb, r, rcond, norm, info);
            //
            if (info != 0) {
                reslts[5 - 1] = one;
            } else {
                if (n != 0) {
                    reslts[5 - 1] = max(reslts[5 - 1], REAL(abs((rcond - rpow[n - 1]) / rpow[n - 1])));
                    reslts[5 - 1] = max(reslts[5 - 1], REAL(abs((norm - mpow[(2 * n + 1) - 1]) / mpow[(2 * n + 1) - 1])));
                    for (i = 1; i <= n; i = i + 1) {
                        reslts[5 - 1] = max(reslts[5 - 1], REAL(abs((r[i - 1] - rpow[(i + 1) - 1]) / rpow[(i + 1) - 1])));
                    }
                }
            }
            if (n != 0) {
                ab[((kl + 1) - 1) + (max(n - 1, (INTEGER)1) - 1) * ldab] = -one;
                Rpbequ("U", n, kl, ab, nszb, r, rcond, norm, info);
                if (info != max(n - 1, (INTEGER)1)) {
                    reslts[5 - 1] = one;
                }
            }
            //
            //           Test lower triangular storage
            //
            for (j = 1; j <= nsz; j = j + 1) {
                for (i = 1; i <= nszb; i = i + 1) {
                    ab[(i - 1) + (j - 1) * ldab] = zero;
                }
            }
            for (j = 1; j <= n; j = j + 1) {
                ab[(j - 1) * ldab] = mpow[(2 * j + 1) - 1];
            }
            //
            Rpbequ("L", n, kl, ab, nszb, r, rcond, norm, info);
            //
            if (info != 0) {
                reslts[5 - 1] = one;
            } else {
                if (n != 0) {
                    reslts[5 - 1] = max(reslts[5 - 1], REAL(abs((rcond - rpow[n - 1]) / rpow[n - 1])));
                    reslts[5 - 1] = max(reslts[5 - 1], REAL(abs((norm - mpow[(2 * n + 1) - 1]) / mpow[(2 * n + 1) - 1])));
                    for (i = 1; i <= n; i = i + 1) {
                        reslts[5 - 1] = max(reslts[5 - 1], REAL(abs((r[i - 1] - rpow[(i + 1) - 1]) / rpow[(i + 1) - 1])));
                    }
                }
            }
            if (n != 0) {
                ab[(max(n - 1, (INTEGER)1) - 1) * ldab] = -one;
                Rpbequ("L", n, kl, ab, nszb, r, rcond, norm, info);
                if (info != max(n - 1, (INTEGER)1)) {
                    reslts[5 - 1] = one;
                }
            }
        }
    }
    reslts[5 - 1] = reslts[5 - 1] / eps;
    bool ok = (reslts[1 - 1] <= thresh) && (reslts[2 - 1] <= thresh) && (reslts[3 - 1] <= thresh) && (reslts[4 - 1] <= thresh) && (reslts[5 - 1] <= thresh);
    if (ok) {
        write(nout, "(1x,'All tests for ',a3,' routines passed the threshold')"), path;
    } else {
        if (reslts[1 - 1] > thresh) {
            sprintnum_short(buf1, reslts[1 - 1]);
            sprintnum_short(buf2, thresh);
            write(nout, "(' Rgeequ failed test with value ',a,' exceeding',' threshold ',"
                        "a)"),
                buf1, buf2;
        }
        if (reslts[2 - 1] > thresh) {
            sprintnum_short(buf1, reslts[2 - 1]);
            sprintnum_short(buf2, thresh);
            write(nout, "(' Rgbequ failed test with value ',a,' exceeding',' threshold ',"
                        "a)"),
                buf1, buf2;
        }
        if (reslts[3 - 1] > thresh) {
            sprintnum_short(buf1, reslts[3 - 1]);
            sprintnum_short(buf2, thresh);
            write(nout, "(' Rpoequ failed test with value ',a,' exceeding',' threshold ',"
                        "a)"),
                buf1, buf2;
        }
        if (reslts[4 - 1] > thresh) {
            sprintnum_short(buf1, reslts[4 - 1]);
            sprintnum_short(buf2, thresh);
            write(nout, "(' Rppequ failed test with value ',a,' exceeding',' threshold ',"
                        "a)"),
                buf1, buf2;
        }
        if (reslts[5 - 1] > thresh) {
            sprintnum_short(buf1, reslts[5 - 1]);
            sprintnum_short(buf2, thresh);
            write(nout, "(' Rpbequ failed test with value ',a,' exceeding',' threshold ',"
                        "a)"),
                buf1, buf2;
        }
    }
    //
    //     End of Rchkeq
    //
}

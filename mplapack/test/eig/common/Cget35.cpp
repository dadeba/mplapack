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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>

using namespace std;
using std::regex;
using std::regex_replace;

void Cget35(REAL &rmax, INTEGER &lmax, INTEGER &ninfo, INTEGER &knt, INTEGER const nin) {
    common cmn;
    common_read read(cmn);
    common_write write(cmn);
    double dtmp;
    std::complex<double> _ctmp;
    char buf[1024];
    REAL eps = 0.0;
    REAL smlnum = 0.0;
    const REAL one = 1.0;
    REAL bignum = 0.0;
    REAL vm1[3];
    const REAL large = 1.0e6;
    REAL vm2[3];
    const REAL two = 2.0;
    const REAL zero = 0.0;
    INTEGER m = 0;
    INTEGER n = 0;
    INTEGER i = 0;
    const INTEGER ldt = 10;
    COMPLEX atmp[ldt * ldt];
    INTEGER ldatmp = ldt;
    INTEGER j = 0;
    COMPLEX btmp[ldt * ldt];
    COMPLEX ctmp[ldt * ldt];
    INTEGER ldbtmp = ldt;
    INTEGER ldctmp = ldt;
    INTEGER imla = 0;
    INTEGER imlad = 0;
    INTEGER imlb = 0;
    INTEGER imlc = 0;
    INTEGER itrana = 0;
    INTEGER itranb = 0;
    INTEGER isgn = 0;
    char trana;
    char tranb;
    REAL tnrm = 0.0;
    COMPLEX a[ldt * ldt];
    COMPLEX b[ldt * ldt];
    COMPLEX c[ldt * ldt];
    COMPLEX csav[ldt * ldt];
    INTEGER lda = ldt;
    INTEGER ldb = ldt;
    INTEGER ldc = ldt;
    INTEGER ldcsav = ldt;
    REAL scale = 0.0;
    INTEGER info = 0;
    REAL dum[1];
    REAL xnrm = 0.0;
    const COMPLEX cone = 1.0;
    COMPLEX rmul = 0.0;
    REAL res1 = 0.0;
    REAL res = 0.0;
    //
    //     Get machine parameters
    //
    eps = Rlamch("P");
    smlnum = Rlamch("S") / eps;
    bignum = one / smlnum;
    //    Rlabad(smlnum, bignum);
    //
    //     Set up test case parameters
    //
    vm1[1 - 1] = sqrt(smlnum);
    vm1[2 - 1] = one;
    vm1[3 - 1] = large;
    vm2[1 - 1] = one;
    vm2[2 - 1] = one + two * eps;
    vm2[3 - 1] = two;
    //
    knt = 0;
    ninfo = 0;
    lmax = 0;
    rmax = zero;
    string str;
    istringstream iss;
    double dtmp_r;
    double dtmp_i;
    //
    //     Begin test loop
    //
    while (getline(cin, str)) {
        stringstream ss(str);
        ss >> m;
        ss >> n;
        printf(" M     %d\n", (int)m);
        printf(" N     %d\n", (int)n);	
        if (n == 0) {
            return;
        }
        for (i = 1; i <= m; i = i + 1) {
            getline(cin, str);
            string ___r = regex_replace(str, regex(","), " ");
            string __r = regex_replace(___r, regex("\\)"), " ");
            string _r = regex_replace(__r, regex("\\("), " ");
            str = regex_replace(_r, regex("D"), "e");
            iss.clear();
            iss.str(str);
            for (j = 1; j <= m; j = j + 1) {
                iss >> dtmp_r;
                iss >> dtmp_i;
                atmp[(i - 1) + (j - 1) * ldatmp] = COMPLEX(dtmp_r, dtmp_i);
            }
        }
        printf("ATMP=");printmat(m, m, atmp, ldatmp);printf("\n");
        for (i = 1; i <= n; i = i + 1) {
            getline(cin, str);
            string ___r = regex_replace(str, regex(","), " ");
            string __r = regex_replace(___r, regex("\\)"), " ");
            string _r = regex_replace(__r, regex("\\("), " ");
            str = regex_replace(_r, regex("D"), "e");
            iss.clear();
            iss.str(str);
            for (j = 1; j <= n; j = j + 1) {
                iss >> dtmp_r;
                iss >> dtmp_i;
                btmp[(i - 1) + (j - 1) * ldbtmp] = COMPLEX(dtmp_r, dtmp_i);
            }
        }
        printf("BTMP="); printmat(n, n, btmp, ldbtmp); printf("\n");
        for (i = 1; i <= m; i = i + 1) {
            getline(cin, str);
            string ___r = regex_replace(str, regex(","), " ");
            string __r = regex_replace(___r, regex("\\)"), " ");
            string _r = regex_replace(__r, regex("\\("), " ");
            str = regex_replace(_r, regex("D"), "e");
            iss.clear();
            iss.str(str);
            for (j = 1; j <= n; j = j + 1) {
                iss >> dtmp_r;
                iss >> dtmp_i;
                ctmp[(i - 1) + (j - 1) * ldctmp] = COMPLEX(dtmp_r, dtmp_i);
            }
        }
        printf("CTMP="); printmat(m, n, ctmp, ldctmp); printf("\n");
	printf("\n"); 
        for (imla = 1; imla <= 3; imla = imla + 1) {
            for (imlad = 1; imlad <= 3; imlad = imlad + 1) {
                for (imlb = 1; imlb <= 3; imlb = imlb + 1) {
                    for (imlc = 1; imlc <= 3; imlc = imlc + 1) {
                        for (itrana = 1; itrana <= 2; itrana = itrana + 1) {
                            for (itranb = 1; itranb <= 2; itranb = itranb + 1) {
                                for (isgn = -1; isgn <= 1; isgn = isgn + 2) {
                                    if (itrana == 1) {
                                        trana = 'N';
                                    }
                                    if (itrana == 2) {
                                        trana = 'C';
                                    }
                                    if (itranb == 1) {
                                        tranb = 'N';
                                    }
                                    if (itranb == 2) {
                                        tranb = 'C';
                                    }
                                    tnrm = zero;
                                    for (i = 1; i <= m; i = i + 1) {
                                        for (j = 1; j <= m; j = j + 1) {
                                            a[(i - 1) + (j - 1) * lda] = atmp[(i - 1) + (j - 1) * ldatmp] * vm1[imla - 1];
                                            tnrm = max(tnrm, abs(a[(i - 1) + (j - 1) * lda]));
                                        }
                                        a[(i - 1) + (i - 1) * lda] = a[(i - 1) + (i - 1) * lda] * vm2[imlad - 1];
                                        tnrm = max(tnrm, abs(a[(i - 1) + (i - 1) * lda]));
                                    }
                                    for (i = 1; i <= n; i = i + 1) {
                                        for (j = 1; j <= n; j = j + 1) {
                                            b[(i - 1) + (j - 1) * ldb] = btmp[(i - 1) + (j - 1) * ldbtmp] * vm1[imlb - 1];
                                            tnrm = max(tnrm, abs(b[(i - 1) + (j - 1) * ldb]));
                                        }
                                    }
                                    if (tnrm == zero) {
                                        tnrm = one;
                                    }
                                    for (i = 1; i <= m; i = i + 1) {
                                        for (j = 1; j <= n; j = j + 1) {
                                            c[(i - 1) + (j - 1) * ldc] = ctmp[(i - 1) + (j - 1) * ldctmp] * vm1[imlc - 1];
                                            csav[(i - 1) + (j - 1) * ldcsav] = c[(i - 1) + (j - 1) * ldc];
                                        }
                                    }
                                    knt++;
                                    Ctrsyl(&trana, &tranb, isgn, m, n, a, ldt, b, ldt, c, ldt, scale, info);
                                    if (info != 0) {
                                        ninfo++;
                                    }
                                    xnrm = Clange("M", m, n, c, ldt, dum);
                                    rmul = cone;
                                    if (xnrm > one && tnrm > one) {
                                        if (xnrm > bignum / tnrm) {
                                            rmul = max(xnrm, tnrm);
                                            rmul = cone / rmul;
                                        }
                                    }
                                    Cgemm(&trana, "N", m, n, m, rmul, a, ldt, c, ldt, -scale * rmul, csav, ldt);
                                    Cgemm("N", &tranb, m, n, n, castREAL(isgn) * rmul, c, ldt, b, ldt, cone, csav, ldt);
                                    res1 = Clange("M", m, n, csav, ldt, dum);
                                    res = res1 / max({smlnum, REAL(smlnum * xnrm), REAL(((abs(rmul) * tnrm) * eps) * xnrm)});
                                    if (res > rmax) {
                                        lmax = knt;
                                        rmax = res;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //
    //     End of Cget35
    //
}

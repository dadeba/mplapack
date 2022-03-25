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

void Rchkbl(INTEGER const nin, INTEGER const nout) {
    common cmn;
    common_read read(cmn);
    common_write write(cmn);
    double dtmp;
    char buf[1024];
    INTEGER lmax[3];
    INTEGER ninfo = 0;
    INTEGER knt = 0;
    const REAL zero = 0.0;
    REAL rmax = 0.0;
    REAL vmax = 0.0;
    REAL sfmin = 0.0;
    REAL meps = 0.0;
    INTEGER n = 0;
    INTEGER i = 0;
    const INTEGER lda = 20;
    REAL a[lda * lda];
    INTEGER j = 0;
    INTEGER iloin = 0;
    INTEGER ihiin = 0;
    REAL ain[lda * lda];
    INTEGER ldain = lda;
    REAL scalin[lda];
    REAL dummy[1];
    REAL anorm = 0.0;
    INTEGER ilo = 0;
    INTEGER ihi = 0;
    REAL scale[lda];
    INTEGER info = 0;
    REAL temp = 0.0;
    //
    lmax[1 - 1] = 0;
    lmax[2 - 1] = 0;
    lmax[3 - 1] = 0;
    ninfo = 0;
    knt = 0;
    rmax = zero;
    vmax = zero;
    sfmin = Rlamch("S");
    string str;
    char line[1024];
    istringstream iss;
    //
    while (getline(cin, str)) {
        stringstream ss(str);
        ss >> n;
        if (n == 0)
            break;
        // printf("n is %d\n", (int)n);
        for (i = 1; i <= n; i = i + 1) {
            getline(cin, str);
            string _r = regex_replace(str, regex("D\\+"), "e+");
            str = regex_replace(_r, regex("D\\-"), "e-");
            iss.clear();
            iss.str(str);
            for (j = 1; j <= n; j = j + 1) {
                iss >> dtmp;
                a[(i - 1) + (j - 1) * lda] = dtmp;
            }
        }
        //  printf("a=");printmat(n,n,a,lda);printf("\n");
        getline(cin, str);
        getline(cin, str);
        istringstream iss(str);
        iss >> iloin;
        iss >> ihiin;
        // printf("iloin: %d ihiin: %d\n", (int)iloin, (int)ihiin);
        for (i = 1; i <= n; i = i + 1) {
            getline(cin, str);
            string _r = regex_replace(str, regex("D\\+"), "e+");
            str = regex_replace(_r, regex("D\\-"), "e-");
            iss.clear();
            iss.str(str);
            for (j = 1; j <= n; j = j + 1) {
                iss >> dtmp;
                ain[(i - 1) + (j - 1) * ldain] = dtmp;
            }
        }
        getline(cin, str);
        getline(cin, str);

        string _r = regex_replace(str, regex("D\\+"), "e+");
        str = regex_replace(_r, regex("D\\-"), "e-");
        iss.clear();
        iss.str(str);
        for (i = 1; i <= n; i = i + 1) {
            iss >> dtmp;
            scalin[i - 1] = dtmp;
        }

        getline(cin, str);
        //
        anorm = Rlange("M", n, n, a, lda, dummy);
        knt++;

        // printf("ain=");printmat(n,n,ain,ldain);printf("\n");
        // printf("aorg=");printmat(n,n,a,lda);printf("\n");
        Rgebal("B", n, a, lda, ilo, ihi, scale, info);
        // printf("aout=");printmat(n,n,a,lda);printf("\n");
        // printf("scale=");printvec(scale,n);printf("\n");
        // printf("scalein=");printvec(scalin,n);printf("\n");
        // printf("\n");
        //
        if (info != 0) {
            ninfo++;
            lmax[1 - 1] = knt;
        }
        //
        if (ilo != iloin || ihi != ihiin) {
            ninfo++;
            lmax[2 - 1] = knt;
        }
        //
        for (i = 1; i <= n; i = i + 1) {
            for (j = 1; j <= n; j = j + 1) {
                temp = max(a[(i - 1) + (j - 1) * lda], ain[(i - 1) + (j - 1) * ldain]);
                temp = max(temp, sfmin);
                vmax = max(vmax, REAL(abs(a[(i - 1) + (j - 1) * lda] - ain[(i - 1) + (j - 1) * ldain]) / temp));
            }
        }
        //
        for (i = 1; i <= n; i = i + 1) {
            temp = max(scale[i - 1], scalin[i - 1]);
            temp = max(temp, sfmin);
            vmax = max(vmax, REAL(abs(scale[i - 1] - scalin[i - 1]) / temp));
        }
        //
        if (vmax > rmax) {
            lmax[3 - 1] = knt;
            rmax = vmax;
        }
        //
    }
    //
    write(nout, "(1x,'.. test output of Rgebal .. ')");
    //
    sprintnum_short(buf, rmax);
    write(nout, "(1x,'value of largest test error            = ',a)"), buf;
    write(nout, "(1x,'example number where info is not zero  = ',i4)"), lmax[1 - 1];
    write(nout, "(1x,'example number where ILO or IHI wrong  = ',i4)"), lmax[2 - 1];
    write(nout, "(1x,'example number having largest error    = ',i4)"), lmax[3 - 1];
    write(nout, "(1x,'number of examples where info is not 0 = ',i4)"), ninfo;
    write(nout, "(1x,'total number of examples tested        = ',i4)"), knt;
    //
    //     End of Rchkbl
    //
    return;
}

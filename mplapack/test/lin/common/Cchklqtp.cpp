/*
 * Copyright (c) 2021-2022
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

void Cchklqtp(REAL const thresh, bool const tsterr, INTEGER const nm, INTEGER *mval, INTEGER const nn, INTEGER *nval, INTEGER const nnb, INTEGER *nbval, INTEGER const nout) {
    common cmn;
    common_write write(cmn);
    //
    //     Initialize constants
    //
    char path[3];
    path[0] = 'C';
    path[1] = 'X';
    path[2] = 'Q';
    char buf[1024];
    INTEGER nrun = 0;
    INTEGER nfail = 0;
    INTEGER nerrs = 0;
    //
    //     Test the error exits
    //
    if (tsterr) {
        Cerrlqtp(path, nout);
    }
    //
    //     Do for each value of M
    //
    INTEGER i = 0;
    INTEGER m = 0;
    INTEGER j = 0;
    INTEGER n = 0;
    INTEGER minmn = 0;
    INTEGER l = 0;
    INTEGER k = 0;
    INTEGER nb = 0;
    const INTEGER ntests = 6;
    REAL result[ntests];
    INTEGER t = 0;
    for (i = 1; i <= nm; i = i + 1) {
        m = mval[i - 1];
        //
        //        Do for each value of N
        //
        for (j = 1; j <= nn; j = j + 1) {
            n = nval[j - 1];
            //
            //           Do for each value of L
            //
            minmn = min(m, n);
            for (l = 0; l <= minmn; l = l + max(minmn, (INTEGER)1)) {
                //
                //              Do for each possible value of NB
                //
                for (k = 1; k <= nnb; k = k + 1) {
                    nb = nbval[k - 1];
                    //
                    //                 Test Rtplqt and Rtpmlqt
                    //
                    if ((nb <= m) && (nb > 0)) {
                        Clqt05(m, n, l, nb, result);
                        //
                        //                    Print information about the tests that did not
                        //                    pass the threshold.
                        //
                        for (t = 1; t <= ntests; t = t + 1) {
                            if (result[t - 1] >= thresh) {
                                if (nfail == 0 && nerrs == 0) {
                                    Alahd(nout, path);
                                }
                                sprintnum_short(buf, result[t - 1]);
                                write(nout, "(' M=',i5,', N=',i5,', NB=',i4,' L=',i4,' test(',i2,')=',"
                                            "a)"),
                                    m, n, nb, l, t, buf;
                                nfail++;
                            }
                        }
                        nrun += ntests;
                    }
                }
            }
        }
    }
    //
    //     Print a summary of the results.
    //
    Alasum(path, nout, nfail, nrun, nerrs);
    //
    //     End of Cchklqtp
    //
}

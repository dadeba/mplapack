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

void Cget10(INTEGER const m, INTEGER const n, COMPLEX *a, INTEGER const lda, COMPLEX *b, INTEGER const ldb, COMPLEX *work, REAL *rwork, REAL &result) {
    //
    //     Quick return if possible
    //
    const REAL zero = 0.0;
    if (m <= 0 || n <= 0) {
        result = zero;
        return;
    }
    //
    REAL unfl = Rlamch("Safe minimum");
    REAL eps = Rlamch("Precision");
    //
    REAL wnorm = zero;
    INTEGER j = 0;
    const REAL one = 1.0;
    for (j = 1; j <= n; j = j + 1) {
        Ccopy(m, &a[(j - 1) * lda], 1, work, 1);
        Caxpy(m, COMPLEX(-one), &b[(j - 1) * ldb], 1, work, 1);
        wnorm = max(wnorm, RCasum(n, work, 1));
    }
    //
    REAL anorm = max(Clange("1", m, n, a, lda, rwork), unfl);
    //
    if (anorm > wnorm) {
        result = (wnorm / anorm) / (m * eps);
    } else {
        if (anorm < one) {
            result = (min(wnorm, REAL(m * anorm)) / anorm) / (castREAL(m) * eps);
        } else {
            result = min(REAL(wnorm / anorm), castREAL(m)) / (castREAL(m) * eps);
        }
    }
    //
    //     End of Cget10
    //
}

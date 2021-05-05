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

void Cglmts(INTEGER const n, INTEGER const m, INTEGER const p, COMPLEX *a, COMPLEX *af, INTEGER const lda, COMPLEX *b, COMPLEX *bf, INTEGER const ldb, COMPLEX *d, COMPLEX *df, COMPLEX *x, COMPLEX *u, COMPLEX *work, INTEGER const lwork, REAL *rwork, REAL &result) {
    a([lda * star]);
    af([lda * star]);
    b([ldb * star]);
    bf([ldb * star]);
    work([lwork]);
    //
    //  -- LAPACK test routine --
    //  -- LAPACK is a software package provided by Univ. of Tennessee,    --
    //  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //
    //     .. Scalar Arguments ..
    //     ..
    //     .. Array Arguments ..
    //
    //  ====================================================================
    //
    //     ..
    //     .. Parameters ..
    //     ..
    //     .. Local Scalars ..
    //     ..
    //     .. External Functions ..
    //     ..
    //     .. External Subroutines ..
    //
    //     ..
    //     .. Intrinsic Functions ..
    //     ..
    //     .. Executable Statements ..
    //
    REAL eps = Rlamch("Epsilon");
    REAL unfl = Rlamch("Safe minimum");
    REAL anorm = max({Clange("1", n, m, a, lda, rwork), unfl});
    REAL bnorm = max({Clange("1", n, p, b, ldb, rwork), unfl});
    //
    //     Copy the matrices A and B to the arrays AF and BF,
    //     and the vector D the array DF.
    //
    Clacpy("Full", n, m, a, lda, af, lda);
    Clacpy("Full", n, p, b, ldb, bf, ldb);
    Ccopy(n, d, 1, df, 1);
    //
    //     Solve GLM problem
    //
    INTEGER info = 0;
    Cggglm(n, m, p, af, lda, bf, ldb, df, x, u, work, lwork, info);
    //
    //     Test the residual for the solution of LSE
    //
    //                       norm( d - A*x - B*u )
    //       RESULT = -----------------------------------------
    //                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
    //
    Ccopy(n, d, 1, df, 1);
    const COMPLEX cone = 1.0;
    Cgemv("No transpose", n, m, -cone, a, lda, x, 1, cone, df, 1);
    //
    Cgemv("No transpose", n, p, -cone, b, ldb, u, 1, cone, df, 1);
    //
    REAL dnorm = RCasum(n, df, 1);
    REAL xnorm = RCasum(m, x, 1) + RCasum(p, u, 1);
    REAL ynorm = anorm + bnorm;
    //
    const REAL zero = 0.0;
    if (xnorm <= zero) {
        result = zero;
    } else {
        result = ((dnorm / ynorm) / xnorm) / eps;
    }
    //
    //     End of Cglmts
    //
}

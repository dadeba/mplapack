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

void Rlattr(INTEGER const imat, const char *uplo, const char *trans, char *diag, INTEGER *iseed, INTEGER const n, REAL *a, INTEGER const lda, REAL *b, REAL *work, INTEGER &info) {
    //
    char path[4] = {};
    path[0] = 'R';
    path[1] = 'T';
    path[2] = 'R';
    REAL unfl = Rlamch("Safe minimum");
    REAL ulp = Rlamch("Epsilon") * Rlamch("Base");
    REAL smlnum = unfl;
    const REAL one = 1.0;
    REAL bignum = (one - ulp) / smlnum;
    if ((imat >= 7 && imat <= 10) || imat == 18) {
        *diag = 'U';
    } else {
        *diag = 'N';
    }
    info = 0;
    //
    //     Quick return if N.LE.0.
    //
    if (n <= 0) {
        return;
    }
    //
    //     Call Rlatb4 to set parameters for SLATMS.
    //
    bool upper = Mlsame(uplo, "U");
    char type;
    INTEGER kl = 0;
    INTEGER ku = 0;
    REAL anorm = 0.0;
    INTEGER mode = 0;
    REAL cndnum = 0.0;
    char dist;
    if (upper) {
        Rlatb4(path, imat, n, n, &type, kl, ku, anorm, mode, cndnum, &dist);
    } else {
        Rlatb4(path, -imat, n, n, &type, kl, ku, anorm, mode, cndnum, &dist);
    }
    //
    //     IMAT <= 6:  Non-unit triangular matrix
    //
    INTEGER j = 0;
    INTEGER i = 0;
    const REAL zero = 0.0;
    REAL star1 = 0.0;
    REAL sfac = 0.0;
    REAL plus1 = 0.0;
    REAL plus2 = 0.0;
    REAL rexp = 0.0;
    REAL x = 0.0;
    REAL y = 0.0;
    REAL z = 0.0;
    REAL ra = 0.0;
    REAL rb = 0.0;
    REAL c = 0.0;
    REAL s = 0.0;
    const REAL two = 2.0e+0;
    INTEGER iy = 0;
    REAL bnorm = 0.0;
    REAL bscal = 0.0;
    REAL tscal = 0.0;
    INTEGER jcount = 0;
    REAL texp = 0.0;
    REAL tleft = 0.0;
    if (imat <= 6) {
        Rlatms(n, n, &dist, iseed, &type, b, mode, cndnum, anorm, kl, ku, "No packing", a, lda, work, info);
        //
        //     IMAT > 6:  Unit triangular matrix
        //     The diagonal is deliberately set to something other than 1.
        //
        //     IMAT = 7:  Matrix is the identity
        //
    } else if (imat == 7) {
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= j - 1; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
                a[(j - 1) + (j - 1) * lda] = j;
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                a[(j - 1) + (j - 1) * lda] = j;
                for (i = j + 1; i <= n; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
            }
        }
        //
        //     IMAT > 7:  Non-trivial unit triangular matrix
        //
        //     Generate a unit triangular matrix T with condition CNDNUM by
        //     forming a triangular matrix with known singular values and
        //     filling in the zero entries with Givens rotations.
        //
    } else if (imat <= 10) {
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= j - 1; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
                a[(j - 1) + (j - 1) * lda] = j;
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                a[(j - 1) + (j - 1) * lda] = j;
                for (i = j + 1; i <= n; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
            }
        }
        //
        //        Since the trace of a unit triangular matrix is 1, the product
        //        of its singular values must be 1.  Let s = sqrt(CNDNUM),
        //        x = sqrt(s) - 1/sqrt(s), y = sqrt(2/(n-2))*x, and z = x**2.
        //        The following triangular matrix has singular values s, 1, 1,
        //        ..., 1, 1/s:
        //
        //        1  y  y  y  ...  y  y  z
        //           1  0  0  ...  0  0  y
        //              1  0  ...  0  0  y
        //                 .  ...  .  .  .
        //                     .   .  .  .
        //                         1  0  y
        //                            1  y
        //                               1
        //
        //        To fill in the zeros, we first multiply by a matrix with small
        //        condition number of the form
        //
        //        1  0  0  0  0  ...
        //           1  +  *  0  0  ...
        //              1  +  0  0  0
        //                 1  +  *  0  0
        //                    1  +  0  0
        //                       ...
        //                          1  +  0
        //                             1  0
        //                                1
        //
        //        Each element marked with a '*' is formed by taking the product
        //        of the adjacent elements marked with '+'.  The '*'s can be
        //        chosen freely, and the '+'s are chosen so that the inverse of
        //        T will have elements of the same magnitude as T.  If the *'s in
        //        both T and inv(T) have small magnitude, T is well conditioned.
        //        The two offdiagonals of T are stored in WORK.
        //
        //        The product of these two matrices has the form
        //
        //        1  y  y  y  y  y  .  y  y  z
        //           1  +  *  0  0  .  0  0  y
        //              1  +  0  0  .  0  0  y
        //                 1  +  *  .  .  .  .
        //                    1  +  .  .  .  .
        //                       .  .  .  .  .
        //                          .  .  .  .
        //                             1  +  y
        //                                1  y
        //                                   1
        //
        //        Now we multiply by Givens rotations, using the fact that
        //
        //              [  c   s ] [  1   w ] [ -c  -s ] =  [  1  -w ]
        //              [ -s   c ] [  0   1 ] [  s  -c ]    [  0   1 ]
        //        and
        //              [ -c  -s ] [  1   0 ] [  c   s ] =  [  1   0 ]
        //              [  s  -c ] [  w   1 ] [ -s   c ]    [ -w   1 ]
        //
        //        where c = w / sqrt(w**2+4) and s = 2 / sqrt(w**2+4).
        //
        star1 = 0.25e0;
        sfac = 0.5e0;
        plus1 = sfac;
        for (j = 1; j <= n; j = j + 2) {
            plus2 = star1 / plus1;
            work[j - 1] = plus1;
            work[(n + j) - 1] = star1;
            if (j + 1 <= n) {
                work[(j + 1) - 1] = plus2;
                work[(n + j + 1) - 1] = zero;
                plus1 = star1 / plus2;
                rexp = Rlarnd(2, iseed);
                star1 = star1 * (pow(sfac, rexp));
                if (rexp < zero) {
                    star1 = -pow(sfac, (one - rexp));
                } else {
                    star1 = pow(sfac, (one + rexp));
                }
            }
        }
        //
        REAL two = 2.0;
        x = sqrt(cndnum) - 1 / sqrt(cndnum);
        if (n > 2) {
            y = sqrt(two / (castREAL(n) - two)) * x;
        } else {
            y = zero;
        }
        z = x * x;
        //
        if (upper) {
            if (n > 3) {
                Rcopy(n - 3, work, 1, &a[(2 - 1) + (3 - 1) * lda], lda + 1);
                if (n > 4) {
                    Rcopy(n - 4, &work[(n + 1) - 1], 1, &a[(2 - 1) + (4 - 1) * lda], lda + 1);
                }
            }
            for (j = 2; j <= n - 1; j = j + 1) {
                a[(j - 1) * lda] = y;
                a[(j - 1) + (n - 1) * lda] = y;
            }
            a[(n - 1) * lda] = z;
        } else {
            if (n > 3) {
                Rcopy(n - 3, work, 1, &a[(3 - 1) + (2 - 1) * lda], lda + 1);
                if (n > 4) {
                    Rcopy(n - 4, &work[(n + 1) - 1], 1, &a[(4 - 1) + (2 - 1) * lda], lda + 1);
                }
            }
            for (j = 2; j <= n - 1; j = j + 1) {
                a[(j - 1)] = y;
                a[(n - 1) + (j - 1) * lda] = y;
            }
            a[(n - 1)] = z;
        }
        //
        //        Fill in the zeros using Givens rotations.
        //
        if (upper) {
            for (j = 1; j <= n - 1; j = j + 1) {
                ra = a[(j - 1) + ((j + 1) - 1) * lda];
                rb = 2.0;
                Rrotg(ra, rb, c, s);
                //
                //              Multiply by [ c  s; -s  c] on the left.
                //
                if (n > j + 1) {
                    Rrot(n - j - 1, &a[(j - 1) + ((j + 2) - 1) * lda], lda, &a[((j + 1) - 1) + ((j + 2) - 1) * lda], lda, c, s);
                }
                //
                //              Multiply by [-c -s;  s -c] on the right.
                //
                if (j > 1) {
                    Rrot(j - 1, &a[((j + 1) - 1) * lda], 1, &a[(j - 1) * lda], 1, -c, -s);
                }
                //
                //              Negate A(J,J+1).
                //
                a[(j - 1) + ((j + 1) - 1) * lda] = -a[(j - 1) + ((j + 1) - 1) * lda];
            }
        } else {
            for (j = 1; j <= n - 1; j = j + 1) {
                ra = a[((j + 1) - 1) + (j - 1) * lda];
                rb = 2.0;
                Rrotg(ra, rb, c, s);
                //
                //              Multiply by [ c -s;  s  c] on the right.
                //
                if (n > j + 1) {
                    Rrot(n - j - 1, &a[((j + 2) - 1) + ((j + 1) - 1) * lda], 1, &a[((j + 2) - 1) + (j - 1) * lda], 1, c, -s);
                }
                //
                //              Multiply by [-c  s; -s -c] on the left.
                //
                if (j > 1) {
                    Rrot(j - 1, &a[(j - 1)], lda, &a[((j + 1) - 1)], lda, -c, s);
                }
                //
                //              Negate A(J+1,J).
                //
                a[((j + 1) - 1) + (j - 1) * lda] = -a[((j + 1) - 1) + (j - 1) * lda];
            }
        }
        //
        //     IMAT > 10:  Pathological test cases.  These triangular matrices
        //     are badly scaled or badly conditioned, so when used in solving a
        //     triangular system they may cause overflow in the solution vector.
        //
    } else if (imat == 11) {
        //
        //        Type 11:  Generate a triangular matrix with elements between
        //        -1 and 1. Give the diagonal norm 2 to make it well-conditioned.
        //        Make the right hand side large so that it requires scaling.
        //
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, j, &a[(j - 1) * lda]);
                a[(j - 1) + (j - 1) * lda] = sign(two, a[(j - 1) + (j - 1) * lda]);
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, n - j + 1, &a[(j - 1) + (j - 1) * lda]);
                a[(j - 1) + (j - 1) * lda] = sign(two, a[(j - 1) + (j - 1) * lda]);
            }
        }
        //
        //        Set the right hand side so that the largest value is BIGNUM.
        //
        Rlarnv(2, iseed, n, b);
        iy = iRamax(n, b, 1);
        bnorm = abs(b[iy - 1]);
        bscal = bignum / max(one, bnorm);
        Rscal(n, bscal, b, 1);
        //
    } else if (imat == 12) {
        //
        //        Type 12:  Make the first diagonal element in the solve small to
        //        cause immediate overflow when dividing by T(j,j).
        //        In type 12, the offdiagonal elements are small (CNORM(j) < 1).
        //
        Rlarnv(2, iseed, n, b);
        tscal = one / max(one, castREAL(n - 1));
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, j, &a[(j - 1) * lda]);
                Rscal(j - 1, tscal, &a[(j - 1) * lda], 1);
                a[(j - 1) + (j - 1) * lda] = sign(one, a[(j - 1) + (j - 1) * lda]);
            }
            a[(n - 1) + (n - 1) * lda] = smlnum * a[(n - 1) + (n - 1) * lda];
        } else {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, n - j + 1, &a[(j - 1) + (j - 1) * lda]);
                if (n > j) {
                    Rscal(n - j, tscal, &a[((j + 1) - 1) + (j - 1) * lda], 1);
                }
                a[(j - 1) + (j - 1) * lda] = sign(one, a[(j - 1) + (j - 1) * lda]);
            }
            a[(1 - 1) + (1 - 1) * lda] = smlnum * a[(1 - 1) + (1 - 1) * lda];
        }
        //
    } else if (imat == 13) {
        //
        //        Type 13:  Make the first diagonal element in the solve small to
        //        cause immediate overflow when dividing by T(j,j).
        //        In type 13, the offdiagonal elements are O(1) (CNORM(j) > 1).
        //
        Rlarnv(2, iseed, n, b);
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, j, &a[(j - 1) * lda]);
                a[(j - 1) + (j - 1) * lda] = sign(one, a[(j - 1) + (j - 1) * lda]);
            }
            a[(n - 1) + (n - 1) * lda] = smlnum * a[(n - 1) + (n - 1) * lda];
        } else {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, n - j + 1, &a[(j - 1) + (j - 1) * lda]);
                a[(j - 1) + (j - 1) * lda] = sign(one, a[(j - 1) + (j - 1) * lda]);
            }
            a[(1 - 1) + (1 - 1) * lda] = smlnum * a[(1 - 1) + (1 - 1) * lda];
        }
        //
    } else if (imat == 14) {
        //
        //        Type 14:  T is diagonal with small numbers on the diagonal to
        //        make the growth factor underflow, but a small right hand side
        //        chosen so that the solution does not overflow.
        //
        if (upper) {
            jcount = 1;
            for (j = n; j >= 1; j = j - 1) {
                for (i = 1; i <= j - 1; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
                if (jcount <= 2) {
                    a[(j - 1) + (j - 1) * lda] = smlnum;
                } else {
                    a[(j - 1) + (j - 1) * lda] = one;
                }
                jcount++;
                if (jcount > 4) {
                    jcount = 1;
                }
            }
        } else {
            jcount = 1;
            for (j = 1; j <= n; j = j + 1) {
                for (i = j + 1; i <= n; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = zero;
                }
                if (jcount <= 2) {
                    a[(j - 1) + (j - 1) * lda] = smlnum;
                } else {
                    a[(j - 1) + (j - 1) * lda] = one;
                }
                jcount++;
                if (jcount > 4) {
                    jcount = 1;
                }
            }
        }
        //
        //        Set the right hand side alternately zero and small.
        //
        if (upper) {
            b[1 - 1] = zero;
            for (i = n; i >= 2; i = i - 2) {
                b[i - 1] = zero;
                b[(i - 1) - 1] = smlnum;
            }
        } else {
            b[n - 1] = zero;
            for (i = 1; i <= n - 1; i = i + 2) {
                b[i - 1] = zero;
                b[(i + 1) - 1] = smlnum;
            }
        }
        //
    } else if (imat == 15) {
        //
        //        Type 15:  Make the diagonal elements small to cause gradual
        //        overflow when dividing by T(j,j).  To control the amount of
        //        scaling needed, the matrix is bidiagonal.
        //
        texp = one / max(one, castREAL(n - 1));
        tscal = pow(smlnum, texp);
        Rlarnv(2, iseed, n, b);
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= j - 2; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = 0.0;
                }
                if (j > 1) {
                    a[((j - 1) - 1) + (j - 1) * lda] = -one;
                }
                a[(j - 1) + (j - 1) * lda] = tscal;
            }
            b[n - 1] = one;
        } else {
            for (j = 1; j <= n; j = j + 1) {
                for (i = j + 2; i <= n; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = 0.0;
                }
                if (j < n) {
                    a[((j + 1) - 1) + (j - 1) * lda] = -one;
                }
                a[(j - 1) + (j - 1) * lda] = tscal;
            }
            b[1 - 1] = one;
        }
        //
    } else if (imat == 16) {
        //
        //        Type 16:  One zero diagonal element.
        //
        iy = n / 2 + 1;
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, j, &a[(j - 1) * lda]);
                if (j != iy) {
                    a[(j - 1) + (j - 1) * lda] = sign(two, a[(j - 1) + (j - 1) * lda]);
                } else {
                    a[(j - 1) + (j - 1) * lda] = zero;
                }
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, n - j + 1, &a[(j - 1) + (j - 1) * lda]);
                if (j != iy) {
                    a[(j - 1) + (j - 1) * lda] = sign(two, a[(j - 1) + (j - 1) * lda]);
                } else {
                    a[(j - 1) + (j - 1) * lda] = zero;
                }
            }
        }
        Rlarnv(2, iseed, n, b);
        Rscal(n, two, b, 1);
        //
    } else if (imat == 17) {
        //
        //        Type 17:  Make the offdiagonal elements large to cause overflow
        //        when adding a column of T.  In the non-transposed case, the
        //        matrix is constructed to cause overflow when adding a column in
        //        every other step.
        //
        tscal = unfl / ulp;
        tscal = (one - ulp) / tscal;
        for (j = 1; j <= n; j = j + 1) {
            for (i = 1; i <= n; i = i + 1) {
                a[(i - 1) + (j - 1) * lda] = 0.0;
            }
        }
        texp = one;
        if (upper) {
            for (j = n; j >= 2; j = j - 2) {
                a[(j - 1) * lda] = -tscal / castREAL(n + 1);
                a[(j - 1) + (j - 1) * lda] = one;
                b[j - 1] = texp * (one - ulp);
                a[((j - 1) - 1) * lda] = -(tscal / castREAL(n + 1)) / castREAL(n + 2);
                a[((j - 1) - 1) + ((j - 1) - 1) * lda] = one;
                b[(j - 1) - 1] = texp * castREAL(n * n + n - 1);
                texp = texp * 2.0;
            }
            b[1 - 1] = (castREAL(n + 1) / castREAL(n + 2)) * tscal;
        } else {
            for (j = 1; j <= n - 1; j = j + 2) {
                a[(n - 1) + (j - 1) * lda] = -tscal / castREAL(n + 1);
                a[(j - 1) + (j - 1) * lda] = one;
                b[j - 1] = texp * (one - ulp);
                a[(n - 1) + ((j + 1) - 1) * lda] = -(tscal / castREAL(n + 1)) / castREAL(n + 2);
                a[((j + 1) - 1) + ((j + 1) - 1) * lda] = one;
                b[(j + 1) - 1] = texp * castREAL(n * n + n - 1);
                texp = texp * 2.0;
            }
            b[n - 1] = (castREAL(n + 1) / castREAL(n + 2)) * tscal;
        }
        //
    } else if (imat == 18) {
        //
        //        Type 18:  Generate a unit triangular matrix with elements
        //        between -1 and 1, and make the right hand side large so that it
        //        requires scaling.
        //
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, j - 1, &a[(j - 1) * lda]);
                a[(j - 1) + (j - 1) * lda] = zero;
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                if (j < n) {
                    Rlarnv(2, iseed, n - j, &a[((j + 1) - 1) + (j - 1) * lda]);
                }
                a[(j - 1) + (j - 1) * lda] = zero;
            }
        }
        //
        //        Set the right hand side so that the largest value is BIGNUM.
        //
        Rlarnv(2, iseed, n, b);
        iy = iRamax(n, b, 1);
        bnorm = abs(b[iy - 1]);
        bscal = bignum / max(one, bnorm);
        Rscal(n, bscal, b, 1);
        //
    } else if (imat == 19) {
        //
        //        Type 19:  Generate a triangular matrix with elements between
        //        BIGNUM/(n-1) and BIGNUM so that at least one of the column
        //        norms will exceed BIGNUM.
        //        1/3/91:  Rlatrs no longer can handle this case
        //
        tleft = bignum / max(one, castREAL(n - 1));
        tscal = bignum * (castREAL(n - 1) / max(one, castREAL(n)));
        if (upper) {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, j, &a[(j - 1) * lda]);
                for (i = 1; i <= j; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = sign(tleft, a[(i - 1) + (j - 1) * lda]) + tscal * a[(i - 1) + (j - 1) * lda];
                }
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                Rlarnv(2, iseed, n - j + 1, &a[(j - 1) + (j - 1) * lda]);
                for (i = j; i <= n; i = i + 1) {
                    a[(i - 1) + (j - 1) * lda] = sign(tleft, a[(i - 1) + (j - 1) * lda]) + tscal * a[(i - 1) + (j - 1) * lda];
                }
            }
        }
        Rlarnv(2, iseed, n, b);
        Rscal(n, two, b, 1);
    }
    //
    //     Flip the matrix if the transpose will be used.
    //
    if (!Mlsame(trans, "N")) {
        if (upper) {
            for (j = 1; j <= n / 2; j = j + 1) {
                Rswap(n - 2 * j + 1, &a[(j - 1) + (j - 1) * lda], lda, &a[((j + 1) - 1) + ((n - j + 1) - 1) * lda], -1);
            }
        } else {
            for (j = 1; j <= n / 2; j = j + 1) {
                Rswap(n - 2 * j + 1, &a[(j - 1) + (j - 1) * lda], 1, &a[((n - j + 1) - 1) + ((j + 1) - 1) * lda], -lda);
            }
        }
    }
    //
    //     End of Rlattr
    //
}

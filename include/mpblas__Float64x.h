/*
 * Copyright (c) 2012-2021
 *	Nakata, Maho
 * 	All rights reserved.
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

#ifndef _MPBLAS__FLOAT64X_H_
#define _MPBLAS__FLOAT64X_H_

#include "mplapack_config.h"
#include "mplapack_utils__Float64x.h"

_Float64x RCabs1(std::complex<_Float64x> const z);
_Float64x RCasum(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx);
_Float64x RCnrm2(mplapackint const n, std::complex<_Float64x> *x, mplapackint const incx);
_Float64x Rasum(mplapackint const n, _Float64x *dx, mplapackint const incx);
_Float64x Rdot(mplapackint const n, _Float64x *dx, mplapackint const incx, _Float64x *dy, mplapackint const incy);
_Float64x Rnrm2(mplapackint const n, _Float64x *x, mplapackint const incx);
bool Mlsame__Float64x(const char *a, const char *b);
mplapackint iCamax(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx);
mplapackint iRamax(mplapackint const n, _Float64x *dx, mplapackint const incx);
std::complex<_Float64x> Cdotc(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx, std::complex<_Float64x> *zy, mplapackint const incy);
std::complex<_Float64x> Cdotu(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx, std::complex<_Float64x> *zy, mplapackint const incy);
void CRrot(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx, std::complex<_Float64x> *zy, mplapackint const incy, _Float64x const c, _Float64x const s);
void CRscal(mplapackint const n, _Float64x const da, std::complex<_Float64x> *zx, mplapackint const incx);
void Caxpy(mplapackint const n, std::complex<_Float64x> const za, std::complex<_Float64x> *zx, mplapackint const incx, std::complex<_Float64x> *zy, mplapackint const incy);
void Ccopy(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx, std::complex<_Float64x> *zy, mplapackint const incy);
void Cgbmv(const char *trans, mplapackint const m, mplapackint const n, mplapackint const kl, mplapackint const ku, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> const beta, std::complex<_Float64x> *y, mplapackint const incy);
void Cgemm(const char *transa, const char *transb, mplapackint const m, mplapackint const n, mplapackint const k, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb, std::complex<_Float64x> const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Cgemv(const char *trans, mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> const beta, std::complex<_Float64x> *y, mplapackint const incy);
void Cgerc(mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> *y, mplapackint const incy, std::complex<_Float64x> *a, mplapackint const lda);
void Cgeru(mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> *y, mplapackint const incy, std::complex<_Float64x> *a, mplapackint const lda);
void Chbmv(const char *uplo, mplapackint const n, mplapackint const k, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> const beta, std::complex<_Float64x> *y, mplapackint const incy);
void Chemm(const char *side, const char *uplo, mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb, std::complex<_Float64x> const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Chemv(const char *uplo, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> const beta, std::complex<_Float64x> *y, mplapackint const incy);
void Cher(const char *uplo, mplapackint const n, _Float64x const alpha, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> *a, mplapackint const lda);
void Cher2(const char *uplo, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> *y, mplapackint const incy, std::complex<_Float64x> *a, mplapackint const lda);
void Cher2k(const char *uplo, const char *trans, mplapackint const n, mplapackint const k, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb, _Float64x const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Cherk(const char *uplo, const char *trans, mplapackint const n, mplapackint const k, _Float64x const alpha, std::complex<_Float64x> *a, mplapackint const lda, _Float64x const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Chpmv(const char *uplo, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *ap, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> const beta, std::complex<_Float64x> *y, mplapackint const incy);
void Chpr(const char *uplo, mplapackint const n, _Float64x const alpha, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> *ap);
void Chpr2(const char *uplo, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *x, mplapackint const incx, std::complex<_Float64x> *y, mplapackint const incy, std::complex<_Float64x> *ap);
void Crotg(std::complex<_Float64x> &ca, std::complex<_Float64x> const cb, _Float64x &c, std::complex<_Float64x> &s);
void Cscal(mplapackint const n, std::complex<_Float64x> const za, std::complex<_Float64x> *zx, mplapackint const incx);
void Cswap(mplapackint const n, std::complex<_Float64x> *zx, mplapackint const incx, std::complex<_Float64x> *zy, mplapackint const incy);
void Csymm(const char *side, const char *uplo, mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb, std::complex<_Float64x> const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Csyr2k(const char *uplo, const char *trans, mplapackint const n, mplapackint const k, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb, std::complex<_Float64x> const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Csyrk(const char *uplo, const char *trans, mplapackint const n, mplapackint const k, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> const beta, std::complex<_Float64x> *c, mplapackint const ldc);
void Ctbmv(const char *uplo, const char *trans, const char *diag, mplapackint const n, mplapackint const k, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx);
void Ctbsv(const char *uplo, const char *trans, const char *diag, mplapackint const n, mplapackint const k, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx);
void Ctpmv(const char *uplo, const char *trans, const char *diag, mplapackint const n, std::complex<_Float64x> *ap, std::complex<_Float64x> *x, mplapackint const incx);
void Ctpsv(const char *uplo, const char *trans, const char *diag, mplapackint const n, std::complex<_Float64x> *ap, std::complex<_Float64x> *x, mplapackint const incx);
void Ctrmm(const char *side, const char *uplo, const char *transa, const char *diag, mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb);
void Ctrmv(const char *uplo, const char *trans, const char *diag, mplapackint const n, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx);
void Ctrsm(const char *side, const char *uplo, const char *transa, const char *diag, mplapackint const m, mplapackint const n, std::complex<_Float64x> const alpha, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *b, mplapackint const ldb);
void Ctrsv(const char *uplo, const char *trans, const char *diag, mplapackint const n, std::complex<_Float64x> *a, mplapackint const lda, std::complex<_Float64x> *x, mplapackint const incx);
void Mxerbla__Float64x(const char *srname, int info);
void Raxpy(mplapackint const n, _Float64x const da, _Float64x *dx, mplapackint const incx, _Float64x *dy, mplapackint const incy);
void Rcopy(mplapackint const n, _Float64x *dx, mplapackint const incx, _Float64x *dy, mplapackint const incy);
void Rgbmv(const char *trans, mplapackint const m, mplapackint const n, mplapackint const kl, mplapackint const ku, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx, _Float64x const beta, _Float64x *y, mplapackint const incy);
void Rgemm(const char *transa, const char *transb, mplapackint const m, mplapackint const n, mplapackint const k, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *b, mplapackint const ldb, _Float64x const beta, _Float64x *c, mplapackint const ldc);
void Rgemv(const char *trans, mplapackint const m, mplapackint const n, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx, _Float64x const beta, _Float64x *y, mplapackint const incy);
void Rger(mplapackint const m, mplapackint const n, _Float64x const alpha, _Float64x *x, mplapackint const incx, _Float64x *y, mplapackint const incy, _Float64x *a, mplapackint const lda);
void Rrot(mplapackint const n, _Float64x *dx, mplapackint const incx, _Float64x *dy, mplapackint const incy, _Float64x const c, _Float64x const s);
void Rrotg(_Float64x &da, _Float64x &db, _Float64x &c, _Float64x &s);
void Rrotm(mplapackint const n, _Float64x *dx, mplapackint const incx, _Float64x *dy, mplapackint const incy, _Float64x *dparam);
void Rrotmg(_Float64x &dd1, _Float64x &dd2, _Float64x &dx1, _Float64x const dy1, _Float64x *dparam);
void Rsbmv(const char *uplo, mplapackint const n, mplapackint const k, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx, _Float64x const beta, _Float64x *y, mplapackint const incy);
void Rscal(mplapackint const n, _Float64x const da, _Float64x *dx, mplapackint const incx);
void Rspmv(const char *uplo, mplapackint const n, _Float64x const alpha, _Float64x *ap, _Float64x *x, mplapackint const incx, _Float64x const beta, _Float64x *y, mplapackint const incy);
void Rspr(const char *uplo, mplapackint const n, _Float64x const alpha, _Float64x *x, mplapackint const incx, _Float64x *ap);
void Rspr2(const char *uplo, mplapackint const n, _Float64x const alpha, _Float64x *x, mplapackint const incx, _Float64x *y, mplapackint const incy, _Float64x *ap);
void Rswap(mplapackint const n, _Float64x *dx, mplapackint const incx, _Float64x *dy, mplapackint const incy);
void Rsymm(const char *side, const char *uplo, mplapackint const m, mplapackint const n, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *b, mplapackint const ldb, _Float64x const beta, _Float64x *c, mplapackint const ldc);
void Rsymv(const char *uplo, mplapackint const n, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx, _Float64x const beta, _Float64x *y, mplapackint const incy);
void Rsyr(const char *uplo, mplapackint const n, _Float64x const alpha, _Float64x *x, mplapackint const incx, _Float64x *a, mplapackint const lda);
void Rsyr2(const char *uplo, mplapackint const n, _Float64x const alpha, _Float64x *x, mplapackint const incx, _Float64x *y, mplapackint const incy, _Float64x *a, mplapackint const lda);
void Rsyr2k(const char *uplo, const char *trans, mplapackint const n, mplapackint const k, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *b, mplapackint const ldb, _Float64x const beta, _Float64x *c, mplapackint const ldc);
void Rsyrk(const char *uplo, const char *trans, mplapackint const n, mplapackint const k, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x const beta, _Float64x *c, mplapackint const ldc);
void Rtbmv(const char *uplo, const char *trans, const char *diag, mplapackint const n, mplapackint const k, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx);
void Rtbsv(const char *uplo, const char *trans, const char *diag, mplapackint const n, mplapackint const k, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx);
void Rtpmv(const char *uplo, const char *trans, const char *diag, mplapackint const n, _Float64x *ap, _Float64x *x, mplapackint const incx);
void Rtpsv(const char *uplo, const char *trans, const char *diag, mplapackint const n, _Float64x *ap, _Float64x *x, mplapackint const incx);
void Rtrmm(const char *side, const char *uplo, const char *transa, const char *diag, mplapackint const m, mplapackint const n, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *b, mplapackint const ldb);
void Rtrmv(const char *uplo, const char *trans, const char *diag, mplapackint const n, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx);
void Rtrsm(const char *side, const char *uplo, const char *transa, const char *diag, mplapackint const m, mplapackint const n, _Float64x const alpha, _Float64x *a, mplapackint const lda, _Float64x *b, mplapackint const ldb);
void Rtrsv(const char *uplo, const char *trans, const char *diag, mplapackint const n, _Float64x *a, mplapackint const lda, _Float64x *x, mplapackint const incx);
#endif

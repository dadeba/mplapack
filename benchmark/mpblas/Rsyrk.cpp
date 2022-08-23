/*
 * Copyright (c) 2008-2022
 *	Nakata, Maho
 * 	All rights reserved.
 *
 * $Id: Rgemm_dd.cpp,v 1.4 2010/08/07 05:50:09 nakatamaho Exp $
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

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <mpblas.h>
#include <mplapack.h>
#include <mplapack_benchmark.h>

// cf. https://netlib.org/lapack/lawnspdf/lawn41.pdf p.120
double flops_syrk(mplapackint k_i, mplapackint n_i) {
    double adds, muls, flops;
    double n, k;
    n = (double)n_i;
    k = (double)k_i;
    muls = k * n * (n + 1) * 0.5 +  n * n + n;
    adds = k * n * (n + 1) * 0.5;
    flops = muls + adds;
    return flops;
}

int main(int argc, char *argv[]) {
    REAL alpha, beta, dummy;
    REAL *dummywork;
    double elapsedtime, t1, t2;
    char uplo, trans, normtype;
    int N0, K0, STEPN = 3, STEPK = 3, LOOP = 3, TOTALSTEPS = 340;
    int lda, ldc;
    int i, n, k, ka, p, q;
    int check_flag = 1;

    ___MPLAPACK_INITIALIZE___

    const char mpblas_sym[] = SYMBOL_GCC_RSYRK;
    const char raxpy_sym[] = SYMBOL_GCC_RAXPY;
    void *handle;
    void (*mpblas_ref)(const char *, const char *, mplapackint, mplapackint, REAL, REAL *, mplapackint, REAL, REAL *, mplapackint);
    void (*raxpy_ref)(mplapackint, REAL, REAL *, mplapackint, REAL *, mplapackint);
    char *error;
    REAL diff;
    double diffr;

    // initialization
    N0 = K0 = 1;
    STEPN = STEPK = 1;
    uplo = 'u';
    trans = 'n';
    normtype = 'm';
    if (argc != 1) {
        for (i = 1; i < argc; i++) {
            if (strcmp("-N", argv[i]) == 0) {
                N0 = atoi(argv[++i]);
            } else if (strcmp("-K", argv[i]) == 0) {
                K0 = atoi(argv[++i]);
            } else if (strcmp("-STEPN", argv[i]) == 0) {
                STEPN = atoi(argv[++i]);
            } else if (strcmp("-STEPK", argv[i]) == 0) {
                STEPK = atoi(argv[++i]);
            } else if (strcmp("-UN", argv[i]) == 0) {
                uplo = 'u', trans = 'n';
            } else if (strcmp("-UT", argv[i]) == 0) {
                uplo = 'u', trans = 't';
            } else if (strcmp("-UC", argv[i]) == 0) {
                uplo = 'u', trans = 'c';
            } else if (strcmp("-LN", argv[i]) == 0) {
                uplo = 'l', trans = 'n';
            } else if (strcmp("-LT", argv[i]) == 0) {
                uplo = 'l', trans = 't';
            } else if (strcmp("-LC", argv[i]) == 0) {
                uplo = 'l', trans = 'c';
            } else if (strcmp("-NOCHECK", argv[i]) == 0) {
                check_flag = 0;
            } else if (strcmp("-LOOP", argv[i]) == 0) {
                LOOP = atoi(argv[++i]);
            } else if (strcmp("-TOTALSTEPS", argv[i]) == 0) {
                TOTALSTEPS = atoi(argv[++i]);
            }
        }
    }

    if (check_flag) {
        handle = dlopen(MPBLAS_REF_LIB DYLIB_SUFFIX, RTLD_LAZY);
        if (!handle) {
            printf("dlopen: %s\n", dlerror());
            return 1;
        }
        mpblas_ref = (void (*)(const char *, const char *, mplapackint, mplapackint, REAL, REAL *, mplapackint, REAL, REAL *, mplapackint))dlsym(handle, mpblas_sym);
        if ((error = dlerror()) != NULL) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }

        raxpy_ref = (void (*)(mplapackint, REAL, REAL *, mplapackint, REAL *, mplapackint))dlsym(handle, raxpy_sym);
        if ((error = dlerror()) != NULL) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
    }
    n = N0;
    k = K0;
    for (p = 0; p < TOTALSTEPS; p++) {
        if (Mlsame(&trans, "n")) {
            ka = k;
            lda = n;
        } else {
            ka = n;
            lda = k;
        }
        ldc = n;

        REAL *A = new REAL[lda * ka];
        REAL *C = new REAL[ldc * n];
        REAL *Cref = new REAL[ldc * n];
        REAL mOne = -1;
        //    alpha = randomnumber (dummy);
        //    beta = randomnumber (dummy);

        alpha = 1.0;
        beta = 0.0;
        for (i = 0; i < lda * ka; i++) {
            A[i] = randomnumber(dummy);
        }
        for (i = 0; i < ldc * n; i++) {
            C[i] = Cref[i] = randomnumber(dummy);
        }

        if (check_flag) {
            t1 = gettime();
            Rsyrk(&uplo, &trans, n, k, alpha, A, lda, beta, C, ldc);
            t2 = gettime();
            elapsedtime = (t2 - t1);
            (*mpblas_ref)(&uplo, &trans, n, k, alpha, A, lda, beta, Cref, ldc);
            (*raxpy_ref)((mplapackint)(ldc * n), mOne, C, (mplapackint)1, Cref, (mplapackint)1);

            diff = Rlange(&normtype, (mplapackint)ldc, (mplapackint)n, Cref, ldc, dummywork);
            diffr = cast2double(diff);
            printf("    n     k      MFLOPS       error    uplo    trans\n");
            printf("%5d %5d  %10.3f    %5.2e       %c        %c\n", (int)n, (int)k, flops_syrk(k, n) / elapsedtime * MFLOPS, diffr, uplo, trans);
        } else {
            elapsedtime = 0.0;
            for (int j = 0; j < LOOP; j++) {
                t1 = gettime();
                Rsyrk(&uplo, &trans, n, k, alpha, A, lda, beta, C, ldc);
                t2 = gettime();
                elapsedtime = elapsedtime + (t2 - t1);
            }
            elapsedtime = elapsedtime / (double)LOOP;
            printf("    n     k      MFLOPS     uplo   trans\n");
            printf("%5d %5d %10.3f      %c      %c\n", (int)n, (int)k, flops_syrk(k, n) / elapsedtime * MFLOPS, uplo, trans);
        }
        delete[] Cref;
        delete[] C;
        delete[] A;
        n = n + STEPN;
        k = k + STEPK;
    }
    if (check_flag)
        dlclose(handle);
}

/*
 * Copyright (c) 2008-2022
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

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <mpblas.h>
#include <mplapack.h>
#include <mplapack_benchmark.h>

int main(int argc, char *argv[]) {
    mplapackint n;
    mplapackint incx = 1, incy = 1, STEP = 97, N0 = 1, LOOP = 3, TOTALSTEPS = 3000;
    REAL alpha, dummy, *dummywork;
    double elapsedtime, t1, t2;
    int i, p;
    int check_flag = 1;
    char normtype = 'm';

    ___MPLAPACK_INITIALIZE___

    const char mpblas_sym[] = SYMBOL_GCC_RAXPY;
    void *handle;
    void (*mpblas_ref)(mplapackint, REAL, REAL *, mplapackint, REAL *, mplapackint);
    char *error;
    REAL diff;
    double diffr;

    if (argc != 1) {
        for (i = 1; i < argc; i++) {
            if (strcmp("-N", argv[i]) == 0) {
                N0 = atoi(argv[++i]);
            } else if (strcmp("-STEP", argv[i]) == 0) {
                STEP = atoi(argv[++i]);
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
        mpblas_ref = (void (*)(mplapackint, REAL, REAL *, mplapackint, REAL *, mplapackint))dlsym(handle, mpblas_sym);
        if ((error = dlerror()) != NULL) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
    }

    n = N0;
    for (p = 0; p < TOTALSTEPS; p++) {
        REAL *x = new REAL[n];
        REAL *y = new REAL[n];
        REAL *yd = new REAL[n];
        if (check_flag) {
            for (i = 0; i < n; i++) {
                x[i] = randomnumber(dummy);
                y[i] = yd[i] = randomnumber(dummy);
            }
            alpha = randomnumber(dummy);
            t1 = gettime();
            Raxpy(n, alpha, x, incx, y, incy);
            t2 = gettime();
            elapsedtime = (t2 - t1);
            (*mpblas_ref)(n, alpha, x, incx, yd, incy);
            diff = Rlange(&normtype, (mplapackint)n, (mplapackint)1, yd, 1, dummywork);
            diffr = cast2double(diff);
            printf("         n       MFLOPS      error\n");
            printf("%10d   %10.3f   %10.3f\n", (int)n, (2.0 * (double)n) / elapsedtime * MFLOPS, diffr);
        } else {
            for (i = 0; i < n; i++) {
                x[i] = randomnumber(dummy);
                y[i] = yd[i] = randomnumber(dummy);
            }
            alpha = randomnumber(dummy);
            elapsedtime = 0.0;
            for (int j = 0; j < LOOP; j++) {
                t1 = gettime();
                Raxpy(n, alpha, x, incx, y, incy);
                t2 = gettime();
                elapsedtime = elapsedtime + (t2 - t1);
            }
            elapsedtime = elapsedtime / (double)LOOP;
            printf("         n       MFLOPS\n");
            printf("%10d   %10.3f\n", (int)n, (2.0 * (double)n) / elapsedtime * MFLOPS);
        }
        delete[] yd;
        delete[] y;
        delete[] x;
        n = n + STEP;
    }
    if (check_flag)
        dlclose(handle);
}

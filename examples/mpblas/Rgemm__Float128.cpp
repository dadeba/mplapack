//public domain
#include <mpblas__Float128.h>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <algorithm>

#define BUFLEN 1024

void printnum(_Float128 rtmp)
{
    int width = 42;
    char buf[BUFLEN];
#if defined ___MPLAPACK_WANT_LIBQUADMATH___
    int n = quadmath_snprintf (buf, sizeof buf, "%+-#*.35Qe", width, rtmp);
#elif defined ___MPLAPACK_LONGDOUBLE_IS_BINARY128___
    snprintf (buf, sizeof buf, "%.35Le", rtmp);
#else
    strfromf128(buf, sizeof(buf), "%.35e", rtmp);
#endif
    if (rtmp >= 0.0)
        printf ("+%s", buf);
    else
        printf ("%s", buf);
    return;
}

//Matlab/Octave format
void printvec(_Float128 *a, int len) {
    _Float128 tmp;
    printf("[ ");
    for (int i = 0; i < len; i++) {
        tmp = a[i];
        printnum(tmp);
        if (i < len - 1)
            printf(", ");
    }
    printf("]");
}

void printmat(int n, int m, _Float128 *a, int lda)
{
    _Float128 mtmp;

    printf("[ ");
    for (int i = 0; i < n; i++) {
        printf("[ ");
        for (int j = 0; j < m; j++) {
            mtmp = a[i + j * lda];
            printnum(mtmp);
            if (j < m - 1)
                printf(", ");
        }
        if (i < n - 1)
            printf("]; ");
        else
            printf("] ");
    }
    printf("]");
}
int main()
{
    mplapackint n = 3;

    _Float128 *a = new _Float128[n * n];
    _Float128 *b = new _Float128[n * n];
    _Float128 *c = new _Float128[n * n];
    _Float128 alpha, beta;

//setting A matrix
    a[0 + 0 * n] = 1;    a[0 + 1 * n] = 8;    a[0 + 2 * n] = 3;
    a[1 + 0 * n] = 2;    a[1 + 1 * n] = 10;   a[1 + 2 * n] = 8;
    a[2 + 0 * n] = 9;    a[2 + 1 * n] = -5;   a[2 + 2 * n] = -1;

    b[0 + 0 * n] = 9;    b[0 + 1 * n] = 8;    b[0 + 2 * n] = 3;
    b[1 + 0 * n] = 3;    b[1 + 1 * n] = -11;  b[1 + 2 * n] = 8;
    b[2 + 0 * n] = -8;   b[2 + 1 * n] = 6;    b[2 + 2 * n] = 1;

    c[0 + 0 * n] = 3;    c[0 + 1 * n] = 3;    c[0 + 2 * n] = -9;
    c[1 + 0 * n] = 8;    c[1 + 1 * n] = 4;    c[1 + 2 * n] = 8;
    c[2 + 0 * n] = 6;    c[2 + 1 * n] = 1;    c[2 + 2 * n] = -2;

    printf("# Rgemm demo...\n");

    printf("a ="); printmat(n, n, a, n); printf("\n");
    printf("b ="); printmat(n, n, b, n); printf("\n");
    printf("c ="); printmat(n, n, c, n); printf("\n");
    alpha = 3.0;
    beta = -2.0;
    Rgemm("n", "n", n, n, n, alpha, a, n, b, n, beta, c, n);

    printf("alpha = "); printnum(alpha); printf("\n");
    printf("beta = "); printnum(beta); printf("\n");
    printf("ans ="); printmat(n, n, c, n); printf("\n");
    printf("#please check by Matlab or Octave following and ans above\n");
    printf("alpha * a * b + beta * c \n");
    delete[]c;
    delete[]b;
    delete[]a;
}

//public domain
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <mpblas_qd.h>
#include <mplapack_qd.h>

#define QD_PRECISION_SHORT 16

inline void printnum(qd_real rtmp) {
    std::cout.precision(QD_PRECISION_SHORT);
    if (rtmp >= 0.0) {
        std::cout << "+" << rtmp;
    } else {
        std::cout << rtmp;
    }
    return;
}

inline void printnum(qd_complex rtmp) {
    std::cout.precision(QD_PRECISION_SHORT);
    if (rtmp.real() >= 0.0) {
        std::cout << "+" << rtmp.real();
    } else {
        std::cout << rtmp.real();
    }
    if (rtmp.imag() >= 0.0) {
        std::cout << "+" << rtmp.imag() << "i";
    } else {
        std::cout << rtmp.imag() << "i";
    }
    return;
}

//Matlab/Octave format
template <class X> void printvec(X *a, int len) {
    X tmp;
    printf("[ ");
    for (int i = 0; i < len; i++) {
        tmp = a[i];
        printnum(tmp);
        if (i < len - 1)
            printf(", ");
    }
    printf("]");
}

template <class X> void printmat(int n, int m, X *a, int lda)
{
    X mtmp;

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
bool cselect(qd_complex a) {
    // sorting rule for eigenvalues.
    return false;
}

int main() {
    mplapackint n = 4;

    qd_complex *a = new qd_complex[n * n];
    mplapackint sdim = 0;
    mplapackint lwork = 2 * n;
    qd_complex *w = new qd_complex[n];
    qd_complex *vs = new qd_complex[n * n];
    qd_complex *work = new qd_complex[lwork];
    qd_real *rwork = new qd_real[n];
    bool bwork[n];
    mplapackint info;

    // setting A matrix
    a[0 + 0 * n] = qd_complex(5.0,  9.0); a[0 + 1 * n] = qd_complex(5.0, 5.0);   a[0 + 2 * n] = qd_complex(-6.0, -6.0); a[0 + 3 * n] = qd_complex(-7.0,-7.0);
    a[1 + 0 * n] = qd_complex(3.0,  3.0); a[1 + 1 * n] = qd_complex(6.0,10.0);   a[1 + 2 * n] = qd_complex(-5.0, -5.0); a[1 + 3 * n] = qd_complex(-6.0,-6.0);
    a[2 + 0 * n] = qd_complex(2.0,  2.0); a[2 + 1 * n] = qd_complex(3.0, 3.0);   a[2 + 2 * n] = qd_complex(-1.0, 3.0);  a[2 + 3 * n] = qd_complex(-5.0,-5.0);
    a[3 + 0 * n] = qd_complex(1.0,  1.0); a[3 + 1 * n] = qd_complex(2.0, 2.0);   a[3 + 2 * n] = qd_complex(-3.0,-3.0);  a[3 + 3 * n] = qd_complex(0.0, 4.0);

    printf("# Ex. 6.5 p. 116, Collection of Matrices for Testing Computational Algorithms, Robert T. Gregory, David L. Karney\n");
    printf("# octave check\n");
    printf("split_long_rows(0)\n");
    printf("a ="); printmat(n, n, a, n); printf("\n");
    Cgees("V", "S", cselect, n, a, n, sdim, w, vs, n, work, lwork, rwork, bwork, info);
    printf("w ="); printvec(w, n); printf("\n");
    printf("vs ="); printmat(n, n, vs, n); printf("\n");
    printf("t ="); printmat(n, n, a, n); printf("\n");
    printf("vs*t*vs'\n");
    printf("eig(a)\n");

    delete[] rwork;
    delete[] work;
    delete[] vs;
    delete[] w;
    delete[] a;
}

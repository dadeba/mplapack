//public domain
#include <mpblas__Float64x.h>
#include <mplapack__Float64x.h>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <algorithm>

#define FLOAT64X_FORMAT "%+25.21Le"
#define FLOAT64X_SHORT_FORMAT "%+20.16Le"

void printnum(_Float64x rtmp)
{
    printf(FLOAT64X_FORMAT, rtmp);
    return;
}

//Matlab/Octave format
void printvec(_Float64x *a, int len) {
    _Float64x tmp;
    printf("[ ");
    for (int i = 0; i < len; i++) {
        tmp = a[i];
        printnum(tmp);
        if (i < len - 1)
            printf(", ");
    }
    printf("]");
}

void printmat(int n, int m, _Float64x *a, int lda)
{
    _Float64x mtmp;

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
bool rselect(_Float64x ar, _Float64x ai) {
    // sorting rule for eigenvalues.
    return false;
}

#include <iostream>
using namespace std;

int main() {
    mplapackint n;

    string str;
    getline(cin, str);
    cout << str << endl;
    getline(cin, str);
    stringstream ss(str);
    ss >> n;
    printf("# n %d\n", (int)n);

    _Float64x *a = new _Float64x[n * n];
    _Float64x *vl = new _Float64x[n * n];
    _Float64x *vr = new _Float64x[n * n];
    mplapackint sdim = 0;
    mplapackint lwork = 4 * n;
    _Float64x *wr = new _Float64x[n];
    _Float64x *wi = new _Float64x[n];
    _Float64x *work = new _Float64x[lwork];
    mplapackint info;
    double dtmp;
    for (int i = 0; i < n; i++) {
        getline(cin, str);
        stringstream ss(str);
        for (int j = 0; j < n; j++) {
            ss >> dtmp;
            a[i + j * n] = dtmp;
        }
    }
    printf("# octave check\n");
    printf("split_long_rows(0)\n");
    printf("a ="); printmat(n, n, a, n); printf("\n");
    Rgeev("V", "V", n, a, n, wr, wi, vl, n, vr, n, work, lwork, info);
    printf("# left vectors\n");
    printf("vl ="); printmat(n, n, vl, n); printf("\n");
    printf("# left vectors\n");
    printf("vr ="); printmat(n, n, vr, n); printf("\n");
    printf("[vl, d, w] = eig(a)\n");
    for (int i = 1; i <= n; i = i + 1) {
        printf("w_%d = ", (int)i); printnum(wr[i - 1]); printf(" "); printnum(wi[i - 1]); printf("i\n");
    }
    delete[] work;
    delete[] wr;
    delete[] wi;
    delete[] vr;
    delete[] vl;
    delete[] a;
}

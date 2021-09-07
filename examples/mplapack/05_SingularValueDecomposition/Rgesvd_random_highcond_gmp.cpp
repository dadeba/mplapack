//public domain
#include <mpblas_gmp.h>
#include <mplapack_gmp.h>
#include <iostream>
#include <cstring>
#include <algorithm>

#define GMP_FORMAT "%+68.64Fe"
#define GMP_SHORT_FORMAT "%+20.16Fe"

inline void printnum(mpf_class rtmp) { gmp_printf(GMP_FORMAT, rtmp.get_mpf_t()); }
inline void printnum_short(mpf_class rtmp) { gmp_printf(GMP_SHORT_FORMAT, rtmp.get_mpf_t()); }

//Matlab/Octave format
void printvec(mpf_class *a, int len) {
    mpf_class tmp;
    printf("[ ");
    for (int i = 0; i < len; i++) {
        tmp = a[i];
        printnum(tmp);
        if (i < len - 1)
            printf(", ");
    }
    printf("]");
}

void printmat(int n, int m, mpf_class * a, int lda)
{
    mpf_class mtmp;

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
#include <random>

int compare_real(const void *a, const void *b)
{
    return *(mpf_class*)a < *(mpf_class*)b;
}

int main(int argc, char *argv[]) {
    mplapackint n = 3;
    mplapackint dispersion = 3;
    if (argc != 1) {
        for (int i = 1; i < argc; i++) {
            if (strcmp("-DIMN", argv[i]) == 0) {
                n = atoi(argv[++i]);
            }
            if (strcmp("-DISPERSION", argv[i]) == 0) {
                dispersion = atoi(argv[++i]);
            }
        }
    }
    printf("#dimension %d, dispersion = %d \n", (int)n, (int)dispersion);

    mpf_class *a = new mpf_class[n * n];
    mpf_class *aorg = new mpf_class[n * n];
    mpf_class *ainv = new mpf_class[n * n];
    mpf_class *at_a = new mpf_class[n * n];
    mpf_class *I_ = new mpf_class[n * n];
    mpf_class *s = new mpf_class[n * n];
    mpf_class *sorg = new mpf_class[n];
    mpf_class *u = new mpf_class[n * n];
    mpf_class *vt = new mpf_class[n * n];

    mpf_class *vl = new mpf_class[n * n];
    mpf_class *vr = new mpf_class[n * n];
    mpf_class *wr = new mpf_class[n];
    mpf_class *wi = new mpf_class[n];

    mplapackint lwork = std::max({(mplapackint)1, 3 * n + n, 5 * n});
    mplapackint *ipiv = new mplapackint[n];
    mplapackint info;

    mpf_class *work = new mpf_class[lwork];

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::normal_distribution<> dist(0, (double)dispersion);

    // Generation of high condition integer matrix.
    // Strategy. Generate a matrix whose elements are integer with determinant = 1.
    // Then the elements of inverse of the matrix is also integer.
    // Prepare a geometric integer series, and obtain a matrix with diagonal elements by the series.
    // Calculate A-1 S A.
    // 1. Set Hessenberg matirx with a[1,n] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) {
                int r = dist(engine);
                a[i + j * n] = r;
                aorg[i + j * n] = r;
            } else if (j == i) {
                int r = dist(engine);
                while (r == 0) {
                    r = dist(engine);
                }
                a[i + j * n] = r;
                aorg[i + j * n] = r;
            } else if (j == i - 1) {
                a[i + j * n] = 1.0;
                aorg[i + j * n] = 1.0;
            } else {
                a[i + j * n] = 0.0;
                aorg[i + j * n] = 0.0;
            }
        }
    }
    a[0 + (n - 1) * n] = 0.0;
    aorg[0 + (n - 1) * n] = 0.0;
    printf("split_long_rows(0)\n");
    printf("aorg ="); printmat(n, n, aorg, n); printf("\n");

    // 2. get determinant via LU factorization
    Rgetrf(n, n, a, n, ipiv, info);
    // printf("aLU ="); printmat(n, n, a, n); printf("\n");
    mpf_class det = 1;
    for (int i = 0; i < n; i++) {
        det = det * a[i + i * n];
        if (ipiv[i] != i + 1)
            det = det * -1.0;
    }
    printf("det="); printnum(det); printf("\n");

    // 3. Set Hessenberg matirx with a[1,n] = 0 to make det of matrix a = 1.
    aorg[0 + (n - 1) * n] = -det + 1;
    printf("anew ="); printmat(n, n, aorg, n); printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ainv[i + j * n] = aorg[i + j * n];
            a[i + j * n] = aorg[i + j * n];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            s[i + j * n] = 0.0;
        }
    }
    // 4. genrarate eigenvalues
    for (int i = 0; i < n; i++) {
         int r = dist(engine);
         s[i + i * n] = mpf_class(r * r);
         while ( r == 0 ) {
             r = (int)dist(engine);
             s[i + i * n] = (mpf_class)(r * r);
	 }
         printf("rand=%d, %d\n", (int)i, (int)r);
    }
    s[0] = 1.0;
    for (int i = 0; i < n; i++)
        sorg[i] = s[i + i * n];
    printf("s = ["); for (int i = 0; i < n; i++) { printnum(s[i + i * n]);  printf(" "); } printf("]\n");

    // 5. inverse matrix. All the elements are integers.
    Rgetrf(n, n, ainv, n, ipiv, info);
    Rgetri(n, ainv, n, ipiv, work, lwork, info);

    // 5.5. verify Ainv * A = I
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mpf_class rtmp = 0.0;
            for (int k = 0; k < n; k++) {
                rtmp = rtmp + ainv[i + k * n] * a[k + j * n];
            }
            I_[i + j * n] = rtmp;
        }
    }
    printf("I ="); printmat(n, n, I_, n); printf("\n");

    // 6. Make a  A <- Ainv * S * A
    printf("ainv ="); printmat(n, n, ainv, n); printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mpf_class rtmp = 0.0;
            for (int k = 0; k < n; k++) {
                for (int l = 0; l < n; l++) {
                    rtmp = rtmp + ainv[i + k * n] * s[k + l * n] * aorg[l + j * n];
                }
            }
            a[i + j * n] = rtmp;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aorg[i + j * n] = a[i + j * n];
        }
    }
    // 7. svd(A)
    Rgesvd("A", "A", n, n, a, n, s, u, n, vt, n, work, lwork, info);
    printf("s=[");
    for (int i = 0; i < n; i++) { printnum(s[i]); printf(" "); } printf(" ] \n");
    printf("s_squared=["); for (int i = 0; i < n; i++) { printnum(s[i] * s[i]); printf(" "); } printf(" ] \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i + j * n] = aorg[i + j * n];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mpf_class rtmp = 0.0;
            for (int k = 0; k < n; k++) {
                rtmp = rtmp + a[k + i * n] * a[k + j * n];
            }
            at_a[i + j * n] = rtmp;
        }
    }
    // 7. eig(A^t A).
    Rgeev("V", "V", n, at_a, n, wr, wi, vl, n, vr, n, work, lwork, info);

    qsort(wr, n, sizeof(mpf_class), compare_real);
    for (int i = 0; i < n; i = i + 1) {
        printf("w_%d = ", (int)i); printnum(wr[i]); printf(" "); printnum(wi[i]); printf("i\n");
    }
    // 8. There is a relation \lambda_i of eig(A^t A) and \sigma_i svd(A)
    // \lambda_i = \sigma_i^2
    // 9. Relative error

    mpf_class relerror;
    for (int i = 0; i < n; i = i + 1) {
        relerror = abs ( (wr[i] - s[i] * s[i]) / (s[i] * s[i]) ) ;
        printf("Relative_error_%d = ", (int)i); printnum(relerror); printf("\n");
    }

    delete[] work;
    delete[] ipiv;
    delete[] wi;
    delete[] wr;
    delete[] vl;
    delete[] vr;
    delete[] vt;
    delete[] u;
    delete[] sorg;
    delete[] s;
    delete[] I_;
    delete[] at_a;
    delete[] ainv;
    delete[] aorg;
    delete[] a;
}

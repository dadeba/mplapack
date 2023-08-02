// This file automatically generated by template-converter
// DO NOT EDIT!

char cpp_kernel_str[] =
"#define __OPENCL__\n"
"\n"
"typedef ulong    u64;\n"
"typedef long     i64;\n"
"typedef uint     u32;\n"
"typedef int      i32;\n"
"\n"
"__constant const u32 nc   = 4;\n"
"__constant const u32 nb   = 30;\n"
"__constant const u32 nexp = 30;\n"
"__constant const u32 nman = nb*nc;\n"
"__constant const u32 nbit = nman + nexp + 1;\n"
"__constant const u32 bias = ((0x1UL<<(nexp - 1))-0x1UL);\n"
"\n"
"__constant const u32 bias30 = ((0x1UL<<(30 - 1))-0x1UL);\n"
"__constant const u32 bias15 = ((0x1UL<<(15 - 1))-0x1UL);\n"
"__constant const u32 bias11 = ((0x1UL<<(11 - 1))-0x1UL);\n"
"__constant const u32 bias8  = ((0x1UL<<(8  - 1))-0x1UL);\n"
"__constant const u32 ncnc   = nc*2;\n"
"\n"
"#define NC 4\n"
"//#define nc 4\n"
"#define NCNC 8\n"
"//#define ncnc 8\n"
"u32 MASK32(const u32 x)\n"
"{\n"
"  return ((0x1<<(x))-0x1);\n"
"}\n"
"\n"
"u64 MASK64(const u64 x)\n"
"{\n"
"  return (((u64)0x1<<(x))-(u64)0x1);\n"
"}\n"
"\n"
"u64 accessn(const u64 x, const u64 hi, const u64 lo)\n"
"{\n"
"  u64 size, res;\n"
"  size = hi - lo + 1;\n"
"  res = (x>>lo) & MASK64(size);\n"
"  return res;\n"
"}\n"
"\n"
"u64 access1(const u64 x, const u64 o)\n"
"{\n"
"  u64 res;\n"
"  res = (x>>o) & (u64)0x1;\n"
"  return res;\n"
"}\n"
"\n"
"struct my_fp {\n"
"  u32 e;\n"
"  u32 m[NC];\n"
"  // m[i](29..0) is the mantissa\n"
"};\n"
"typedef struct my_fp FP[1];\n"
"#ifdef __OPENCL__\n"
"__constant\n"
"#endif\n"
"const struct my_fp zero_fp = {.e = 0, .m = {0,0,0,0} };\n"
"\n"
"#ifdef __OPENCL__\n"
"inline void fp_zero_clear(FP x)\n"
"{\n"
"  x->e = 0x0;\n"
"  for(u32 i = 0; i < nc; i++) x->m[i] = 0;\n"
"}\n"
"\n"
"inline void fp_dup(const FP src, FP dst)\n"
"{\n"
"  dst->e = src->e;\n"
"  for(u32 i = 0; i < nc; i++) dst->m[i] = src->m[i];\n"
"}\n"
"#else\n"
"inline void fp_dup(const FP src, FP dst)\n"
"{\n"
"  bcopy(src, dst, sizeof(FP));\n"
"}\n"
"\n"
"inline void fp_zero_clear(FP x)\n"
"{\n"
"  //  fp_dup(fp_zero, x);\n"
"  //  memset(x, 0x0, sizeof(FP));\n"
"  *x = zero_fp;\n"
"}\n"
"#endif\n"
"\n"
"inline void fp_dup_and_flip_sign(const FP src, FP dst)\n"
"{\n"
"  fp_dup(src, dst);\n"
"  dst->e = dst->e ^ (0x1 << nexp);\n"
"}\n"
"\n"
"inline u32 get_exponent(const FP x)\n"
"{\n"
"  return (x->e & MASK32(nexp));\n"
"}\n"
"\n"
"inline u32 get_sign(const FP x) {\n"
"  return (x->e >> nexp) & MASK32(1);\n"
"}\n"
"\n"
"inline void set_sign(const u32 s, FP x) {\n"
"  x->e = (s << nexp) | get_exponent(x);\n"
"}\n"
"\n"
"inline void set_exponent(const u32 e0, FP x) {\n"
"  x->e = (get_sign(x) << nexp) | (e0 & MASK32(nexp));\n"
"}\n"
"\n"
"void fp_set_double(const double xx, FP x)\n"
"{\n"
"  fp_zero_clear(x);\n"
"\n"
"  if (xx == 0.0) {\n"
"    // its really zero!\n"
"    return; \n"
"  }\n"
"\n"
"  u64 *p, s0, e0, m0;\n"
"  p = (u64 *) (&xx);\n"
"  s0 = access1(*p, 63); \n"
"  e0 = accessn(*p, 62, 52);\n"
"  m0 = accessn(*p, 51, 0);\n"
"\n"
"  u32 ee = (e0 - bias11) + bias;\n"
"  set_sign((u32)s0, x);\n"
"  set_exponent(ee, x);\n"
"\n"
"  if (e0 != 0) m0 |= (u64)0x1<<52;\n"
"  x->m[0] = m0 >> (53 - nb);\n"
"  m0 = m0 & MASK64(53 - nb);\n"
"  x->m[1] = m0 << (nb - (53 - nb));\n"
"}\n"
"\n"
"void fp_swap_xy(FP x, FP y) {\n"
"  if ( get_exponent(y) > get_exponent(x) ) {\n"
"    FP tmp;\n"
"    fp_dup(x, tmp);\n"
"    fp_dup(y, x);\n"
"    fp_dup(tmp, y);\n"
"  }\n"
"}\n"
"\n"
"u32 diff(FP x, FP y)\n"
"{\n"
"  if (y->e == 0) return 0;\n"
"  return get_exponent(x) - get_exponent(y);\n"
"}\n"
"\n"
"void r_shift(const u32 s, FP x)\n"
"{\n"
"  //    force-1 rounding\n"
"  u32 q = 0;\n"
"  u32 skip = s/nb;\n"
"  u32 ss = s % nb;\n"
"\n"
"  u32 sbit = 0; \n"
"  {\n"
"    u32 j = nc-1;\n"
"    u32 k = s;\n"
"    while(k > 29 && j > 0) {\n"
"      sbit |= x->m[j];\n"
"      k = k - nb;\n"
"      j--;\n"
"    }\n"
"    sbit = x->m[j] & MASK32(k);\n"
"    sbit = (sbit == 0) ? 0 : 0x1;\n"
"  }\n"
"\n"
"  u32 mm0[NC];\n"
"  for(u32 i = 0; i < nc; i++) mm0[i] = 0;\n"
"\n"
"  for(u32 i = skip; i < nc; i++) {\n"
"    u32 mm = x->m[i-skip];\n"
"    u32 u = q | (mm >> ss);\n"
"    q = (mm & MASK32(ss)) << (nb-ss);\n"
"    mm0[i] = u;\n"
"  }\n"
"\n"
"  u32 mask = (s < nbit) ? (u32)MASK64(32) : 0x0;\n"
"  for(u32 i = 0; i < nc; i++) x->m[i] = mm0[i] & mask;\n"
"  x->m[nc-1] |= sbit;\n"
"}\n"
"\n"
"void l_shift(const u32 s, FP x)\n"
"{\n"
"  u32 mm[8];\n"
"  for(u32 i = 0; i < nc; i++) mm[i] = 0;\n"
"\n"
"  u32 ss = s % nb;\n"
"  u32 stride = s/nb;\n"
"  u32 cc = nc - s/nb;\n"
"\n"
"  for(u32 i = 0; i < cc; i++) {\n"
"    u32 h, l;\n"
"    u32 mm_h = x->m[i+stride];\n"
"    u32 mm_l = (i == cc-1) ? 0 : x->m[i+stride+1];\n"
"\n"
"    h = mm_h << ss;\n"
"    l = (mm_l >> (nb-ss)) & MASK32(ss);\n"
"    mm[i] = (h | l) & MASK32(nb);\n"
"  }\n"
"\n"
"  for(u32 i = 0; i < nc; i++) x->m[i] = mm[i];\n"
"}\n"
"\n"
"bool negative(FP x)\n"
"{\n"
"  return (get_sign(x) == 0x1);\n"
"}\n"
"\n"
"void inv(FP x) {\n"
"  x->m[0] = ~x->m[0];\n"
"  for(u32 i = 1; i < nc; i++) {\n"
"    x->m[i] = (~x->m[i]) & MASK32(nb);\n"
"  }\n"
"}\n"
"\n"
"void c_adder_primitive(u32 *x, u32 *carry, i32 mask)\n"
"{\n"
"  *x = *x + *carry;\n"
"  *carry = (*x  >> nb) & MASK32(1);\n"
"  if (mask) *x &= MASK32(nb);\n"
"}\n"
"\n"
"void c_adder(FP x, FP y, u32 carry, FP z) {\n"
"  // for full-adder\n"
"  for(i32 i = nc-1; i >= 0; i--) {\n"
"    u32 res = x->m[i] + y->m[i];\n"
"    c_adder_primitive(&res, &carry, i);\n"
"    z->m[i] = res;\n"
"  }\n"
"}\n"
"\n"
"void c_adder_inv(FP x, u32 carry) {\n"
"  // for inversion\n"
"  for(i32 i = nc-1; i >= 0; i--) {\n"
"    x->m[i] = x->m[i] + carry;\n"
"    carry = (x->m[i] >> nb) & MASK32(1);\n"
"    x->m[i] = x->m[i] & MASK32(nb);\n"
"  }\n"
"}\n"
"\n"
"void fp_negate(FP x) {\n"
"  u32 flag = x->m[0] >> 31;\n"
"  if (flag == 0x1) {\n"
"    inv(x);\n"
"    c_adder_inv(x, 0x1);\n"
"    set_sign(0x1, x);\n"
"  }\n"
"}\n"
"\n"
"u32 nlz0(u32 x) \n"
"{\n"
"  u32 y;\n"
"  i32 n;\n"
"\n"
"  n = 32;\n"
"  y = x >> 16; if (y != 0) {n = n - 16; x = y; }\n"
"  y = x >>  8; if (y != 0) {n = n -  8; x = y; }\n"
"  y = x >>  4; if (y != 0) {n = n -  4; x = y; }\n"
"  y = x >>  2; if (y != 0) {n = n -  2; x = y; }\n"
"  y = x >>  1; if (y != 0) return n - 2;\n"
"  \n"
"  return n - x;\n"
"}\n"
"  \n"
"i32 nlz(FP x)\n"
"{\n"
"  i32 pos = 0;\n"
"  for(u32 i = 0; i < nc; i++) {\n"
"    i32 tmp = nlz0(x->m[i]);\n"
"    i32 tmp_c = tmp - (32 - nb);\n"
"    pos += tmp_c;\n"
"    if (tmp < 32) {\n"
"      break;\n"
"    }\n"
"  }\n"
"  return pos;\n"
"}\n"
"\n"
"void exp_plus1(FP x) {\n"
"  set_exponent(get_exponent(x) + 1, x);\n"
"}\n"
"\n"
"void exp_minus1(FP x) {\n"
"  set_exponent(get_exponent(x) - 1, x);\n"
"}\n"
"\n"
"void fp_normalize(FP x) {\n"
"  u32 sum = 0x0;\n"
"  for(u32 i = 0; i < nc; i++) {\n"
"    sum |= x->m[i];\n"
"  }\n"
"\n"
"  if (sum == 0x0) {\n"
"    fp_zero_clear(x);\n"
"    return;\n"
"  }\n"
"\n"
"  i32 pos = nlz(x);\n"
"  if (pos == -1) {\n"
"    r_shift(1, x);\n"
"    exp_plus1(x);\n"
"    return;\n"
"  }\n"
"\n"
"  if (pos == 0) {\n"
"    return;\n"
"  }\n"
"\n"
"  l_shift(pos, x);\n"
"  set_exponent(get_exponent(x) - pos, x);\n"
"}\n"
"\n"
"inline void flip_sign(FP x)\n"
"{\n"
"  //  u32 s = get_sign(x);\n"
"  //  set_sign(x, s^0x1);\n"
"  x->e = x->e ^ (0x1<<nexp);\n"
"  //  x->e ^= (0x1<<nexp);\n"
"}\n"
"\n"
"#ifdef __OPENCL__\n"
"#else\n"
"template <int ncomponent>\n"
"void mul_u32xu32(const u32 xm[], const u32 ym [], u64 mmp[])\n"
"{\n"
"  puts(\"not supported nc\");\n"
"  //  assert(false);\n"
"}\n"
"\n"
"template<> void mul_u32xu32<7>(const u32 xm[], const u32 ym [], u64 mmp[])\n"
"{\n"
"  u64 mm[nc][nc];\n"
"  for(u32 j = 0; j < nc; j++) {\n"
"    for(u32 i = 0; i < nc; i++) {\n"
"      mm[j][i] = (u64)xm[i]*(u64)ym[j];\n"
"    }\n"
"  }\n"
"\n"
"  u64 mmp0[ncnc];\n"
"  mmp0[0] = mm[0][0];\n"
"  mmp0[1] = mm[1][0] + mm[0][1]; \n"
"  mmp0[2] = mm[2][0] + mm[1][1] + mm[0][2];\n"
"  mmp0[3] = mm[3][0] + mm[2][1] + mm[1][2] + mm[0][3];\n"
"  mmp0[4] = mm[4][0] + mm[3][1] + mm[2][2] + mm[1][3] + mm[0][4];\n"
"  mmp0[5] = mm[5][0] + mm[4][1] + mm[3][2] + mm[2][3] + mm[1][4] + mm[0][5];\n"
"  mmp0[6] = mm[6][0] + mm[5][1] + mm[4][2] + mm[3][3] + mm[2][4] + mm[1][5] + mm[0][6];\n"
"  mmp0[7] =          + mm[6][1] + mm[5][2] + mm[4][3] + mm[3][4] + mm[2][5] + mm[1][6];\n"
"  mmp0[8] =                     + mm[6][2] + mm[5][3] + mm[4][4] + mm[3][5] + mm[2][6];\n"
"  mmp0[9] =                	      	   + mm[6][3] + mm[5][4] + mm[4][5] + mm[3][6];\n"
"  mmp0[10]=         	      		              + mm[6][4] + mm[5][5] + mm[4][6];\n"
"  mmp0[11]=         	      		  	                 + mm[6][5] + mm[5][6];\n"
"  mmp0[12]=         	      					              mm[6][6];\n"
"\n"
"  u64 mmp1[ncnc];\n"
"  mmp1[12] = mmp0[12];\n"
"  mmp1[11] = mmp0[11] + (mmp0[12] >> nb);\n"
"  // carry in \n"
"  mmp1[10] = mmp0[10] + (mmp1[11] >> nb);\n"
"  mmp1[ 9] = mmp0[ 9] + (mmp1[10] >> nb);\n"
"  mmp1[ 8] = mmp0[ 8] + (mmp1[ 9] >> nb);\n"
"  mmp1[ 7] = mmp0[ 7] + (mmp1[ 8] >> nb);\n"
"  mmp1[ 6] = mmp0[ 6] + (mmp1[ 7] >> nb);\n"
"  mmp1[ 5] = mmp0[ 5] + (mmp1[ 6] >> nb);\n"
"  mmp1[ 4] = mmp0[ 4] + (mmp1[ 5] >> nb);\n"
"  mmp1[ 3] = mmp0[ 3] + (mmp1[ 4] >> nb);\n"
"  mmp1[ 2] = mmp0[ 2] + (mmp1[ 3] >> nb);\n"
"  mmp1[ 1] = mmp0[ 1] + (mmp1[ 2] >> nb);\n"
"  mmp1[ 0] = mmp0[ 0] + (mmp1[ 1] >> nb);\n"
"\n"
"  mmp[0] = mmp1[0];\n"
"  for(u32 i = 1; i < ncnc-1; i++) {\n"
"    mmp[i] = mmp1[i] & MASK64(nb);\n"
"  }\n"
"}\n"
"#endif\n"
"\n"
"#ifdef __OPENCL__\n"
"void mul_u32xu32(const u32 xm[], const u32 ym [], u64 mmp[])\n"
"#else\n"
"template<> void mul_u32xu32<4>(const u32 xm[], const u32 ym [], u64 mmp[])\n"
"#endif\n"
"{\n"
"#ifdef __OPENCL__\n"
"  u64 mm[NC][NC];\n"
"#else\n"
"  u64 mm[nc][nc];\n"
"#endif\n"
"  for(u32 j = 0; j < nc; j++) {\n"
"    for(u32 i = 0; i < nc; i++) {\n"
"      mm[j][i] = (u64)xm[i]*(u64)ym[j];\n"
"    }\n"
"  }\n"
"\n"
"  u64 mmp0[14];\n"
"  mmp0[0] = mm[0][0];\n"
"  mmp0[1] = mm[1][0] + mm[0][1]; \n"
"  mmp0[2] = mm[2][0] + mm[1][1] + mm[0][2];\n"
"  mmp0[3] = mm[3][0] + mm[2][1] + mm[1][2] + mm[0][3];\n"
"  mmp0[4] =            mm[3][1] + mm[2][2] + mm[1][3];\n"
"  mmp0[5] =                       mm[3][2] + mm[2][3];\n"
"  mmp0[6] =                                  mm[3][3];\n"
"\n"
"  u64 mmp1[14];\n"
"  mmp1[ 6] = mmp0[ 6];\n"
"  mmp1[ 5] = mmp0[ 5] + (mmp1[ 6] >> nb);\n"
"  mmp1[ 4] = mmp0[ 4] + (mmp1[ 5] >> nb);\n"
"  mmp1[ 3] = mmp0[ 3] + (mmp1[ 4] >> nb);\n"
"  mmp1[ 2] = mmp0[ 2] + (mmp1[ 3] >> nb);\n"
"  mmp1[ 1] = mmp0[ 1] + (mmp1[ 2] >> nb);\n"
"  mmp1[ 0] = mmp0[ 0] + (mmp1[ 1] >> nb);\n"
"\n"
"  mmp[0] = mmp1[0];\n"
"  for(u32 i = 1; i < ncnc; i++) {\n"
"    mmp[i] = mmp1[i] & MASK64(nb);\n"
"  }\n"
"}\n"
"\n"
"void add_body(FP x0, FP y0, FP z)\n"
"{\n"
"  fp_swap_xy(x0, y0);\n"
"\n"
"  z->e = get_exponent(x0);\n"
"\n"
"  u32 rs = diff(x0, y0);\n"
"  if ( rs > 0 ) r_shift(rs, y0);\n"
"\n"
"  u32 c1 = 0;\n"
"  u32 c2 = 0;\n"
"  u32 flag = get_sign(x0) ^ get_sign(y0);\n"
"  if (negative(x0) && flag == 0x1) {\n"
"    inv(x0);\n"
"    c1 = 1;\n"
"  }\n"
"\n"
"  if (negative(y0) && flag == 0x1) {\n"
"    inv(y0);\n"
"    c2 = 1;\n"
"  }\n"
"\n"
"  c_adder(x0, y0, c1+c2, z);\n"
"\n"
"  fp_negate(z);\n"
"  \n"
"  if ( negative(x0) && negative(y0) ) {\n"
"    set_sign(0x1, z);\n"
"  }\n"
"  \n"
"  fp_normalize(z);\n"
"}\n"
"\n"
"void add(const FP x, const FP y, FP z)\n"
"{\n"
"  FP x0, y0;\n"
"  fp_dup(x, x0);\n"
"  fp_dup(y, y0);\n"
"\n"
"  add_body(x0, y0, z);\n"
"}\n"
"\n"
"void sub(FP x, FP y, FP z)\n"
"{\n"
"  FP x0, y0;\n"
"  fp_dup(x, x0);\n"
"  //  fp_dup_and_flip_sign(x, x0);\n"
"  fp_dup_and_flip_sign(y, y0);\n"
"  //  flip_sign(y0);\n"
"  add_body(x0, y0, z);\n"
"}\n"
"\n"
"u32 mul_exponent(const u32 x, const u32 y)\n"
"{\n"
"  u32 z_e = x + y - bias;\n"
"\n"
"  if (z_e > MASK32(nexp)) z_e = MASK32(nexp);\n"
"\n"
"  return z_e;\n"
"}\n"
"\n"
"void mul(FP x, FP y, FP z)\n"
"{\n"
"  FP x0, y0;\n"
"  fp_dup(x, x0);\n"
"  fp_dup(y, y0);\n"
"  \n"
"  set_sign(get_sign(x0) ^ get_sign(y0), z);\n"
"  \n"
"  u32 x_e = get_exponent(x0);\n"
"  u32 y_e = get_exponent(y0);\n"
"  u32 z_e = mul_exponent(x_e, y_e);\n"
"  if (x_e == 0 || y_e == 0) {\n"
"    fp_zero_clear(z);\n"
"    return;\n"
"  }\n"
"  set_exponent(z_e, z);\n"
"\n"
"\n"
"#ifdef __OPENCL__\n"
"  u64 mmp[NCNC];\n"
"  mul_u32xu32(x0->m, y0->m, mmp);\n"
"#else\n"
"  u64 mmp[ncnc];\n"
"  mul_u32xu32<nc>(x0->m, y0->m, mmp);\n"
"#endif\n"
"  \n"
"  // used mantissa is \n"
"  //   mmp[0] + ... + mmp[5] + mmp[6](high) : nb*7\n"
"  // remaider is rounded off\n"
"  // force-1 rounding\n"
"  u64 sbit = 0x0;\n"
"  u32 pos = nlz0((u32)(mmp[0] >> 32));\n"
"  if (pos == 4) {\n"
"    // mmp[0] is 60 bit\n"
"    z->m[0] = mmp[0] >> nb;\n"
"    z->m[1] = mmp[0] & MASK32(nb);\n"
"    z->m[2] = mmp[1];\n"
"    z->m[3] = mmp[2];\n"
"    exp_plus1(z);\n"
"    sbit = mmp[3];\n"
"  } else { // pos == 5\n"
"    // mmp[0] is 59 bit\n"
"    z->m[0] = mmp[0] >> (nb-1);\n"
"    z->m[1] = (mmp[0] & MASK32(nb-1))<<1 | ((mmp[1] >> (nb-1)) & MASK32(1));\n"
"    z->m[2] = (mmp[1] & MASK32(nb-1))<<1 | ((mmp[2] >> (nb-1)) & MASK32(1));\n"
"    z->m[3] = (mmp[2] & MASK32(nb-1))<<1 | ((mmp[3] >> (nb-1)) & MASK32(1));\n"
"    sbit   = (mmp[3] & MASK32(nb-1));\n"
"  }\n"
"\n"
"  sbit |= mmp[4];\n"
"  sbit |= mmp[5];\n"
"  sbit |= mmp[6];\n"
"  sbit = ( sbit != 0x0 ) ? 0x1 : 0x0;\n"
"\n"
"  z->m[3] |= sbit;\n"
"}\n"
"\n"
"bool eq(const FP x, const FP y, const i32 c)\n"
"{\n"
"  bool res = false;\n"
"\n"
"  res = (x->e == y->e);\n"
"  for(i32 i = 0; i < c; i++) {\n"
"    res &= (x->m[i] == y->m[i]); \n"
"  }\n"
"\n"
"  return res;\n"
"}\n"
"struct b128 {\n"
"  u64 d[2];\n"
"};\n"
"__constant const struct b128 zero_b128 = { .d = {0,0} };\n"
"\n"
"void fp_to_b128(const FP x, struct b128 *y) \n"
"{\n"
"  u64 x_sign, x_exp;\n"
"  i64 e;\n"
"\n"
"  x_sign = get_sign(x);\n"
"  x_exp  = get_exponent(x);\n"
"\n"
"  if (x_exp == 0x0) {\n"
"    *y = zero_b128;\n"
"    return;\n"
"  }\n"
"\n"
"  e = (((i32)x_exp - bias) + bias15) & MASK64(15);\n"
"\n"
"  u64 x_man[NC];\n"
"  \n"
"  for(i32 i = 0; i < (i32)nc; i++) {\n"
"    x_man[i] = (u64)x->m[i];\n"
"  }\n"
"  \n"
"  //  1,15,[29],19|11,30,23\n"
"  //  u64 man1 = (x->m[0] & MASK32(29)) <<19 | (x->m[1]>>11)&MASK32(19);\n"
"  u64 man1 = (x_man[0] & MASK64(29)) <<19 | ((x_man[1]>>11)&MASK64(19));\n"
"  u64 man2 = (x_man[1] & MASK64(11)) <<53 | (x_man[2]<<23) | (x_man[3]>>7 & MASK64(23));\n"
"  u64 rbit = (x_man[3] & MASK64(7)) == 0x0 ? 0 : 0x1;\n"
"\n"
"  man2 = man2 | rbit;\n"
"  \n"
"  y->d[0] = man2;\n"
"  y->d[1] = x_sign << 63 | e << 48 | man1;\n"
"  \n"
"  return;\n"
"}\n"
"\n"
"void b128_to_fp(const struct b128 x, FP y)\n"
"{\n"
"  u32 x_sign, x_exp;\n"
"  u64 man1, man2;\n"
"\n"
"  x_sign = (x.d[1] >> 63) & MASK32(1);\n"
"  x_exp  = (x.d[1] >> 48) & MASK32(15);\n"
"  \n"
"  if (x_exp == 0x0) {\n"
"    *y = zero_fp;\n"
"    return;\n"
"  }\n"
"\n"
"  man1   = (x.d[1] >>  0) & MASK64(48);\n"
"  man2   = x.d[0];\n"
"\n"
"  i64 e = (((i64)x_exp - bias15) + bias) & MASK64(30);  \n"
"\n"
"  u32 x_man[NC];\n"
"\n"
"  x_man[0] = (0x1 << 29) | ((man1 >> 19) & MASK32(29));\n"
"  x_man[1] = (man1 & MASK32(19))<<11 | ((man2 >> 53) & MASK32(11));\n"
"  x_man[2] = (man2 >> 23) & MASK32(30);\n"
"  x_man[3] = (man2 & MASK32(23)) << 7;\n"
"\n"
"  set_sign(x_sign, y);\n"
"  set_exponent((u32)e, y);\n"
"  \n"
"  for(i32 i = 0; i < (i32)nc; i++) {\n"
"    y->m[i] = x_man[i];\n"
"  }\n"
"  \n"
"  return;\n"
"}\n"
"\n"
"// blocking https://cnugteren.github.io/tutorial/pages/page4.html\n"
"\n"
"#define BLOCKA As[TS][TS]\n"
"#define BLOCKB Bs[TS][TS]\n"
"\n"
"#define STOREBLOCK { As[l_i][l_j] = aa; Bs[l_j][l_i] = bb; }\n"
"#define LOADBLOCK  { aa = As[l_i][p];  bb = Bs[l_j][p]; }\n"
"\n"
"//#define STOREBLOCK { As[l_j][l_i] = aa; Bs[l_j][l_i] = bb; }\n"
"//#define LOADBLOCK  { aa = As[p][l_i];  bb = Bs[l_j][p]; }\n"
"\n"
"//            i   j\n"
"// Matrix A : m * k  \n"
"// Matrix B : k * n\n"
"// Matrix C : m * n\n"
"\n"
"// Column major layout\n"
"//\n"
"// if A is not transposed : shape m-by-k, buffer lda*k\n"
"// if A is transposed     : shape k-by-m, buffer lda*m\n"
"//\n"
"// if B is not transposed : shape k-by-n, buffer ldb*n\n"
"// if B is transposed     : shape n-by-k, buffer ldb*k\n"
"//\n"
"// C : shape m-by-n, buffer ldc*n\n"
"//\n"
"\n"
"// m x k\n"
"#define LOAD_A_N { aa_128 = ((i   < m) && (t_j < k)) ? A[i   + t_j*lda] : zero_b128; }\n"
"// k x m \n"
"#define LOAD_A_T { aa_128 = ((i   < m) && (t_j < k)) ? A[t_j + i*lda] : zero_b128; }\n"
"\n"
"// k x n\n"
"#define LOAD_B_N { bb_128 = ((t_i < k) && (  j < n)) ? B[t_i + j*ldb] : zero_b128; }\n"
"// n x k\n"
"#define LOAD_B_T { bb_128 = ((t_i < k) && (  j < n)) ? B[j + t_i*ldb] : zero_b128; }\n"
"\n"
"//#define STORERESULT { if (i < m && j < n) { struct b128 res; fp_to_b128(&temp, &res); C[i + j * ldc] = res; } }\n"
"\n"
"#define STORERESULT { if (i < m && j < n) { struct b128 ctmp = C[i + j * ldc]; C[i + j * ldc] = store_al_be(C[i + j * ldc], alpha, beta, &temp); } }\n"
"\n"
"struct b128 store_al_be(struct b128 C, const struct b128 alpha, const struct b128 beta, const FP res1)\n"
"{\n"
"  FP Ctmp, altmp, betmp;\n"
"  b128_to_fp(C,     Ctmp);\n"
"  b128_to_fp(alpha, altmp);\n"
"  b128_to_fp(beta,  betmp);\n"
"\n"
"  FP tmp1, tmp2, tmp3;\n"
"  mul(res1, altmp, tmp1);\n"
"  mul(Ctmp, betmp, tmp2);\n"
"  add(tmp1, tmp2, tmp3);\n"
"\n"
"  struct b128 res;\n"
"  fp_to_b128(tmp3, &res);\n"
"\n"
"  return res;\n"
"}\n"
"\n"
"__kernel void gemm_NN_X(const int m, const int n, const int k,\n"
"        	        const struct b128 alpha, \n"
"			__global struct b128 *restrict A, const int lda,\n"
"			__global struct b128 *restrict B, const int ldb,\n"
"			const struct b128 beta, \n"
"			__global struct b128 *restrict C, const int ldc\n"
"		      )\n"
"{\n"
"  const int l_i = get_local_id(0);  // row\n"
"  const int l_j = get_local_id(1);  // col\n"
"  const int i = TS*get_group_id(0) + l_i; //const int i = get_global_id(0);\n"
"  const int j = TS*get_group_id(1) + l_j; //const int j = get_global_id(1);\n"
"\n"
"  __local struct my_fp BLOCKA;\n"
"  __local struct my_fp BLOCKB;\n"
"\n"
"  struct my_fp temp = zero_fp;\n"
"\n"
"  int numtiles = k/TS;\n"
"  if (k % TS != 0) numtiles++;\n"
"  for (int t = 0; t < numtiles; t++) {\n"
"    struct b128 aa_128, bb_128;\n"
"\n"
"    const int t_i = TS*t + l_i;\n"
"    const int t_j = TS*t + l_j;\n"
"\n"
"    LOAD_A_N;\n"
"    LOAD_B_N;\n"
"    \n"
"    struct my_fp aa, bb;\n"
"    b128_to_fp(aa_128, &aa);\n"
"    b128_to_fp(bb_128, &bb);\n"
"\n"
"    STOREBLOCK;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for(int p = 0; p < TS; p++) {\n"
"      struct my_fp aa, bb, tmp;\n"
"\n"
"      LOADBLOCK;\n"
"      \n"
"      mul(&aa, &bb, &tmp);\n"
"      add(&tmp, &temp, &temp);\n"
"    }\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"\n"
"  STORERESULT;\n"
"}\n"
"\n"
"__kernel void gemm_TN_X(const int m, const int n, const int k,\n"
"        	        const struct b128 alpha, \n"
"			__global struct b128 *restrict A, const int lda,\n"
"			__global struct b128 *restrict B, const int ldb,\n"
"			const struct b128 beta, \n"
"			__global struct b128 *restrict C, const int ldc\n"
"		      )\n"
"{\n"
"  const int l_i = get_local_id(0);  // row\n"
"  const int l_j = get_local_id(1);  // col\n"
"  const int i = TS*get_group_id(0) + l_i; //const int i = get_global_id(0);\n"
"  const int j = TS*get_group_id(1) + l_j; //const int j = get_global_id(1);\n"
"\n"
"  __local struct my_fp BLOCKA;\n"
"  __local struct my_fp BLOCKB;\n"
"\n"
"  struct my_fp temp = zero_fp;\n"
"\n"
"  int numtiles = k/TS;\n"
"  if (k % TS != 0) numtiles++;\n"
"  for (int t = 0; t < numtiles; t++) {\n"
"    struct b128 aa_128, bb_128;\n"
"\n"
"    const int t_i = TS*t + l_i;\n"
"    const int t_j = TS*t + l_j;\n"
"\n"
"    LOAD_A_T;\n"
"    LOAD_B_N;\n"
"    \n"
"    struct my_fp aa, bb;\n"
"    b128_to_fp(aa_128, &aa);\n"
"    b128_to_fp(bb_128, &bb);\n"
"\n"
"    STOREBLOCK;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for(int p = 0; p < TS; p++) {\n"
"      struct my_fp aa, bb, tmp;\n"
"\n"
"      LOADBLOCK;\n"
"\n"
"      mul(&aa, &bb, &tmp);\n"
"      add(&tmp, &temp, &temp);\n"
"    }\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"\n"
"  STORERESULT;\n"
"}\n"
"\n"
"__kernel void gemm_NT_X(const int m, const int n, const int k,\n"
"        	        const struct b128 alpha, \n"
"			__global struct b128 *restrict A, const int lda,\n"
"			__global struct b128 *restrict B, const int ldb,\n"
"			const struct b128 beta, \n"
"			__global struct b128 *restrict C, const int ldc\n"
"		      )\n"
"{\n"
"  const int l_i = get_local_id(0);  // row\n"
"  const int l_j = get_local_id(1);  // col\n"
"  const int i = TS*get_group_id(0) + l_i; //const int i = get_global_id(0);\n"
"  const int j = TS*get_group_id(1) + l_j; //const int j = get_global_id(1);\n"
"\n"
"  __local struct my_fp BLOCKA;\n"
"  __local struct my_fp BLOCKB;\n"
"\n"
"  struct my_fp temp = zero_fp;\n"
"\n"
"  int numtiles = k/TS;\n"
"  if (k % TS != 0) numtiles++;\n"
"  for (int t = 0; t < numtiles; t++) {\n"
"    struct b128 aa_128, bb_128;\n"
"\n"
"    const int t_i = TS*t + l_i;\n"
"    const int t_j = TS*t + l_j;\n"
"\n"
"    LOAD_A_N;\n"
"    LOAD_B_T;\n"
"    \n"
"    struct my_fp aa, bb;\n"
"    b128_to_fp(aa_128, &aa);\n"
"    b128_to_fp(bb_128, &bb);\n"
"\n"
"    STOREBLOCK;  \n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for(int p = 0; p < TS; p++) {\n"
"      struct my_fp aa, bb, tmp;\n"
"\n"
"      LOADBLOCK;\n"
"      \n"
"      mul(&aa, &bb, &tmp);\n"
"      add(&tmp, &temp, &temp);\n"
"    }\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"\n"
"  STORERESULT;\n"
"}\n"
"\n"
"__kernel void gemm_TT_X(const int m, const int n, const int k,\n"
"        	        const struct b128 alpha, \n"
"			__global struct b128 *restrict A, const int lda,\n"
"			__global struct b128 *restrict B, const int ldb,\n"
"			const struct b128 beta, \n"
"			__global struct b128 *restrict C, const int ldc\n"
"			)\n"
"{\n"
"  const int l_i = get_local_id(0);  // row\n"
"  const int l_j = get_local_id(1);  // col\n"
"  const int i = TS*get_group_id(0) + l_i; //const int i = get_global_id(0);\n"
"  const int j = TS*get_group_id(1) + l_j; //const int j = get_global_id(1);\n"
"\n"
"  __local struct my_fp BLOCKA;\n"
"  __local struct my_fp BLOCKB;\n"
"\n"
"  struct my_fp temp = zero_fp;\n"
"\n"
"  int numtiles = k/TS;\n"
"  if (k % TS != 0) numtiles++;\n"
"  for (int t = 0; t < numtiles; t++) {\n"
"    struct b128 aa_128, bb_128;\n"
"\n"
"    const int t_i = TS*t + l_i;\n"
"    const int t_j = TS*t + l_j;\n"
"\n"
"    LOAD_A_T;\n"
"    LOAD_B_T;\n"
"    \n"
"    struct my_fp aa, bb;\n"
"    b128_to_fp(aa_128, &aa);\n"
"    b128_to_fp(bb_128, &bb);\n"
"\n"
"    STOREBLOCK; \n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for(int p = 0; p < TS; p++) {\n"
"      struct my_fp aa, bb, tmp;\n"
"\n"
"      LOADBLOCK;\n"
"\n"
"      mul(&aa, &bb, &tmp);\n"
"      add(&tmp, &temp, &temp);\n"
"    }\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"\n"
"  STORERESULT;\n"
"}\n"
"\n"
"    \n"
;


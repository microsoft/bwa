/*++

Module Name:

    ksw_align_avx2.cpp

Abstract:

    KSW Align AVX2 implementation.

Authors:

    Roman Snytsar, June, 2018

Environment:
`
    User mode service.

Revision History:


--*/

// Turn off the MSC warnings on standard crt 'unsafe' functions
#ifdef _DEBUG
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>

#define UNIT_TEST_ALIGN
//#define UNIT_DUMP_ALIGN
#endif

#include <assert.h>

#include "ksw_perf.h"
#include "ksw_avx.h"

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

//
// AVX2-vectorized implementations.
//

class kswq {
public:
    int qlen, slen;
    uint8_t shift, mdiff, max, size;
private:
    __m256i *qp;
    __m256i *H;

public:
    kswq(int size, int qlen, const uint8_t *query, int m, const int8_t *mat);
    ~kswq();

    __m256i* S(uint8_t t)
    {
        return (qp + t * slen);
    }

    __m256i* H0()
    {
        return (H);
    }

    __m256i* H1()
    {
        return (H + slen);
    }

    __m256i* E()
    {
        return (H + 2 * slen);
    }

    __m256i* Hmax()
    {
        return (H + 3 * slen);
    }
};

kswq::kswq(int size, int qlen, const uint8_t *query, int m, const int8_t *mat)
{
    int slen, a, tmp, p;

    size = size > 1 ? 2 : 1;
    p = 16 * (3 - size); // # values per __m256i
    slen = (qlen + p - 1) / p; // segmented length
    this->qp = (__m256i*) _mm_malloc(sizeof(__m256i) * slen * (m + 4), 64);
    this->H = this->qp + slen * m;
    this->slen = slen; this->qlen = qlen; this->size = size;
    // compute shift
    tmp = m * m;
    for (a = 0, this->shift = 127, this->mdiff = 0; a < tmp; ++a) { // find the minimum and maximum score
        if (mat[a] < (int8_t)this->shift) this->shift = mat[a];
        if (mat[a] > (int8_t)this->mdiff) this->mdiff = mat[a];
    }
    this->max = this->mdiff;
    this->shift = 256 - this->shift; // NB: q->shift is uint8_t
    this->mdiff += this->shift; // this is the difference between the min and max scores
                                // An example: p=8, qlen=19, slen=3 and segmentation:
                                //  {{0,3,6,9,12,15,18,-1},{1,4,7,10,13,16,-1,-1},{2,5,8,11,14,17,-1,-1}}
    if (size == 1) {
        int8_t *t = (int8_t*)this->qp;
        for (a = 0; a < m; ++a) {
            int i, k, nlen = slen * p;
            const int8_t *ma = mat + a * m;
            for (i = 0; i < slen; ++i)
                for (k = i; k < nlen; k += slen) // p iterations
                    *t++ = (k >= qlen ? 0 : ma[query[k]]) + this->shift;
        }
    }
    else {
        int16_t *t = (int16_t*)this->qp;
        for (a = 0; a < m; ++a) {
            int i, k, nlen = slen * p;
            const int8_t *ma = mat + a * m;
            for (i = 0; i < slen; ++i)
                for (k = i; k < nlen; k += slen) // p iterations
                    *t++ = (k >= qlen ? 0 : ma[query[k]]);
        }
    }
}

kswq::~kswq()
{
    _mm_free(qp);
}

const kswr_t g_defr = { 0, -1, -1, -1, -1, -1, -1 };

__forceinline __m256i _mm256_mask_qlen_epu8(const __m256i _h, const __m256i _idx, const __m256i _qlen)
{
    __m256i _ret = _mm256_andnot_si256(// mask out-of-range values
        _mm256_cmpeq_epi8(//emulate unsigned comparison
            _idx,
            _mm256_max_epu8(
                _qlen,
                _idx)),
        _h);

    return _ret;
}

__forceinline void _mm256_update_max_qe_epu8(__m256i &_max, __m256i &_qe, const __m256i _h, const __m256i _idx)
{
    __m256i _cmp0 = _mm256_cmpeq_epi8(_max, _h);

    _max = _mm256_max_epu8(_max, _h);

    __m256i _cmp1 = _mm256_cmpeq_epi8(_max, _h);

    _qe = _mm256_blendv_epi8(
        _qe,
        _mm256_blendv_epi8(
            _mm256_min_epu8(
                _idx,
                _qe),
            _idx,
            _mm256_andnot_si256(
                _cmp0,
                _cmp1)),
        _cmp1);
}

kswr_t ksw_u8(kswq &q, int tlen, const uint8_t *target, int gapo, int gape, int xtra) // the first gap costs -(_o+_e)
{
    int te = -1, gmax = 0, imax = 0;
    __m256i *H0, *H1, *E, *Hmax;

    // initialization
    int slen = q.slen;
    kswr_t r = g_defr;
    int minsc = (xtra&KSW_XSUBO) ? xtra & 0xffff : 0x10000;
    int endsc = (xtra&KSW_XSTOP) ? xtra & 0xffff : 0x10000;
    int n_b = 0;
    uint64_t *b = nullptr;
    __m256i _zero = _mm256_setzero_si256();
    __m256i _shift = _mm256_set1_epi8(q.shift);
    __m256i _gapoe = _mm256_set1_epi8(gapo + gape);
    __m256i _gape = _mm256_set1_epi8(gape);

    __m256i _gape_x_slen = _mm256_set1_epi8(gape * slen);

    __m256i _gape_x_slen_m_1 = _mm256_subs_epu8(
        _gape_x_slen,
        _gape);

    __m256i _gape_scan = _mm256_exclusive_add_epu8(_gape_x_slen);

    H0 = q.H0(); H1 = q.H1();
    E = q.E(); Hmax = q.Hmax();

    __m256i _f_scan = _zero;

    for (int j = 0; j < slen; ++j) {
        _mm256_store_si256(E + j, _zero);
        _mm256_store_si256(H0 + j, _zero);
        _mm256_store_si256(Hmax + j, _zero);
    }

    __m256i _one = _mm256_set1_epi8(1);
    __m256i _qlen = _mm256_set1_epi8(q.qlen);

    // prefix sum the slen
    __m256i _idx0 = _mm256_set1_epi8(slen);;
    _idx0 = _mm256_adds_epu8(_idx0, _mm256_shift_left_si256<1>(_idx0));
    _idx0 = _mm256_adds_epu8(_idx0, _mm256_shift_left_si256<2>(_idx0));
    _idx0 = _mm256_adds_epu8(_idx0, _mm256_shift_left_si256<4>(_idx0));
    _idx0 = _mm256_adds_epu8(_idx0, _mm256_shift_left_si256<8>(_idx0));
    _idx0 = _mm256_adds_epu8(_idx0, _mm256_permute2x128_si256(_idx0, _idx0, 0x08));

    // make it exclusive
    _idx0 = _mm256_shift_left_si256<1>(_idx0);

    // the core loop
    for (int i = 0; i < tlen; ++i) {
        __m256i f = _zero, max = _zero;
        __m256i *S = q.S(target[i]); // s is the 1st score vector
        __m256i _idx = _idx0;

        __m256i h = _mm256_load_si256(H0 + slen - 1); // h={2,5,8,11,14,17,-1,-1} in the above example
        h = _mm256_max_epu8(h,
            _mm256_subs_epu8(
                _f_scan,
                _gape_x_slen_m_1));

        h = _mm256_shift_left_si256<1>(h); // h=H(i-1,-1); << instead of >> because x64 is little-endian

        for (int j = 0; LIKELY(j < slen); ++j) {
            /* SW cells are computed in the following order:
            *   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
            *   E(i+1,j) = max{H(i,j)-q, E(i,j)-r}
            *   F(i,j+1) = max{H(i,j)-q, F(i,j)-r}
            */
            // compute H'(i,j); note that at the beginning, h=H'(i-1,j-1)
            h = _mm256_adds_epu8(h, _mm256_load_si256(S + j));
            h = _mm256_subs_epu8(h, _shift); // h=H'(i-1,j-1)+S(i,j)

            __m256i e = _mm256_load_si256(E + j); // e=E'(i,j)
            h = _mm256_max_epu8(h, e);
            h = _mm256_max_epu8(h, f); // h=H'(i,j)

            max = _mm256_max_epu8( // set max
                max,
                _mm256_mask_qlen_epu8(
                    h,
                    _idx,
                    _qlen));

            _mm256_store_si256(H1 + j, h); // save to H'(i,j)

            __m256i t = _mm256_subs_epu8(h, _gapoe); // h=H'(i,j) - o_del - e_del

            // now compute E'(i+1,j)
            e = _mm256_subs_epu8(e, _gape); // e=E'(i,j) - e_del
            e = _mm256_max_epu8(e, t); // e=E'(i+1,j)
            _mm256_store_si256(E + j, e); // save to E'(i+1,j)

            // now compute F'(i,j+1)
            f = _mm256_subs_epu8(f, _gape);
            f = _mm256_max_epu8(f, t);

            // get H'(i-1,j) and prepare for the next j
            h = _mm256_load_si256(H0 + j); // h=H'(i-1,j)

            // Lazy-F update
            h = _mm256_max_epu8(h, _f_scan); // h=H'(i,j)

            _f_scan = _mm256_subs_epu8(
                _f_scan,
                _gape);

            _idx = _mm256_adds_epu8(_idx, _one);
        }

        // Prefix Scan
        _f_scan = prefix_scan_F_u8(f, _gape_x_slen, _gape_scan);

        if (imax > gmax) {
            // te is the end position on the target
            gmax = imax; te = i - 1;

            // keep the H0 vector as maximun candidate
            __m256i *Htemp = H0; H0 = Hmax; Hmax = Htemp;

            if (gmax + q.shift >= 255 || gmax >= endsc)
                break;
        }

        // imax is the maximum number in max
        imax = _mm_extract_epi32(
            _mm_cvtepu8_epi32(
                _mm256_hmax_epu8(
                    max)),
            0);

        if (imax >= minsc) { // write the b array; this condition adds branching unfortunately
            if (n_b == 0 || (int32_t)b[n_b - 1] + 1 != i) { // then append
                if (n_b == 0) {
                    b = (uint64_t*)_mm_malloc(tlen * sizeof(uint64_t), 64);
                }
                b[n_b++] = (uint64_t)imax << 32 | i;
            }
            else
            {
                if ((int)(b[n_b - 1] >> 32) < imax)
                    b[n_b - 1] = (uint64_t)imax << 32 | i; // modify the last
            }
        }

        __m256i *Hnext = H0;
        H0 = H1; H1 = Hnext; // roll H0 and H1
    }

    // check the last row for max
    if (imax > gmax) {
        // te is the end position on the target
        gmax = imax; te = tlen - 1;

        // keep the H0 vector as maximun candidate
        Hmax = H0;
    }

    r.te = te;

    r.score = gmax + q.shift < 255 ? gmax : 255;
    if (r.score != 255)
    {
        // get a qe, the end of query match
        __m256i _idx = _idx0, _max = _zero, _qe = _zero;

        // Loop2
        for (int j = 0; LIKELY(j < slen); ++j) {
            __m256i _h = _mm256_load_si256(Hmax + j);

            _h = _mm256_mask_qlen_epu8(_h, _idx, _qlen);

            _mm256_update_max_qe_epu8(_max, _qe, _h, _idx);

            _idx = _mm256_adds_epu8(_idx, _one);
        }

        // Reduce2
        _mm256_update_max_qe_epu8(
            _max,
            _qe,
            _mm256_srli_si256(_max, 1),
            _mm256_srli_si256(_qe, 1));

        _mm256_update_max_qe_epu8(
            _max,
            _qe,
            _mm256_srli_si256(_max, 2),
            _mm256_srli_si256(_qe, 2));

        _mm256_update_max_qe_epu8(
            _max,
            _qe,
            _mm256_srli_si256(_max, 4),
            _mm256_srli_si256(_qe, 4));

        _mm256_update_max_qe_epu8(
            _max,
            _qe,
            _mm256_srli_si256(_max, 8),
            _mm256_srli_si256(_qe, 8));

        _mm256_update_max_qe_epu8(
            _max,
            _qe,
            _mm256_permute2x128_si256(_max, _max, 0x81),
            _mm256_permute2x128_si256(_qe, _qe, 0x81));

        r.qe = _mm_extract_epi32(
            _mm_cvtepu8_epi32(
                _mm256_castsi256_si128(
                    _qe)),
            0);

        // find the 2nd best score
        if (b != nullptr) {
            int delta = (r.score + q.max - 1) / q.max;
            int low = te - delta, high = te + delta;
            for (int i = 0; i < n_b; ++i)
            {
                int e = (int32_t)b[i];
                if ((e < low || e > high) && (int)(b[i] >> 32) > r.score2)
                {
                    r.score2 = b[i] >> 32;
                    r.te2 = e;
                }
            }
        }
    }

    if (b != nullptr)
        _mm_free(b);

    return r;
}

__forceinline __m256i _mm256_mask_qlen_epi16(const __m256i _h, const __m256i _idx, const __m256i _qlen)
{
    __m256i _ret = _mm256_and_si256(
        _mm256_cmpgt_epi16(
            _qlen,
            _idx),
        _h);

    return _ret;
}

__forceinline void _mm256_update_max_qe_epi16(__m256i &_max, __m256i &_qe, const __m256i _h, const __m256i _idx)
{
    __m256i _cmp0 = _mm256_cmpeq_epi16(_max, _h);

    _max = _mm256_max_epi16(_max, _h);

    __m256i _cmp1 = _mm256_cmpeq_epi16(_max, _h);

    _qe = _mm256_blendv_epi8(
        _qe,
        _mm256_blendv_epi8(
            _mm256_min_epi16(
                _idx,
                _qe),
            _idx,
            _mm256_andnot_si256(
                _cmp0,
                _cmp1)),
        _cmp1);
}

kswr_t ksw_i16(kswq &q, int tlen, const uint8_t *target, int gapo, int gape, int xtra) // the first gap costs -(_o+_e)
{
    int te = -1, gmax = 0, imax = 0;
    __m256i *H0, *H1, *E, *Hmax;

    // initialization
    int slen = q.slen;
    kswr_t r = g_defr;
    int minsc = (xtra&KSW_XSUBO) ? xtra & 0xffff : 0x10000;
    int endsc = (xtra&KSW_XSTOP) ? xtra & 0xffff : 0x10000;
    int n_b = 0;
    uint64_t *b = nullptr;
    __m256i zero = _mm256_setzero_si256();
    __m256i _gapoe = _mm256_set1_epi16(gapo + gape);
    __m256i _gape = _mm256_set1_epi16(gape);

    __m256i _slen = _mm256_set1_epi16(slen);

    __m256i _gape_x_slen = _mm256_mullo_epi16(
        _gape,
        _slen);

    __m256i _gape_x_slen_m_1 = _mm256_subs_epi16(
        _gape_x_slen,
        _gape);

    __m256i _gape_scan = _mm256_exclusive_add_epi16(_gape_x_slen);

    H0 = q.H0(); H1 = q.H1();
    E = q.E(); Hmax = q.Hmax();

    __m256i _f_scan = _mm256_setzero_si256();

    for (int j = 0; j < slen; ++j) {
        _mm256_store_si256(E + j, zero);
        _mm256_store_si256(H0 + j, zero);
        _mm256_store_si256(Hmax + j, zero);
    }

    __m256i _one = _mm256_set1_epi16(1);
    __m256i _qlen = _mm256_set1_epi16(q.qlen);

    // prefix sum the slen
    __m256i _idx0 = _mm256_set1_epi16(slen);;
    _idx0 = _mm256_adds_epu16(_idx0, _mm256_shift_left_si256<2>(_idx0));
    _idx0 = _mm256_adds_epu16(_idx0, _mm256_shift_left_si256<4>(_idx0));
    _idx0 = _mm256_adds_epu16(_idx0, _mm256_shift_left_si256<8>(_idx0));
    _idx0 = _mm256_adds_epu16(_idx0, _mm256_permute2x128_si256(_idx0, _idx0, 0x08));

    // make it exclusive
    _idx0 = _mm256_shift_left_si256<2>(_idx0);

#ifdef UNIT_DUMP_ALIGN
    FILE* fTable = fopen("ksw_align_avx2.csv", "w");
    assert(fTable != NULL);
#endif // UNIT_DUMP_ALIGN

    // the core loop
    for (int i = 0; i < tlen; ++i) {
        __m256i f = zero, max = zero;
        __m256i *S = q.S(target[i]); // s is the 1st score vector
        __m256i _idx = _idx0;

        __m256i h = _mm256_load_si256(H0 + slen - 1); // h={2,5,8,11,14,17,-1,-1} in the above example
        h = _mm256_max_epu16(h,
            _mm256_subs_epu16(
                _f_scan,
                _gape_x_slen_m_1));

        h = _mm256_shift_left_si256<2>(h);
        for (int j = 0; LIKELY(j < slen); ++j) {
            h = _mm256_adds_epi16(h, _mm256_load_si256(S + j));
            __m256i e = _mm256_load_si256(E + j);
            h = _mm256_max_epi16(h, e);
            h = _mm256_max_epi16(h, f);

            max = _mm256_max_epi16(
                max,
                _mm256_mask_qlen_epi16(
                    h,
                    _idx,
                    _qlen));

            _mm256_store_si256(H1 + j, h);

            __m256i t = _mm256_subs_epu16(h, _gapoe);

            e = _mm256_subs_epu16(e, _gape);
            e = _mm256_max_epi16(e, t);
            _mm256_store_si256(E + j, e);

            f = _mm256_subs_epu16(f, _gape);
            f = _mm256_max_epi16(f, t);

            // get H'(i-1,j) and prepare for the next j
            h = _mm256_load_si256(H0 + j); // h=H'(i-1,j)

            // Lazy-F update
            h = _mm256_max_epu8(h, _f_scan); // h=H'(i,j)
            _f_scan = _mm256_subs_epu16(
                _f_scan,
                _gape);

#ifdef UNIT_DUMP_ALIGN
            for (size_t cc = 0; cc < 16; cc++)
            {
                fprintf(fTable, "%d, ", h.m256i_i16[cc]);
        }
#endif // UNIT_DUMP_ALIGN

            _idx = _mm256_adds_epu16(_idx, _one);
        }
#ifdef UNIT_DUMP_ALIGN
        fprintf(fTable, "\n\n");

        for (int j = 0; LIKELY(j < slen); ++j)
        {
            h = _mm256_load_si256(H1 + j);
            for (size_t cc = 0; cc < 16; cc++)
            {
                fprintf(fTable, "%d, ", h.m256i_i16[cc]);
            }
    }
        fprintf(fTable, "\n");
#endif // UNIT_DUMP_ALIGN

        // Prefix Scan
        _f_scan = prefix_scan_F_i16(f, _gape_x_slen, _gape_scan);

        if (imax > gmax) {
            // te is the end position on the target
            gmax = imax; te = i - 1;

            // keep the H0 vector as maximun candidate
            __m256i *Htemp = H0; H0 = Hmax; Hmax = Htemp;

            if (gmax >= endsc)
                break;
        }

        imax = _mm_extract_epi32(
            _mm_cvtepi16_epi32(
                _mm256_hmax_epi16(
                    max)),
            0);

        if (imax >= minsc) {
            if (n_b == 0 || (int32_t)b[n_b - 1] + 1 != i) {
                if (n_b == 0) {
                    b = (uint64_t*)_mm_malloc(tlen * sizeof(uint64_t), 64);
                }
                b[n_b++] = (uint64_t)imax << 32 | i;
            }
            else if ((int)(b[n_b - 1] >> 32) < imax) b[n_b - 1] = (uint64_t)imax << 32 | i; // modify the last
        }

        __m256i *Hnext = H0;
        H0 = H1; H1 = Hnext; // roll H0 and H1
}

    // check the last row for max
    if (imax > gmax) {
        // te is the end position on the target
        gmax = imax; te = tlen - 1;

        // keep the H0 vector as maximun candidate
        Hmax = H0;
    }

#ifdef UNIT_DUMP_ALIGN
    fclose(fTable);
#endif // UNIT_DUMP_ALIGN

    r.score = gmax; r.te = te;
    {
        // get a qe, the end of query match
        __m256i _idx = _idx0, _max = _mm256_setzero_si256(), _qe = _mm256_setzero_si256();

        // Loop2
        for (int j = 0; LIKELY(j < slen); ++j) {
            __m256i _h = _mm256_load_si256(Hmax + j);

            _h = _mm256_mask_qlen_epi16(_h, _idx, _qlen);

            _mm256_update_max_qe_epi16(_max, _qe, _h, _idx);

            _idx = _mm256_adds_epu16(_idx, _one);
        }

        // Reduce2
        _mm256_update_max_qe_epi16(
            _max,
            _qe,
            _mm256_srli_si256(_max, 2),
            _mm256_srli_si256(_qe, 2));

        _mm256_update_max_qe_epi16(
            _max,
            _qe,
            _mm256_srli_si256(_max, 4),
            _mm256_srli_si256(_qe, 4));

        _mm256_update_max_qe_epi16(
            _max,
            _qe,
            _mm256_srli_si256(_max, 8),
            _mm256_srli_si256(_qe, 8));

        _mm256_update_max_qe_epi16(
            _max,
            _qe,
            _mm256_permute2x128_si256(_max, _max, 0x81),
            _mm256_permute2x128_si256(_qe, _qe, 0x81));

        r.qe = _mm_extract_epi32(
            _mm_cvtepi16_epi32(
                _mm256_castsi256_si128(
                    _qe)),
            0);

        // find the 2nd best score
        if (b != nullptr)
        {
            int delta = (r.score + q.max - 1) / q.max;
            int low = te - delta, high = te + delta;
            for (int i = 0; i < n_b; ++i)
            {
                int e = (int32_t)b[i];
                if ((e < low || e > high) && (int)(b[i] >> 32) > r.score2)
                {
                    r.score2 = b[i] >> 32, r.te2 = e;
                }
            }
        }
    }

    if (b != nullptr)
        _mm_free(b);

    return r;
    }

static inline void revseq(int l, uint8_t *s)
{
    int i, t;
    for (i = 0; i < l >> 1; ++i)
        t = s[i], s[i] = s[l - 1 - i], s[l - 1 - i] = t;
}

kswr_t ksw_align_avx2(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int gapo, int gape, int xtra, kswq_t **qry)
{
    assert(qry == NULL);

    int size = (xtra&KSW_XBYTE) ? 1 : 2;

    kswr_t(*func)(kswq&, int, const uint8_t*, int, int, int) = (size == 2) ? ksw_i16 : ksw_u8;

    kswq q(size, qlen, query, m, mat);

    kswr_t r = func(q, tlen, target, gapo, gape, xtra);

    if (!((xtra&KSW_XSTART) == 0 || ((xtra&KSW_XSUBO) && r.score < (xtra & 0xffff))))
    {
        revseq(r.qe + 1, query); revseq(r.te + 1, target); // +1 because qe/te points to the exact end, not the position after the end

        kswq qq(size, r.qe + 1, query, 5, mat);
        kswr_t rr = func(qq, tlen, target, gapo, gape, KSW_XSTOP | r.score);

        revseq(r.qe + 1, query); revseq(r.te + 1, target);

        if (r.score == rr.score)
        {
            r.tb = r.te - rr.te;
            r.qb = r.qe - rr.qe;
        }
    }

#ifdef UNIT_TEST_ALIGN
    kswr_t r2 = ksw_align(qlen, query, tlen, target, m, mat, gapo, gape, xtra, qry);

    assert(r.qb == r2.qb);
    assert(r.qe == r2.qe);
    assert(r.score == r2.score);
    if ((r.score2 != r2.score2) || (r.te2 != r2.te2))
    {
        r2 = ksw_align(qlen, query, tlen, target, m, mat, gapo, gape, xtra, qry);

        r = func(q, tlen, target, gapo, gape, xtra);

        if (!((xtra&KSW_XSTART) == 0 || ((xtra&KSW_XSUBO) && r.score < (xtra & 0xffff))))
        {
            revseq(r.qe + 1, query); revseq(r.te + 1, target); // +1 because qe/te points to the exact end, not the position after the end

            kswq qq(size, r.qe + 1, query, 5, mat);
            kswr_t rr = func(qq, tlen, target, gapo, gape, KSW_XSTOP | r.score);

            revseq(r.qe + 1, query); revseq(r.te + 1, target);

            if (r.score == rr.score)
            {
                r.tb = r.te - rr.te;
                r.qb = r.qe - rr.qe;
            }
        }
    }
    assert(r.score2 == r2.score2);
    assert(r.te2 == r2.te2);
    assert(r.tb == r2.tb);
    assert(r.te == r2.te);
#endif

    return r;
}

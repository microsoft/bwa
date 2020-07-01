/*++

Module Name:

    ksw_extend_avx2.cpp

Abstract:

    KSW Extend AVX2 implementation.

Authors:

    Roman Snytsar, June, 2018

Environment:
`
    User mode service.

Revision History:


--*/

#include <cassert>
#include <cstdlib>
#include <cstring>

// Turn off the MSC warnings on standard crt 'unsafe' functions
#ifdef _DEBUG
#define _CRT_SECURE_NO_WARNINGS
#define UNIT_TEST_EXTEND
#include "stdio.h"
#endif // _DEBUG

#ifdef USE_MALLOC_WRAPPERS
# include "malloc_wrap.h"
#endif

#include "ksw_perf.h"
#include "ksw_avx.h"

__forceinline __m128i _mm_prefix_add_epi8(const __m128i _base)
{
    __m128i _prefix_sum = _base;

    _prefix_sum = _mm_add_epi8(
        _mm_slli_si128(_prefix_sum, 1),
        _prefix_sum);

    _prefix_sum = _mm_add_epi8(
        _mm_slli_si128(_prefix_sum, 2),
        _prefix_sum);

    _prefix_sum = _mm_add_epi8(
        _mm_slli_si128(_prefix_sum, 4),
        _prefix_sum);

    _prefix_sum = _mm_add_epi8(
        _mm_slli_si128(_prefix_sum, 8),
        _prefix_sum);

    return _prefix_sum;
}

__forceinline __m256i _mm256_prefix_add_epi16(const __m256i _base)
{
    __m256i _prefix_sum = _base;

    _prefix_sum = _mm256_add_epi16(
        _mm256_shift_left_si256<2>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi16(
        _mm256_shift_left_si256<4>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi16(
        _mm256_shift_left_si256<8>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi16(
        _mm256_permute2x128_si256(_prefix_sum, _prefix_sum, 0x8),
        _prefix_sum);

    return _prefix_sum;
}

__forceinline __m256i _mm256_prefix_add_epi8(const __m256i _base)
{
    __m256i _prefix_sum = _base;

    _prefix_sum = _mm256_add_epi8(
        _mm256_shift_left_si256<1>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi8(
        _mm256_shift_left_si256<2>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi8(
        _mm256_shift_left_si256<4>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi8(
        _mm256_shift_left_si256<8>(_prefix_sum),
        _prefix_sum);

    _prefix_sum = _mm256_add_epi8(
        _mm256_permute2x128_si256(_prefix_sum, _prefix_sum, 0x8),
        _prefix_sum);

    return _prefix_sum;
}

__forceinline void update_global_score_i16(int& gscore, int& max_ie, __m256i _gscore, size_t bandIndex)
{
    __m256i _max = _gscore;

    _max = _mm256_max_epi16(_max, _mm256_srli_si256(_max, 2));
    _max = _mm256_max_epi16(_max, _mm256_srli_si256(_max, 4));
    _max = _mm256_max_epi16(_max, _mm256_srli_si256(_max, 8));
    _max = _mm256_max_epi16(_max, _mm256_permute2x128_si256(_max, _max, 0x01));

    int max_score_candidate = _mm256_extract_epi16(_max, 0);

    if (max_score_candidate > gscore)
    {
        gscore = max_score_candidate;

        int mask = _mm256_movemask_epi8(
            _mm256_cmpeq_epi16(
                _gscore,
                _mm256_set1_epi16(
                    gscore)));

        int max_index = (31 - __lzcnt(mask)) >> 1;

        max_ie = ((int)bandIndex << 4) + max_index;
    }
}


__forceinline void update_global_score_u8(int& gscore, int& max_ie, __m256i _gscore, size_t bandIndex)
{
    __m256i _max = _gscore;

    _max = _mm256_max_epu8(_max, _mm256_srli_si256(_max, 1));
    _max = _mm256_max_epu8(_max, _mm256_srli_si256(_max, 2));
    _max = _mm256_max_epu8(_max, _mm256_srli_si256(_max, 4));
    _max = _mm256_max_epu8(_max, _mm256_srli_si256(_max, 8));
    _max = _mm256_max_epu8(_max, _mm256_permute2x128_si256(_max, _max, 0x01));

    int max_score_candidate = _mm256_extract_epi8(_max, 0);

    if (max_score_candidate > gscore)
    {
        gscore = max_score_candidate;

        int mask = _mm256_movemask_epi8(
            _mm256_cmpeq_epi8(
                _gscore,
                _mm256_broadcastb_epi8(
                    _mm256_castsi256_si128(
                        _max))));

        int max_index = 31 - __lzcnt(mask);

        max_ie = ((int)bandIndex << 5) + max_index;
    }
}

__forceinline __m256i separate_M_i16(const __m128i _score_matrix, const __m256i _H, const __m128i _query, const __m128i _target)
{
    __m256i _M = advance_H_i16(_score_matrix, _H, _query, _target);

    _M = _mm256_max_epi16(_M, _mm256_setzero_si256());
    _M = _mm256_andnot_si256(
        _mm256_cmpeq_epi16(_H, _mm256_setzero_si256()),
        _M);

    return _M;
}

__forceinline __m128i separate_M_u8(const __m128i _score_matrix, const __m128i _score_shift, const __m128i _H, const __m128i _query, const __m128i _target)
{
    __m128i _M = advance_H_u8(_score_matrix, _score_shift, _H, _query, _target);

    _M = _mm_andnot_si128(
        _mm_cmpeq_epi8(_H, _mm_setzero_si128()),
        _M);

    return _M;
}

__forceinline __m256i separate_M_u8(const __m256i _score_matrix, const __m256i _score_shift, const __m256i _H, const __m256i _query, const __m256i _target)
{
    __m256i _M = advance_H_u8(_score_matrix, _score_shift, _H, _query, _target);

    _M = _mm256_andnot_si256(
        _mm256_cmpeq_epi8(_H, _mm256_setzero_si256()),
        _M);

    return _M;
}

#ifdef UNIT_TEST_EXTEND

typedef struct {
    int32_t h, e;
} eh_t;

int* ksw_extend_debug(int qlen, const uint8_t* query, int tlen, const uint8_t* target, __m256i _score_matrix, int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0, int& d_max, int& d_max_i, int& d_max_j, int& d_max_ie, int& d_gscore, int& d_max_off)
{
    eh_t* eh; // score array
    int i, j;

    // allocate memory
    int* pMem = (int*)malloc(6ull * qlen * tlen * sizeof(int));

    eh = (eh_t*)calloc(qlen + 1ull, 8);

    // fill the first row
    int oe_ins = o_ins + e_ins, oe_del = o_del + e_del;
    eh[0].h = h0; eh[1].h = h0 > oe_ins ? h0 - oe_ins : 0;
    for (j = 2; j <= qlen && eh[j - 1].h > e_ins; ++j)
        eh[j].h = eh[j - 1].h - e_ins;
    // adjust $w if it is too large

    // DP loop
    d_max = h0, d_max_i = d_max_j = -1; d_max_ie = -1, d_gscore = -1;
    d_max_off = 0;

    int* pM = pMem;

    bool foundZeroRow = false;

    for (i = 0; LIKELY(i < tlen); ++i) {
        int t, f = 0, h1, m = 0, mj = -1;

        // compute the first column
        h1 = h0 - (o_del + e_del * (i + 1));
        if (h1 < 0) h1 = 0;

        for (j = 0; LIKELY(j < qlen); ++j) {
            // At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
            // Similar to SSE2-SW, cells are computed in the following order:
            //   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
            //   E(i+1,j) = max{H(i,j)-gapo, E(i,j)} - gape
            //   F(i,j+1) = max{H(i,j)-gapo, F(i,j)} - gape
            eh_t* p = &eh[j];
            int h, M = p->h, e = p->e; // get H(i-1,j-1) and E(i-1,j)
            p->h = h1;           // set H(i,j-1) for the next row
            char s = ((target[i] <= 3) && (query[j] <= 3)) ? _score_matrix.m256i_i8[target[i] * 4 + query[j]] : -1;
            *pM++ = query[j];
            *pM++ = s;
            M = M ? M + s : 0;// separating H and M to disallow a cigar like "100M3I3D20M"
            M = M > 0 ? M : 0;
            *pM++ = M;
            *pM++ = e;
            h = M > e ? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
            *pM++ = f;
            h = h > f ? h : f;
            *pM++ = h;
            h1 = h;              // save H(i,j) to h1 for the next column
            mj = m > h ? mj : j; // record the position where max score is achieved
            m = m > h ? m : h;   // m is stored at eh[mj+1]

            t = M - oe_del;
            t = t > 0 ? t : 0;
            e -= e_del;
            e = e > t ? e : t;   // computed E(i+1,j)
            p->e = e;            // save E(i+1,j) for the next row

            t = M - oe_ins;
            t = t > 0 ? t : 0;
            f -= e_ins;
            f = f > t ? f : t;   // computed F(i,j+1)
        }

        eh[qlen].h = h1; eh[qlen].e = 0;

        if (!foundZeroRow)
        {
            d_max_ie = d_gscore > h1 ? d_max_ie : i;
            d_gscore = d_gscore > h1 ? d_gscore : h1;

            if (m > d_max) {
                d_max = m, d_max_i = i, d_max_j = mj;
                d_max_off = d_max_off > abs(mj - i) ? d_max_off : abs(mj - i);
            }

            if (m == 0)
                foundZeroRow = true;
        }
    }

    free(eh);

    return pMem;
}

void ksw_extend_dump(const char* dumpName, const int* pMem, int qlen, int tlen)
{
    int i, j;

    FILE* fTable = fopen(dumpName, "w");

    fprintf(fTable, ",,");
    for (j = 0; LIKELY(j < qlen); ++j)
    {
        fprintf(fTable, "%d, ", j);
    }
    fprintf(fTable, "\n\n");

    for (i = 0; LIKELY(i < tlen); ++i)
    {
        fprintf(fTable, "%d,,", i);
        for (j = 0; LIKELY(j < qlen); ++j)
        {
            size_t dm = 6ull * (i * qlen + j);
            fprintf(fTable, "%d, ", pMem[dm + 3]);
        }
        fprintf(fTable, "\n");

        fprintf(fTable, "%d,,", i);
        for (j = 0; LIKELY(j < qlen); ++j)
        {
            size_t dm = 6ull * (i * qlen + j);
            fprintf(fTable, "%d, ", pMem[dm + 4]);
        }
        fprintf(fTable, "\n");

        fprintf(fTable, "%d,,", i);
        int hMax = 0;
        for (j = 0; LIKELY(j < qlen); ++j)
        {
            size_t dm = 6ull * (i * qlen + j);
            if (hMax < pMem[dm + 5]) hMax = pMem[dm + 5];
            fprintf(fTable, "%d, ", pMem[dm + 5]);
        }
        fprintf(fTable, ", %d", hMax);
        fprintf(fTable, "\n\n");
    }

    fclose(fTable);
}
#endif

void compute_band_i16(
    const int qlen, const int tlen,
    const uint8_t* i_q,
    const int gapoe, const int gape,
    const __m128i _score_matrix,
    __m256i*& pRow0, __m256i*& pRow1,
    __m256i& _hCol, const __m128i _target,
    __m256i& _mm, __m256i& _mj, __m256i& _gscore,
    size_t bandStart, size_t bandWidth
#ifdef UNIT_TEST_EXTEND
    , const int* pMem
#endif
)
{
    size_t COLS_PADDED = qlen + 32ull;

    __m256i* pInput = pRow1;
    __m256i* pOutput = pRow1 = pRow0;
    pRow0 = pInput;

    pInput += 1;

    __m256i _H0;

    if (bandStart == 0)
    {
        _H0 = _mm256_shift_left_si256<2>(
            _mm256_setzero_si256(),
            _hCol);

        __m256i _hLadder = _mm256_prefix_add_epi16(
            _mm256_set1_epi16(
                gape));

        _hCol = _mm256_subs_epu16(
            _mm256_subs_epu16(
                _hCol,
                _mm256_set1_epi16(gapoe - gape)),
            _hLadder);
    }
    else
    {
        _hCol = _mm256_broadcastw_epi16(
            _mm256_castsi256_si128(
                _mm256_shift_left_si256<2>(_hCol, _hCol)));

        _H0 = _mm256_shift_left_si256<2>(
            _mm256_setzero_si256(),
            _hCol);

        __m256i _hLadder = _mm256_prefix_add_epi16(
            _mm256_set1_epi16(
                gape));

        _hCol = _mm256_subs_epu16(
            _hCol,
            _hLadder);
    }

    __m256i _F = _mm256_setzero_si256();
    __m256i _E = _mm256_setzero_si256();
    __m256i _H1 = _mm256_setzero_si256();

    __m256i _hMask = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1);

    int j = 0, jj = COLS_PADDED - 17;
    for (; j < 15; j++)
    {
        _H1 = _mm256_blendv_epi8(
            _H1,
            _hCol,
            _hMask);

        _hMask = _mm256_shift_left_si256<2>(_hMask,
            _mm256_setzero_si256());

        _H1 = _mm256_shift_left_si256<2>(_H1,
            *pInput++);

        __m128i _query = _mm_loadu_si128((const __m128i*)(i_q + jj));

        __m256i _M = separate_M_i16(_score_matrix, _H0, _query, _target);

        _E = _mm256_shift_left_si256<2>(_E,
            *pInput++);

        _H0 = _mm256_max_epi16(_M, _E);

        _H0 = _mm256_max_epi16(_H0, _F);

        update_row_maximums_i16(_mm, _mj, j, _H0);

#ifdef UNIT_TEST_EXTEND
        for (size_t di = 0; (di <= j) && (di < bandWidth); di++)
        {
            size_t dm = 6 * ((bandStart + di) * qlen + j - di);

            assert(_query.m128i_u8[di] == pMem[dm]);
            assert(_M.m256i_i16[di] == pMem[dm + 2]);
            assert(_E.m256i_i16[di] == pMem[dm + 3]);
            assert(_F.m256i_i16[di] == pMem[dm + 4]);
            assert(_H0.m256i_i16[di] == pMem[dm + 5]);
        }
#endif
        __m256i _gape = _mm256_set1_epi16(gape);
        __m256i _gapoe = _mm256_set1_epi16(gapoe);

        // Compute E'(i+1,j)
        _E = _mm256_max_epu16(
            _mm256_subs_epu16(_E, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        // Compute F'(i,j+1)
        _F = _mm256_max_epu16(
            _mm256_subs_epu16(_F, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        __m256i _Temp = _H1; _H1 = _H0; _H0 = _Temp;

        jj--;
    }

    __m128i _query;

    for (; j < qlen; j++)
    {
        _H1 = _mm256_shift_left_si256<2>(_H1,
            *pInput++);

        _query = _mm_loadu_si128((const __m128i*)(i_q + jj));

        __m256i _M = separate_M_i16(_score_matrix, _H0, _query, _target);

        _mm256_store_si256(pOutput++, _E);

        _E = _mm256_shift_left_si256<2>(_E,
            *pInput++);

        _H0 = _mm256_max_epi16(_M, _E);

        _H0 = _mm256_max_epi16(_H0, _F);

        _mm256_store_si256(pOutput++, _H0);

        update_row_maximums_i16(_mm, _mj, j, _H0);

#ifdef UNIT_TEST_EXTEND
        for (size_t di = 0; di < bandWidth; di++)
        {
            size_t dm = 6 * ((bandStart + di) * qlen + j - di);

            assert(_query.m128i_u8[di] == pMem[dm]);
            assert(_M.m256i_i16[di] == pMem[dm + 2]);
            assert(_E.m256i_i16[di] == pMem[dm + 3]);
            assert(_F.m256i_i16[di] == pMem[dm + 4]);
            assert(_H0.m256i_i16[di] == pMem[dm + 5]);
        }
#endif

        __m256i _gape = _mm256_set1_epi16(gape);
        __m256i _gapoe = _mm256_set1_epi16(gapoe);

        // Compute E'(i+1,j)
        _E = _mm256_max_epu16(
            _mm256_subs_epu16(_E, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        // Compute F'(i,j+1)
        _F = _mm256_max_epu16(
            _mm256_subs_epu16(_F, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        __m256i _Temp = _H1; _H1 = _H0; _H0 = _Temp;

        jj--;
    }

    __m256i _mMask = _mm256_set1_epi8(-1);

    __m256i _gMask = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1);

    for (; j < qlen + 16; j++)
    {
        _gscore = _mm256_blendv_epi8(
            _gscore,
            _H1,
            _gMask);

        _gMask = _mm256_shift_left_si256<2>(_gMask,
            _mm256_setzero_si256());

        _H1 = _mm256_shift_left_si256<2>(_H1,
            _mm256_setzero_si256());

        _query = _mm_slli_si128(
            _query,
            1);

        __m256i _M = separate_M_i16(_score_matrix, _H0, _query, _target);

        _mMask = _mm256_shift_left_si256<2>(
            _mMask,
            _mm256_setzero_si256());

        _mm256_store_si256(pOutput++, _E);

        _E = _mm256_shift_left_si256<2>(_E,
            _mm256_setzero_si256());

        _H0 = _mm256_max_epi16(_M, _E);

        _H0 = _mm256_max_epi16(_H0, _F);

        _mm256_store_si256(pOutput++, _H0);

        update_row_maximums_i16(_mm, _mj, j, _H0, _mMask);

#ifdef UNIT_TEST_EXTEND
        for (size_t di = j - qlen + 1; di < bandWidth; di++)
        {
            size_t dm = 6 * ((bandStart + di) * qlen + j - di);

            assert(_query.m128i_u8[di] == pMem[dm]);
            assert(_M.m256i_i16[di] == pMem[dm + 2]);
            assert(_E.m256i_i16[di] == pMem[dm + 3]);
            assert(_F.m256i_i16[di] == pMem[dm + 4]);
            assert(_H0.m256i_i16[di] == pMem[dm + 5]);
        }
#endif

        __m256i _gape = _mm256_set1_epi16(gape);
        __m256i _gapoe = _mm256_set1_epi16(gapoe);

        // Compute E'(i+1,j)
        _E = _mm256_max_epu16(
            _mm256_subs_epu16(_E, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        // Compute F'(i,j+1)
        _F = _mm256_max_epu16(
            _mm256_subs_epu16(_F, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        __m256i _Temp = _H1; _H1 = _H0; _H0 = _Temp;

        jj--;
    }
}

__forceinline void update_max_g_i16(__m128i& _max_g, __m128i& _max_ie, const __m128i _gscore, const __m128i _mi, const __m128i _kmask)
{
    __m128i _cmp0 = _mm_cmpgt_epi16(_gscore, _max_g);

    __m128i _cmp1 = _mm_and_si128(
        _mm_cmpeq_epi16(_gscore, _max_g),
        _mm_cmpgt_epi16(_mi, _max_ie));

    __m128i _cmp = _mm_and_si128(
        _mm_or_si128(_cmp0, _cmp1),
        _kmask);

    _max_g = _mm_blendv_epi8(
        _max_g,
        _gscore,
        _cmp0);

    _max_ie = _mm_blendv_epi8(
        _max_ie,
        _mi,
        _cmp);
}

__forceinline void update_max_ij_i16(__m128i& _max_score, __m128i& _max_i, __m128i& _max_j, const __m128i _mm, const __m128i _mi, const __m128i _mj, const __m128i _kmask)
{
    __m128i _cmp0 = _mm_cmpgt_epi16(_mm, _max_score);

    __m128i _cmp1 = _mm_and_si128(
        _mm_or_si128(
            _mm_cmpgt_epi16(_max_i, _mi),
            _mm_and_si128(
                _mm_cmpeq_epi16(_max_i, _mi),
                _mm_cmpgt_epi16(_mj, _max_j))),
        _mm_cmpeq_epi16(_mm, _max_score));

    __m128i _cmp = _mm_and_si128(
        _mm_or_si128(_cmp0, _cmp1),
        _kmask);

    _max_score = _mm_blendv_epi8(
        _max_score,
        _mm,
        _cmp);

    _max_i = _mm_blendv_epi8(
        _max_i,
        _mi,
        _cmp);

    _max_j = _mm_blendv_epi8(
        _max_j,
        _mj,
        _cmp);
}

int ksw_extend_i16(int qlen, const uint8_t* query, int tlen, const uint8_t* target, __m256i _score_matrix, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int* _qle, int* _tle, int* _gtle, int* _gscore, int* _max_off)
{
    int gscore = -1;
    int max_ie = -1;

    int gapoe = gapo + gape;

    assert(h0 > 0 && w > 0);

#ifdef UNIT_TEST_EXTEND
    int d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off;

    int* pMem = ksw_extend_debug(qlen, query, tlen, target, _score_matrix, gapo, gape, gapo, gape, w, end_bonus, zdrop, h0, d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off);
#endif

    size_t COLS_PADDED = qlen + 32ull;

    // Initialize memory
    size_t bandSize = 64 * COLS_PADDED;

    __m256i* pRow0 = (__m256i*)_mm_malloc(bandSize, 32);
    __m256i* pRow1 = (__m256i*)_mm_malloc(bandSize, 32);

    uint8_t* i_q = (uint8_t*)_mm_malloc(COLS_PADDED, 32);
    memset(i_q, -5, COLS_PADDED);

    for (int j = 0; j < qlen; j++)
    {
        size_t jj = COLS_PADDED - j - 17;
        i_q[jj] = query[j];
    }

    __m128i _mm_score_matrix = _mm256_castsi256_si128(
        _score_matrix);

    __m256i _hCol = _mm256_set1_epi16(h0);

    // initialize the antidiagonals
    __m256i _E = _mm256_setzero_si256();

    __m256i* pOutput = pRow1;

    __m256i _H1 = _mm256_subs_epi16(
        _hCol,
        _mm256_set1_epi16(gapoe));

    for (size_t j = 0; j < COLS_PADDED; j++)
    {
        _mm256_store_si256(pOutput++, _E);
        _mm256_store_si256(pOutput++, _H1);

        _H1 = _mm256_subs_epi16(
            _H1,
            _mm256_set1_epi16(gape));
    }

    int max_score = h0, max_i = -1, max_j = -1, max_off = 0;

    for (int bandStart = 0; bandStart < tlen; bandStart += 16)
    {
        __m256i _mm = _mm256_setzero_si256(), _mj = _mm256_setzero_si256(), _g_score = _mm256_setzero_si256();

        int bandWidth = 16;

        if (bandStart + bandWidth > tlen)
        {
            bandWidth = tlen - bandStart;
        }

        int bandComplement = 16 - bandWidth;

        __m128i _target = _mm_loadu_si128((const __m128i*) (target + bandStart - bandComplement));


        while (bandComplement-- > 0)
        {
            _target = _mm_srli_si128(_target, 1);
        }

        compute_band_i16(
            qlen, tlen,
            i_q,
            gapoe, gape,
            _mm_score_matrix,
            pRow0, pRow1,
            _hCol, _target,
            _mm, _mj, _g_score,
            bandStart, bandWidth
#ifdef UNIT_TEST_EXTEND
            , pMem
#endif
        );

        __m256i _in = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        _mj = _mm256_subs_epi16(_mj, _in);
        __m256i _mi = _mm256_add_epi16(_mm256_set1_epi16(bandStart), _in);

        __m256i _tlen = _mm256_set1_epi16(tlen);

        __m256i _cmask = _mm256_cmpeq_epi16(_mm256_max_epu16(_tlen, _mi), _mi);

        _mm = _mm256_andnot_si256(_cmask, _mm);

        // First zero element should be in the mask
        __m256i _mmask256 = _mm256_cmpgt_epi16(
            _mm,
            _mm256_setzero_si256());

        const __m256i _true = _mm256_set1_epi16(-1);

        _mmask256 = _mm256_shift_left_si256<2>(
            _mmask256, _true);

        // Parallel scan
        _mmask256 = _mm256_and_si256(
            _mmask256,
            _mm256_shift_left_si256<2>(
                _mmask256, _true));

        _mmask256 = _mm256_and_si256(
            _mmask256,
            _mm256_shift_left_si256<4>(
                _mmask256, _true));

        _mmask256 = _mm256_and_si256(
            _mmask256,
            _mm256_shift_left_si256<8>(
                _mmask256, _true));

        _mmask256 = _mm256_and_si256(
            _mmask256,
            _mm256_shift_left_si256<16>(
                _mmask256, _true));

        _mmask256 = _mm256_andnot_si256(_cmask, _mmask256);

        __m128i _mmask = _mm256_castsi256_si128(_mmask256);

        __m128i _kmask = _mm256_extracti128_si256(_mmask256, 1);

        // Two parallel reduces running, well, in parallel
        _g_score = _mm256_andnot_si256(_cmask, _g_score);

        __m128i _max_g = _mm256_castsi256_si128(_g_score);
        __m128i _max_ie = _mm256_castsi256_si128(_mi);

        __m128i _max_score = _mm256_castsi256_si128(_mm);
        __m128i _max_i = _mm256_castsi256_si128(_mi);
        __m128i _max_j = _mm256_castsi256_si128(_mj);

        update_max_g_i16(_max_g, _max_ie,
            _mm256_extracti128_si256(_g_score, 1),
            _mm256_extracti128_si256(_mi, 1),
            _kmask);

        update_max_ij_i16(_max_score, _max_i, _max_j,
            _mm256_extracti128_si256(_mm, 1),
            _mm256_extracti128_si256(_mi, 1),
            _mm256_extracti128_si256(_mj, 1),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 8);

        update_max_g_i16(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 8),
            _mm_srli_si128(_max_ie, 8),
            _kmask);

        update_max_ij_i16(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 8),
            _mm_srli_si128(_max_i, 8),
            _mm_srli_si128(_max_j, 8),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 4);

        update_max_g_i16(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 4),
            _mm_srli_si128(_max_ie, 4),
            _kmask);

        update_max_ij_i16(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 4),
            _mm_srli_si128(_max_i, 4),
            _mm_srli_si128(_max_j, 4),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 2);

        update_max_g_i16(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 2),
            _mm_srli_si128(_max_ie, 2),
            _kmask);

        update_max_ij_i16(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 2),
            _mm_srli_si128(_max_i, 2),
            _mm_srli_si128(_max_j, 2),
            _kmask);

        int gscore_candidate = _mm_extract_epi16(_max_g, 0);

        if (gscore <= gscore_candidate)
        {
            max_ie = _mm_extract_epi16(_max_ie, 0);
            gscore = gscore_candidate;
        }

        int max_score_candidate = _mm_extract_epi16(_max_score, 0);

        if (max_score < max_score_candidate)
        {
            max_score = max_score_candidate;

            max_i = _mm_extract_epi16(_max_i, 0);
            max_j = _mm_extract_epi16(_max_j, 0);

            int off = abs(max_j - max_i);
            if (off > max_off)
            {
                max_off = off;
            }
        }

        __m256i _cmp = _mm256_cmpeq_epi16(_mm, _mm256_setzero_si256());
        if (!_mm256_testz_si256(_cmp, _cmp))
        {
            break;
        }
    }

    _mm_free(i_q);

    _mm_free(pRow1);
    _mm_free(pRow0);

    if (_qle)*_qle = max_j + 1;
    if (_tle)*_tle = max_i + 1;
    if (_gtle)*_gtle = (int)max_ie + 1;
    if (_gscore)*_gscore = gscore;
    if (_max_off)*_max_off = max_off;

#ifdef UNIT_TEST_EXTEND

    if (max_score != d_max)
    {
        ksw_extend_dump("ksw_extend_i16.csv", pMem, qlen, tlen);
        assert(max_score == d_max);
    }

    if (max_i != d_max_i)
    {
        ksw_extend_dump("ksw_extend_i16.csv", pMem, qlen, tlen);

        assert(max_i == d_max_i);
    }

    if (max_j != d_max_j)
    {
        ksw_extend_dump("ksw_extend_i16.csv", pMem, qlen, tlen);

        assert(max_j == d_max_j);
    }

    if (gscore != d_gscore)
    {
        ksw_extend_dump("ksw_extend_i16.csv", pMem, qlen, tlen);

        assert(gscore == d_gscore);
    }

    if (max_ie != d_max_ie)
    {
        ksw_extend_dump("ksw_extend_i16.csv", pMem, qlen, tlen);

        assert(max_ie == d_max_ie);
    }

    free(pMem);
#endif
    return max_score;
}

int ksw_extend_i16_16(int qlen, const uint8_t* query, int tlen, const uint8_t* target, __m256i _score_matrix, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int* _qle, int* _tle, int* _gtle, int* _gscore, int* _max_off)
{
    assert(qlen < 16);

    int gscore = -1, max_ie = -1;

    assert(h0 > 0 && w > 0);

#ifdef UNIT_TEST_EXTEND
    int d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off;

    int* pMem = ksw_extend_debug(qlen, query, tlen, target, _score_matrix, gapo, gape, gapo, gape, w, end_bonus, zdrop, h0, d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off);
#endif

    // initialization
    __m256i _gape = _mm256_set1_epi16(gape);
    __m256i _gapoe = _mm256_set1_epi16(gapo + gape);

    __m256i _gape_scan = _mm256_exclusive_add_epi16(_gape);

    __m256i _in = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m256i _cmask = _mm256_cmpgt_epi16(
        _mm256_set1_epi16(qlen),
        _in);

    __m128i _mm_score_matrix = _mm256_castsi256_si128(
        _score_matrix);

    alignas(64) short q_mem[16];

    memset(q_mem, 0xff, 32);
    memcpy(q_mem, query, qlen << 1);
    __m128i _query = _mm_load_si128((__m128i*)q_mem);

    // initialize the first column
    __m256i _h0 = _mm256_set1_epi16(h0);

    __m256i _h1 = _mm256_subs_epu16(
        _h0,
        _mm256_set1_epi16(gapo));

    // initialize the first row
    __m256i _H0 = _mm256_prefix_add_epi16(_gape);

    _H0 = _mm256_shift_left_si256<2>(
        _mm256_subs_epu16(
            _mm256_subs_epu16(
                _h0,
                _mm256_set1_epi16(gapo)),
            _H0),
        _h0);

    __m256i _E = _mm256_setzero_si256();

    __m256i _mm = _mm256_setzero_si256(), _mi = _mm256_setzero_si256();

    for (int i = 0; LIKELY(i < tlen); ++i) {
        __m256i _h = _H0;

        __m128i _target = _mm_set1_epi8(target[i]);

        __m256i _M = separate_M_i16(_mm_score_matrix, _h, _query, _target);

        _h = _mm256_max_epu16(_M, _E);

        // Prefix Scan
        __m256i _f = prefix_scan_F_i16(_M, _gape, _gape_scan);

        _h = _mm256_max_epi16(
            _h,
            _mm256_subs_epu16(_f, _gapoe));

        _mm256_store_si256((__m256i*)q_mem, _h);
        short lscore = q_mem[qlen - 1];
        if (lscore >= gscore)
        {
            gscore = lscore;
            max_ie = i;
        }

        if (_mm256_testz_si256(_h, _cmask))
            break;

        update_column_maximums_i16(_mm, _mi, i, _h);

#ifdef UNIT_TEST_EXTEND
        for (size_t dj = 0; dj < qlen; dj++)
        {
            size_t dm = 6ull * (i * qlen + dj);

            assert(_query.m128i_u8[dj] == pMem[dm]);
            assert(_M.m256i_i16[dj] == pMem[dm + 2]);
            assert(_E.m256i_i16[dj] == pMem[dm + 3]);
            //           assert(_F.m256i_u8[di] == pMem[dm + 4]);
            assert(_h.m256i_i16[dj] == pMem[dm + 5]);
        }
#endif
        // now compute E'(i+1,j)
        _E = _mm256_max_epu16(
            _mm256_subs_epu16(_E, _gape),
            _mm256_subs_epu16(_M, _gapoe));

        // Prepare for the next row
        _h1 = _mm256_subs_epu16(_h1, _gape);

        _H0 = _mm256_shift_left_si256<2>(_h, _h1);
    }

    int max_score = h0, max_i = -1, max_j = -1, max_off = 0;

    __m256i _mj = _in;

    __m256i _mmask256 = _cmask;

    __m128i _mmask = _mm256_castsi256_si128(_mmask256);

    __m128i _kmask = _mm256_extracti128_si256(_mmask256, 1);

    // Parallel reduce
    __m128i _max_score = _mm256_castsi256_si128(_mm);
    __m128i _max_i = _mm256_castsi256_si128(_mi);
    __m128i _max_j = _mm256_castsi256_si128(_mj);

    update_max_ij_i16(_max_score, _max_i, _max_j,
        _mm256_extracti128_si256(_mm, 1),
        _mm256_extracti128_si256(_mi, 1),
        _mm256_extracti128_si256(_mj, 1),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 8);

    update_max_ij_i16(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 8),
        _mm_srli_si128(_max_i, 8),
        _mm_srli_si128(_max_j, 8),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 4);

    update_max_ij_i16(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 4),
        _mm_srli_si128(_max_i, 4),
        _mm_srli_si128(_max_j, 4),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 2);

    update_max_ij_i16(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 2),
        _mm_srli_si128(_max_i, 2),
        _mm_srli_si128(_max_j, 2),
        _kmask);

    int max_score_candidate = _mm_extract_epi16(_max_score, 0);

    if (max_score < max_score_candidate)
    {
        max_score = max_score_candidate;

        max_i = _mm_extract_epi16(_max_i, 0);
        max_j = _mm_extract_epi16(_max_j, 0);

        int off = abs(max_j - max_i);
        if (off > max_off)
        {
            max_off = off;
        }
    }

    if (_qle)*_qle = max_j + 1;
    if (_tle)*_tle = max_i + 1;
    if (_gtle)*_gtle = max_ie + 1;
    if (_gscore)*_gscore = gscore;
    if (_max_off)*_max_off = max_off;

#ifdef UNIT_TEST_EXTEND

    if (max_score != d_max)
    {
        ksw_extend_dump("ksw_extend_i16_16.csv", pMem, qlen, tlen);

        assert(max_score == d_max);
    }

    if (max_i != d_max_i)
    {
        ksw_extend_dump("ksw_extend_i16_16.csv", pMem, qlen, tlen);

        assert(max_i == d_max_i);
    }

    if (max_j != d_max_j)
    {
        ksw_extend_dump("ksw_extend_i16_16.csv", pMem, qlen, tlen);

        assert(max_j == d_max_j);
    }

    if (gscore != d_gscore)
    {
        ksw_extend_dump("ksw_extend_i16_16.csv", pMem, qlen, tlen);

        assert(gscore == d_gscore);
    }

    assert(max_ie == d_max_ie);

    free(pMem);
#endif

    return max_score;
}

void compute_band_u8(
    const int qlen, const int tlen,
    const uint8_t* i_q,
    const int gapoe, const int gape,
    const __m256i _score_matrix, const __m256i _score_shift,
    __m256i*& pRow0, __m256i*& pRow1,
    __m256i& _hCol, const __m256i _target,
    __m256i& _mm, __m256i& _mj, __m256i& _gscore,
    size_t bandStart, size_t bandWidth
#ifdef UNIT_TEST_EXTEND
    , const int* pMem
#endif
)
{
    size_t COLS_PADDED = qlen + 64ull;

    __m256i* pInput = pRow1;
    __m256i* pOutput = pRow1 = pRow0;
    pRow0 = pInput;

    pInput += 1;

    __m256i _H0;

    if (bandStart == 0)
    {
        _H0 = _mm256_shift_left_si256<1>(
            _mm256_setzero_si256(),
            _hCol);

        __m256i _hLadder = _mm256_prefix_add_epi8(
            _mm256_set1_epi8(
                gape));

        _hCol = _mm256_subs_epu8(
            _mm256_subs_epu8(
                _hCol,
                _mm256_set1_epi8(
                    gapoe - gape)),
            _hLadder);
    }
    else
    {
        _hCol = _mm256_broadcastb_epi8(
            _mm256_castsi256_si128(
                _mm256_shift_left_si256<1>(_hCol, _hCol)));

        _H0 = _mm256_shift_left_si256<1>(
            _mm256_setzero_si256(),
            _hCol);

        __m256i _hLadder = _mm256_prefix_add_epi8(
            _mm256_set1_epi8(
                gape));

        _hCol = _mm256_subs_epu8(
            _hCol,
            _hLadder);
    }

    __m256i _F = _mm256_setzero_si256();
    __m256i _E = _mm256_setzero_si256();
    __m256i _H1 = _mm256_setzero_si256();

    __m256i _hMask = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);

    int j = 0, jj = COLS_PADDED - 33;
    for (; j < 31; j++)
    {
        _H1 = _mm256_blendv_epi8(
            _H1,
            _hCol,
            _hMask);

        _hMask = _mm256_shift_left_si256<1>(_hMask,
            _mm256_setzero_si256());

        _H1 = _mm256_shift_left_si256<1>(_H1,
            *pInput++);

        __m256i _query = _mm256_loadu_si256((const __m256i*)(i_q + jj));

        __m256i _M = separate_M_u8(_score_matrix, _score_shift, _H0, _query, _target);

        _E = _mm256_shift_left_si256<1>(_E,
            *pInput++);

        _H0 = _mm256_max_epu8(_M, _E);

        _H0 = _mm256_max_epu8(_H0, _F);

        update_row_maximums_u8(_mm, _mj, j, _H0);

#ifdef UNIT_TEST_EXTEND
        for (size_t di = 0; (di <= j) && (di < bandWidth); di++)
        {
            size_t dm = 6 * ((bandStart + di) * qlen + j - di);

            assert(_query.m256i_u8[di] == pMem[dm]);
            assert(_M.m256i_u8[di] == pMem[dm + 2]);
            assert(_E.m256i_u8[di] == pMem[dm + 3]);
            assert(_F.m256i_u8[di] == pMem[dm + 4]);
            assert(_H0.m256i_u8[di] == pMem[dm + 5]);
        }
#endif

        __m256i _gape = _mm256_set1_epi8(gape);
        __m256i _gapoe = _mm256_set1_epi8(gapoe);

        // Compute E'(i+1,j)
        _E = _mm256_max_epu8(
            _mm256_subs_epu8(_E, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        // Compute F'(i,j+1)
        _F = _mm256_max_epu8(
            _mm256_subs_epu8(_F, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        __m256i _Temp = _H1; _H1 = _H0; _H0 = _Temp;

        jj--;
    }

    __m256i _query;

    for (; j < qlen; j++)
    {
        _H1 = _mm256_shift_left_si256<1>(_H1,
            *pInput++);

        _query = _mm256_loadu_si256((const __m256i*)(i_q + jj));

        __m256i _M = separate_M_u8(_score_matrix, _score_shift, _H0, _query, _target);

        _mm256_store_si256(pOutput++, _E);

        _E = _mm256_shift_left_si256<1>(_E,
            *pInput++);

        _H0 = _mm256_max_epu8(_M, _E);

        _H0 = _mm256_max_epu8(_H0, _F);

        _mm256_store_si256(pOutput++, _H0);

        update_row_maximums_u8(_mm, _mj, j, _H0);

#ifdef UNIT_TEST_EXTEND
        for (size_t di = 0; di < bandWidth; di++)
        {
            size_t dm = 6 * ((bandStart + di) * qlen + j - di);

            assert(_query.m256i_u8[di] == pMem[dm]);
            assert(_M.m256i_u8[di] == pMem[dm + 2]);
            assert(_E.m256i_u8[di] == pMem[dm + 3]);
            assert(_F.m256i_u8[di] == pMem[dm + 4]);
            assert(_H0.m256i_u8[di] == pMem[dm + 5]);
        }
#endif

        __m256i _gape = _mm256_set1_epi8(gape);
        __m256i _gapoe = _mm256_set1_epi8(gapoe);

        // Compute E'(i+1,j)
        _E = _mm256_max_epu8(
            _mm256_subs_epu8(_E, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        // Compute F'(i,j+1)
        _F = _mm256_max_epu8(
            _mm256_subs_epu8(_F, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        __m256i _Temp = _H1; _H1 = _H0; _H0 = _Temp;

        jj--;
    }

    __m256i _mMask = _mm256_set1_epi8(-1);

    __m256i _gMask = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);

    _gscore = _mm256_setzero_si256();

    for (; j < qlen + 32; j++)
    {
        _gscore = _mm256_blendv_epi8(
            _gscore,
            _H1,
            _gMask);

        _gMask = _mm256_shift_left_si256<1>(_gMask,
            _mm256_setzero_si256());

        _H1 = _mm256_shift_left_si256<1>(_H1,
            _mm256_setzero_si256());

        _query = _mm256_shift_left_si256<1>(
            _query,
            _mm256_setzero_si256());

        __m256i _M = separate_M_u8(_score_matrix, _score_shift, _H0, _query, _target);

        _mMask = _mm256_shift_left_si256<1>(
            _mMask,
            _mm256_setzero_si256());

        _mm256_store_si256(pOutput++, _E);

        _E = _mm256_shift_left_si256<1>(_E,
            _mm256_setzero_si256());

        _H0 = _mm256_max_epu8(_M, _E);

        _H0 = _mm256_max_epu8(_H0, _F);

        _mm256_store_si256(pOutput++, _H0);

        update_row_maximums_u8(_mm, _mj, j, _H0, _mMask);

#ifdef UNIT_TEST_EXTEND
        for (size_t di = j - qlen + 1; di < bandWidth; di++)
        {
            size_t dm = 6 * ((bandStart + di) * qlen + j - di);

            assert(_query.m256i_u8[di] == pMem[dm]);
            assert(_M.m256i_u8[di] == pMem[dm + 2]);
            assert(_E.m256i_u8[di] == pMem[dm + 3]);
            assert(_F.m256i_u8[di] == pMem[dm + 4]);
            assert(_H0.m256i_u8[di] == pMem[dm + 5]);
        }
#endif

        __m256i _gape = _mm256_set1_epi8(gape);
        __m256i _gapoe = _mm256_set1_epi8(gapoe);

        // Compute E'(i+1,j)
        _E = _mm256_max_epu8(
            _mm256_subs_epu8(_E, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        // Compute F'(i,j+1)
        _F = _mm256_max_epu8(
            _mm256_subs_epu8(_F, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        __m256i _Temp = _H1; _H1 = _H0; _H0 = _Temp;

        jj--;
    }
}

__forceinline void update_max_g_i8(__m128i& _max_g, __m128i& _max_ie, const __m128i _gscore, const __m128i _mi, const __m128i _kmask)
{
    __m128i _cmpg = _mm_cmpeq_epi8(_gscore, _max_g);

    __m128i _cmp0 = _mm_andnot_si128(
        _cmpg,
        _mm_cmpeq_epi8(_mm_max_epu8(_gscore, _max_g), _gscore));

    __m128i _cmp1 = _mm_andnot_si128(
        _mm_cmpeq_epi8(_mm_max_epu8(_mi, _max_ie), _max_ie),
        _cmpg);

    __m128i _cmp = _mm_and_si128(
        _mm_or_si128(_cmp0, _cmp1),
        _kmask);

    _max_g = _mm_blendv_epi8(
        _max_g,
        _gscore,
        _cmp0);

    _max_ie = _mm_blendv_epi8(
        _max_ie,
        _mi,
        _cmp);
}

__forceinline void update_max_ij_i8(__m128i& _max_score, __m128i& _max_i, __m128i& _max_j, const __m128i _mm, const __m128i _mi, const __m128i _mj, const __m128i _kmask)
{
    __m128i _cmpm = _mm_cmpeq_epi8(_mm, _max_score);

    __m128i _cmpi = _mm_cmpeq_epi8(_max_i, _mi);

    __m128i _cmp0 = _mm_andnot_si128(
        _cmpm,
        _mm_cmpeq_epi8(_mm_max_epu8(_mm, _max_score), _mm));

    __m128i _cmp1 = _mm_and_si128(
        _cmpm,
        _mm_or_si128(
            _mm_andnot_si128(
                _cmpi,
                _mm_cmpeq_epi8(_mm_max_epu8(_max_i, _mi), _max_i)),
            _mm_andnot_si128(
                _mm_cmpeq_epi8(_mm_max_epu8(_mj, _max_j), _max_j),
                _cmpi)));

    __m128i _cmp = _mm_and_si128(
        _mm_or_si128(_cmp0, _cmp1),
        _kmask);

    _max_score = _mm_blendv_epi8(
        _max_score,
        _mm,
        _cmp);

    _max_i = _mm_blendv_epi8(
        _max_i,
        _mi,
        _cmp);

    _max_j = _mm_blendv_epi8(
        _max_j,
        _mj,
        _cmp);
}

int ksw_extend_u8(int qlen, const uint8_t* query, int tlen, const uint8_t* target, __m256i _score_matrix, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int* _qle, int* _tle, int* _gtle, int* _gscore, int* _max_off)
{
    int gscore = -1;
    int max_ie = -1;

    int gapoe = gapo + gape;

    assert(h0 > 0 && w > 0);

#ifdef UNIT_TEST_EXTEND
    int d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off;

    int* pMem = ksw_extend_debug(qlen, query, tlen, target, _score_matrix, gapo, gape, gapo, gape, w, end_bonus, zdrop, h0, d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off);
#endif

    __m256i _score_shift = setup_score_shift(
        _score_matrix);

    __m256i _mat_shifted = _mm256_add_epi8(
        _score_matrix,
        _score_shift);

    size_t COLS_PADDED = qlen + 64ull;

    // Initialize memory
    size_t bandSize = 64 * COLS_PADDED;

    __m256i* pRow0 = (__m256i*)_mm_malloc(bandSize, 32);
    __m256i* pRow1 = (__m256i*)_mm_malloc(bandSize, 32);

    uint8_t* i_q = (uint8_t*)_mm_malloc(COLS_PADDED, 32);
    memset(i_q, -1, COLS_PADDED);

    for (int j = 0; j < qlen; j++)
    {
        size_t jj = COLS_PADDED - j - 33;
        i_q[jj] = query[j];
    }

    __m256i _hCol = _mm256_set1_epi8(h0);

    // initialize the antidiagonals
    __m256i _E = _mm256_setzero_si256();

    __m256i* pOutput = pRow1;

    __m256i _H1 = _mm256_subs_epu8(
        _hCol,
        _mm256_set1_epi8(gapo + gape));

    for (size_t j = 0; j < COLS_PADDED; j++)
    {
        _mm256_store_si256(pOutput++, _E);
        _mm256_store_si256(pOutput++, _H1);

        _H1 = _mm256_subs_epu8(
            _H1,
            _mm256_set1_epi8(gape));
    }

    int max_score = h0, max_i = -1, max_j = -1, max_off = 0;

    for (int bandStart = 0; bandStart < tlen; bandStart += 32)
    {
        __m256i _mm = _mm256_setzero_si256(), _mj = _mm256_setzero_si256(), _g_score = _mm256_setzero_si256();

        int bandWidth = 32;

        if (bandStart + bandWidth > tlen)
        {
            bandWidth = tlen - bandStart;
        }

        int bandComplement = 32 - bandWidth;

        __m256i _target = _mm256_loadu_si256((const __m256i*) (target + bandStart - bandComplement));

        while (bandComplement-- > 0)
        {
            _target = _mm256_shift_right_si256<1>(_target);
        }

        compute_band_u8(
            qlen, tlen,
            i_q,
            gapoe, gape,
            _mat_shifted, _score_shift,
            pRow0, pRow1,
            _hCol, _target,
            _mm, _mj, _g_score,
            bandStart, bandWidth
#ifdef UNIT_TEST_EXTEND
            , pMem
#endif
        );

        __m256i _in = _mm256_set_epi8(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        _mj = _mm256_subs_epu8(_mj, _in);
        __m256i _mi = _mm256_add_epi8(_mm256_set1_epi8(bandStart), _in);

        __m256i _tlen = _mm256_set1_epi8(tlen);

        __m256i _cmask = _mm256_cmpeq_epi8(_mm256_max_epu8(_tlen, _mi), _mi);

        _mm = _mm256_andnot_si256(_cmask, _mm);

        // First zero element should be in the mask
        __m256i _mmask256 = _mm256_cmpeq_epi8(
            _mm,
            _mm256_setzero_si256());

        _mmask256 = _mm256_shift_left_si256<1>(_mmask256);

        // Parallel scan
        _mmask256 = _mm256_or_si256(
            _mmask256,
            _mm256_shift_left_si256<1>(_mmask256));

        _mmask256 = _mm256_or_si256(
            _mmask256,
            _mm256_shift_left_si256<2>(_mmask256));

        _mmask256 = _mm256_or_si256(
            _mmask256,
            _mm256_shift_left_si256<4>(_mmask256));

        _mmask256 = _mm256_or_si256(
            _mmask256,
            _mm256_shift_left_si256<8>(_mmask256));

        _mmask256 = _mm256_or_si256(
            _mmask256,
            _mm256_shift_left_si256<16>(_mmask256));

        _mmask256 = _mm256_xor_si256(
            _mmask256,
            _mm256_set1_epi8(-1));

        _mmask256 = _mm256_andnot_si256(_cmask, _mmask256);

        __m128i _mmask = _mm256_castsi256_si128(_mmask256);

        __m128i _kmask = _mm256_extracti128_si256(_mmask256, 1);

        // Two parallel reduces running, well, in parallel
        _g_score = _mm256_andnot_si256(_cmask, _g_score);

        __m128i _max_g = _mm256_castsi256_si128(_g_score);
        __m128i _max_ie = _mm256_castsi256_si128(_mi);

        __m128i _max_score = _mm256_castsi256_si128(_mm);
        __m128i _max_i = _mm256_castsi256_si128(_mi);
        __m128i _max_j = _mm256_castsi256_si128(_mj);

        update_max_g_i8(_max_g, _max_ie,
            _mm256_extracti128_si256(_g_score, 1),
            _mm256_extracti128_si256(_mi, 1),
            _kmask);

        update_max_ij_i8(_max_score, _max_i, _max_j,
            _mm256_extracti128_si256(_mm, 1),
            _mm256_extracti128_si256(_mi, 1),
            _mm256_extracti128_si256(_mj, 1),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 8);

        update_max_g_i8(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 8),
            _mm_srli_si128(_max_ie, 8),
            _kmask);

        update_max_ij_i8(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 8),
            _mm_srli_si128(_max_i, 8),
            _mm_srli_si128(_max_j, 8),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 4);

        update_max_g_i8(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 4),
            _mm_srli_si128(_max_ie, 4),
            _kmask);

        update_max_ij_i8(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 4),
            _mm_srli_si128(_max_i, 4),
            _mm_srli_si128(_max_j, 4),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 2);

        update_max_g_i8(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 2),
            _mm_srli_si128(_max_ie, 2),
            _kmask);

        update_max_ij_i8(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 2),
            _mm_srli_si128(_max_i, 2),
            _mm_srli_si128(_max_j, 2),
            _kmask);

        _mmask = _mm_or_si128(_mmask, _kmask);
        _kmask = _mm_srli_si128(_mmask, 1);

        update_max_g_i8(_max_g, _max_ie,
            _mm_srli_si128(_max_g, 1),
            _mm_srli_si128(_max_ie, 1),
            _kmask);

        update_max_ij_i8(_max_score, _max_i, _max_j,
            _mm_srli_si128(_max_score, 1),
            _mm_srli_si128(_max_i, 1),
            _mm_srli_si128(_max_j, 1),
            _kmask);

        int gscore_candidate = _mm_extract_epi8(_max_g, 0);

        if (gscore <= gscore_candidate)
        {
            max_ie = _mm_extract_epi8(_max_ie, 0);
            gscore = gscore_candidate;
        }

        int max_score_candidate = _mm_extract_epi8(_max_score, 0);

        if (max_score < max_score_candidate)
        {
            max_score = max_score_candidate;

            max_i = _mm_extract_epi8(_max_i, 0);
            max_j = _mm_extract_epi8(_max_j, 0);

            int off = abs(max_j - max_i);
            if (off > max_off)
            {
                max_off = off;
            }
        }

        __m256i _cmp = _mm256_cmpeq_epi8(_mm, _mm256_setzero_si256());
        if (!_mm256_testz_si256(_cmp, _cmp))
        {
            break;
        }
    }

    _mm_free(i_q);

    _mm_free(pRow1);
    _mm_free(pRow0);

    if (_qle)*_qle = max_j + 1;
    if (_tle)*_tle = max_i + 1;
    if (_gtle)*_gtle = (int)max_ie + 1;
    if (_gscore)*_gscore = gscore;
    if (_max_off)*_max_off = max_off;

#ifdef UNIT_TEST_EXTEND

    if (max_score != d_max)
    {
        ksw_extend_dump("ksw_extend_u8.csv", pMem, qlen, tlen);

        assert(max_score == d_max);
    }

    if (max_i != d_max_i)
    {
        ksw_extend_dump("ksw_extend_u8.csv", pMem, qlen, tlen);

        assert(max_i == d_max_i);
    }

    if (max_j != d_max_j)
    {
        ksw_extend_dump("ksw_extend_u8.csv", pMem, qlen, tlen);

        assert(max_j == d_max_j);
    }

    if (gscore != d_gscore)
    {
        ksw_extend_dump("ksw_extend_u8.csv", pMem, qlen, tlen);

        assert(gscore == d_gscore);
    }

    if (max_ie != d_max_ie)
    {
        ksw_extend_dump("ksw_extend_u8.csv", pMem, qlen, tlen);

        assert(max_ie == d_max_ie);
    }

    free(pMem);
#endif
    return max_score;
}

int ksw_extend_u8_32(int qlen, const uint8_t* query, int tlen, const uint8_t* target, __m256i _score_matrix, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int* _qle, int* _tle, int* _gtle, int* _gscore, int* _max_off)
{
    assert(qlen < 32);

    int gscore = -1, max_ie = -1;

    assert(h0 > 0 && w > 0);

#ifdef UNIT_TEST_EXTEND
    int d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off;

    int* pMem = ksw_extend_debug(qlen, query, tlen, target, _score_matrix, gapo, gape, gapo, gape, w, end_bonus, zdrop, h0, d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off);
#endif

    // initialization
    __m256i _gape = _mm256_set1_epi8(gape);
    __m256i _gapoe = _mm256_set1_epi8(gapo + gape);

    __m256i _gape_scan = _mm256_exclusive_add_epu8(_gape);

    __m256i _in = _mm256_set_epi8(
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m256i _cmask = _mm256_cmpgt_epi8(
        _mm256_set1_epi8(qlen),
        _in);

    __m256i _score_shift = setup_score_shift(_score_matrix);

    __m256i _mat_shifted = _mm256_add_epi8(
        _score_matrix,
        _score_shift);

    alignas(64) uint8_t q_mem[32];

    std::memset(q_mem, 0xff, 32);
    std::memcpy(q_mem, query, qlen);
    __m256i _query = _mm256_load_si256((__m256i*)q_mem);

    // initialize the first column
    __m256i _h0 = _mm256_set1_epi8(h0);

    __m256i _h1 = _mm256_subs_epu8(
        _h0,
        _mm256_set1_epi8(gapo));

    // initialize the first row
    __m256i _H0 = _mm256_prefix_add_epi8(_gape);

    _H0 = _mm256_shift_left_si256<1>(
        _mm256_subs_epu8(
            _mm256_subs_epu8(
                _h0,
                _mm256_set1_epi8(gapo)),
            _H0),
        _h0);

    __m256i _E = _mm256_setzero_si256();

    __m256i _mm = _mm256_setzero_si256(), _mi = _mm256_setzero_si256();

    for (int i = 0; LIKELY(i < tlen); ++i) {
        __m256i _h = _H0;

        __m256i _target = _mm256_set1_epi8(target[i]);

        __m256i _M = separate_M_u8(_mat_shifted, _score_shift, _h, _query, _target);

        _h = _mm256_max_epu8(_M, _E);

        // Prefix Scan
        __m256i _f = prefix_scan_F_u8(_M, _gape, _gape_scan);

        _h = _mm256_max_epu8(
            _h,
            _mm256_subs_epu8(_f, _gapoe));

        _mm256_store_si256((__m256i*)q_mem, _h);
        uint8_t lscore = q_mem[qlen - 1];

        if (lscore >= gscore)
        {
            gscore = lscore;
            max_ie = i;
        }

        if (_mm256_testz_si256(_h, _cmask))
            break;

        update_column_maximums_u8(_mm, _mi, i, _h);

#ifdef UNIT_TEST_EXTEND
        for (size_t dj = 0; dj < qlen; dj++)
        {
            size_t dm = 6ull * (i * qlen + dj);

            assert(_query.m256i_u8[dj] == pMem[dm]);
            assert(_M.m256i_u8[dj] == pMem[dm + 2]);
            assert(_E.m256i_u8[dj] == pMem[dm + 3]);
            //           assert(_F.m256i_u8[di] == pMem[dm + 4]);
            assert(_h.m256i_u8[dj] == pMem[dm + 5]);
        }
#endif
        // now compute E'(i+1,j)
        _E = _mm256_max_epu8(
            _mm256_subs_epu8(_E, _gape),
            _mm256_subs_epu8(_M, _gapoe));

        // Prepare for the next row
        _h1 = _mm256_subs_epu8(_h1, _gape);

        _H0 = _mm256_shift_left_si256<1>(_h, _h1);
    }

    int max_score = h0, max_i = -1, max_j = -1, max_off = 0;

    __m256i _mj = _in;

    __m256i _mmask256 = _cmask;

    __m128i _mmask = _mm256_castsi256_si128(_mmask256);

    __m128i _kmask = _mm256_extracti128_si256(_mmask256, 1);

    // Parallel reduce
    __m128i _max_score = _mm256_castsi256_si128(_mm);
    __m128i _max_i = _mm256_castsi256_si128(_mi);
    __m128i _max_j = _mm256_castsi256_si128(_mj);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm256_extracti128_si256(_mm, 1),
        _mm256_extracti128_si256(_mi, 1),
        _mm256_extracti128_si256(_mj, 1),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 8);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 8),
        _mm_srli_si128(_max_i, 8),
        _mm_srli_si128(_max_j, 8),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 4);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 4),
        _mm_srli_si128(_max_i, 4),
        _mm_srli_si128(_max_j, 4),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 2);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 2),
        _mm_srli_si128(_max_i, 2),
        _mm_srli_si128(_max_j, 2),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 1);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 1),
        _mm_srli_si128(_max_i, 1),
        _mm_srli_si128(_max_j, 1),
        _kmask);

    int max_score_candidate = _mm_extract_epi8(_max_score, 0);

    if (max_score < max_score_candidate)
    {
        max_score = max_score_candidate;

        max_i = _mm_extract_epi8(_max_i, 0);
        max_j = _mm_extract_epi8(_max_j, 0);

        int off = abs(max_j - max_i);
        if (off > max_off)
        {
            max_off = off;
        }
    }

    if (_qle)*_qle = max_j + 1;
    if (_tle)*_tle = max_i + 1;
    if (_gtle)*_gtle = max_ie + 1;
    if (_gscore)*_gscore = gscore;
    if (_max_off)*_max_off = max_off;

#ifdef UNIT_TEST_EXTEND

    if (max_score != d_max)
    {
        ksw_extend_dump("ksw_extend_u8_32.csv", pMem, qlen, tlen);

        assert(max_score == d_max);
    }

    if (max_i != d_max_i)
    {
        ksw_extend_dump("ksw_extend_u8_32.csv", pMem, qlen, tlen);

        assert(max_i == d_max_i);
    }

    if (max_j != d_max_j)
    {
        ksw_extend_dump("ksw_extend_u8_32.csv", pMem, qlen, tlen);

        assert(max_j == d_max_j);
    }

    if (gscore != d_gscore)
    {
        ksw_extend_dump("ksw_extend_u8_32.csv", pMem, qlen, tlen);

        assert(gscore == d_gscore);
    }

    if (max_ie != d_max_ie)
    {
        ksw_extend_dump("ksw_extend_u8_32.csv", pMem, qlen, tlen);

        assert(max_ie == d_max_ie);
    }

    free(pMem);
#endif

    return max_score;
}

int ksw_extend_u8_16(int qlen, const uint8_t* query, int tlen, const uint8_t* target, __m256i _score_matrix, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int* _qle, int* _tle, int* _gtle, int* _gscore, int* _max_off)
{
    assert(qlen < 16);

    int gscore = -1, max_ie = -1;

    __m128i _zero = _mm_setzero_si128();

    assert(h0 > 0 && w > 0);

#ifdef UNIT_TEST_EXTEND
    int d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off;

    int* pMem = ksw_extend_debug(qlen, query, tlen, target, _score_matrix, gapo, gape, gapo, gape, w, end_bonus, zdrop, h0, d_max, d_max_i, d_max_j, d_max_ie, d_gscore, d_max_off);
#endif

    // initialization
    __m128i _gape = _mm_set1_epi8(gape);
    __m128i _gapoe = _mm_set1_epi8(gapo + gape);


    __m128i _in = _mm_set_epi8(
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m128i _cmask = _mm_cmpgt_epi8(
        _mm_set1_epi8(qlen),
        _in);

    __m128i _mm_score_shift = _mm256_castsi256_si128(
        setup_score_shift(
            _score_matrix));

    __m128i _mm_score_matrix = _mm_add_epi8(
        _mm256_castsi256_si128(
            _score_matrix),
        _mm_score_shift);

    alignas(64) uint8_t q_mem[16];

    std::memset(q_mem, 0xff, 16);
    std::memcpy(q_mem, query, qlen);
    __m128i _query = _mm_load_si128((__m128i*)q_mem);

    // initialize the first column
    __m128i _h0 = _mm_set1_epi8(h0);

    __m128i _h1 = _mm_subs_epu8(
        _h0,
        _mm_set1_epi8(gapo));

    // initialize the first row
    __m128i _H0 = _mm_prefix_add_epi8(_gape);

    _H0 = _mm_shift_left_si128<1>(
        _mm_subs_epu8(
            _mm_subs_epu8(
                _h0,
                _mm_set1_epi8(gapo)),
            _H0),
        _h0);

    __m128i _E = _mm_setzero_si128();

    __m128i _mm = _zero, _mi = _zero;

    for (int i = 0; LIKELY(i < tlen); ++i) {
        __m128i _h = _H0;

        __m128i _target = _mm_set1_epi8(target[i]);

        __m128i _M = separate_M_u8(_mm_score_matrix, _mm_score_shift, _h, _query, _target);

        _h = _mm_max_epu8(_M, _E);

        // Prefix Scan
        __m128i _f = prefix_scan_F_u8(_M, _gape);

        _h = _mm_max_epu8(
            _h,
            _mm_subs_epu8(_f, _gapoe));

        _mm_store_si128((__m128i*)q_mem, _h);
        uint8_t lscore = q_mem[qlen - 1];

        if (lscore >= gscore)
        {
            gscore = lscore;
            max_ie = i;
        }

        if (_mm_testz_si128(_h, _cmask))
            break;

        update_column_maximums_u8(_mm, _mi, i, _h);

#ifdef UNIT_TEST_EXTEND
        for (size_t dj = 0; dj < qlen; dj++)
        {
            size_t dm = 6ull * (i * qlen + dj);

            assert(_query.m128i_u8[dj] == pMem[dm]);
            assert(_M.m128i_u8[dj] == pMem[dm + 2]);
            assert(_E.m128i_u8[dj] == pMem[dm + 3]);
            assert(_h.m128i_u8[dj] == pMem[dm + 5]);
        }
#endif

        // now compute E'(i+1,j)
        _E = _mm_max_epu8(
            _mm_subs_epu8(_E, _gape),
            _mm_subs_epu8(_M, _gapoe));

        // Prepare for the next row
        _h1 = _mm_subs_epu8(_h1, _gape);

        _H0 = _mm_shift_left_si128<1>(_h, _h1);
    }

    int max_score = h0, max_i = -1, max_j = -1, max_off = 0;

    __m128i _mj = _in;

    __m128i _mmask = _cmask;

    __m128i _kmask = _mm_srli_si128(_mmask, 8);

    // Parallel reduce
    __m128i _max_score = _mm;
    __m128i _max_i = _mi;
    __m128i _max_j = _mj;

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 8),
        _mm_srli_si128(_max_i, 8),
        _mm_srli_si128(_max_j, 8),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 4);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 4),
        _mm_srli_si128(_max_i, 4),
        _mm_srli_si128(_max_j, 4),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 2);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 2),
        _mm_srli_si128(_max_i, 2),
        _mm_srli_si128(_max_j, 2),
        _kmask);

    _mmask = _mm_or_si128(_mmask, _kmask);
    _kmask = _mm_srli_si128(_mmask, 1);

    update_max_ij_i8(_max_score, _max_i, _max_j,
        _mm_srli_si128(_max_score, 1),
        _mm_srli_si128(_max_i, 1),
        _mm_srli_si128(_max_j, 1),
        _kmask);

    int max_score_candidate = _mm_extract_epi8(_max_score, 0);

    if (max_score < max_score_candidate)
    {
        max_score = max_score_candidate;

        max_i = _mm_extract_epi8(_max_i, 0);
        max_j = _mm_extract_epi8(_max_j, 0);

        int off = abs(max_j - max_i);
        if (off > max_off)
        {
            max_off = off;
        }
    }

    if (_qle)*_qle = max_j + 1;
    if (_tle)*_tle = max_i + 1;
    if (_gtle)*_gtle = max_ie + 1;
    if (_gscore)*_gscore = gscore;
    if (_max_off)*_max_off = max_off;

#ifdef UNIT_TEST_EXTEND

    if (max_score != d_max)
    {
        ksw_extend_dump("ksw_extend_u8_16.csv", pMem, qlen, tlen);

        assert(max_score == d_max);
    }

    if (max_i != d_max_i)
    {
        ksw_extend_dump("ksw_extend_u8_16.csv", pMem, qlen, tlen);

        assert(max_i == d_max_i);
    }

    if (max_j != d_max_j)
    {
        ksw_extend_dump("ksw_extend_u8_16.csv", pMem, qlen, tlen);

        assert(max_j == d_max_j);
    }

    if (gscore != d_gscore)
    {
        ksw_extend_dump("ksw_extend_u8_16.csv", pMem, qlen, tlen);

        assert(gscore == d_gscore);
    }

    assert(max_ie == d_max_ie);
    free(pMem);
#endif

    return max_score;
}

int ksw_extend_avx2(int qlen, const uint8_t* query, int tlen, const uint8_t* target, int m, const int8_t* mat, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int* qle, int* tle, int* gtle, int* gscore, int* max_off)
{
    int ret;

    assert(m == 5);

    __m256i _q = _mm256_set_epi8(
        3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0,
        3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

    __m256i _t = _mm256_set_epi8(
        3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0,
        3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

    __m256i _mask = _mm256_cmpeq_epi8(
        _q,
        _t);

    __m256i _score_matrix = _mm256_blendv_epi8(
        _mm256_set1_epi8(mat[1]),
        _mm256_set1_epi8(mat[0]),
        _mask);

    if ((h0 + qlen < 250) && (tlen < 250)) {
        // row position is uint8_t
        if (qlen < 16) {
            ret = ksw_extend_u8_16(qlen, query, tlen, target, _score_matrix, gapo, gape, w, end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
        }
        else if (qlen < 32) {
            ret = ksw_extend_u8_32(qlen, query, tlen, target, _score_matrix, gapo, gape, w, end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
        }
        else {
            ret = ksw_extend_u8(qlen, query, tlen, target, _score_matrix, gapo, gape, w, end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
        }
    }
    else {
        if (qlen < 16) {
            ret = ksw_extend_i16_16(qlen, query, tlen, target, _score_matrix, gapo, gape, w, end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
        }
        else {
            ret = ksw_extend_i16(qlen, query, tlen, target, _score_matrix, gapo, gape, w, end_bonus, zdrop, h0, qle, tle, gtle, gscore, max_off);
        }
    }

    return ret;
}

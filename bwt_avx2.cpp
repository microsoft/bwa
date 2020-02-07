/*++

Module Name:

    bwt_avx2.cpp

Abstract:

    BWT AVX2 performance functions.

Authors:

    Roman Snytsar, June, 2018

Environment:
`
    User mode service.

Revision History:


--*/
#include <assert.h>

#include <chrono>

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>

#include "kvec.h"
#include "utils.h"

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

#include "bwt_perf.h"
#include "bwt_avx.h"

#define _mm_permute_epi64(a, imm) _mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(a), imm))

#define _mm256_permute_epi64(a, imm) _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(a), imm))

#define _mm256_shuffle_epi64(a, b, imm) _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), imm))

#define _mm256_blend_epi64(a, b, imm) _mm256_castpd_si256(_mm256_blend_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), imm))

const __m256i _mm256_true_si256 = _mm256_set_epi32(
    -1, -1, -1, -1, -1, -1, -1, -1);

const __m256i _mm256_occ_lut = _mm256_set_epi8(
    0x80, 0x50, 0x44, 0x41, 0x50, 0x20, 0x14, 0x11, 0x44, 0x14, 0x08, 0x05, 0x41, 0x11, 0x05, 0x02,
    0x80, 0x50, 0x44, 0x41, 0x50, 0x20, 0x14, 0x11, 0x44, 0x14, 0x08, 0x05, 0x41, 0x11, 0x05, 0x02);

const __m256i _mm256_occ_lut_inv = _mm256_set_epi8(
    0x7f, 0xaf, 0xbb, 0xbe, 0xaf, 0xdf, 0xeb, 0xee, 0xbb, 0xeb, 0xf7, 0xfa, 0xbe, 0xee, 0xfa, 0xfd,
    0x7f, 0xaf, 0xbb, 0xbe, 0xaf, 0xdf, 0xeb, 0xee, 0xbb, 0xeb, 0xf7, 0xfa, 0xbe, 0xee, 0xfa, 0xfd);

const __m256i _mm256_0f_epi8 = _mm256_set1_epi8(
    0x0f);

const __m256i _mm256_fc_epi8 = _mm256_set1_epi8(
    0xfc);

const __m256i _mm256_0246_epi32 = _mm256_set_epi32(
    0, 0, 2, 2, 4, 4, 6, 6);

const __m256i _mm256_2064_epi32 = _mm256_set_epi32(
    2, 2, 0, 0, 6, 6, 4, 4);

const __m256i _mm256_4602_epi32 = _mm256_set_epi32(
    4, 4, 6, 6, 0, 0, 2, 2);

const __m256i _mm256_6420_epi32 = _mm256_set_epi32(
    6, 6, 4, 4, 2, 2, 0, 0);

const __m256i _mm256_c_inc_epi32 = _mm256_set_epi32(
    1, 0, 1, 0, 1, 0, 1, 0);

__forceinline __m256i _occ_lookup(__m256i *p, const __m256i _L2, const __m128i _pre_mask, const __m128i _adjust, __m256i& _bwt, __m256i& _occlo, __m256i& _occhi)
{
    // Load counts
    __m256i _cnt = _mm256_load_si256(p++);

    // Prepare bwt mask
    __m256i _mask = _mm256_sllv_epi32(
        _mm256_true_si256,
        _mm256_cvtepi8_epi32(
            _pre_mask));

    // Load bwt
    _bwt = _mm256_load_si256(p);

    // Pre-adjust total count
    __m256i _total = _mm256_add_epi64(
        _cnt,
        _mm256_add_epi64(
            _mm256_cvtepi16_epi64(_adjust),
            _L2));

    // Mask the bwt
    __m256i _bwt_masked = _mm256_and_si256(
        _mask,
        _bwt);

    // Lookup
    _occlo = _mm256_shuffle_epi8(
        _mm256_occ_lut_inv,
        _mm256_and_si256(
            _bwt_masked,
            _mm256_0f_epi8));

    _occhi = _mm256_shuffle_epi8(
        _mm256_occ_lut,
        _mm256_and_si256(
            _mm256_srli_epi32(
                _bwt_masked,
                4),
            _mm256_0f_epi8));

    return _total;
}

__forceinline __m256i _occ_extract(const __m256i _total, const __m256i _occlo, const __m256i _occhi, const __m256i _c_x_2)
{
    // Extraction 
    __m256i _occ_1 = _mm256_sad_epu8(
        _mm256_or_si256(
            _mm256_fc_epi8,
            _mm256_srlv_epi32(
                _occlo,
                _c_x_2)),
        _mm256_andnot_si256(
            _mm256_fc_epi8,
            _mm256_srlv_epi32(
                _occhi,
                _c_x_2)));

    // Additions
    __m256i _occ_2 = _mm256_add_epi64(
        _occ_1,
        _mm256_permute_epi64(
            _occ_1,
            0x5));

    __m256i _occ_3 = _mm256_add_epi64(
        _occ_2,
        _mm256_permute4x64_epi64(
            _occ_2,
            0x4e));

    __m256i _occ_4 = _mm256_sub_epi64(
        _total,
        _occ_3);

    __m256i _cntc = _mm256_permutevar8x32_epi32(
        _occ_4,
        _mm256_add_epi32(
            _mm256_c_inc_epi32,
            _c_x_2));

    return _cntc;
}

__forceinline void _occ_epilogue(const bwt_t *bwt, __m128i _kl, __m128i& _kl2, const char* &p_k, const char* &p_l)
{
    _kl = _mm_blendv_epi8(
        _kl,
        _mm_add_epi64(
            _kl,
            _mm_one_epi64),
        _mm_cmpgt_epi64(
            _mm_set1_epi64x(bwt->primary),
            _kl));

    _kl2 = _mm_and_si128(
        _mm_occ_intv_mask,
        _kl);

    __m128i _kl1 = _mm_add_epi64(
        _mm_srli_epi64(
            _mm_andnot_si128(
                _mm_occ_intv_mask,
                _kl),
            1),
        _mm_set1_epi64x((int64_t)bwt->bwt));

    p_k = (const char*)_mm_extract_epi64(_kl1, 0);
    p_l = (const char*)_mm_extract_epi64(_kl1, 1);
};

__forceinline __m256i _occ_kernel(__m256i *p, const __m256i _L2, const __m128i _pre_mask, const __m128i _adjust, const __m256i _c_x_2)
{
    __m256i _bwt, _occlo, _occhi;

    // Lookup
    __m256i _total = _occ_lookup(p, _L2, _pre_mask, _adjust, _bwt, _occlo, _occhi);

    // Extraction and additions
    __m256i _cntc = _occ_extract(_total, _occlo, _occhi, _c_x_2);

    return _cntc;
}

__forceinline bwtint_t bwt_invPsi_avx2(const bwt_t *bwt, bwtint_t k)
{
    __m256i _L2 = _mm256_loadu_si256((__m256i *)bwt->L2);

    __m128i _k = _mm_cvtsi64_si128(k);

    __m128i _kl = _mm_blendv_epi8(
        _k,
        _mm_add_epi64(
            _k,
            _mm_one_epi64),
        _mm_cmpgt_epi64(
            _mm_set1_epi64x(bwt->primary),
            _k));

    __m128i _k2 = _mm_and_si128(
        _mm_occ_intv_mask,
        _kl);

    __m128i _k1 = _mm_srli_epi64(
        _mm_andnot_si128(
            _mm_occ_intv_mask,
            _kl),
        1);

    __m256i *p = (__m256i *)((uint8_t *)(bwt->bwt) + _mm_cvtsi128_si64(_k1));

    __m256i _k2_broadcast = _mm256_broadcastb_epi8(_k2);

    __m128i _pre_mask = _occ_premask(
        _mm256_castsi256_si128(
            _k2_broadcast));

    __m128i _adjust = _mm_add_epi16(
        _mm_occ_adjust,
        _k2);

    __m256i _bwt, _occlo, _occhi;

    // Lookup
    __m256i _total = _occ_lookup(p, _L2, _pre_mask, _adjust, _bwt, _occlo, _occhi);

    __m256i _c_x_2 = _mm256_slli_epi32(
        _mm256_permutevar8x32_epi32(
            _mm256_andnot_si256(
                _mm256_srai_epi32(
                    _mm256_fc_epi8,
                    24),
                _mm256_srlv_epi32(
                    _bwt,
                    _mm256_srli_epi32(
                        _mm256_andnot_si256(
                            _k2_broadcast,
                            _mm256_0f_epi8),
                        23))),
            _mm256_srli_epi32(
                _k2_broadcast,
                28)),
        1);

    // Extraction and additions
    __m256i _cntc = _occ_extract(_total, _occlo, _occhi, _c_x_2);

    bwtint_t psi = _mm_cvtsi128_si64(
        _mm_andnot_si128(
            _mm_cmpeq_epi64(
                _k,
                _mm_cvtsi64_si128(bwt->primary)),
            _mm256_castsi256_si128(_cntc)));

    return psi;
}

__forceinline bwtint_t bwt_invPsi_avx2(const bwt_t *bwt, const bwtint_t k, __m128i &_k2, const char* &p_k)
{
    __m256i _L2 = _mm256_loadu_si256((__m256i *)bwt->L2);

    __m128i _k = _mm_cvtsi64_si128(k);

    __m256i _k2_broadcast = _mm256_broadcastb_epi8(_k2);

    __m128i _pre_mask = _occ_premask(
        _mm256_castsi256_si128(
            _k2_broadcast));

    __m128i _adjust = _mm_add_epi16(
        _mm_occ_adjust,
        _k2);

    __m256i _bwt, _occlo, _occhi;

    // Lookup
    __m256i _total = _occ_lookup((__m256i*)p_k, _L2, _pre_mask, _adjust, _bwt, _occlo, _occhi);

    __m256i _c_x_2 = _mm256_slli_epi32(
        _mm256_permutevar8x32_epi32(
            _mm256_andnot_si256(
                _mm256_srai_epi32(
                    _mm256_fc_epi8,
                    24),
                _mm256_srlv_epi32(
                    _bwt,
                    _mm256_srli_epi32(
                        _mm256_andnot_si256(
                            _k2_broadcast,
                            _mm256_0f_epi8),
                        23))),
            _mm256_srli_epi32(
                _k2_broadcast,
                28)),
        1);

    // Extraction and additions
    __m256i _cntc = _occ_extract(_total, _occlo, _occhi, _c_x_2);

    bwtint_t psi = _mm_cvtsi128_si64(
        _mm_andnot_si128(
            _mm_cmpeq_epi64(
                _k,
                _mm_cvtsi64_si128(bwt->primary)),
            _mm256_castsi256_si128(_cntc)));

    // Dummy pointer
    const char *p_l;

    _occ_epilogue(
        bwt,
        _mm_cvtsi64_si128(psi),
        _k2,
        p_k,
        p_l);

    return psi;
}

#ifdef UNIT_TEST_SEED
bwt_t *bwt_debug;
#endif

static inline bwtint_t bwt_invPsi(const bwt_t *bwt, bwtint_t k) // compute inverse CSA
{
    bwtint_t x = k - (k > bwt->primary);
    x = bwt_B0(bwt, x);
    x = bwt->L2[x] + bwt_occ(bwt, k, x);
    return (k == bwt->primary) ? 0 : x;
}

bwtint_t bwt_sa_avx2(const bwt_t *bwt, bwtint_t k)
{
#ifdef UNIT_TEST_SEED
    bwtint_t d_ret = bwt_sa(bwt_debug, k);
#endif	


    bwtint_t sa = 0, mask = bwt->sa_intv - 1;
    while (k & mask) {
        ++sa;

#ifdef UNIT_TEST_SEED
        bwtint_t d_k = bwt_invPsi(bwt_debug, k);
#endif	

        k = bwt_invPsi_avx2(bwt, k);

#ifdef UNIT_TEST_SEED
        assert(d_k == k);
#endif	
    }
    /* without setting bwt->sa[0] = -1, the following line should be
    changed to (sa + bwt->sa[k/bwt->sa_intv]) % (bwt->seq_len + 1) */
    bwtint_t ret = sa + bwt->sa[k / bwt->sa_intv];

#ifdef UNIT_TEST_SEED
    assert(d_ret == ret);
#endif

    return ret;
}

void bwt_sa_bulk_avx2(const bwt_t *bwt, bwtint_t *indices, bwtint_t *values, bwtint_t count)
{
    bwtint_t intv_mask = bwt->sa_intv - 1;
    bwtint_t intv_shift = _tzcnt_u64(bwt->sa_intv);

    bwtint_t c = 0;

    // Dummy pointer
    const char *p_l;

    if (c < count)
    {
        bwtint_t sa0 = 0;
        bwtint_t index0 = c++;
        bwtint_t k0 = indices[index0];

        __m128i _kl0;
        const char *p_k0;
        _occ_epilogue(bwt, _mm_cvtsi64_si128(k0), _kl0, p_k0, p_l);
        _mm_prefetch(p_k0, _MM_HINT_T1);

        if (c < count)
        {
            bwtint_t sa1 = 0;
            bwtint_t index1 = c++;
            bwtint_t k1 = indices[index1];

            __m128i _kl1;
            const char *p_k1;
            _occ_epilogue(bwt, _mm_cvtsi64_si128(k1), _kl1, p_k1, p_l);
            _mm_prefetch(p_k1, _MM_HINT_T1);

            if (c < count)
            {
                bwtint_t sa2 = 0;
                bwtint_t index2 = c++;
                bwtint_t k2 = indices[index2];

                __m128i _kl2;
                const char *p_k2;
                _occ_epilogue(bwt, _mm_cvtsi64_si128(k2), _kl2, p_k2, p_l);
                _mm_prefetch(p_k2, _MM_HINT_T1);

                if (c < count)
                {
                    bwtint_t sa3 = 0;
                    bwtint_t index3 = c++;
                    bwtint_t k3 = indices[index3];

                    __m128i _kl3;
                    const char *p_k3;
                    _occ_epilogue(bwt, _mm_cvtsi64_si128(k3), _kl3, p_k3, p_l);
                    _mm_prefetch(p_k3, _MM_HINT_T1);

                    if (c < count)
                    {
                        bwtint_t sa4 = 0;
                        bwtint_t index4 = c++;
                        bwtint_t k4 = indices[index4];

                        __m128i _kl4;
                        const char *p_k4;
                        _occ_epilogue(bwt, _mm_cvtsi64_si128(k4), _kl4, p_k4, p_l);
                        _mm_prefetch(p_k4, _MM_HINT_T1);

                        if (c < count)
                        {
                            bwtint_t sa5 = 0;
                            bwtint_t index5 = c++;
                            bwtint_t k5 = indices[index5];

                            __m128i _kl5;
                            const char *p_k5;
                            _occ_epilogue(bwt, _mm_cvtsi64_si128(k5), _kl5, p_k5, p_l);
                            _mm_prefetch(p_k5, _MM_HINT_T1);

                            if (c < count)
                            {
                                bwtint_t sa6 = 0;
                                bwtint_t index6 = c++;
                                bwtint_t k6 = indices[index6];

                                __m128i _kl6;
                                const char *p_k6;
                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k6), _kl6, p_k6, p_l);
                                _mm_prefetch(p_k6, _MM_HINT_T1);
                                _mm_prefetch(p_k0, _MM_HINT_T0);

                                if (c < count)
                                {
                                    bwtint_t sa7 = 0;
                                    bwtint_t index7 = c++;
                                    bwtint_t k7 = indices[index7];

                                    __m128i _kl7;
                                    const char *p_k7;
                                    _occ_epilogue(bwt, _mm_cvtsi64_si128(k7), _kl7, p_k7, p_l);
                                    _mm_prefetch(p_k7, _MM_HINT_T1);
                                    _mm_prefetch(p_k1, _MM_HINT_T0);

                                    // Unrolled loop of 8
                                    for (;;)
                                    {
                                        if ((k0 & intv_mask) == 0)
                                        {
                                            values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa0 = 0;
                                                index0 = c++;
                                                k0 = indices[index0];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k0), _kl0, p_k0, p_l);
                                                _mm_prefetch(p_k0, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa0 = sa7;
                                                index0 = index7;
                                                k0 = k7;
                                                _kl0 = _kl7;
                                                p_k0 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa0++;
                                            k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                                            if ((k0 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k0, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k2, _MM_HINT_T0);

                                        if ((k1 & intv_mask) == 0)
                                        {
                                            values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa1 = 0;
                                                index1 = c++;
                                                k1 = indices[index1];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k1), _kl1, p_k1, p_l);
                                                _mm_prefetch(p_k1, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa1 = sa7;
                                                index1 = index7;
                                                k1 = k7;
                                                _kl1 = _kl7;
                                                p_k1 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa1++;
                                            k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                                            if ((k1 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k1, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k3, _MM_HINT_T0);

                                        if ((k2 & intv_mask) == 0)
                                        {
                                            values[index2] = sa2 + bwt->sa[k2 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa2 = 0;
                                                index2 = c++;
                                                k2 = indices[index2];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k2), _kl2, p_k2, p_l);
                                                _mm_prefetch(p_k2, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa2 = sa7;
                                                index2 = index7;
                                                k2 = k7;
                                                _kl2 = _kl7;
                                                p_k2 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa2++;
                                            k2 = bwt_invPsi_avx2(bwt, k2, _kl2, p_k2);

                                            if ((k2 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k2 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k2, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k4, _MM_HINT_T0);

                                        if ((k3 & intv_mask) == 0)
                                        {
                                            values[index3] = sa3 + bwt->sa[k3 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa3 = 0;
                                                index3 = c++;
                                                k3 = indices[index3];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k3), _kl3, p_k3, p_l);
                                                _mm_prefetch(p_k3, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa3 = sa7;
                                                index3 = index7;
                                                k3 = k7;
                                                _kl3 = _kl7;
                                                p_k3 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa3++;
                                            k3 = bwt_invPsi_avx2(bwt, k3, _kl3, p_k3);

                                            if ((k3 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k3 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k3, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k5, _MM_HINT_T0);

                                        if ((k4 & intv_mask) == 0)
                                        {
                                            values[index4] = sa4 + bwt->sa[k4 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa4 = 0;
                                                index4 = c++;
                                                k4 = indices[index4];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k4), _kl4, p_k4, p_l);
                                                _mm_prefetch(p_k4, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa4 = sa7;
                                                index4 = index7;
                                                k4 = k7;
                                                _kl4 = _kl7;
                                                p_k4 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa4++;
                                            k4 = bwt_invPsi_avx2(bwt, k4, _kl4, p_k4);

                                            if ((k4 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k4 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k4, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k6, _MM_HINT_T0);

                                        if ((k5 & intv_mask) == 0)
                                        {
                                            values[index5] = sa5 + bwt->sa[k5 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa5 = 0;
                                                index5 = c++;
                                                k5 = indices[index5];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k5), _kl5, p_k5, p_l);
                                                _mm_prefetch(p_k5, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa5 = sa7;
                                                index5 = index7;
                                                k5 = k7;
                                                _kl5 = _kl7;
                                                p_k5 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa5++;
                                            k5 = bwt_invPsi_avx2(bwt, k5, _kl5, p_k5);

                                            if ((k5 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k5 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k5, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k7, _MM_HINT_T0);

                                        if ((k6 & intv_mask) == 0)
                                        {
                                            values[index6] = sa6 + bwt->sa[k6 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa6 = 0;
                                                index6 = c++;
                                                k6 = indices[index6];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k6), _kl6, p_k6, p_l);
                                                _mm_prefetch(p_k6, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                // Copy last task to the now empty slot
                                                sa6 = sa7;
                                                index6 = index7;
                                                k6 = k7;
                                                _kl6 = _kl7;
                                                p_k6 = p_k7;

                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa6++;
                                            k6 = bwt_invPsi_avx2(bwt, k6, _kl6, p_k6);

                                            if ((k6 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k6 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k6, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k0, _MM_HINT_T0);

                                        if ((k7 & intv_mask) == 0)
                                        {
                                            values[index7] = sa7 + bwt->sa[k7 >> intv_shift];

                                            if (c < count)
                                            {
                                                // Pull new task
                                                sa7 = 0;
                                                index7 = c++;
                                                k7 = indices[index7];

                                                _occ_epilogue(bwt, _mm_cvtsi64_si128(k7), _kl7, p_k7, p_l);
                                                _mm_prefetch(p_k7, _MM_HINT_T1);
                                            }
                                            else
                                            {
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            sa7++;
                                            k7 = bwt_invPsi_avx2(bwt, k7, _kl7, p_k7);

                                            if ((k7 & intv_mask) == 0)
                                            {
                                                _mm_prefetch((const char*)(bwt->sa + (k7 >> intv_shift)), _MM_HINT_NTA);
                                            }
                                            else
                                            {
                                                _mm_prefetch(p_k7, _MM_HINT_T1);
                                            }
                                        }
                                        _mm_prefetch(p_k1, _MM_HINT_T0);
                                    }
                                }

                                // Unrolled loop of 7
                                for (;;)
                                {
                                    if ((k0 & intv_mask) == 0)
                                    {
                                        values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                                        // Copy last task to the now empty slot
                                        sa0 = sa6;
                                        index0 = index6;
                                        k0 = k6;
                                        _kl0 = _kl6;
                                        p_k0 = p_k6;

                                        break;
                                    }
                                    else
                                    {
                                        sa0++;
                                        k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                                        if ((k0 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k0, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k2, _MM_HINT_T0);

                                    if ((k1 & intv_mask) == 0)
                                    {
                                        values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                                        // Copy last task to the now empty slot
                                        sa1 = sa6;
                                        index1 = index6;
                                        k1 = k6;
                                        _kl1 = _kl6;
                                        p_k1 = p_k6;

                                        break;
                                    }
                                    else
                                    {
                                        sa1++;
                                        k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                                        if ((k1 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k1, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k3, _MM_HINT_T0);

                                    if ((k2 & intv_mask) == 0)
                                    {
                                        values[index2] = sa2 + bwt->sa[k2 >> intv_shift];

                                        // Copy last task to the now empty slot
                                        sa2 = sa6;
                                        index2 = index6;
                                        k2 = k6;
                                        _kl2 = _kl6;
                                        p_k2 = p_k6;

                                        break;
                                    }
                                    else
                                    {
                                        sa2++;
                                        k2 = bwt_invPsi_avx2(bwt, k2, _kl2, p_k2);

                                        if ((k2 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k2 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k2, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k4, _MM_HINT_T0);

                                    if ((k3 & intv_mask) == 0)
                                    {
                                        values[index3] = sa3 + bwt->sa[k3 >> intv_shift];

                                        // Copy last task to the now empty slot
                                        sa3 = sa6;
                                        index3 = index6;
                                        k3 = k6;
                                        _kl3 = _kl6;
                                        p_k3 = p_k6;

                                        break;
                                    }
                                    else
                                    {
                                        sa3++;
                                        k3 = bwt_invPsi_avx2(bwt, k3, _kl3, p_k3);

                                        if ((k3 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k3 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k3, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k5, _MM_HINT_T0);

                                    if ((k4 & intv_mask) == 0)
                                    {
                                        values[index4] = sa4 + bwt->sa[k4 >> intv_shift];

                                        // Copy last task to the now empty slot
                                        sa4 = sa6;
                                        index4 = index6;
                                        k4 = k6;
                                        _kl4 = _kl6;
                                        p_k4 = p_k6;

                                        break;
                                    }
                                    else
                                    {
                                        sa4++;
                                        k4 = bwt_invPsi_avx2(bwt, k4, _kl4, p_k4);

                                        if ((k4 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k4 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k4, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k6, _MM_HINT_T0);

                                    if ((k5 & intv_mask) == 0)
                                    {
                                        values[index5] = sa5 + bwt->sa[k5 >> intv_shift];

                                        // Copy last task to the now empty slot
                                        sa5 = sa6;
                                        index5 = index6;
                                        k5 = k6;
                                        _kl5 = _kl6;
                                        p_k5 = p_k6;

                                        break;
                                    }
                                    else
                                    {
                                        sa5++;
                                        k5 = bwt_invPsi_avx2(bwt, k5, _kl5, p_k5);

                                        if ((k5 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k5 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k5, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k0, _MM_HINT_T0);

                                    if ((k6 & intv_mask) == 0)
                                    {
                                        values[index6] = sa6 + bwt->sa[k6 >> intv_shift];

                                        break;
                                    }
                                    else
                                    {
                                        sa6++;
                                        k6 = bwt_invPsi_avx2(bwt, k6, _kl6, p_k6);

                                        if ((k6 & intv_mask) == 0)
                                        {
                                            _mm_prefetch((const char*)(bwt->sa + (k6 >> intv_shift)), _MM_HINT_NTA);
                                        }
                                        else
                                        {
                                            _mm_prefetch(p_k6, _MM_HINT_T1);
                                        }
                                    }
                                    _mm_prefetch(p_k1, _MM_HINT_T0);
                                }
                            }

                            // Unrolled loop of 6
                            for (;;)
                            {
                                if ((k0 & intv_mask) == 0)
                                {
                                    values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                                    // Copy last task to the now empty slot
                                    sa0 = sa5;
                                    index0 = index5;
                                    k0 = k5;
                                    _kl0 = _kl5;
                                    p_k0 = p_k5;

                                    break;
                                }
                                else
                                {
                                    sa0++;
                                    k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                                    if ((k0 & intv_mask) == 0)
                                    {
                                        _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                                    }
                                    else
                                    {
                                        _mm_prefetch(p_k0, _MM_HINT_T1);
                                    }
                                }
                                _mm_prefetch(p_k2, _MM_HINT_T0);

                                if ((k1 & intv_mask) == 0)
                                {
                                    values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                                    // Copy last task to the now empty slot
                                    sa1 = sa5;
                                    index1 = index5;
                                    k1 = k5;
                                    _kl1 = _kl5;
                                    p_k1 = p_k5;

                                    break;
                                }
                                else
                                {
                                    sa1++;
                                    k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                                    if ((k1 & intv_mask) == 0)
                                    {
                                        _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                                    }
                                    else
                                    {
                                        _mm_prefetch(p_k1, _MM_HINT_T1);
                                    }
                                }
                                _mm_prefetch(p_k3, _MM_HINT_T0);

                                if ((k2 & intv_mask) == 0)
                                {
                                    values[index2] = sa2 + bwt->sa[k2 >> intv_shift];

                                    // Copy last task to the now empty slot
                                    sa2 = sa5;
                                    index2 = index5;
                                    k2 = k5;
                                    _kl2 = _kl5;
                                    p_k2 = p_k5;

                                    break;
                                }
                                else
                                {
                                    sa2++;
                                    k2 = bwt_invPsi_avx2(bwt, k2, _kl2, p_k2);

                                    if ((k2 & intv_mask) == 0)
                                    {
                                        _mm_prefetch((const char*)(bwt->sa + (k2 >> intv_shift)), _MM_HINT_NTA);
                                    }
                                    else
                                    {
                                        _mm_prefetch(p_k2, _MM_HINT_T1);
                                    }
                                }
                                _mm_prefetch(p_k4, _MM_HINT_T0);

                                if ((k3 & intv_mask) == 0)
                                {
                                    values[index3] = sa3 + bwt->sa[k3 >> intv_shift];

                                    // Copy last task to the now empty slot
                                    sa3 = sa5;
                                    index3 = index5;
                                    k3 = k5;
                                    _kl3 = _kl5;
                                    p_k3 = p_k5;

                                    break;
                                }
                                else
                                {
                                    sa3++;
                                    k3 = bwt_invPsi_avx2(bwt, k3, _kl3, p_k3);

                                    if ((k3 & intv_mask) == 0)
                                    {
                                        _mm_prefetch((const char*)(bwt->sa + (k3 >> intv_shift)), _MM_HINT_NTA);
                                    }
                                    else
                                    {
                                        _mm_prefetch(p_k3, _MM_HINT_T1);
                                    }
                                }
                                _mm_prefetch(p_k5, _MM_HINT_T0);

                                if ((k4 & intv_mask) == 0)
                                {
                                    values[index4] = sa4 + bwt->sa[k4 >> intv_shift];

                                    // Copy last task to the now empty slot
                                    sa4 = sa5;
                                    index4 = index5;
                                    k4 = k5;
                                    _kl4 = _kl5;
                                    p_k4 = p_k5;

                                    break;
                                }
                                else
                                {
                                    sa4++;
                                    k4 = bwt_invPsi_avx2(bwt, k4, _kl4, p_k4);

                                    if ((k4 & intv_mask) == 0)
                                    {
                                        _mm_prefetch((const char*)(bwt->sa + (k4 >> intv_shift)), _MM_HINT_NTA);
                                    }
                                    else
                                    {
                                        _mm_prefetch(p_k4, _MM_HINT_T1);
                                    }
                                }
                                _mm_prefetch(p_k0, _MM_HINT_T0);

                                if ((k5 & intv_mask) == 0)
                                {
                                    values[index5] = sa5 + bwt->sa[k5 >> intv_shift];

                                    break;
                                }
                                else
                                {
                                    sa5++;
                                    k5 = bwt_invPsi_avx2(bwt, k5, _kl5, p_k5);

                                    if ((k5 & intv_mask) == 0)
                                    {
                                        _mm_prefetch((const char*)(bwt->sa + (k5 >> intv_shift)), _MM_HINT_NTA);
                                    }
                                    else
                                    {
                                        _mm_prefetch(p_k5, _MM_HINT_T1);
                                    }
                                }
                                _mm_prefetch(p_k1, _MM_HINT_T0);
                            }
                        }

                        // Unrolled loop of 5
                        for (;;)
                        {
                            if ((k0 & intv_mask) == 0)
                            {
                                values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                                // Copy last task to the now empty slot
                                sa0 = sa4;
                                index0 = index4;
                                k0 = k4;
                                _kl0 = _kl4;
                                p_k0 = p_k4;

                                break;
                            }
                            else
                            {
                                sa0++;
                                k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                                if ((k0 & intv_mask) == 0)
                                {
                                    _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                                }
                                else
                                {
                                    _mm_prefetch(p_k0, _MM_HINT_T1);
                                }
                            }
                            _mm_prefetch(p_k2, _MM_HINT_T0);

                            if ((k1 & intv_mask) == 0)
                            {
                                values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                                // Copy last task to the now empty slot
                                sa1 = sa4;
                                index1 = index4;
                                k1 = k4;
                                _kl1 = _kl4;
                                p_k1 = p_k4;

                                break;
                            }
                            else
                            {
                                sa1++;
                                k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                                if ((k1 & intv_mask) == 0)
                                {
                                    _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                                }
                                else
                                {
                                    _mm_prefetch(p_k1, _MM_HINT_T1);
                                }
                            }
                            _mm_prefetch(p_k3, _MM_HINT_T0);

                            if ((k2 & intv_mask) == 0)
                            {
                                values[index2] = sa2 + bwt->sa[k2 >> intv_shift];

                                // Copy last task to the now empty slot
                                sa2 = sa4;
                                index2 = index4;
                                k2 = k4;
                                _kl2 = _kl4;
                                p_k2 = p_k4;

                                break;
                            }
                            else
                            {
                                sa2++;
                                k2 = bwt_invPsi_avx2(bwt, k2, _kl2, p_k2);

                                if ((k2 & intv_mask) == 0)
                                {
                                    _mm_prefetch((const char*)(bwt->sa + (k2 >> intv_shift)), _MM_HINT_NTA);
                                }
                                else
                                {
                                    _mm_prefetch(p_k2, _MM_HINT_T1);
                                }
                            }
                            _mm_prefetch(p_k4, _MM_HINT_T0);

                            if ((k3 & intv_mask) == 0)
                            {
                                values[index3] = sa3 + bwt->sa[k3 >> intv_shift];

                                // Copy last task to the now empty slot
                                sa3 = sa4;
                                index3 = index4;
                                k3 = k4;
                                _kl3 = _kl4;
                                p_k3 = p_k4;

                                break;
                            }
                            else
                            {
                                sa3++;
                                k3 = bwt_invPsi_avx2(bwt, k3, _kl3, p_k3);

                                if ((k3 & intv_mask) == 0)
                                {
                                    _mm_prefetch((const char*)(bwt->sa + (k3 >> intv_shift)), _MM_HINT_NTA);
                                }
                                else
                                {
                                    _mm_prefetch(p_k3, _MM_HINT_T1);
                                }
                            }
                            _mm_prefetch(p_k0, _MM_HINT_T0);

                            if ((k4 & intv_mask) == 0)
                            {
                                values[index4] = sa4 + bwt->sa[k4 >> intv_shift];

                                break;
                            }
                            else
                            {
                                sa4++;
                                k4 = bwt_invPsi_avx2(bwt, k4, _kl4, p_k4);

                                if ((k4 & intv_mask) == 0)
                                {
                                    _mm_prefetch((const char*)(bwt->sa + (k4 >> intv_shift)), _MM_HINT_NTA);
                                }
                                else
                                {
                                    _mm_prefetch(p_k4, _MM_HINT_T1);
                                }
                            }
                            _mm_prefetch(p_k1, _MM_HINT_T0);
                        }
                    }

                    // Unrolled loop of 4
                    for (;;)
                    {
                        if ((k0 & intv_mask) == 0)
                        {
                            values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                            // Copy last task to the now empty slot
                            sa0 = sa3;
                            index0 = index3;
                            k0 = k3;
                            _kl0 = _kl3;
                            p_k0 = p_k3;

                            break;
                        }
                        else
                        {
                            sa0++;
                            k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                            if ((k0 & intv_mask) == 0)
                            {
                                _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                            }
                            else
                            {
                                _mm_prefetch(p_k0, _MM_HINT_T1);
                            }
                        }
                        _mm_prefetch(p_k2, _MM_HINT_T0);

                        if ((k1 & intv_mask) == 0)
                        {
                            values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                            // Copy last task to the now empty slot
                            sa1 = sa3;
                            index1 = index3;
                            k1 = k3;
                            _kl1 = _kl3;
                            p_k1 = p_k3;

                            break;
                        }
                        else
                        {
                            sa1++;
                            k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                            if ((k1 & intv_mask) == 0)
                            {
                                _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                            }
                            else
                            {
                                _mm_prefetch(p_k1, _MM_HINT_T1);
                            }
                        }
                        _mm_prefetch(p_k3, _MM_HINT_T0);

                        if ((k2 & intv_mask) == 0)
                        {
                            values[index2] = sa2 + bwt->sa[k2 >> intv_shift];

                            // Copy last task to the now empty slot
                            sa2 = sa3;
                            index2 = index3;
                            k2 = k3;
                            _kl2 = _kl3;
                            p_k2 = p_k3;

                            break;
                        }
                        else
                        {
                            sa2++;
                            k2 = bwt_invPsi_avx2(bwt, k2, _kl2, p_k2);

                            if ((k2 & intv_mask) == 0)
                            {
                                _mm_prefetch((const char*)(bwt->sa + (k2 >> intv_shift)), _MM_HINT_NTA);
                            }
                            else
                            {
                                _mm_prefetch(p_k2, _MM_HINT_T1);
                            }
                        }
                        _mm_prefetch(p_k0, _MM_HINT_T0);

                        if ((k3 & intv_mask) == 0)
                        {
                            values[index3] = sa3 + bwt->sa[k3 >> intv_shift];

                            break;
                        }
                        else
                        {
                            sa3++;
                            k3 = bwt_invPsi_avx2(bwt, k3, _kl3, p_k3);

                            if ((k3 & intv_mask) == 0)
                            {
                                _mm_prefetch((const char*)(bwt->sa + (k3 >> intv_shift)), _MM_HINT_NTA);
                            }
                            else
                            {
                                _mm_prefetch(p_k3, _MM_HINT_T1);
                            }
                        }
                        _mm_prefetch(p_k1, _MM_HINT_T0);
                    }
                }

                // Unrolled loop of 3
                for (;;)
                {
                    if ((k0 & intv_mask) == 0)
                    {
                        values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                        // Copy last task to the now empty slot
                        sa0 = sa2;
                        index0 = index2;
                        k0 = k2;
                        _kl0 = _kl2;
                        p_k0 = p_k2;

                        break;
                    }
                    else
                    {
                        sa0++;
                        k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                        if ((k0 & intv_mask) == 0)
                        {
                            _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                        }
                        else
                        {
                            _mm_prefetch(p_k0, _MM_HINT_T1);
                        }
                    }
                    _mm_prefetch(p_k2, _MM_HINT_T0);

                    if ((k1 & intv_mask) == 0)
                    {
                        values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                        // Copy last task to the now empty slot
                        sa1 = sa2;
                        index1 = index2;
                        k1 = k2;
                        _kl1 = _kl2;
                        p_k1 = p_k2;

                        break;
                    }
                    else
                    {
                        sa1++;
                        k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                        if ((k1 & intv_mask) == 0)
                        {
                            _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                        }
                        else
                        {
                            _mm_prefetch(p_k1, _MM_HINT_T1);
                        }
                    }
                    _mm_prefetch(p_k0, _MM_HINT_T0);

                    if ((k2 & intv_mask) == 0)
                    {
                        values[index2] = sa2 + bwt->sa[k2 >> intv_shift];

                        break;
                    }
                    else
                    {
                        sa2++;
                        k2 = bwt_invPsi_avx2(bwt, k2, _kl2, p_k2);

                        if ((k2 & intv_mask) == 0)
                        {
                            _mm_prefetch((const char*)(bwt->sa + (k2 >> intv_shift)), _MM_HINT_NTA);
                        }
                        else
                        {
                            _mm_prefetch(p_k2, _MM_HINT_T1);
                        }
                    }
                    _mm_prefetch(p_k1, _MM_HINT_T0);
                }
            }

            // Unrolled loop of 2
            for (;;)
            {
                if ((k0 & intv_mask) == 0)
                {
                    values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                    // Copy last task to the now empty slot
                    sa0 = sa1;
                    index0 = index1;
                    k0 = k1;
                    _kl0 = _kl1;
                    p_k0 = p_k1;

                    break;
                }
                else
                {
                    sa0++;
                    k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);

                    _mm_prefetch(p_k1, _MM_HINT_T0);
                    if ((k0 & intv_mask) == 0)
                    {
                        _mm_prefetch((const char*)(bwt->sa + (k0 >> intv_shift)), _MM_HINT_NTA);
                    }
                    else
                    {
                        _mm_prefetch(p_k0, _MM_HINT_T1);
                    }

                }

                if ((k1 & intv_mask) == 0)
                {
                    values[index1] = sa1 + bwt->sa[k1 >> intv_shift];

                    break;
                }
                else
                {
                    sa1++;
                    k1 = bwt_invPsi_avx2(bwt, k1, _kl1, p_k1);

                    _mm_prefetch(p_k0, _MM_HINT_T0);
                    if ((k1 & intv_mask) == 0)
                    {
                        _mm_prefetch((const char*)(bwt->sa + (k1 >> intv_shift)), _MM_HINT_NTA);
                    }
                    else
                    {
                        _mm_prefetch(p_k1, _MM_HINT_T1);
                    }
                }
            }
        }

        // Loop of 1
        for (;;)
        {
            if ((k0 & intv_mask) == 0)
            {
                values[index0] = sa0 + bwt->sa[k0 >> intv_shift];

                break;
            }
            else
            {
                _mm_prefetch(p_k0, _MM_HINT_T0);
                sa0++;
                k0 = bwt_invPsi_avx2(bwt, k0, _kl0, p_k0);
                _mm_prefetch(p_k0, _MM_HINT_T1);
            }
        }
    }

#ifdef UNIT_TEST_SEED
    bwtint_t *dv = (bwtint_t *)alloca(count * sizeof(bwtint_t));
    bwt_sa_bulk(bwt_debug, indices, dv, count);
    for (int di = 0; di < count; di++)
    {
        assert(values[di] == dv[di]);
    }
#endif
}

__forceinline __m256i _bwt_create_intv(const bwt_t *bwt, const uint8_t *readData, const int spanIndex, __m128i &_kl2, const char* &p_k, const char* &p_l)
{
    const uint8_t c = readData[spanIndex];

    __m256i _out = _mm256_setzero_si256();

    _out = _mm256_insert_epi64(_out, bwt->L2[c] + 1, 0);
    _out = _mm256_insert_epi64(_out, bwt->L2[c + 1] + 1, 1);
    _out = _mm256_insert_epi64(_out, bwt->L2[c + 1] - bwt->L2[c], 2);

    _out = _mm256_insert_epi32(_out, spanIndex, 6);
    _out = _mm256_insert_epi32(_out, spanIndex + 1, 7);

    _occ_epilogue(
        bwt,
        _mm_sub_epi64(
            _mm256_castsi256_si128(_out),
            _mm_one_epi64),
        _kl2,
        p_k,
        p_l);

    return _out;
}

__forceinline __m256i _bwt_extend_intv(const bwt_t *bwt, const uint8_t *readData, const int intervalStart, const int spanIndex, __m128i &_kl2, const char* &p_k, const char* &p_l)
{
    const uint8_t c = readData[intervalStart];

    __m256i _L2 = _mm256_loadu_si256((__m256i *)bwt->L2);

    __m128i _adjust = _mm_add_epi16(
            _mm_occ_adjust,
            _kl2);

    __m128i _mask_kl = _occ_premask(
        _mm_shuffle_epi8(
            _kl2,
            _mm_occ_broadcast_mask));

    __m256i _c_x_2 = _mm256_slli_epi32(
        _mm256_set1_epi32(c),
        1);

    __m256i _cntk = _occ_kernel(
        (__m256i*)p_k,
        _L2,
        _mask_kl,
        _adjust,
        _c_x_2);

    __m256i _cntl = _occ_kernel(
        (__m256i*)p_l,
        _L2,
        _mm_srli_si128(_mask_kl, 8),
        _mm_srli_si128(_adjust, 8),
        _c_x_2);

    __m256i _hits = _mm256_sub_epi64(
        _cntl,
        _cntk);

    __m256i _out = _mm256_shuffle_epi64(_cntk, _cntl, 0x00);

    _out = _mm256_inserti128_si256(
        _hits,
        _mm_add_epi64(
            _mm256_castsi256_si128(_out),
            _mm_one_epi64),
        0);

    _out = _mm256_insert_epi32(_out, intervalStart, 7);
    _out = _mm256_insert_epi32(_out, spanIndex + 1, 6);

    _occ_epilogue(
        bwt,
        _mm_sub_epi64(
            _mm256_castsi256_si128(_out),
            _mm_one_epi64),
        _kl2,
        p_k,
        p_l);

    return _out;
}

__forceinline void _kv_push(bwtintv_v *vec, __m256i _intv)
{
    if (vec->n == vec->m)
    {
        vec->m = vec->m ? vec->m << 1 : 32;
        vec->a = (bwtintv_t*)realloc(vec->a, sizeof(bwtintv_t) * vec->m);
    }

    _mm256_storeu_si256((__m256i*)(vec->a + vec->n), _intv);

    vec->n++;
}

// Interval cache at every position 
bwtintv_v * bwt_intv_forest_avx2(const bwt_t *bwt, const uint8_t *readData, const int spanStart, const int spanEnd)
{
    int spanSize = spanEnd - spanStart;
    bwtintv_v* intervalForest = (bwtintv_v *)calloc(spanSize, sizeof(bwtintv_v));

    __m256i _workingInterval;

    int spanIndex0 = spanStart;

    if (spanIndex0 < spanEnd)
    {
        bwtintv_v* currentSpan0 = intervalForest + spanIndex0 - spanStart;

        __m128i _kl0;
        const char *p_k0, *p_l0;

        // the initial interval of a single base
        _workingInterval = _bwt_create_intv(bwt, readData, spanIndex0, _kl0, p_k0, p_l0);
        _kv_push(currentSpan0, _workingInterval);
        _mm_prefetch(p_k0, _MM_HINT_T1);
        _mm_prefetch(p_l0, _MM_HINT_T1);

        int intervalStart0 = spanIndex0;

        int spanIndex1 = spanIndex0 + 1;

        if (spanIndex1 < spanEnd)
        {
            bwtintv_v* currentSpan1 = intervalForest + spanIndex1 - spanStart;

            __m128i _kl1;
            const char *p_k1, *p_l1;

            // the initial interval of a single base
            _workingInterval = _bwt_create_intv(bwt, readData, spanIndex1, _kl1, p_k1, p_l1);
            _kv_push(currentSpan1, _workingInterval);
            _mm_prefetch(p_k1, _MM_HINT_T1);
            _mm_prefetch(p_l1, _MM_HINT_T1);

            int intervalStart1 = spanIndex1;

            int spanIndex2 = spanIndex1 + 1;

            if (spanIndex2 < spanEnd)
            {
                bwtintv_v* currentSpan2 = intervalForest + spanIndex2 - spanStart;

                __m128i _kl2;
                const char *p_k2, *p_l2;

                // the initial interval of a single base
                _workingInterval = _bwt_create_intv(bwt, readData, spanIndex2, _kl2, p_k2, p_l2);
                _kv_push(currentSpan2, _workingInterval);
                _mm_prefetch(p_k2, _MM_HINT_T1);
                _mm_prefetch(p_l2, _MM_HINT_T1);

                int intervalStart2 = spanIndex2;

                int spanIndex3 = spanIndex2 + 1;

                if (spanIndex3 < spanEnd)
                {
                    bwtintv_v* currentSpan3 = intervalForest + spanIndex3 - spanStart;

                    __m128i _kl3;
                    const char *p_k3, *p_l3;

                    // the initial interval of a single base
                    _workingInterval = _bwt_create_intv(bwt, readData, spanIndex3, _kl3, p_k3, p_l3);
                    _kv_push(currentSpan3, _workingInterval);
                    _mm_prefetch(p_k3, _MM_HINT_T1);
                    _mm_prefetch(p_l3, _MM_HINT_T1);

                    int intervalStart3 = spanIndex3;

                    int spanIndex4 = spanIndex3 + 1;

                    if (spanIndex4 < spanEnd)
                    {
                        bwtintv_v* currentSpan4 = intervalForest + spanIndex4 - spanStart;

                        __m128i _kl4;
                        const char *p_k4, *p_l4;

                        // the initial interval of a single base
                        _workingInterval = _bwt_create_intv(bwt, readData, spanIndex4, _kl4, p_k4, p_l4);
                        _kv_push(currentSpan4, _workingInterval);
                        _mm_prefetch(p_k4, _MM_HINT_T1);
                        _mm_prefetch(p_l4, _MM_HINT_T1);

                        int intervalStart4 = spanIndex4;

                        for (;;)
                        {
                            if (intervalStart0-- > spanStart)
                            {
                                _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart0, spanIndex0, _kl0, p_k0, p_l0);

                                if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                                {
                                    intervalStart0 = spanStart;
                                }
                                else
                                {
                                    _kv_push(currentSpan0, _workingInterval);
                                    _mm_prefetch(p_k0, _MM_HINT_T1);
                                    _mm_prefetch(p_l0, _MM_HINT_T1);
                                }
                            }
                            else
                            {
                                // Copy slot 4 to slot 0
                                spanIndex0 = spanIndex4;
                                currentSpan0 = currentSpan4;
                                intervalStart0 = intervalStart4;
                                _kl0 = _kl4;
                                p_k0 = p_k4;
                                p_l0 = p_l4;

                                // This would caise slot 4 to reload
                                intervalStart4 = spanStart;
                            }
                            _mm_prefetch(p_k2, _MM_HINT_T0);
                            _mm_prefetch(p_l2, _MM_HINT_T0);

                            if (intervalStart1-- > spanStart)
                            {
                                _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart1, spanIndex1, _kl1, p_k1, p_l1);

                                if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                                {
                                    intervalStart1 = spanStart;
                                }
                                else
                                {
                                    _kv_push(currentSpan1, _workingInterval);
                                    _mm_prefetch(p_k1, _MM_HINT_T1);
                                    _mm_prefetch(p_l1, _MM_HINT_T1);
                                }
                            }
                            else
                            {
                                // Copy slot 4 to slot 1
                                spanIndex1 = spanIndex4;
                                currentSpan1 = currentSpan4;
                                intervalStart1 = intervalStart4;
                                _kl1 = _kl4;
                                p_k1 = p_k4;
                                p_l1 = p_l4;

                                // This would caise slot 4 to reload
                                intervalStart4 = spanStart;
                            }
                            _mm_prefetch(p_k3, _MM_HINT_T0);
                            _mm_prefetch(p_l3, _MM_HINT_T0);

                            if (intervalStart2-- > spanStart)
                            {
                                _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart2, spanIndex2, _kl2, p_k2, p_l2);

                                if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                                {
                                    intervalStart2 = spanStart;
                                }
                                else
                                {
                                    _kv_push(currentSpan2, _workingInterval);
                                    _mm_prefetch(p_k2, _MM_HINT_T1);
                                    _mm_prefetch(p_l2, _MM_HINT_T1);
                                }
                            }
                            else
                            {
                                // Copy slot 4 to slot 2
                                spanIndex2 = spanIndex4;
                                currentSpan2 = currentSpan4;
                                intervalStart2 = intervalStart4;
                                _kl2 = _kl4;
                                p_k2 = p_k4;
                                p_l2 = p_l4;

                                // This would caise slot 4 to reload
                                intervalStart4 = spanStart;
                            }
                            _mm_prefetch(p_k4, _MM_HINT_T0);
                            _mm_prefetch(p_l4, _MM_HINT_T0);

                            if (intervalStart3-- > spanStart)
                            {
                                _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart3, spanIndex3, _kl3, p_k3, p_l3);

                                if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                                {
                                    intervalStart3 = spanStart;
                                }
                                else
                                {
                                    _kv_push(currentSpan3, _workingInterval);
                                    _mm_prefetch(p_k3, _MM_HINT_T1);
                                    _mm_prefetch(p_l3, _MM_HINT_T1);
                                }
                            }
                            else
                            {
                                // Copy slot 4 to slot 3
                                spanIndex3 = spanIndex4;
                                currentSpan3 = currentSpan4;
                                intervalStart3 = intervalStart4;
                                _kl3 = _kl4;
                                p_k3 = p_k4;
                                p_l3 = p_l4;

                                // This would caise slot 4 to reload
                                intervalStart4 = spanStart;
                            }
                            _mm_prefetch(p_k0, _MM_HINT_T0);
                            _mm_prefetch(p_l0, _MM_HINT_T0);

                            if (intervalStart4-- > spanStart)
                            {
                                _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart4, spanIndex4, _kl4, p_k4, p_l4);

                                if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                                {
                                    intervalStart4 = spanStart;
                                }
                                else
                                {
                                    _kv_push(currentSpan4, _workingInterval);
                                    _mm_prefetch(p_k4, _MM_HINT_T1);
                                    _mm_prefetch(p_l4, _MM_HINT_T1);
                                }
                            }
                            else
                            {
                                if (++spanIndex4 < spanEnd)
                                {
                                    currentSpan4 = intervalForest + spanIndex4 - spanStart;

                                    // the initial interval of a single base
                                    _workingInterval = _bwt_create_intv(bwt, readData, spanIndex4, _kl4, p_k4, p_l4);
                                    _kv_push(currentSpan4, _workingInterval);
                                    _mm_prefetch(p_k4, _MM_HINT_T1);
                                    _mm_prefetch(p_l4, _MM_HINT_T1);

                                    intervalStart4 = spanIndex4;
                                }
                                else
                                {
                                    break;
                                }
                            }
                            _mm_prefetch(p_k1, _MM_HINT_T0);
                            _mm_prefetch(p_l1, _MM_HINT_T0);
                        }
                    }

                    // 4 slots left
                    for (;;)
                    {
                        if (intervalStart0-- > spanStart)
                        {
                            _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart0, spanIndex0, _kl0, p_k0, p_l0);

                            if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                            {
                                intervalStart0 = spanStart;
                            }
                            else
                            {
                                _kv_push(currentSpan0, _workingInterval);
                                _mm_prefetch(p_k0, _MM_HINT_T1);
                                _mm_prefetch(p_l0, _MM_HINT_T1);
                            }
                        }
                        else
                        {
                            // Copy slot 3 to slot 0
                            spanIndex0 = spanIndex3;
                            currentSpan0 = currentSpan3;
                            intervalStart0 = intervalStart3;
                            _kl0 = _kl3;
                            p_k0 = p_k3;
                            p_l0 = p_l3;

                            break;
                        }
                        _mm_prefetch(p_k2, _MM_HINT_T0);
                        _mm_prefetch(p_l2, _MM_HINT_T0);

                        if (intervalStart1-- > spanStart)
                        {
                            _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart1, spanIndex1, _kl1, p_k1, p_l1);

                            if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                            {
                                intervalStart1 = spanStart;
                            }
                            else
                            {
                                _kv_push(currentSpan1, _workingInterval);
                                _mm_prefetch(p_k1, _MM_HINT_T1);
                                _mm_prefetch(p_l1, _MM_HINT_T1);
                            }
                        }
                        else
                        {
                            // Copy slot 3 to slot 1
                            spanIndex1 = spanIndex3;
                            currentSpan1 = currentSpan3;
                            intervalStart1 = intervalStart3;
                            _kl1 = _kl3;
                            p_k1 = p_k3;
                            p_l1 = p_l3;

                            break;
                        }
                        _mm_prefetch(p_k3, _MM_HINT_T0);
                        _mm_prefetch(p_l3, _MM_HINT_T0);

                        if (intervalStart2-- > spanStart)
                        {
                            _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart2, spanIndex2, _kl2, p_k2, p_l2);

                            if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                            {
                                intervalStart2 = spanStart;
                            }
                            else
                            {
                                _kv_push(currentSpan2, _workingInterval);
                                _mm_prefetch(p_k2, _MM_HINT_T1);
                                _mm_prefetch(p_l2, _MM_HINT_T1);
                            }
                        }
                        else
                        {
                            // Copy slot 3 to slot 2
                            spanIndex2 = spanIndex3;
                            currentSpan2 = currentSpan3;
                            intervalStart2 = intervalStart3;
                            _kl2 = _kl3;
                            p_k2 = p_k3;
                            p_l2 = p_l3;

                            break;
                        }
                        _mm_prefetch(p_k0, _MM_HINT_T0);
                        _mm_prefetch(p_l0, _MM_HINT_T0);

                        if (intervalStart3-- > spanStart)
                        {
                            _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart3, spanIndex3, _kl3, p_k3, p_l3);

                            if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                            {
                                intervalStart3 = spanStart;
                            }
                            else
                            {
                                _kv_push(currentSpan3, _workingInterval);
                                _mm_prefetch(p_k3, _MM_HINT_T1);
                                _mm_prefetch(p_l3, _MM_HINT_T1);
                            }
                        }
                        else
                        {
                            break;
                        }
                        _mm_prefetch(p_k1, _MM_HINT_T0);
                        _mm_prefetch(p_l1, _MM_HINT_T0);
                    }
                }

                // 3 slots left
                for (;;)
                {
                    if (intervalStart0-- > spanStart)
                    {
                        _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart0, spanIndex0, _kl0, p_k0, p_l0);

                        if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                        {
                            intervalStart0 = spanStart;
                        }
                        else
                        {
                            _kv_push(currentSpan0, _workingInterval);
                            _mm_prefetch(p_k0, _MM_HINT_T1);
                            _mm_prefetch(p_l0, _MM_HINT_T1);
                        }
                    }
                    else
                    {
                        // Copy slot 2 to slot 0
                        spanIndex0 = spanIndex2;
                        currentSpan0 = currentSpan2;
                        intervalStart0 = intervalStart2;
                        _kl0 = _kl2;
                        p_k0 = p_k2;
                        p_l0 = p_l2;

                        break;
                    }
                    _mm_prefetch(p_k2, _MM_HINT_T0);
                    _mm_prefetch(p_l2, _MM_HINT_T0);

                    if (intervalStart1-- > spanStart)
                    {
                        _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart1, spanIndex1, _kl1, p_k1, p_l1);

                        if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                        {
                            intervalStart1 = spanStart;
                        }
                        else
                        {
                            _kv_push(currentSpan1, _workingInterval);
                            _mm_prefetch(p_k1, _MM_HINT_T1);
                            _mm_prefetch(p_l1, _MM_HINT_T1);
                        }
                    }
                    else
                    {
                        // Copy slot 2 to slot 1
                        spanIndex1 = spanIndex2;
                        currentSpan1 = currentSpan2;
                        intervalStart1 = intervalStart2;
                        _kl1 = _kl2;
                        p_k1 = p_k2;
                        p_l1 = p_l2;

                        break;
                    }
                    _mm_prefetch(p_k0, _MM_HINT_T0);
                    _mm_prefetch(p_l0, _MM_HINT_T0);

                    if (intervalStart2-- > spanStart)
                    {
                        _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart2, spanIndex2, _kl2, p_k2, p_l2);

                        if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                        {
                            intervalStart2 = spanStart;
                        }
                        else
                        {
                            _kv_push(currentSpan2, _workingInterval);
                            _mm_prefetch(p_k2, _MM_HINT_T1);
                            _mm_prefetch(p_l2, _MM_HINT_T1);
                        }
                    }
                    else
                    {
                        break;
                    }
                    _mm_prefetch(p_k1, _MM_HINT_T0);
                    _mm_prefetch(p_l1, _MM_HINT_T0);
                }
            }

            // 2 slots left
            for (;;)
            {
                if (intervalStart0-- > spanStart)
                {
                    _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart0, spanIndex0, _kl0, p_k0, p_l0);

                    if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                    {
                        intervalStart0 = spanStart;
                    }
                    else
                    {
                        _kv_push(currentSpan0, _workingInterval);
                        _mm_prefetch(p_k0, _MM_HINT_T1);
                        _mm_prefetch(p_l0, _MM_HINT_T1);
                    }
                }
                else
                {
                    // Copy slot 1 to slot 0
                    spanIndex0 = spanIndex1;
                    currentSpan0 = currentSpan1;
                    intervalStart0 = intervalStart1;
                    _kl0 = _kl1;
                    p_k0 = p_k1;
                    p_l0 = p_l1;

                    break;
                }

                if (intervalStart1-- > spanStart)
                {
                    _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart1, spanIndex1, _kl1, p_k1, p_l1);

                    if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                    {
                        intervalStart1 = spanStart;
                    }
                    else
                    {
                        _kv_push(currentSpan1, _workingInterval);
                        _mm_prefetch(p_k1, _MM_HINT_T1);
                        _mm_prefetch(p_l1, _MM_HINT_T1);
                    }
                }
                else
                {
                    break;
                }
            }
        }

        // 1 slot left
        for (;;)
        {
            if (intervalStart0-- > spanStart)
            {
                _workingInterval = _bwt_extend_intv(bwt, readData, intervalStart0, spanIndex0, _kl0, p_k0, p_l0);

                if (_mm256_extract_epi64(_workingInterval, 2) < 1)
                {
                    intervalStart0 = spanStart;
                }
                else
                {
                    _kv_push(currentSpan0, _workingInterval);
                    _mm_prefetch(p_k0, _MM_HINT_T1);
                    _mm_prefetch(p_l0, _MM_HINT_T1);
                }
            }
            else
            {
                break;
            }
        }
    }

    return intervalForest;
}

bwt_t* bwt_restore_bwt_avx2(const char *fn)
{
#ifdef UNIT_TEST_SEED
    bwt_debug = bwt_restore_bwt(fn);
#endif

    auto start = std::chrono::high_resolution_clock::now();

    bwt_t *bwt = (bwt_t*)scalable_aligned_malloc(sizeof(bwt_t), 64);

    FILE *fp = xopen(fn, "rb");
    err_fseek(fp, 0, SEEK_END);

    // Size of BWT on disk in bytes
    bwtint_t bwtSize = (err_ftell(fp) - sizeof(bwtint_t) * 5);
    // Save the size in 32 bit dwords
    bwt->bwt_size = bwtSize >> 2;
    // Align on 64 bytes (512 bits), the size of BWT bucket
    // and add space for one more bucket
    bwtSize += 64 + ((bwtSize % 64 == 0) ? 0 : (64 - bwtSize % 64));
    assert(bwtSize % 64 == 0);

    // Allocate
    bwt->bwt = (uint32_t*)scalable_aligned_malloc(bwtSize, 64);

    // Read
    err_fseek(fp, 0, SEEK_SET);
    err_fread_noeof(&bwt->primary, sizeof(bwtint_t), 1, fp);
    bwt->L2[0] = 0;
    err_fread_noeof(bwt->L2 + 1, sizeof(bwtint_t), 4, fp);
    err_fread_noeof(bwt->bwt, bwt->bwt_size << 2, 1, fp);
    bwt->seq_len = bwt->L2[4];
    err_fclose(fp);

    // Fill the extra bucket
    memmove(bwt->bwt + bwt->bwt_size, bwt->L2 + 1, 32);
    memset(bwt->bwt + bwt->bwt_size + 32, 0, 32);

    uint32_t* new_bwt = (uint32_t*)scalable_aligned_malloc(bwtSize, 64);

    __m256i *p = (__m256i *)(bwt->bwt);
    __m256i *q = (__m256i *)(new_bwt);

    __m256i _carryover = _mm256_setzero_si256();

    for (uint32_t i = 0; i < bwtSize; i += 64)
    {
        __m256i _cnt = _mm256_load_si256(p++);

        __m256i _carry_mask = _mm256_cmpeq_epi32(
            _carryover,
            _mm256_setr_epi32(0, 0, 0x40000000, 0x40000000, 0x80000000, 0x80000000, 0xc0000000, 0xc0000000)
        );

        __m256i _cntNew = _mm256_add_epi64(
            _cnt,
            _carry_mask
        );

        _mm256_store_si256(q++, _cntNew);

        __m256i _bwt = _mm256_load_si256(p++);

        __m256i _carry = _mm256_slli_epi32(_bwt, 30);

        __m256i _bwt_new = _mm256_or_si256(
            _mm256_srli_epi32(
                _bwt,
                2),
            _mm256_shift_left_si256<4>(
                _carry,
                _carryover));

        _mm256_store_si256(q++, _bwt_new);

        _carryover = _mm256_permutevar8x32_epi32(
            _carry,
            _mm256_set1_epi32(7)
        );

    }

    scalable_aligned_free(bwt->bwt);

    bwt->bwt = new_bwt;

    auto end = std::chrono::high_resolution_clock::now();

    auto occ4Time = std::chrono::duration<double>(end - start).count();

    fprintf(stderr, "BWT restored in %f seconds.\n", occ4Time);

    return bwt;
}

void bwt_restore_sa_avx2(const char *fn, bwt_t *bwt)
{
    char skipped[256];
    bwtint_t primary;

#ifdef UNIT_TEST_SEED
    bwt_restore_sa(fn, bwt_debug);
#endif

    auto start = std::chrono::high_resolution_clock::now();

    FILE *fp = xopen(fn, "rb");
    err_fread_noeof(&primary, sizeof(bwtint_t), 1, fp);
    xassert(primary == bwt->primary, "SA-BWT inconsistency: primary is not the same.");
    err_fread_noeof(skipped, sizeof(bwtint_t), 4, fp); // skip
    err_fread_noeof(&bwt->sa_intv, sizeof(bwtint_t), 1, fp);
    err_fread_noeof(&primary, sizeof(bwtint_t), 1, fp);
    xassert(primary == bwt->seq_len, "SA-BWT inconsistency: seq_len is not the same.");

    bwt->n_sa = (bwt->seq_len + bwt->sa_intv) / bwt->sa_intv;
    bwt->sa = (bwtint_t*)scalable_aligned_malloc(bwt->n_sa * sizeof(bwtint_t), 64);
    bwt->sa[0] = -1;

    err_fread_noeof(bwt->sa + 1, sizeof(bwtint_t), bwt->n_sa - 1, fp);
    err_fclose(fp);

    auto end = std::chrono::high_resolution_clock::now();

    auto occ4Time = std::chrono::duration<double>(end - start).count();

    fprintf(stderr, "Suffix Array loaded in %f seconds.\n", occ4Time);
}

void bwt_restore_sa_full_avx2(const char *fn, bwt_t *bwt)
{
    bwt_restore_sa_avx2(fn, bwt);

    auto start = std::chrono::high_resolution_clock::now();

    if (bwt->sa_intv != 1)
    {
        bwtint_t full_sa_len = bwt->seq_len + 1;
        bwtint_t* full_sa = (bwtint_t*)scalable_aligned_malloc(full_sa_len * sizeof(bwtint_t), 64);

        bwt->sa[0] = bwt->seq_len;

        // Unleash the parallelism
        tbb::parallel_for(tbb::blocked_range<bwtint_t>(0, bwt->n_sa, 1000),
            [bwt, full_sa](const tbb::blocked_range<bwtint_t>& r) {
                bwtint_t mask = bwt->sa_intv - 1;

                static const bwtint_t maxBranches = 16;

                bwtint_t sa_value[maxBranches];
                bwtint_t isa[maxBranches];

                bwtint_t b = 0, c = r.begin();

                // Setup
                for (; (b < maxBranches) && (c != r.end()); b++, c++)
                {
                    isa[b] = c * bwt->sa_intv;
                    sa_value[b] = bwt->sa[c];
                }

                // Multi loop
                while (b > 0)
                {
                    for (bwtint_t i = 0; i < b; i++)
                    {
                        full_sa[isa[i]] = sa_value[i]--;
                        isa[i] = bwt_invPsi_avx2(bwt, isa[i]);

                        if ((isa[i] & mask) == 0)
                        {
                            if (c != r.end())
                            {
                                isa[i] = c * bwt->sa_intv;
                                sa_value[i] = bwt->sa[c];

                                c++;
                            }
                            else
                            {
                                b--;

                                isa[i] = isa[b];
                                sa_value[i] = sa_value[b];

                                break;
                            }
                        }
                    }
                }
            });

        // before this line, full_sa[0] is set to bwt->seq_len
        full_sa[0] = (bwtint_t)-1;

        // Replace SA
        scalable_aligned_free(bwt->sa);
        bwt->sa = full_sa;
        bwt->n_sa = full_sa_len;
        bwt->sa_intv = 1;

#ifdef UNIT_TEST_SEED
        int nStragglers = 0;
        bwtint_t disa = 0, dsa = bwt_debug->seq_len;
        bwtint_t mask = bwt_debug->sa_intv - 1;

        for (bwtint_t di = 0; di < bwt_debug->seq_len; ++di)
        {
            disa = bwt_invPsi(bwt_debug, disa);
            dsa--;

            //if ((disa & mask) == 0)
            //{
            //	fprintf(stderr, "*");
            //}

            if (dsa != full_sa[disa])
            {
                fprintf(stderr, "\n%lld\t%lld", disa, dsa);
                fprintf(stderr, "\t%lld", full_sa[disa]);

                nStragglers++;
            }

            //assert(dsa == full_sa[di]);
            //if (di % bwt_debug->sa_intv == 0)
            //{
            //	assert(bwt_debug->sa[di / bwt_debug->sa_intv] == full_sa[di]);
            //}
        }
        assert(nStragglers == 0);
#endif
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto occ4Time = std::chrono::duration<double>(end - start).count();

    fprintf(stderr, "Suffix Array restored in %f seconds.\n", occ4Time);
}

void bwt_restore_sa_full(const char *fn, bwt_t *bwt)
{
    auto start = std::chrono::high_resolution_clock::now();

    bwt_restore_sa(fn, bwt);

    if (bwt->sa_intv != 1)
    {
        bwtint_t full_sa_len = bwt->seq_len + 1;
        bwtint_t* full_sa = (bwtint_t*)scalable_aligned_malloc(full_sa_len * sizeof(bwtint_t), 64);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, bwt->n_sa, 1000),
            [bwt, full_sa](const tbb::blocked_range<size_t>& r) {
                for (size_t sa_index = r.begin(); sa_index != r.end(); sa_index++)
                {
                    bwtint_t mask = bwt->sa_intv - 1;

                    bwtint_t isa = sa_index * bwt->sa_intv;
                    bwtint_t sa_value = (sa_index == 0) ? bwt->seq_len : bwt->sa[sa_index];

                    while (true)
                    {
                        full_sa[isa] = sa_value--;
                        isa = bwt_invPsi(bwt, isa);

                        if ((isa & mask) == 0)
                        {
                            break;
                        }
                    };
                }
            });

        // before this line, full_sa[0] is not set
        full_sa[0] = (bwtint_t)-1;

#ifdef UNIT_TEST_SEED
        for (bwtint_t di = 0; di < bwt->seq_len; ++di)
        {
            if (di % bwt->sa_intv == 0)
            {
                assert(bwt->sa[di / bwt->sa_intv] == full_sa[di]);
            }
        }
#endif

        // Replace SA
        scalable_aligned_free(bwt->sa);
        bwt->sa = full_sa;
        bwt->n_sa = full_sa_len;
        bwt->sa_intv = 1;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto occ4Time = std::chrono::duration<double>(end - start).count();

    fprintf(stderr, "Suffix Array restored in %f seconds.\n", occ4Time);
}

bwtint_t bwt_sa_full(const bwt_t *bwt, bwtint_t k)
{
    bwtint_t sa_value = bwt->sa[k];

    return sa_value;
}

void bwt_sa_bulk_full(const bwt_t *bwt, bwtint_t *indices, bwtint_t *values, bwtint_t count)
{
    for (bwtint_t i = 0; i < count; i++)
    {
        values[i] = bwt->sa[indices[i]];
    }
}

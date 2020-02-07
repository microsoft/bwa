/*++

Module Name:

	ksw_avx.h

Abstract:

	KSW AVX performance snippets.

Authors:

	Roman Snytsar, June, 2018

Environment:
`
	User mode service.

Revision History:


--*/
#pragma once

#ifdef __GNUC__
#include <immintrin.h>
#define __lzcnt(x) __builtin_clz(x)
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#define __forceinline inline
#else
#include <intrin.h>
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

__forceinline __m128i _mm256_hmax_epi16(__m256i a) {
	__m128i _max = _mm_max_epi16(
		_mm256_castsi256_si128(a),
		_mm256_extracti128_si256(a,
			1));

	_max = _mm_max_epi16(_max, _mm_srli_si128(_max, 2));
	_max = _mm_max_epi16(_max, _mm_srli_si128(_max, 4));
	_max = _mm_max_epi16(_max, _mm_srli_si128(_max, 8));

	return _max;
}

__forceinline __m128i _mm256_hmax_epu8(__m256i a) {
	__m128i _max = _mm_max_epu8(
		_mm256_castsi256_si128(a),
		_mm256_extracti128_si256(a,
			1));

	_max = _mm_max_epu8(_max, _mm_srli_si128(_max, 1));
	_max = _mm_max_epu8(_max, _mm_srli_si128(_max, 2));
	_max = _mm_max_epu8(_max, _mm_srli_si128(_max, 4));
	_max = _mm_max_epu8(_max, _mm_srli_si128(_max, 8));

	return _max;
}

__forceinline __m128i _mm256_hmax_epi8(__m256i a) {
	__m128i _max = _mm_max_epi8(
		_mm256_castsi256_si128(a),
		_mm256_extracti128_si256(a,
			1));

	_max = _mm_max_epi8(_max, _mm_srli_si128(_max, 1));
	_max = _mm_max_epi8(_max, _mm_srli_si128(_max, 2));
	_max = _mm_max_epi8(_max, _mm_srli_si128(_max, 4));
	_max = _mm_max_epi8(_max, _mm_srli_si128(_max, 8));

	return _max;
}

__forceinline __m128i _mm256_hmin_epi8(__m256i a) {
	__m128i _min = _mm_min_epi8(
		_mm256_castsi256_si128(a),
		_mm256_extracti128_si256(a,
			1));

	_min = _mm_min_epi8(_min, _mm_srli_si128(_min, 1));
	_min = _mm_min_epi8(_min, _mm_srli_si128(_min, 2));
	_min = _mm_min_epi8(_min, _mm_srli_si128(_min, 4));
	_min = _mm_min_epi8(_min, _mm_srli_si128(_min, 8));

	return _min;
}

template <size_t __N>
__forceinline __m256i _mm256_shift_left_si256(__m256i a, __m256i b) {
	__m256i c = _mm256_permute2x128_si256(a, b, 0x03);
	return _mm256_alignr_epi8(a, c, 16 - __N);
}

template <size_t __N>
__forceinline __m256i _mm256_shift_left_si256(__m256i a) {
	__m256i c = _mm256_permute2x128_si256(a, _mm256_setzero_si256(), 0x03);
	return _mm256_alignr_epi8(a, c, 16 - __N);
}

template <size_t __N>
__forceinline __m128i _mm_shift_left_si128(__m128i a, __m128i b) {
	return _mm_alignr_epi8(a, b, 16 - __N);
}

template <size_t __N>
__forceinline __m256i _mm256_shift_right_si256(__m256i a) {
	__m256i b = _mm256_permute2x128_si256(a, _mm256_setzero_si256(), 0x81);
	return _mm256_alignr_epi8(b, a, __N);
}

__forceinline __m256i setup_score_shift(const __m256i _score_matrix)
{
	__m256i _score_shift = _mm256_broadcastb_epi8(
		_mm_sub_epi8(
			_mm_setzero_si128(),
			_mm_min_epi8(
				_mm_set1_epi8(-1),
				_mm256_hmin_epi8(
					_score_matrix))));

	return _score_shift;
}

__forceinline __m256i advance_H_i16(const __m128i _score_matrix, const __m256i _H, const __m128i _query, const __m128i _target)
{
	__m128i _i1 = _mm_add_epi8(
		_query,
		_mm_slli_epi32(
			_target,
			2));

	__m128i _i2 = _mm_slli_epi32(
		_mm_or_si128(
			_query,
			_target),
		5);

	__m256i _s = _mm256_cvtepi8_epi16(
		_mm_blendv_epi8(
			_mm_shuffle_epi8(
				_score_matrix,
				_i1),
			_mm_set1_epi8(-1),
			_i2));

	__m256i _H1 = _mm256_adds_epi16(_H, _s);

	return _H1;
}

__forceinline __m128i advance_H_u8(const __m128i _score_matrix, const __m128i _score_shift, const __m128i _H, const __m128i _query, const __m128i _target)
{
	__m128i _i1 = _mm_add_epi8(
		_query,
		_mm_slli_epi32(
			_target,
			2));

	__m128i _i2 = _mm_slli_epi32(
		_mm_or_si128(
			_query,
			_target),
		5);

	__m128i _s = _mm_blendv_epi8(
		_mm_shuffle_epi8(
			_score_matrix,
			_i1),
		_mm_add_epi8(
			_score_shift,
			_mm_set1_epi8(
				-1)),
		_i2);

	__m128i _H1 = _mm_adds_epu8(_H, _s);

	_H1 = _mm_subs_epu8(_H1, _score_shift);

	return _H1;
}

__forceinline __m256i advance_H_u8(const __m256i _score_matrix, const __m256i _score_shift, const __m256i _H, const __m256i _query, const __m256i _target)
{
	__m256i _i1 = _mm256_add_epi8(
		_query,
		_mm256_slli_epi32(
			_target,
			2));

	__m256i _i2 = _mm256_slli_epi32(
		_mm256_or_si256(
			_query,
			_target),
		5);

	__m256i _s = _mm256_blendv_epi8(
		_mm256_shuffle_epi8(
			_score_matrix,
			_i1),
		_mm256_add_epi8(
			_score_shift,
			_mm256_set1_epi8(
				-1)),
		_i2);

	__m256i _H1 = _mm256_adds_epu8(_H, _s);

	_H1 = _mm256_subs_epu8(_H1, _score_shift);

	return _H1;
}

// _e_scan[j] = _gape * j
__forceinline __m256i _mm256_exclusive_add_epi16(const __m256i _gape)
{
    __m256i _e_scan = _gape;

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 2));

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 4));

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 8));

    _e_scan = _mm256_bslli_epi128(_e_scan, 2);

    return _e_scan;
}

__forceinline __m256i prefix_scan_F_i16(const __m256i _f0, const __m256i _gape, const __m256i _gape_scan)
{
	// Prefix Scan for F
	__m256i _e_k = _gape;

	__m256i _f = _f0;

    // First pass
    _f = _mm256_max_epu16(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu16(
                _f,
                _e_k),
            2));

    _e_k = _mm256_adds_epu16(_e_k, _e_k);

    _f = _mm256_max_epu16(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu16(
                _f,
                _e_k),
            4));

    _e_k = _mm256_adds_epu16(_e_k, _e_k);

    _f = _mm256_max_epu16(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu16(
                _f,
                _e_k),
            8));

    // Second pass
    __m256i _f2 = _mm256_permute2x128_si256(
        _mm256_broadcastw_epi16(
            _mm256_castsi256_si128(
                _mm256_bsrli_epi128(_f, 14))),
        _f,
        0x08);

    // Final pass
    _f = _mm256_max_epu16(
        _mm256_bslli_epi128(_f, 2),
        _mm256_subs_epu16(
            _f2,
            _gape_scan));

    return _f;
}

// _e_scan[j] = _gape * j
__forceinline __m256i _mm256_exclusive_add_epu8(const __m256i _gape)
{
    __m256i _e_scan = _gape;

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 1));

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 2));

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 4));

    _e_scan = _mm256_adds_epu8(
        _e_scan,
        _mm256_bslli_epi128(_e_scan, 8));

    _e_scan = _mm256_bslli_epi128(_e_scan, 1);

    return _e_scan;
}

__forceinline __m256i prefix_scan_F_u8(const __m256i _f0, const __m256i _gape, const __m256i _gape_scan)
{
	__m256i _e_k = _gape;

	__m256i _f = _f0;

    // First pass
    _f = _mm256_max_epu8(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu8(
                _f,
                _e_k),
            1));

    _e_k = _mm256_adds_epu8(_e_k, _e_k);

    _f = _mm256_max_epu8(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu8(
                _f,
                _e_k),
            2));

    _e_k = _mm256_adds_epu8(_e_k, _e_k);

    _f = _mm256_max_epu8(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu8(
                _f,
                _e_k),
            4));

    _e_k = _mm256_adds_epu8(_e_k, _e_k);

    _f = _mm256_max_epu8(
        _f,
        _mm256_bslli_epi128(
            _mm256_subs_epu8(
                _f,
                _e_k),
            8));

    // Second pass
    __m256i _f2 = _mm256_permute2x128_si256(
        _mm256_broadcastb_epi8(
            _mm256_castsi256_si128(
                _mm256_bsrli_epi128(_f, 15))),
        _f,
        0x08);

    // Final pass
    _f = _mm256_max_epu8(
        _mm256_bslli_epi128(_f, 1),
        _mm256_subs_epu8(
            _f2,
            _gape_scan));

	return _f;
}

__forceinline __m128i prefix_scan_F_u8(const __m128i _f0, const __m128i _gape)
{
	// Prefix Scan for F
	__m128i e_k = _gape;

	__m128i _f = _f0;

	_f = _mm_max_epu8(
		_f,
		_mm_slli_si128(
			_mm_subs_epu8(
				_f,
				e_k),
			1));

	e_k = _mm_adds_epu8(e_k, e_k);

	_f = _mm_max_epu8(
		_f,
		_mm_slli_si128(
			_mm_subs_epu8(
				_f,
				e_k),
			2));

	e_k = _mm_adds_epu8(e_k, e_k);

	_f = _mm_max_epu8(
		_f,
		_mm_slli_si128(
			_mm_subs_epu8(
				_f,
				e_k),
			4));

	e_k = _mm_adds_epu8(e_k, e_k);

	_f = _mm_max_epu8(
		_f,
		_mm_slli_si128(
			_mm_subs_epu8(
				_f,
				e_k),
			8));

    // Make it exclusive
    _f = _mm_slli_si128(_f, 1);

	return _f;
}

__forceinline void update_row_maximums_i16(__m256i& _mm, __m256i& _mj, const size_t j, const __m256i _H)
{
	_mm = _mm256_max_epi16(
		_H,
		_mm);

	__m256i _max_mask = _mm256_cmpeq_epi16(_H, _mm);

	_mj = _mm256_blendv_epi8(
		_mj,
		_mm256_set1_epi16((short)j),
		_max_mask);
}

__forceinline void update_row_maximums_i16(__m256i& _mm, __m256i& _mj, const size_t j, const __m256i _H, const __m256i _mMask)
{
	update_row_maximums_i16(_mm, _mj, j, _mm256_and_si256(_H, _mMask));
}

__forceinline void update_row_maximums_u8(__m256i& _mm, __m256i& _mj, const size_t j, const __m256i _H)
{
	_mm = _mm256_max_epu8(
		_mm,
		_H);

	__m256i _max_mask = _mm256_cmpeq_epi8(_H, _mm);

	_mj = _mm256_blendv_epi8(
		_mj,
		_mm256_set1_epi8((int8_t)j),
		_max_mask);
}

__forceinline void update_row_maximums_u8(__m256i& _mm, __m256i& _mj, const size_t j, const __m256i _H, const __m256i _mMask)
{
	update_row_maximums_u8(_mm, _mj, j, _mm256_and_si256(_H, _mMask));
}

__forceinline void update_column_maximums_i16(__m256i& _mm, __m256i& _mi, const size_t i, const __m256i _H)
{
    __m256i _m0 = _mm;

	_mm = _mm256_max_epi16(
		_mm,
		_H);

    // If stays the same
	__m256i _max_mask = _mm256_cmpeq_epi16(_m0, _mm);

	_mi = _mm256_blendv_epi8(
		_mm256_set1_epi16((short)i),
		_mi,
		_max_mask);
}

__forceinline void update_column_maximums_u8(__m256i& _mm, __m256i& _mi, const size_t i, const __m256i _H)
{
    __m256i _m0 = _mm;

	_mm = _mm256_max_epu8(
		_H,
		_mm);

    // If stays the same
    __m256i _max_mask = _mm256_cmpeq_epi8(_m0, _mm);

	_mi = _mm256_blendv_epi8(
		_mm256_set1_epi8((int8_t)i),
		_mi,
		_max_mask);
}

__forceinline void update_column_maximums_u8(__m128i& _mm, __m128i& _mi, const int i, const __m128i _H)
{
    __m128i _m0 = _mm;

	_mm = _mm_max_epu8(
		_H,
		_mm);

    // If stays the same
	__m128i _max_mask = _mm_cmpeq_epi8(_m0, _mm);

	_mi = _mm_blendv_epi8(
		_mm_set1_epi8(i),
		_mi,
		_max_mask);
}

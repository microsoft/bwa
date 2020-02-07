/*++

Module Name:

	bwt_avx.h

Abstract:

	Common BWT AVX definitions.

Authors:

	Roman Snytsar, September, 2018

Environment:
`
	User mode service.

Revision History:


--*/
#pragma once

#ifdef __GNUC__
#include <immintrin.h>
#define __lzcnt(x) __builtin_clz(x)
#define __forceinline inline
#else
#include <intrin.h>
#endif

#ifdef _DEBUG
#define UNIT_TEST_SEED
#endif

template <size_t __N>
__forceinline __m256i _mm256_shift_left_si256(__m256i a, __m256i b) {
	__m256i c = _mm256_permute2x128_si256(a, b, 0x03);
	return _mm256_alignr_epi8(a, c, 16 - __N);
}

template <size_t __N>
__forceinline __m256i _mm256_shift_left_si256(__m256i a) {
	__m256i c = _mm256_permute2x128_si256(a, a, 0x08);
	return _mm256_alignr_epi8(a, c, 16 - __N);
}

template <size_t __N>
__forceinline __m256i _mm256_shift_right_si256(__m256i a, __m256i b) {
	__m256i c = _mm256_permute2x128_si256(a, b, 0x21);
	return _mm256_alignr_epi8(c, a, __N);
}

template <size_t __N>
__forceinline __m256i _mm256_shift_right_si256(__m256i a) {
	__m256i c = _mm256_permute2x128_si256(a, a, 0x81);
	return _mm256_alignr_epi8(c, a, __N);
}

const __m128i _mm_one_epi64 = _mm_set_epi32(
	0, 1, 0, 1);

const __m128i _mm_occ_adjust = _mm_set_epi32(
	0x1fe01fe0, 0x1fe01f61, 0x1fe01fe0, 0x1fe01f61);

const __m128i _mm_occ_intv_mask = _mm_set_epi64x(
	0x7fll, 0x7fll);

const __m128i _mm_occ_mask_shift = _mm_set_epi64x(
	0xfedebe9e7e5e3e1ell, 0xfedebe9e7e5e3e1ell);

const __m128i _mm_occ_broadcast_mask = _mm_set_epi64x(
	0x0808080808080808ll, 0ll);

__forceinline void _occ_preamble(bwtint_t primary, __m128i _kl, __m128i& _kl1, __m128i& _kl2, __m128i& _adjust)
{
	_kl = _mm_blendv_epi8(
		_kl,
		_mm_add_epi64(
			_kl,
			_mm_one_epi64),
		_mm_cmpgt_epi64(
			_mm_set1_epi64x(primary),
			_kl));

	_kl1 = _mm_srli_epi64(
		_mm_andnot_si128(
			_mm_occ_intv_mask,
			_kl),
		1);

	_kl2 = _mm_and_si128(
		_mm_occ_intv_mask,
		_kl);

	_adjust = _mm_add_epi16(
		_mm_occ_adjust,
		_kl2);
};

__forceinline __m128i _occ_premask(__m128i _k2_broacast)
{
	__m128i _premask = _mm_subs_epu8(
		_mm_occ_mask_shift,
		_mm_slli_epi32(
			_k2_broacast,
			1));

	return _premask;
}

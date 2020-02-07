/*++

Module Name:

	bwt_perf.h

Abstract:

	BWT performance functions definitions.

Authors:

	Roman Snytsar, June, 2018

Environment:
`
	User mode service.

Revision History:


--*/
#pragma once

#include "bwt.h"

#ifdef __cplusplus
extern "C" {
#endif
	// Basic BWT optimizations
	bwt_t* bwt_restore_bwt_avx2(const char *fn);
	void bwt_extend_avx2(const bwt_t *bwt, const bwtintv_t *ik, bwtintv_t ok[4], int is_back);

	void bwt_restore_sa_avx2(const char *fn, bwt_t *bwt);
	bwtint_t bwt_sa_avx2(const bwt_t *bwt, bwtint_t k);
	void bwt_sa_bulk_avx2(const bwt_t *bwt, bwtint_t *indices, bwtint_t *values, bwtint_t count);

	// Restoring the full Suffix Array
	void bwt_restore_sa_full(const char *fn, bwt_t *bwt);
	bwtint_t bwt_sa_full(const bwt_t *bwt, bwtint_t k);
	void bwt_sa_bulk_full(const bwt_t *bwt, bwtint_t *indices, bwtint_t *values, bwtint_t count);
	void bwt_restore_sa_full_avx2(const char *fn, bwt_t *bwt);

	// Extend optimizations
    bwtintv_v * bwt_intv_forest_avx2(const bwt_t *bwt, const uint8_t *readData, const int spanStart, const int spanEnd);

#if _MSC_VER >= 1914
#endif
#ifdef __cplusplus
}
#endif


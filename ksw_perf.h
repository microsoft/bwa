/*++

Module Name:

	ksw_perf.h

Abstract:

	KSW preformance functions definitions.

Authors:

	Roman Snytsar, June, 2018

Environment:
`
	User mode service.

Revision History:


--*/
#pragma once

#include "ksw.h"

#ifdef __cplusplus
extern "C" {
#endif
	kswr_t ksw_align_avx2(int qlen, uint8_t *query, int tlen, uint8_t *target, int m, const int8_t *mat, int gapo, int gape, int xtra, kswq_t **qry);
	int ksw_extend_avx2(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int gapo, int gape, int w, int end_bonus, int zdrop, int h0, int *qle, int *tle, int *gtle, int *gscore, int *max_off);
#ifdef __cplusplus
}
#endif

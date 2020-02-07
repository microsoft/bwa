#ifndef BWAPERF_H_
#define BWAPERF_H_

#include "stdint.h"

enum bwaperf_opt
{
	AVX2 = 0x01ull,
	AVX512 = 0x02ull,

	AVX_ANY = AVX2 | AVX512,

	FAST_EXTEND = 0x0100ull,
	FAST_ALIGN = 0x0200ull,
	FAST_GLOBAL = 0x0400ull,

	FAST_SEED = 0x0800ull,

	FAST_SA = 0x010000ull,

	DEFAULT = FAST_SEED | FAST_EXTEND | FAST_ALIGN | FAST_GLOBAL
};

#ifdef __cplusplus
extern "C" {
#endif
void bwaperf_config(uint64_t options);
#ifdef __cplusplus
}
#endif

#endif

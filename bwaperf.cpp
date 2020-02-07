#include "bwaperf.h"

#include "bwt_perf.h"
#include "ksw_perf.h"

void bwaperf_config(uint64_t options)
{
	if (options & bwaperf_opt::FAST_EXTEND)
	{
		if (options & bwaperf_opt::AVX_ANY)
		{
			g_ksw_extend = ksw_extend_avx2;
		}
	}

	if (options & bwaperf_opt::FAST_ALIGN)
	{
		if (options & bwaperf_opt::AVX_ANY)
		{
			g_ksw_align = ksw_align_avx2;
		}
	}

	if (options & bwaperf_opt::FAST_SEED)
	{
		if (options & bwaperf_opt::AVX_ANY)
		{
			g_bwt_restore_bwt = bwt_restore_bwt_avx2;
			g_bwt_restore_sa = bwt_restore_sa_avx2;

			g_bwt_sa = bwt_sa_avx2;
			g_bwt_sa_bulk = bwt_sa_bulk_avx2;

#if _MSC_VER >= 1914
			if (options & bwaperf_opt::AVX512)
			{
                g_bwt_intv_forest = bwt_intv_forest_avx2;
            }
			else 
#endif
			{
                g_bwt_intv_forest = bwt_intv_forest_avx2;
			}
		}
	}

	if (options & bwaperf_opt::FAST_SA)
	{
		if (options & bwaperf_opt::AVX_ANY)
		{
			g_bwt_restore_sa = bwt_restore_sa_full_avx2;
		}
		else
		{
			g_bwt_restore_sa = bwt_restore_sa_full;
		}

		g_bwt_sa = bwt_sa_full;
		g_bwt_sa_bulk = bwt_sa_bulk_full;
	}
}

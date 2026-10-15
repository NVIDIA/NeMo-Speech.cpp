#pragma once
#ifndef MK2_INFLIGHT
#define MK2_INFLIGHT 0
#endif
#if MK2_INFLIGHT
#define MK2_PROGRESS_PARAMETER(name) , const unsigned char *name = nullptr
#define MK2_PROGRESS_ARGUMENT(value) , value
#define MK2_PROGRESS_GUARD(mask, row) if ((mask) && !(mask)[row]) continue
#define MK2_PROGRESS_ENABLED(mask, row) (!(mask) || (mask)[row] != 0)
#define MK2_PROGRESS_MEMBER const unsigned char *native_progress_mask = nullptr;
#else
#define MK2_PROGRESS_PARAMETER(name)
#define MK2_PROGRESS_ARGUMENT(value)
#define MK2_PROGRESS_GUARD(mask, row)
#define MK2_PROGRESS_ENABLED(mask, row) true
#define MK2_PROGRESS_MEMBER
#endif

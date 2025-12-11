/**
 *
 * @Name:
 * @Description: xxxxxxx
 * @Version: v1.0
 * @Date: 2019-03-23
 * @Copyright (c) 2019 BYD Co.,Ltd
 * @Author: luo peng <luopengchn@yeah.net>
 *
 */

#ifndef LIBCBDETECT_LOG_DEFS_H
#define LIBCBDETECT_LOG_DEFS_H

#include <assert.h>

#ifndef SVM_LOG_ERROR
#define SVM_LOG_ERROR(format, ...)    \
    SVM_print_log ("SVM ERROR %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif

#ifndef SVM_LOG_WARNING
#define SVM_LOG_WARNING(format, ...)   \
    SVM_print_log ("SVM WARNING %s:%d:" format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif

#ifndef SVM_LOG_INFO
#define SVM_LOG_INFO(format, ...)   \
    SVM_print_log ("SVM INFO %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif

#ifdef DEBUG

#ifndef SVM_LOG_DEBUG
#define SVM_LOG_DEBUG(format, ...)   \
      SVM_print_log ("SVM DEBUG %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif
#else
#define SVM_LOG_DEBUG(...)
#endif //end ifdef DEBUG

#define SVM_ASSERT(exp)  assert(exp)

#ifdef  __cplusplus
#define SVM_BEGIN_DECLARE  extern "C" {
#define SVM_END_DECLARE    }
#else
#define SVM_BEGIN_DECLARE
#define SVM_END_DECLARE
#endif

#ifndef __user
#define __user
#endif

#endif //LIBCBDETECT_LOG_DEFS_H

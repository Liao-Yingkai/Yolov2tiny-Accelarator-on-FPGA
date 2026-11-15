#ifndef CAMERA_INPUT_H
#define CAMERA_INPUT_H
#include <stdint.h>
#include "xaxivdma.h"
#include "camera_geometry.h"
enum { CAMERA_OK, CAMERA_CONFIG, CAMERA_SLOT, CAMERA_PARK,
       CAMERA_DMA_ERROR, CAMERA_TIMEOUT, CAMERA_OWNERSHIP };
int camera_input_check(XAxiVdma *vdma);
/* frames为连续三帧、cacheline对齐的DDR，启动前flush，启动后CPU只读。
 * snapshot是CPU自有的416×416紧密BGR ROI，与frames不重叠，DMA不访问它。
 * 检查配置→READ停车→等两次新事件→确认槽所有权→invalidate源帧→
 * 按行复制中心ROI→复查槽所有权及DMA错误→请求StopParking恢复READ循环。
 * 成功得到原尺寸BGR888 ROI；预览可能短暂定格，不保证独立新画面。
 * 失败输出可能尚未写入或只写入部分，禁止使用。所有停车后出口均
 * 请求StopParking；DMA故障可能需用户复位重启，不保证画面恢复，
 * 不自动清错/reset，不使用中断或动态分配。 */
int camera_input_capture(XAxiVdma *vdma, const uint8_t *frames,
                         uint8_t snapshot[CAMERA_ROI_BYTES]);
const char *camera_input_reason(int result);
#endif

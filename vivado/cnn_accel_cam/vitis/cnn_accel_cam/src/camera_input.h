#ifndef CAMERA_INPUT_H
#define CAMERA_INPUT_H
#include <stdint.h>
#include "xaxivdma.h"
/* 每行640像素，与当前摄像头及显示模式一致，单位：像素。 */
#define CAMERA_WIDTH 640U
/* 每帧480行，单位：行。 */
#define CAMERA_HEIGHT 480U
/* 每像素3字节，BGR888物理低地址依次存B、G、R，各8位。 */
#define CAMERA_BYTES_PER_PIXEL 3U
/* VDMA循环使用3个物理帧槽，不是新画面的序号。 */
#define CAMERA_FRAMES 3U
/* 行跨度=640像素×3字节/像素=1920字节，无行尾填充。 */
#define CAMERA_STRIDE (CAMERA_WIDTH*CAMERA_BYTES_PER_PIXEL)
/* 帧大小=1920字节/行×480行=921600字节。 */
#define CAMERA_FRAME_BYTES (CAMERA_STRIDE*CAMERA_HEIGHT)
enum { CAMERA_OK, CAMERA_CONFIG, CAMERA_SLOT, CAMERA_PARK,
       CAMERA_DMA_ERROR, CAMERA_TIMEOUT, CAMERA_OWNERSHIP, CAMERA_COPY_MISMATCH };
int camera_input_check(XAxiVdma *vdma);
/* frames为连续三帧、cacheline对齐的DDR，启动前flush，启动后CPU只读。
 * snapshot是CPU自有的一整帧缓冲，与frames不重叠，DMA不访问它。
 * 检查配置→READ停车→等两次新事件→确认槽所有权→invalidate源帧→
 * memcpy整帧→停车期间memcmp逐字节验证→请求StopParking恢复READ循环。
 * 成功得到完整原始BGR888；预览可能短暂定格，不保证独立新画面。
 * 失败输出可能尚未写入或只写入部分，禁止使用。所有停车后出口均
 * 请求StopParking；DMA故障可能需用户复位重启，不保证画面恢复，
 * 不自动清错/reset，不使用中断或动态分配。 */
int camera_input_capture(XAxiVdma *vdma, const uint8_t *frames,
                         uint8_t snapshot[CAMERA_FRAME_BYTES]);
const char *camera_input_reason(int result);
#endif

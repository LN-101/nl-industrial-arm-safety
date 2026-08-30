#ifndef __PUMP_H___
#define __PUMP_H___

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */


/* USER CODE BEGIN Private defines */
enum busy_flag{
    busy = 0,
    free = 1
};
enum send_flag{
    ok,
    error
};
extern enum busy_flag pump_dma_busy; // 枚举标志位，表示DMA是否忙
extern enum send_flag pump_send_flag; // 枚举标志位，表示泵是否忙
extern uint8_t pump_data_DMA[100];//气泵数据缓冲区
/* USER CODE END Private defines */


/* USER CODE BEGIN Prototypes */
void pump_init();
enum send_flag pump_send(uint8_t *data, uint16_t length);
void pump_on(uint8_t time_num);
void pump_task(void);
/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __PUMP_H___ */


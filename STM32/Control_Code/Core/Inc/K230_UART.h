#ifndef __K230_UART_H___
#define __K230_UART_H___
/* Define -*/
#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */


/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */


/* USER CODE BEGIN Prototypes */
int K230_UART_RxHandler(const uint8_t *rx_buf, uint16_t Size);
void K230_UART_ClearTargets(void);
void K230_UART_RequestRxRestart(void);
void K230_UART_Task(void);
void K230_UART_Init(void);
/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif
#endif /* __K230_UART_H___ */

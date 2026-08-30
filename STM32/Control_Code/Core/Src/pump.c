#include "pump.h"
#include "main.h"
#include "stm32f4xx_hal.h"
#include "stm32f4xx_hal_uart.h"
#include "stm32f4xx_hal_dma.h"
#include "usart.h"
#include "oled.h"
#include <string.h>

extern UART_HandleTypeDef huart4;
enum send_flag pump_send_flag = ok;
enum busy_flag pump_dma_busy = free;
uint8_t pump_data_DMA[100];
uint8_t pump_data[20]= "#005P2500T0000!";

#define PUMP_CMD_LEN                    15U
#define PUMP_CONTINUOUS_TIME_NUM        9U
#define PUMP_CONTINUOUS_REFRESH_MS      8000U

static uint8_t s_pump_continuous = 0U;
static uint32_t s_pump_last_refresh_tick = 0U;

static void pump_set_time_cmd(uint8_t time_num)
{
    pump_data[10] = (uint8_t)('0' + time_num);
    pump_data[11] = '0';
    pump_data[12] = '0';
    pump_data[13] = '0';
}

void pump_init(void)
{
    MX_UART4_Init();
}

enum send_flag pump_send(uint8_t *data, uint16_t length)
{
    if ((data == NULL) || (length == 0U) || (length > sizeof(pump_data_DMA)))
    {
        return error;
    }

    if (pump_dma_busy == busy)
    {
        return error;
    }

    pump_dma_busy = busy;
    memcpy(pump_data_DMA, data, length);

    if (HAL_UART_Transmit_DMA(&huart4, pump_data_DMA, length) != HAL_OK)
    {
        pump_dma_busy = free;   /* 启动失败，恢复空闲，防止死锁 */
        return error;
    }

    return ok;
}


/*
num   time
0     stop refreshing continuous command
1~9   1~9 seconds
10    continuous
*/
enum time_pump_flag{
    time_on,
    time_off,
};
enum time_pump_flag pump_time_flag = time_on;

void pump_on(uint8_t time_num)
{
    enum send_flag ret;

    switch (time_num)
    {
    case 0:
        s_pump_continuous = 0U;          /* PUT: stop sending new pump commands */
        break;
    case 10:
        s_pump_continuous = 1U;
        pump_set_time_cmd(PUMP_CONTINUOUS_TIME_NUM);  /* "#005P2500T9000!" */
        ret = pump_send(pump_data, PUMP_CMD_LEN);
        if (ret == ok)
        {
            s_pump_last_refresh_tick = HAL_GetTick();
        }
        else
        {
            s_pump_last_refresh_tick = HAL_GetTick() - PUMP_CONTINUOUS_REFRESH_MS;
        }
        break;
    default:
        if (time_num >= 1 && time_num <= 9)
        {
            s_pump_continuous = 0U;
            pump_set_time_cmd(time_num);   /* "#005P2500Tx000!" x=1..9 */
            pump_send(pump_data, PUMP_CMD_LEN);
        }
        break;
    }
}

void pump_task(void)
{
    uint32_t now;

    if (s_pump_continuous == 0U)
    {
        return;
    }

    now = HAL_GetTick();
    if ((now - s_pump_last_refresh_tick) >= PUMP_CONTINUOUS_REFRESH_MS)
    {
        pump_set_time_cmd(PUMP_CONTINUOUS_TIME_NUM);
        if (pump_send(pump_data, PUMP_CMD_LEN) == ok)
        {
            s_pump_last_refresh_tick = now;
        }
    }
}

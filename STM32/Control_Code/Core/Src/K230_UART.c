#include "K230_UART.h"
#include "main.h"
#include <stdio.h>
#include <string.h>

uint8_t K230_UART_RX_BUF[256];
uint8_t K230_TO_OUT[64] = "E0,0,0,0,0,0,0,0P";
extern UART_HandleTypeDef huart3;
static int slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
static char s_k230_stream[257];
static uint16_t s_k230_stream_len = 0U;
static volatile uint8_t s_k230_rx_restart = 0U;

void K230_UART_ClearTargets(void)
{
    int k;

    for (k = 0; k < 8; k++)
    {
        slot[k] = 0;
    }
    (void)snprintf((char *)K230_TO_OUT, sizeof(K230_TO_OUT), "E0,0,0,0,0,0,0,0P");
}

void K230_UART_Init(void)
{
    MX_USART3_UART_Init();
    K230_UART_ClearTargets();
    if ((HAL_UARTEx_ReceiveToIdle_DMA(&huart3, K230_UART_RX_BUF, 256) == HAL_OK) && (huart3.hdmarx != NULL))
    {
        __HAL_DMA_DISABLE_IT(huart3.hdmarx, DMA_IT_HT);
    }
    else
    {
        K230_UART_RequestRxRestart();
    }
}

void K230_UART_RequestRxRestart(void)
{
    s_k230_rx_restart = 1U;
}

void K230_UART_Task(void)
{
    HAL_StatusTypeDef ret;

    if (s_k230_rx_restart == 0U)
    {
        return;
    }

    s_k230_rx_restart = 0U;
    (void)HAL_UART_AbortReceive(&huart3);

    if (huart3.hdmarx != NULL)
    {
        huart3.hdmarx->State = HAL_DMA_STATE_READY;
        huart3.hdmarx->ErrorCode = HAL_DMA_ERROR_NONE;
    }
    huart3.RxState = HAL_UART_STATE_READY;
    huart3.ReceptionType = HAL_UART_RECEPTION_STANDARD;
    huart3.ErrorCode = HAL_UART_ERROR_NONE;
    huart3.RxXferCount = 0U;

    ret = HAL_UARTEx_ReceiveToIdle_DMA(&huart3, K230_UART_RX_BUF, 256);
    if ((ret == HAL_OK) && (huart3.hdmarx != NULL))
    {
        __HAL_DMA_DISABLE_IT(huart3.hdmarx, DMA_IT_HT);
    }
    else
    {
        s_k230_rx_restart = 1U;
    }
}

int K230_UART_RxHandler(const uint8_t *rx_buf, uint16_t Size)
{
    char *cur;
    char *end;
    char type;
    char *payload;
    char *tail;
    int x1 = 0;
    int y1 = 0;
    int parsed;
    int updated = 0;

    if ((rx_buf == NULL) || (Size == 0U))
    {
        return 0;
    }

    if (Size > (uint16_t)(sizeof(s_k230_stream) - 1U - s_k230_stream_len))
    {
        s_k230_stream_len = 0U;
        if (Size > (uint16_t)(sizeof(s_k230_stream) - 1U))
        {
            rx_buf = &rx_buf[Size - (uint16_t)(sizeof(s_k230_stream) - 1U)];
            Size = (uint16_t)(sizeof(s_k230_stream) - 1U);
        }
    }

    memcpy(&s_k230_stream[s_k230_stream_len], rx_buf, Size);
    s_k230_stream_len = (uint16_t)(s_k230_stream_len + Size);
    s_k230_stream[s_k230_stream_len] = '\0';
    cur = s_k230_stream;
    end = s_k230_stream + s_k230_stream_len;

    while (cur < end)
    {
        while ((cur < end) && (*cur != 'E'))
        {
            cur++;
        }

        if ((cur + 2) >= end)
        {
            break;
        }

        if ((cur[1] >= 'A') && (cur[1] <= 'D'))
        {
            type = cur[1];
            payload = &cur[2];
            if (*payload == ',')
            {
                payload++;
            }
        }
        else if ((cur[1] == ',') && ((cur + 3) < end) && (cur[2] >= 'A') && (cur[2] <= 'D'))
        {
            type = cur[2];
            payload = &cur[3];
            if (*payload == ',')
            {
                payload++;
            }
        }
        else
        {
            cur++;
            continue;
        }

        tail = strchr(payload, 'P');
        if (tail == NULL)
        {
            break;
        }
        *tail = '\0';

        parsed = sscanf(payload, "%d,%d", &x1, &y1);
        if (parsed == 2)
        {
            switch (type)
            {
            case 'A':
                slot[0] = x1;
                slot[1] = y1;
                updated = 1;
                break;
            case 'B':
                slot[2] = x1;
                slot[3] = y1;
                updated = 1;
                break;
            case 'C':
                slot[4] = x1;
                slot[5] = y1;
                updated = 1;
                break;
            case 'D':
                slot[6] = x1;
                slot[7] = y1;
                updated = 1;
                break;
            default:
                break;
            }
        }

        cur = tail + 1;
    }

    if (cur > s_k230_stream)
    {
        uint16_t remain = (uint16_t)(end - cur);
        if (remain > 0U)
        {
            memmove(s_k230_stream, cur, remain);
        }
        s_k230_stream_len = remain;
        s_k230_stream[s_k230_stream_len] = '\0';
    }

    if (updated != 0)
    {
        return snprintf((char *)K230_TO_OUT, sizeof(K230_TO_OUT),
                        "E%d,%d,%d,%d,%d,%d,%d,%dP",
                        slot[0], slot[1], slot[2], slot[3],
                        slot[4], slot[5], slot[6], slot[7]);
    }

    return 0;
}

/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "can.h"
#include "dma.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "dma.h"
#include "tim.h"
#include "smd.h"
#include "oled.h"
#include "stm32f4xx_hal.h"
#include "pump.h"
#include "K230_UART.h"
/* ---- 全局变量 ---------------------------------------------------------- */
/* 本文件是6轴机械臂主控逻辑，通过CAN总线与SMD电机驱动器通信，
 * 接收PC端ASCII协议命令，解析目标角度，控制机械臂运动 *///起始 0 0 360000 0 -10000 0--》0 380000 360000 0 -10000 0  末尾

can_frame_t g_can_frame = {0};          /* CAN发送/接收帧缓存（与SMD电机驱动通信） */
smd_data_t data[12];                 /* 6轴电机回传数据缓冲区，下标1-6对应电机1-6 */
SMD_Response receive_data_from[10];  /* SMD协议解析后的响应结构体数组 */
uint8_t catch_flag = 0;              /* 夹爪触发计数器：每收到一帧'A'...'B'命令自增，奇数次=抓取，偶数次=释放 */
uint8_t catch_time_flag = 0;         /* 抓取延时阶段标记：0=未开始计时，1=已开始计时 */
uint8_t free_flag = 0;               /* 保留位（当前未使用） */
uint8_t free_time_flag = 0;          /* 释放延时阶段标记：0=未开始计时，1=已开始计时 */
uint8_t grip_executed = 0;           /* 抓取动作已执行标志，防止重复触发 */
uint32_t time_ms1 = 0;               /* 抓取延时计时起点（ms），用于6秒超时判断 */
uint32_t time_ms2 = 0;               /* 释放延时计时起点（ms），用于6秒超时判断 */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
extern TIM_HandleTypeDef htim1;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
int32_t begin_out_run_angle[6][6] = {{0,0,360000,0,-10000,0},{0,380000,360000,0,-10000,0},{0,0,0,0,50000,0}}; /* 读取6轴实际位置时的缓冲区 *///,{0,-370000,350000,0,-10000,0}
int32_t begin_out_run_real_angle[6][6] = {{0,0,0,0,0,0},{0,0,0,0,0,0}}; /* 读取6轴实际位置时的缓冲区 */
int32_t angle_data[6] = {0, 0, 0, 0, 0, 0};   /* 解析后的6轴目标角度（相对于零点的偏置值） */
int32_t test_data[2][6] = {{0, 120000, 0, 0, 0, 0},{0, 0, 0, 0, 0, 0}}; /* 测试用预置角度 */
uint8_t test_data_flag = 0;          /* 测试数据切换标志：偶数/奇数切换两组预置角度 */
static uint8_t s_uart_rx_buf[2][128];        /* DMA双缓冲区：交替接收PC端ASCII命令帧 */
static volatile uint8_t s_uart_rx_active = 0U; /* 当前DMA填充的缓冲区索引(0或1) */
static volatile uint8_t s_uart_rx_restart = 0U;/* DMA需重启标志：ISR置1，主循环处理 */
static uint8_t  s_rx_remainder[128];          /* 跨回调的帧分片残留 */
static volatile uint16_t s_rx_remainder_len = 0U; /* 残留缓冲区有效字节数 */
static volatile uint32_t s_rx_remainder_tick = 0U;/* 残留缓冲区更新时间，用于超时丢弃 */
static volatile uint8_t s_uart_rx_drop_remainder = 0U; /* UART错误恢复时丢弃残包 */
static volatile uint8_t s_uart_confirm_ready = 0U; /* UART1确认回复待发送标志，主循环处理 */
static int32_t s_uart_confirm_angles[6] = {0, 0, 0, 0, 0, 0}; /* 确认回复角度快照 */
static volatile uint8_t s_id_reply_pending_count = 0U; /* 只读固件身份查询回复待发送计数 */
static volatile uint8_t s_uart_rx_discard_until_lf = 0U; /* 畸形身份查询行丢弃状态 */
static volatile uint32_t s_uart_rx_discard_tick = 0U; /* 畸形行丢弃状态更新时间 */
static const char s_firmware_build_id[] = "stm32-f407-unified-v2-p6";
#define P6_ENABLE_LEGACY_UART1_CONTROL 0U
#define IDENTITY_REPLY_PENDING_MAX 255U
#define K230_RX_QUEUE_DEPTH 32U
#define K230_AGGREGATE_QUIET_MS 40U
#define K230_AGGREGATE_MAX_WAIT_MS 120U
#define UART1_RX_REMAINDER_TIMEOUT_MS 200U
static uint8_t s_k230_rx_queue[K230_RX_QUEUE_DEPTH][256]; /* USART3收到的K230帧队列 */
static uint16_t s_k230_rx_queue_size[K230_RX_QUEUE_DEPTH]; /* K230帧长度队列 */
static uint8_t s_k230_parse_frame[256];       /* K230解析用本地副本，避免与DMA缓冲竞争 */
static volatile uint8_t s_k230_rx_head = 0U;  /* K230队列写入位置 */
static volatile uint8_t s_k230_rx_tail = 0U;  /* K230队列读取位置 */
static volatile uint8_t s_k230_rx_count = 0U; /* K230待解析帧数量 */
static volatile uint32_t s_k230_last_rx_tick = 0U; /* 最近一次收到K230帧的时刻，用于聚合多目标 */
uint8_t receive_data_test[10];       /* 辅助接收缓冲（保留） */
uint16_t read_real_angle[6];         /* 6轴实际角度暂存（通过CAN读取） */
static volatile uint8_t s_angle_cmd_ready = 0U;    /* 角度命令已就绪：main循环收到此标志后执行运动 */
static volatile uint8_t s_emergency_stop = 0U;     /* 急停信号：1=正常急停"CD"，2=DMA重启失败 */
static volatile uint8_t s_robotic_move_trigger = 0U;/* 机械臂"移动到预设位置"触发信号（"EF"命令） */
static volatile uint8_t s_pump_cmd_trigger = 0U;   /* 气泵命令触发：串口1收到"CPQE"/"PUT"后主循环执行 */
static volatile uint8_t s_pump_cmd_value = 0U;     /* 10=持续吸，0=停止 */
static volatile uint8_t s_speed_scale_ready = 0U;  /* 速度倍率命令就绪：串口1收到"S<num>Q"后主循环修改速度 */
static volatile uint16_t s_speed_scale_percent = 100U; /* 速度倍率百分比：100=1倍, 50=0.5倍 */
static int16_t s_robotic_speed_base = 50;     /* 速度百分比基准值，避免重复发送150时累乘 */
static volatile uint32_t s_arm_cmd_rx_count = 0U;  /* 已接收完整"A...B"机械臂数据帧次数 */
static volatile uint8_t s_arm_cmd_rx_display = 0U; /* 机械臂数据帧计数需要刷新到OLED */
static volatile uint32_t s_last_ok_tick = 0U;      /* 最后收到心跳/命令的时刻（看门狗计时基准） */
static int32_t s_zero_offset[6] = {0, 0, 0, 0, 0, 0}; /* 上电零点偏置 */
extern int test_flag;
extern uint8_t K230_UART_RX_BUF[256];
extern uint8_t K230_TO_OUT[64];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
/**
  * @brief  读取当前6轴实际位置并记录为零点偏置（上电原点标定）。
  *         调用后所有下发角度均为相对于上电位置的增量值。
  *         Read current position of all 6 motors and store as zero-point offset.
  *         After calling this, all positions are relative to the power-on position.
  */
static void capture_zero_offset(void)
{
  uint8_t i;
  /* 通过CAN总线读取全部6轴电机当前位置 */
  /* Read real position for all motors */
  Read_robotic_arm_real_angle();
  for (i = 0U; i < 6U; i++)
  {
    s_zero_offset[i] = data[i + 1U].real_pos_pulse;
  }
}

/**
  * @brief  验证零点偏置是否在容差范围内，超限则重试（防止上电时电机处于异常位置）。
  *         采用"读取-校验-延时重试"循环，确保零点标定可靠。
  *         Verify that all captured zero-offsets are within tolerance,
  *         retrying read + capture if any motor exceeds the threshold.
  * @param  max_retries  max additional attempts after the first read
  * @param  tolerance    allowed absolute deviation from zero (pulses)
  * @retval 0  all offsets within tolerance
  * @retval 1  still out-of-tolerance after max_retries exhausted
  */
static uint8_t verify_and_capture_zero_offset(uint8_t max_retries, int32_t tolerance)
{
  uint8_t retry;
  uint8_t i;

  for (retry = 0U; retry <= max_retries; retry++)
  {
    Read_robotic_arm_real_angle();
    for (i = 0U; i < 6U; i++)
    {
      s_zero_offset[i] = data[i + 1U].real_pos_pulse;
    }

    /* 检查每个电机的偏置值是否在阈值内 */
    /* Check every motor is within tolerance */
    uint8_t all_ok = 1U;
    for (i = 0U; i < 6U; i++)
    {
      int32_t v = s_zero_offset[i];
      if (v < 0L) { v = -v; }
      if (v > tolerance)
      {
        all_ok = 0U;
        break;
      }
    }

    if (all_ok != 0U)
    {
      return 0U;
    }

    /* 延迟200ms后重试，给电机驱动器留出响应时间 */
    if (retry < max_retries)
    {
      HAL_Delay(200);
    }
  }

  return 1U;
}

extern void robotic_move_to(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/**
  * @brief  将解析出的6轴角度数据显示在OLED屏幕上（4行布局，每行2轴）。
  *         方便现场调试时直观查看PC下发的目标角度。
  */
static void OLED_ShowUartData(const int32_t *angles)
{
  char buf[17];

  OLED_ShowString(0, 0, "UART CMD DATA   ");

  snprintf(buf, sizeof(buf), "1:%-5ld 2:%-5ld", (long)angles[0], (long)angles[1]);
  OLED_ShowString(0, 1, buf);

  snprintf(buf, sizeof(buf), "3:%-5ld 4:%-5ld", (long)angles[2], (long)angles[3]);
  OLED_ShowString(0, 2, buf);

  snprintf(buf, sizeof(buf), "5:%-5ld 6:%-5ld", (long)angles[4], (long)angles[5]);
  OLED_ShowString(0, 3, buf);
}

/**
  * @brief  判断字符是否为角度值分隔符（空格、逗号、回车、换行、制表符、空字符）。
  *         用于ASCII协议解析时跳过字段间隔。
  */
static uint8_t is_angle_delimiter(uint8_t ch)
{
  return (uint8_t)((ch == ' ') || (ch == ',') || (ch == '\r') || (ch == '\n') || (ch == '\t') || (ch == '\0'));
}

/**
  * @brief  解析ASCII格式的角度命令："-100,200,0,-50,300,400" 或 "100 -200 0 50 -300 400"。
  *         支持可选正负号，以分隔符（空格/逗号/回车等）区分各轴数值。
  *         必须解析出恰好6个整数才返回成功，否则视为无效帧。
  */
static uint8_t decode_angle_command_ascii(const uint8_t *rx_buf, uint16_t size, int32_t *out_angle)
{
  uint16_t idx = 0U;
  uint8_t count = 0U;

  if ((rx_buf == NULL) || (out_angle == NULL))
  {
    return 0U;
  }

  /* 逐位扫描，跳过前导分隔符，解析符号+数字，存入out_angle */
  while ((idx < size) && (count < 6U))
  {
    uint32_t value = 0U;
    uint8_t have_digit = 0U;
    int8_t sign = 1;

    /* 跳过前导分隔符 */
    while ((idx < size) && (is_angle_delimiter(rx_buf[idx]) != 0U))
    {
      idx++;
    }

    if (idx >= size)
    {
      break;
    }

    /* 处理可选的负号/正号 */
    /* Handle optional negative sign */
    if ((idx < size) && (rx_buf[idx] == '-'))
    {
      sign = -1;
      idx++;
    }
    else if ((idx < size) && (rx_buf[idx] == '+'))
    {
      idx++;
    }

    /* 解析数字部分（十进制） */
    while ((idx < size) && (rx_buf[idx] >= '0') && (rx_buf[idx] <= '9'))
    {
      have_digit = 1U;
      value = (uint32_t)(value * 10U + (uint32_t)(rx_buf[idx] - '0'));
      idx++;
    }

    /* 无有效数字则视为解析失败 */
    if (have_digit == 0U)
    {
      return 0U;
    }

    /* 保存带符号的结果 */
    out_angle[count] = (int32_t)(sign * (int32_t)value);
    count++;

    /* 跳过数字后的分隔符 */
    while ((idx < size) && (is_angle_delimiter(rx_buf[idx]) != 0U))
    {
      idx++;
    }
    /* 如果后面还有非数字、非符号、非分隔符的内容，则格式非法 */
    if ((idx < size) && ((rx_buf[idx] < '0') || (rx_buf[idx] > '9')) &&
        (rx_buf[idx] != '-') && (rx_buf[idx] != '+') &&
        (is_angle_delimiter(rx_buf[idx]) == 0U))
    {
      return 0U;
    }
  }

  /* 6个角度解析完毕后，剩余字符必须全部是分隔符，否则判定为无效 */
  while (idx < size)
  {
    if (is_angle_delimiter(rx_buf[idx]) == 0U)
    {
      return 0U;
    }
    idx++;
  }

  return (uint8_t)(count == 6U);
}

/**
  * @brief  统一的角度命令解码入口：优先尝试ASCII文本格式；
  *         失败后若数据长度为6字节（6个int8）或12字节（6个int16），
  *         则按二进制格式解码（向后兼容旧版协议）。
  */
static uint8_t decode_angle_command(const uint8_t *rx_buf, uint16_t size, int32_t *out_angle)
{
  uint8_t i;

  if ((rx_buf == NULL) || (out_angle == NULL))
  {
    return 0U;
  }

  /* 优先尝试ASCII文本解析（当前主用协议） */
  if (decode_angle_command_ascii(rx_buf, size, out_angle) != 0U)
  {
    return 1U;
  }

  /* 二进制兼容：6字节 = 6个int8 */
  if (size == 6U)
  {
    for (i = 0U; i < 6U; i++)
    {
      out_angle[i] = (int32_t)(int8_t)rx_buf[i];
    }
    return 1U;
  }

  /* 二进制兼容：12字节 = 6个int16（大端序） */
  if (size == 12U)
  {
    for (i = 0U; i < 6U; i++)
    {
      out_angle[i] = (int32_t)(((uint16_t)rx_buf[i * 2U] << 8) |
                     ((uint16_t)rx_buf[i * 2U + 1U]));
    }
    return 1U;
  }

  return 0U;
}

static uint8_t token_equals(const uint8_t *token, uint16_t len, const char *text)
{
  uint16_t i = 0U;

  if ((token == NULL) || (text == NULL))
  {
    return 0U;
  }

  while (text[i] != '\0')
  {
    if ((i >= len) || (token[i] != (uint8_t)text[i]))
    {
      return 0U;
    }
    i++;
  }

  return (uint8_t)(i == len);
}

static uint8_t parse_i32_token(const uint8_t *token, uint16_t len, int32_t *out_value)
{
  uint16_t idx = 0U;
  uint8_t have_digit = 0U;
  int8_t sign = 1;
  int32_t value = 0L;

  if ((token == NULL) || (out_value == NULL) || (len == 0U))
  {
    return 0U;
  }

  if (token[idx] == '-')
  {
    sign = -1;
    idx++;
  }
  else if (token[idx] == '+')
  {
    idx++;
  }

  while (idx < len)
  {
    if ((token[idx] < '0') || (token[idx] > '9'))
    {
      return 0U;
    }
    have_digit = 1U;
    value = (int32_t)(value * 10L + (int32_t)(token[idx] - '0'));
    idx++;
  }

  if (have_digit == 0U)
  {
    return 0U;
  }

  *out_value = (sign < 0) ? -value : value;
  return 1U;
}

static uint8_t parse_u16_token(const uint8_t *token, uint16_t len, uint16_t *out_value)
{
  uint16_t idx;
  uint32_t value = 0U;

  if ((token == NULL) || (out_value == NULL) || (len == 0U))
  {
    return 0U;
  }

  for (idx = 0U; idx < len; idx++)
  {
    if ((token[idx] < '0') || (token[idx] > '9'))
    {
      return 0U;
    }
    value = (uint32_t)(value * 10U + (uint32_t)(token[idx] - '0'));
    if (value > 1000U)
    {
      value = 1000U;
    }
  }

  *out_value = (uint16_t)value;
  return 1U;
}

static uint8_t unified_packet_tail_ok(const uint8_t *frame, uint16_t frame_len, uint16_t star_pos)
{
  if ((frame == NULL) || (star_pos >= frame_len))
  {
    return 0U;
  }

  return (uint8_t)(((star_pos + 4U) == frame_len) &&
                   (token_equals(&frame[star_pos + 1U], 3U, "CHK") != 0U));
}

static uint8_t identity_query_prefix_possible(const uint8_t *data, uint16_t len)
{
  static const uint8_t query[] = "Q,ID*CHK\r\n";
  uint16_t i;

  if ((data == NULL) || (len == 0U) || (len > (uint16_t)(sizeof(query) - 1U)))
  {
    return 0U;
  }

  for (i = 0U; i < len; i++)
  {
    if (data[i] != query[i])
    {
      return 0U;
    }
  }

  return 1U;
}

static uint8_t parse_identity_query_packet(const uint8_t *frame, uint16_t frame_len)
{
  static const uint8_t query[] = "Q,ID*CHK\r\n";
  uint16_t i;

  if ((frame == NULL) || (frame_len != (uint16_t)(sizeof(query) - 1U)))
  {
    return 0U;
  }

  for (i = 0U; i < frame_len; i++)
  {
    if (frame[i] != query[i])
    {
      return 0U;
    }
  }

  return 1U;
}

static void identity_reply_pending_add(void)
{
  uint32_t primask = __get_PRIMASK();

  __disable_irq();
  if (s_id_reply_pending_count < IDENTITY_REPLY_PENDING_MAX)
  {
    s_id_reply_pending_count++;
  }
  if (primask == 0U)
  {
    __enable_irq();
  }
}

static void identity_reply_pending_consume_one(void)
{
  uint32_t primask = __get_PRIMASK();

  __disable_irq();
  if (s_id_reply_pending_count != 0U)
  {
    s_id_reply_pending_count--;
  }
  if (primask == 0U)
  {
    __enable_irq();
  }
}

static uint16_t uart1_discard_until_lf(const uint8_t *data, uint16_t len)
{
  uint16_t i;

  if ((data == NULL) || (len == 0U))
  {
    return 0U;
  }

  for (i = 0U; i < len; i++)
  {
    if (data[i] == '\n')
    {
      s_uart_rx_discard_until_lf = 0U;
      s_uart_rx_discard_tick = 0U;
      return (uint16_t)(i + 1U);
    }
  }

  s_uart_rx_discard_until_lf = 1U;
  s_uart_rx_discard_tick = HAL_GetTick();
  return len;
}

static uint8_t firmware_build_id_is_valid(void)
{
  uint8_t len = 0U;

  while (s_firmware_build_id[len] != '\0')
  {
    char ch = s_firmware_build_id[len];
    uint8_t ok = (uint8_t)(((ch >= 'A') && (ch <= 'Z')) ||
                           ((ch >= 'a') && (ch <= 'z')) ||
                           ((ch >= '0') && (ch <= '9')) ||
                           (ch == '.') || (ch == '_') || (ch == '-'));
    if ((ok == 0U) || (len >= 31U))
    {
      return 0U;
    }
    len++;
  }

  return (uint8_t)(len >= 1U);
}

static uint8_t parse_unified_uart1_packet(const uint8_t *frame, uint16_t frame_len)
{
  uint16_t star_pos = 0U;
  uint16_t field_start[11];
  uint16_t field_len[11];
  uint8_t field_count = 0U;
  uint16_t start = 0U;
  uint16_t i;
  uint8_t dog_ok = 0U;
  uint8_t angle_numeric_count = 0U;
  uint8_t angle_mm_count = 0U;
  int32_t parsed_angles[6] = {0, 0, 0, 0, 0, 0};
  uint8_t stop_action = 0U; /* 0=MM, 1=CD, 2=EF */
  uint8_t pump_action = 0U; /* 0=MM, 1=PUT, 10=CPQE */
  uint8_t speed_has_value = 0U;
  uint16_t speed_percent = 100U;

  if ((frame == NULL) || (frame_len < 7U) || (frame[0] != 'U'))
  {
    return 0U;
  }

  while ((star_pos < frame_len) && (frame[star_pos] != '*'))
  {
    star_pos++;
  }
  if ((star_pos >= frame_len) ||
      (unified_packet_tail_ok(frame, frame_len, star_pos) == 0U))
  {
    return 0U;
  }

  for (i = 0U; i <= star_pos; i++)
  {
    if ((i == star_pos) || (frame[i] == ','))
    {
      if (field_count >= 11U)
      {
        return 0U;
      }
      field_start[field_count] = start;
      field_len[field_count] = (uint16_t)(i - start);
      field_count++;
      start = (uint16_t)(i + 1U);
    }
  }

  if ((field_count != 11U) ||
      (token_equals(&frame[field_start[0]], field_len[0], "U") == 0U))
  {
    return 0U;
  }

  if (token_equals(&frame[field_start[1]], field_len[1], "OK") != 0U)
  {
    dog_ok = 1U;
  }
  else if (token_equals(&frame[field_start[1]], field_len[1], "MM") == 0U)
  {
    return 0U;
  }

  for (i = 0U; i < 6U; i++)
  {
    uint8_t field = (uint8_t)(i + 2U);
    if (token_equals(&frame[field_start[field]], field_len[field], "MM") != 0U)
    {
      angle_mm_count++;
    }
    else if (parse_i32_token(&frame[field_start[field]], field_len[field], &parsed_angles[i]) != 0U)
    {
      angle_numeric_count++;
    }
    else
    {
      return 0U;
    }
  }

  if (!((angle_numeric_count == 6U) || (angle_mm_count == 6U)))
  {
    return 0U;
  }

  if (token_equals(&frame[field_start[8]], field_len[8], "CD") != 0U)
  {
    stop_action = 1U;
  }
  else if (token_equals(&frame[field_start[8]], field_len[8], "EF") != 0U)
  {
    stop_action = 2U;
  }
  else if (token_equals(&frame[field_start[8]], field_len[8], "MM") == 0U)
  {
    return 0U;
  }

  if (token_equals(&frame[field_start[9]], field_len[9], "CPQE") != 0U)
  {
    pump_action = 10U;
  }
  else if (token_equals(&frame[field_start[9]], field_len[9], "PUT") != 0U)
  {
    pump_action = 1U;
  }
  else if (token_equals(&frame[field_start[9]], field_len[9], "MM") == 0U)
  {
    return 0U;
  }

  if (token_equals(&frame[field_start[10]], field_len[10], "MM") == 0U)
  {
    if (parse_u16_token(&frame[field_start[10]], field_len[10], &speed_percent) == 0U)
    {
      return 0U;
    }
    speed_has_value = 1U;
  }

  if ((dog_ok != 0U) || (angle_numeric_count == 6U) ||
      (stop_action != 0U) || (pump_action != 0U) || (speed_has_value != 0U))
  {
    s_last_ok_tick = HAL_GetTick();
  }

  if (angle_numeric_count == 6U)
  {
    uint8_t ai;
    catch_flag++;
    s_arm_cmd_rx_count++;
    s_arm_cmd_rx_display = 1U;
    for (ai = 0U; ai < 6U; ai++)
    {
      angle_data[ai] = parsed_angles[ai];
      s_uart_confirm_angles[ai] = parsed_angles[ai];
    }
    s_angle_cmd_ready = 1U;
    s_uart_confirm_ready = 1U;
  }

  if (stop_action == 1U)
  {
    s_emergency_stop = 1U;
  }
  else if (stop_action == 2U)
  {
    s_robotic_move_trigger = 1U;
  }

  if (pump_action != 0U)
  {
    s_pump_cmd_value = (pump_action == 10U) ? 10U : 0U;
    s_pump_cmd_trigger = 1U;
  }

  if (speed_has_value != 0U)
  {
    s_speed_scale_percent = speed_percent;
    s_speed_scale_ready = 1U;
  }

  return 1U;
}

/**
  * @brief  通过UART1向PC发送"OK <角度列表>"确认应答。
  *         每次成功解析角度命令后调用，供上位机确认接收无误。
  *         Send a confirmation response via UART1 with the decoded angle data.
  *         Called in RxEventCallback to acknowledge receipt and echo parsed values.
  */
static void uart_send_confirm(const int32_t *angles)
{
  char buf[64];
  int len;

  if (angles == NULL)
  {
    return;
  }

  len = snprintf(buf, sizeof(buf), "OK %ld,%ld,%ld,%ld,%ld,%ld\r\n",
         (long)angles[0], (long)angles[1], (long)angles[2],
         (long)angles[3], (long)angles[4], (long)angles[5]);

  if (len > 0 && len < (int)sizeof(buf))
  {
    (void)uart1_transmit_dma_copy((const uint8_t *)buf, (uint16_t)len);
  }
}

static uint8_t uart_send_identity_response(void)
{
  char buf[48];
  int len;

  if (firmware_build_id_is_valid() == 0U)
  {
    return 0U;
  }

  len = snprintf(buf, sizeof(buf), "I,1,%s*CHK\r\n", s_firmware_build_id);
  if ((len <= 0) || (len >= (int)sizeof(buf)))
  {
    return 0U;
  }

  return uart1_transmit_dma_copy((const uint8_t *)buf, (uint16_t)len);
}

static uint8_t uart1_start_rx_dma(void)
{
  HAL_StatusTypeDef ret;

  ret = HAL_UARTEx_ReceiveToIdle_DMA(&huart1,
                                     s_uart_rx_buf[s_uart_rx_active],
                                     sizeof(s_uart_rx_buf[0]));
  if ((ret == HAL_OK) && (huart1.hdmarx != NULL))
  {
    __HAL_DMA_DISABLE_IT(huart1.hdmarx, DMA_IT_HT);
    return 1U;
  }

  return 0U;
}

static void uart1_store_rx_remainder(const uint8_t *data, uint16_t len)
{
  if ((data != NULL) && (len > 0U) && (len <= sizeof(s_rx_remainder)))
  {
    (void)memcpy(s_rx_remainder, data, len);
    s_rx_remainder_len = len;
    s_rx_remainder_tick = HAL_GetTick();
  }
  else
  {
    s_rx_remainder_len = 0U;
    s_rx_remainder_tick = 0U;
  }
}

static void uart1_drop_stale_remainder(void)
{
  uint32_t now = HAL_GetTick();

  if ((s_rx_remainder_len != 0U) &&
      ((now - s_rx_remainder_tick) > UART1_RX_REMAINDER_TIMEOUT_MS))
  {
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    if ((s_rx_remainder_len != 0U) &&
        ((now - s_rx_remainder_tick) > UART1_RX_REMAINDER_TIMEOUT_MS))
    {
      s_rx_remainder_len = 0U;
      s_rx_remainder_tick = 0U;
    }
    if (primask == 0U)
    {
      __enable_irq();
    }
  }

  if ((s_uart_rx_discard_until_lf != 0U) &&
      ((now - s_uart_rx_discard_tick) > UART1_RX_REMAINDER_TIMEOUT_MS))
  {
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    if ((s_uart_rx_discard_until_lf != 0U) &&
        ((now - s_uart_rx_discard_tick) > UART1_RX_REMAINDER_TIMEOUT_MS))
    {
      s_uart_rx_discard_until_lf = 0U;
      s_uart_rx_discard_tick = 0U;
    }
    if (primask == 0U)
    {
      __enable_irq();
    }
  }
}

void uart1_request_rx_restart(void)
{
  s_uart_rx_drop_remainder = 1U;
  s_uart_rx_discard_until_lf = 0U;
  s_uart_rx_discard_tick = 0U;
  s_uart_rx_restart = 1U;
}

static void k230_drain_rx_queue(void)
{
  while (s_k230_rx_count != 0U)
  {
    uint16_t k230_size;
    uint32_t primask;

    primask = __get_PRIMASK();
    __disable_irq();
    k230_size = s_k230_rx_queue_size[s_k230_rx_tail];
    (void)memcpy(s_k230_parse_frame, s_k230_rx_queue[s_k230_rx_tail], k230_size);
    s_k230_rx_tail = (uint8_t)((s_k230_rx_tail + 1U) % K230_RX_QUEUE_DEPTH);
    s_k230_rx_count--;
    if (primask == 0U)
    {
      __enable_irq();
    }

    (void)K230_UART_RxHandler(s_k230_parse_frame, k230_size);
  }
}

static void k230_wait_quiet_and_drain(uint32_t quiet_ms, uint32_t max_wait_ms)
{
  uint32_t start = HAL_GetTick();

  do
  {
    k230_drain_rx_queue();
    if ((HAL_GetTick() - s_k230_last_rx_tick) >= quiet_ms)
    {
      break;
    }
  } while ((HAL_GetTick() - start) < max_wait_ms);

  k230_drain_rx_queue();
}

/**
  * @brief  UART1 DMA空闲/传输完成回调：
  *         优先在回调中快速重启RX DMA，减少接收空窗；
  *         若HAL忙或异常，则标记主循环执行AbortReceive后兜底恢复。
  */
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
  if (huart->Instance == USART1)
  {
    
  

  /* ---- 第1步：快照当前缓冲区，切换指针，标记需重启 ---- */
  uint8_t *rx_data = s_uart_rx_buf[s_uart_rx_active];
  uint16_t  rx_size = Size;

  s_uart_rx_active ^= 1U;
  if (uart1_start_rx_dma() == 0U)
  {
    s_uart_rx_restart = 1U;  /* 主循环看到此标志后会安全地重启DMA */
  }
  else
  {
    if (s_uart_rx_drop_remainder == 0U)
    {
      s_uart_rx_restart = 0U;
    }
  }

  /* ---- 第2步：拼接残留 + 新数据 ---- */
  {
    uint8_t work[256];
    uint16_t work_len;
    uint16_t rem_len = s_rx_remainder_len;

    if ((rem_len > 0U) &&
        ((HAL_GetTick() - s_rx_remainder_tick) > UART1_RX_REMAINDER_TIMEOUT_MS))
    {
      rem_len = 0U;
      s_rx_remainder_len = 0U;
      s_rx_remainder_tick = 0U;
    }

    if ((rem_len > 0U) && ((uint32_t)rem_len + (uint32_t)rx_size <= sizeof(work)))
    {
      (void)memcpy(&work[0], s_rx_remainder, rem_len);
      (void)memcpy(&work[rem_len], rx_data, rx_size);
      work_len = rem_len + rx_size;
      s_rx_remainder_len = 0U;
      s_rx_remainder_tick = 0U;
    }
    else
    {
      (void)memcpy(work, rx_data, rx_size);
      work_len = rx_size;
      s_rx_remainder_len = 0U;
      s_rx_remainder_tick = 0U;
    }

    /* ---- 第3步：逐帧解析 ---- */
    uint16_t pos = 0U;
    while (pos < work_len)
    {
      uint16_t remaining = work_len - pos;
      uint8_t *cur = &work[pos];

      if (s_uart_rx_discard_until_lf != 0U)
      {
        pos = (uint16_t)(pos + uart1_discard_until_lf(cur, remaining));
        continue;
      }

      if ((cur[0] == '\r') || (cur[0] == '\n') || (cur[0] == ' ') || (cur[0] == '\t'))
      {
        pos++;
        continue;
      }

      /* “U,OK,P1,P2,P3,P4,P5,P6,STOP,PUMP,SPD*CHK\r\n” 统一控制包 */
      if (cur[0] == 'U')
      {
        uint16_t scan = 0U;
        uint16_t frame_len = 0U;
        uint16_t consume_len = 0U;

        while ((scan + 3U) < remaining)
        {
          if ((cur[scan] == '*') && (cur[scan + 1U] == 'C') &&
              (cur[scan + 2U] == 'H') && (cur[scan + 3U] == 'K'))
          {
            frame_len = (uint16_t)(scan + 4U);
            consume_len = frame_len;
            while ((consume_len < remaining) &&
                   ((cur[consume_len] == '\r') || (cur[consume_len] == '\n')))
            {
              consume_len++;
            }
            break;
          }
          scan++;
        }

        if (frame_len != 0U)
        {
          (void)parse_unified_uart1_packet(cur, frame_len);
          pos += consume_len;
          continue;
        }
        else
        {
          uart1_store_rx_remainder(cur, remaining);
          break;
        }
      }

      /* “Q,ID*CHK\r\n” 只读固件身份查询: 只置回复计数, 不刷新看门狗, 不触发运动/泵/速度 */
      if (cur[0] == 'Q')
      {
        const uint16_t identity_len = 10U;

        if (remaining >= identity_len)
        {
          if (parse_identity_query_packet(cur, identity_len) != 0U)
          {
            identity_reply_pending_add();
            pos = (uint16_t)(pos + identity_len);
            continue;
          }

          s_uart_rx_discard_until_lf = 1U;
          pos = (uint16_t)(pos + uart1_discard_until_lf(cur, remaining));
          continue;
        }

        if (identity_query_prefix_possible(cur, remaining) != 0U)
        {
          uart1_store_rx_remainder(cur, remaining);
          break;
        }

        s_uart_rx_discard_until_lf = 1U;
        pos = (uint16_t)(pos + uart1_discard_until_lf(cur, remaining));
        continue;
      }

#if P6_ENABLE_LEGACY_UART1_CONTROL
      /* “OK” 心跳 */
      if ((remaining >= 2U) && (cur[0] == 'O') && (cur[1] == 'K'))
      {
        s_last_ok_tick = HAL_GetTick();
        pos += 2U;
        continue;
      }

      /* “EF” 移动到预设位置 */
      if ((remaining >= 2U) && (cur[0] == 'E') && (cur[1] == 'F'))
      {
        s_robotic_move_trigger = 1U;
        s_last_ok_tick = HAL_GetTick();
        pos += 2U;
        continue;
      }

      /* “CPQE” 气泵持续吸 */
      if ((remaining >= 4U) && (cur[0] == 'C') && (cur[1] == 'P') &&
          (cur[2] == 'Q') && (cur[3] == 'E'))
      {
        s_pump_cmd_value = 10U;
        s_pump_cmd_trigger = 1U;
        s_last_ok_tick = HAL_GetTick();
        pos += 4U;
        continue;
      }

      /* “PUT” 停止续发气泵指令 */
      if ((remaining >= 3U) && (cur[0] == 'P') && (cur[1] == 'U') && (cur[2] == 'T'))
      {
        s_pump_cmd_value = 0U;
        s_pump_cmd_trigger = 1U;
        s_last_ok_tick = HAL_GetTick();
        pos += 3U;
        continue;
      }

      /* “S<num>Q” 速度倍率命令：S100Q=当前速度1倍, S50Q=当前速度0.5倍 */
      if (cur[0] == 'S')
      {
        uint16_t scan = 1U;
        uint32_t percent = 0U;
        uint8_t have_digit = 0U;

        while ((scan < remaining) && (cur[scan] != 'Q'))
        {
          if ((cur[scan] < '0') || (cur[scan] > '9'))
          {
            break;
          }
          have_digit = 1U;
          percent = (uint32_t)(percent * 10U + (uint32_t)(cur[scan] - '0'));
          scan++;
        }

        if ((scan < remaining) && (cur[scan] == 'Q') && (have_digit != 0U))
        {
          if (percent > 1000U)
          {
            percent = 1000U;
          }
          s_speed_scale_percent = (uint16_t)percent;
          s_speed_scale_ready = 1U;
          s_last_ok_tick = HAL_GetTick();
          pos += scan + 1U;
          continue;
        }
        else if (scan >= remaining)
        {
          uart1_store_rx_remainder(cur, remaining);
          break;
        }
      }

      /* “CD” 紧急停止 */
      if ((remaining >= 2U) && (cur[0] == 'C') && (cur[1] == 'D'))
      {
        s_emergency_stop = 1U;
        pos += 2U;
        continue;
      }

      /* “A...B” 角度命令帧 */
      if (cur[0] == 'A')
      {
        uint16_t scan = 1U;
        while ((scan < remaining) && (cur[scan] != 'B'))
        {
          scan++;
        }

        if (scan < remaining)
        {
          catch_flag++;
          s_arm_cmd_rx_count++;
          s_arm_cmd_rx_display = 1U;
          if (decode_angle_command(&cur[1], scan - 1U, angle_data) != 0U)
          {
            uint8_t ai;
            s_angle_cmd_ready = 1U;
            for (ai = 0U; ai < 6U; ai++)
            {
              s_uart_confirm_angles[ai] = angle_data[ai];
            }
            s_uart_confirm_ready = 1U;
          }
          pos += scan + 1U;
          continue;
        }
        else
        {
          uart1_store_rx_remainder(cur, remaining);
          break;
        }
      }
#endif

      pos++;
    }
  }
  }
  else if(huart->Instance == USART3)
  {
    uint16_t copy_size = Size;
    if (copy_size > sizeof(s_k230_rx_queue[0]))
    {
      copy_size = sizeof(s_k230_rx_queue[0]);
    }
    if (s_k230_rx_count >= K230_RX_QUEUE_DEPTH)
    {
      s_k230_rx_tail = (uint8_t)((s_k230_rx_tail + 1U) % K230_RX_QUEUE_DEPTH);
      s_k230_rx_count--;
    }
    (void)memcpy(s_k230_rx_queue[s_k230_rx_head], K230_UART_RX_BUF, copy_size);
    s_k230_rx_queue_size[s_k230_rx_head] = copy_size;
    s_k230_rx_head = (uint8_t)((s_k230_rx_head + 1U) % K230_RX_QUEUE_DEPTH);
    s_k230_rx_count++;
    s_k230_last_rx_tick = HAL_GetTick();
    {
      HAL_StatusTypeDef kret;
      kret = HAL_UARTEx_ReceiveToIdle_DMA(&huart3, K230_UART_RX_BUF, 256);
      if ((kret == HAL_OK) && (huart3.hdmarx != NULL))
      {
        __HAL_DMA_DISABLE_IT(huart3.hdmarx, DMA_IT_HT);
      }
      else
      {
        K230_UART_RequestRxRestart();
      }
    }
  }
  else{
    return;
  }
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  smd_init();     /* 初始化SMD协议栈（CAN滤波器、报文回调等） */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_CAN2_Init();
  MX_I2C1_Init();
  MX_USART1_UART_Init();
  MX_TIM1_Init();
  MX_UART4_Init();
  // MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */
  pump_init();
  OLED_Init();
  K230_UART_Init();
  s_robotic_speed_base = robotic_run_speed;
  OLED_Clear();
  HAL_Delay(7000);   /* 等待7秒让SMD驱动器上电完成 */
  OLED_ShowString(0, 0, "WOSHI");  /* 显示"READY"提示，等待PC连接 */

  /* 上电零点标定：读取当前位置作为软件零点，验证在±500脉冲内，最多重试3次 */
  /* Capture current positions as software zero-point offset,
     verify within ±500 pulses, retry up to 3 times if out of tolerance */
  (void)verify_and_capture_zero_offset(3U, 500L);
  HAL_Delay(3000);
  // int k;
  // for (k = 0U; k < 6U; k++)
  // {
  //   begin_out_run_real_angle[0][k] = begin_out_run_angle[0][k] + s_zero_offset[k];
  // }
  // robotic_arm_control(begin_out_run_real_angle[0]); /* 启动时先移动到一个安全的预设位置，避免机械臂处于奇异点 */
  // HAL_Delay(1000);
  //  for (k = 0U; k < 6U; k++)
  // {
  //   begin_out_run_real_angle[1][k] = begin_out_run_angle[1][k] + s_zero_offset[k];
  // }
  // robotic_arm_control(begin_out_run_real_angle[1]);
  // HAL_Delay(1000);
  // HAL_UART_Transmit(&huart1,data[6].data,data[6].len,1000);
  //  for (k = 0U; k < 6U; k++)
  // {
  //     s_zero_offset[k]= begin_out_run_angle[1][k] + s_zero_offset[k];
  // }
  // HAL_Delay(1000);
  //   for (k = 0U; k < 6U; k++)
  // {
  //   begin_out_run_real_angle[2][k] = begin_out_run_angle[2][k] + s_zero_offset[k];
  // }
  // robotic_arm_control(begin_out_run_real_angle[2]);
  // HAL_Delay(1000);
  /* 启动UART1 DMA空闲接收（双缓冲，ISR只标记需重启，主循环安全重启） */
  {
    s_uart_rx_active = 0U;
    if (uart1_start_rx_dma() == 0U)
    {
      HAL_UART_Transmit(&huart1, (uint8_t *)"INIT_ERR\r\n", 10, 1000);
    }
  }
  /* 发送启动标记，确认系统已就绪（调试用） */
  /* DEBUG: send boot marker to confirm UART TX works */
HAL_UART_Transmit(&huart1, (uint8_t *)"BOOT\r\n", 6, 1000);
  /* 启动看门狗计时（上电后首次记录心跳基准） */
  s_last_ok_tick = HAL_GetTick();

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */
    pump_task();
    K230_UART_Task();
    uart1_drop_stale_remainder();

    /* ---- K230数据解析：USART3回调只入队，主循环执行sscanf等较重操作 ---- */
    k230_drain_rx_queue();

    /* ---- UART1确认回复：避免在USART1接收中断中调用阻塞发送 ---- */
    if ((s_uart_confirm_ready != 0U) && (uart1_tx_dma_is_busy() == 0U))
    {
      int32_t confirm_angles[6];
      uint8_t ci;
      uint32_t primask;

      primask = __get_PRIMASK();
      __disable_irq();
      for (ci = 0U; ci < 6U; ci++)
      {
        confirm_angles[ci] = s_uart_confirm_angles[ci];
      }
      s_uart_confirm_ready = 0U;
      if (primask == 0U)
      {
        __enable_irq();
      }

      uart_send_confirm(confirm_angles);
    }

    /* ---- 固件身份只读查询回复：RX回调只置标志, 主循环在TX DMA空闲时发送 ---- */
    if ((s_id_reply_pending_count != 0U) && (uart1_tx_dma_is_busy() == 0U))
    {
      if (uart_send_identity_response() != 0U)
      {
        identity_reply_pending_consume_one();
      }
    }

    /* ---- DMA重装：ISR标记需要重启，主循环安全执行（SysTick有效） ---- */
    if (s_uart_rx_restart != 0U)
    {
      {
        HAL_StatusTypeDef rret;
        uint8_t drop_remainder;
        uint32_t primask;

        primask = __get_PRIMASK();
        __disable_irq();
        s_uart_rx_restart = 0U;
        drop_remainder = s_uart_rx_drop_remainder;
        s_uart_rx_drop_remainder = 0U;
        if (primask == 0U)
        {
          __enable_irq();
        }

        if (drop_remainder != 0U)
        {
          primask = __get_PRIMASK();
          __disable_irq();
          s_rx_remainder_len = 0U;
          s_rx_remainder_tick = 0U;
          if (primask == 0U)
          {
            __enable_irq();
          }
        }

        /* AbortReceive在主循环中调用是安全的（SysTick正常递增，timeout有效） */
        rret = HAL_UART_AbortReceive(&huart1);
        /* 即使Abort失败也强制复位状态后重装 */
        if (rret != HAL_OK && huart1.hdmarx != NULL)
        {
          huart1.hdmarx->State = HAL_DMA_STATE_READY;
        }
        if (huart1.hdmarx != NULL)
        {
          huart1.hdmarx->State = HAL_DMA_STATE_READY;
          huart1.hdmarx->ErrorCode = HAL_DMA_ERROR_NONE;
        }
        huart1.RxState       = HAL_UART_STATE_READY;
        huart1.ReceptionType = HAL_UART_RECEPTION_STANDARD;
        huart1.ErrorCode     = HAL_UART_ERROR_NONE;
        huart1.RxXferCount   = 0U;

        if (uart1_start_rx_dma() == 0U)
        {
          /* 重装失败（极少见）：再复位一次后重试 */
          if (huart1.hdmarx != NULL)
          {
            huart1.hdmarx->State = HAL_DMA_STATE_READY;
            huart1.hdmarx->ErrorCode = HAL_DMA_ERROR_NONE;
          }
          huart1.RxState       = HAL_UART_STATE_READY;
          huart1.ReceptionType = HAL_UART_RECEPTION_STANDARD;
          huart1.ErrorCode     = HAL_UART_ERROR_NONE;
          huart1.RxXferCount   = 0U;
          if (uart1_start_rx_dma() == 0U)
          {
            s_emergency_stop = 2U;
          }
        }
      }
    }

    /* ---- 处理角度运动命令：s_angle_cmd_ready由UART回调置位 ---- */
    /* 将解析后的增量角度加上零点偏置，换算为绝对脉冲值后下发各电机 */
    if (s_angle_cmd_ready != 0U)
    {
      s_angle_cmd_ready = 0U;
      // OLED_ShowUartData(angle_data);        /* 在OLED上刷新显示目标角度 */
      robotic_arm_reset_target_cache();     /* 清除上次运动的目标缓存 */
      /* 增量角度 + 零点偏置 = 绝对脉冲位置 */
      /* Add zero offset: convert relative position to absolute position */
      {
        int32_t abs_pulse[6];
        uint8_t j;
        for (j = 0U; j < 6U; j++)
        {
          abs_pulse[j] = angle_data[j] + s_zero_offset[j];
        }
        robotic_arm_control(abs_pulse);     /* 通过CAN总线下发6轴目标位置 */
      }
    }

    /* ---- 看门狗检查：超过2秒未收到任何命令则自动急停 ---- */
    // if ((HAL_GetTick() - s_last_ok_tick) > 2000U)
    // {
    //   s_last_ok_tick = HAL_GetTick();
    //   robotic_stop();
    // }

    /* ---- 急停处理：stop_reason=1为正常急停（刹车），=2为DMA故障（上报DMA_ERR） ---- */
    if (s_emergency_stop != 0U)
    {
      uint8_t stop_reason = s_emergency_stop;
      s_emergency_stop = 0U;
      if (stop_reason == 2U)
      {
        /* DMA重装失败：报告错误后强制复位恢复 */
        HAL_UART_Transmit(&huart1, (uint8_t *)"DMA_ERR\r\n", 9, 1000);

        {
          DMA_Stream_TypeDef *dma_s;

          if (huart1.hdmarx != NULL) {
            dma_s = huart1.hdmarx->Instance;
            dma_s->CR &= ~(DMA_SxCR_TCIE | DMA_SxCR_TEIE | DMA_SxCR_DMEIE | DMA_SxCR_HTIE);
            dma_s->NDTR = 0U;
            huart1.hdmarx->State     = HAL_DMA_STATE_READY;
            huart1.hdmarx->ErrorCode = HAL_DMA_ERROR_NONE;
          }
          huart1.RxState       = HAL_UART_STATE_READY;
          huart1.ReceptionType = HAL_UART_RECEPTION_STANDARD;
          huart1.ErrorCode     = HAL_UART_ERROR_NONE;
          huart1.RxXferCount   = 0U;

          (void)uart1_start_rx_dma();
        }
      }
      else
      {
        /* 正常急停（收到"CD"命令）：立即刹车所有电机 */
        robotic_stop();
        HAL_UART_Transmit(&huart1, (uint8_t *)"OK\r\n", 4, 1000);
      }
    }

    /* ---- "EF"命令处理：机械臂移动到预设目标位置（如抓取预备位） ---- */
    if (s_robotic_move_trigger != 0U)
    {
      s_robotic_move_trigger = 0U;
      robotic_move_to();     /* 执行预设轨迹运动 */
    }

    /* ---- "CPQE"/"PUT"命令处理：持续吸/停止续发气泵指令 ---- */
    if (s_pump_cmd_trigger != 0U)
    {
      s_pump_cmd_trigger = 0U;
      pump_on(s_pump_cmd_value);
    }

    /* ---- "S<num>Q"命令处理：按百分比修改机械臂运行速度 ---- */
    if (s_speed_scale_ready != 0U)
    {
      uint16_t percent;
      int32_t new_speed;

      s_speed_scale_ready = 0U;
      percent = s_speed_scale_percent;

      new_speed = ((int32_t)s_robotic_speed_base * (int32_t)percent + 50L) / 100L;
      if (new_speed > 32767L)
      {
        new_speed = 32767L;
      }
      else if (new_speed < 0L)
      {
        new_speed = 0L;
      }
      robotic_run_speed = (int16_t)new_speed;
    }

    /* ---- OLED显示机械臂数据帧接收次数：收到A...B后显示ABX ---- */
    if (s_arm_cmd_rx_display != 0U)
    {
      char oled_buf[17];
      uint32_t count;

      s_arm_cmd_rx_display = 0U;
      count = s_arm_cmd_rx_count;
      (void)snprintf(oled_buf, sizeof(oled_buf), "AB%lu", (unsigned long)count);
      OLED_ShowString(0, 4, "                ");
      OLED_ShowString(0, 4, oled_buf);
    }

    /* ---- 读取实际角度反馈：当CAN读取完成标志置位时 ---- */
    /* 将电机绝对位置减去零点偏置，按"A<角度>B"格式通过DMA发送给PC */
    if(Read_pos_flag == FLAG_OK){
      Read_pos_flag = FLAG_ERROR;
      Read_robotic_arm_real_angle();
      char buf[192];
      char k230_buf[64];
      uint32_t primask;
      /* 绝对位置 - 零点偏置 = 相对位置（以"A<角度>B"帧格式上报），后接最新K230数据 */
      /* Subtract zero offset: convert absolute position to relative position */

      k230_wait_quiet_and_drain(K230_AGGREGATE_QUIET_MS, K230_AGGREGATE_MAX_WAIT_MS);

      primask = __get_PRIMASK();
      __disable_irq();
      (void)strncpy(k230_buf, (const char *)K230_TO_OUT, sizeof(k230_buf));
      k230_buf[sizeof(k230_buf) - 1U] = '\0';
      if (primask == 0U)
      {
        __enable_irq();
      }

      int len = snprintf(buf, sizeof(buf), "A%ld,%ld,%ld,%ld,%ld,%ldB%s\r\n",
        (long)(data[1].real_pos_pulse - s_zero_offset[0]),
        (long)(data[2].real_pos_pulse - s_zero_offset[1]),
        (long)(data[3].real_pos_pulse - s_zero_offset[2]),
        (long)(data[4].real_pos_pulse - s_zero_offset[3]),
        (long)(data[5].real_pos_pulse - s_zero_offset[4]),
        (long)(data[6].real_pos_pulse - s_zero_offset[5]),
        k230_buf);
      if ((len > 0) && (len < (int)sizeof(buf)))
      {
        /* 使用DMA复制发送，不阻塞主循环；发送成功排队后清空一次性视觉坐标 */
        if (uart1_transmit_dma_copy((const uint8_t *)buf, (uint16_t)len) != 0U)
        {
          K230_UART_ClearTargets();
        }
      }
    }

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

#ifndef __SMD_H
#define __SMD_H
#define COMM_TYPE               1           
#define CAN_EXTID               0x1000      

#define FRAME_HEAD              0xC5        
#define FRAME_TAIL              0x5C


#include "stdbool.h"
#include "string.h"
#include "can.h"
#include "stdint.h"

typedef struct {
    uint8_t data[256];
    uint16_t len;
} SMD_Response;

typedef void (*smd_addr_cmd_fn_t)(uint8_t addr);

extern int16_t robotic_run_speed;       /* 机械臂运行速度 */
extern int16_t robotic_run_acc;         /* 机械臂运行加速度 */

#define SMD_TRANS_OK            1U
#define SMD_TRANS_FAIL          0U

typedef enum
{
    FCT_IDLE                    = 0x00,      
    FCT_CAL_ENCODER             = 0x01,      
    FCT_RESTART                 = 0x02,      
    FCT_RESET_FACTORY           = 0x03,      
    FCT_PARAM_SAVE              = 0x04,      

    FCT_READ_SOFT_HARD_VER      = 0x20,      
    FCT_READ_PSI                = 0x21,      
    FCT_READ_PHASE_RES_IND      = 0x22,      
    FCT_READ_PHASE_MA           = 0x23,      
    FCT_READ_VOL                = 0x24,      
    FCT_READ_MA_PID             = 0x25,      
    FCT_READ_SPEED_PID          = 0x26,      
    FCT_READ_POS_PID            = 0x27,      
    FCT_READ_TOTAL_PULSE        = 0x28,      
    FCT_READ_ROTATE_SPEED       = 0x29,      
    FCT_READ_POS                = 0x2A,      
    FCT_READ_POS_ERROR          = 0x2B,      
    FCT_READ_MOTOR_STA          = 0x2C,      
    FCT_READ_CLOG_FLAG          = 0x2D,      
    FCT_READ_CLOG_CUR           = 0x2E,      
    FCT_READ_ENABLE_STA         = 0x2F,      
    FCT_READ_ARRIVED_STA        = 0x30,      
    FCT_READ_SYS_PARAM          = 0x31,      
    FCT_READ_DRIVE_PARAMS       = 0x32,      

    FCT_SET_SLAVE_ADD           = 0x60,      
    FCT_SET_GROUP_ADD           = 0x61,      
    FCT_SET_MODE                = 0x62,      
    FCT_SET_POS_PID             = 0x63,      
    FCT_SET_POS_TORQUE          = 0x64,      
    FCT_SET_STEP                = 0x65,      
    FCT_SET_MA                  = 0x66,      
    FCT_SET_UART_BAUD           = 0x67,      
    FCT_SET_CAN_BAUD            = 0x68,      
    FCT_SET_MODBUS              = 0x69,      
    FCT_SET_CLOG_PRO            = 0x6A,      
    FCT_SET_CLOG_CUR            = 0x6B,      
    FCT_SET_CAN_ID              = 0x6C,      
    FCT_SET_DIR_LEVEL           = 0x6D,      
    FCT_SET_EN_LEVEL            = 0x6E,      
    FCT_SET_CMD_ECHO            = 0x6F,      
    FCT_SET_KEY_LOCK            = 0x70,      
    FCT_SET_AUTO_NOT_DISPLAY    = 0x71,      
    FCT_SET_IO_START_LEVEL      = 0x72,      
    FCT_SET_SPEED_PID           = 0x73,      

    FCT_ORIGIN_SET_LEFT_POS     = 0x90,      
    FCT_ORIGIN_LIMIT_HOME       = 0x91,      
    FCT_ORIGIN_TRIG             = 0x92,      
    FCT_ORIGIN_BREAK            = 0x93,      
    FCT_ORIGIN_READ_PARAMS      = 0x94,      
    FCT_ORIGIN_SET_PARAMS       = 0x95,      
    FCT_ORIGIN_READ_STA         = 0x96,      
    FCT_ORIGIN_AOTO_ZERO        = 0x97,      
    FCT_ORIGIN_SET_RIGHT_POS    = 0x98,      
    FCT_ORIGIN_SWITCH           = 0x99,      

    FCT_OL_SPEED_MODE           = 0xE0,      
    FCT_OL_POS_MODE             = 0xE1,      
    FCT_OL_POS_REL_MODE         = 0xE2,      
    FCT_OL_PULSES_MODE          = 0xE3,      
    
    FCT_IO_RUN_MODE             = 0xE4,      
    
    FCT_TORQUE_MODE             = 0xF0,      
    FCT_SPEED_MODE              = 0xF1,      
    FCT_POS_MODE                = 0xF2,      
    FCT_POS_REL_MODE            = 0xF3,      
    FCT_PULSES_MODE             = 0xF4,      
    FCT_PULSE_WIDTH_POS_MODE    = 0xF5,      
    FCT_PULSE_WIDTH_MA_MODE     = 0xF6,      
    FCT_PULSE_WIDTH_SPEED_MODE  = 0xF7,      
    FCT_ANGLE_ZERO              = 0xF8,      
    FCT_CLEAR_CLOG_PRO          = 0xF9,      
    FCT_MOTOR_ENABLE            = 0xFA,      
    FCT_CLEAR_STATE             = 0xFB,      
    FCT_STOP_NOW                = 0xFC,      
}FUN_CODE_TYPE;
extern CAN_HandleTypeDef hcan2;                /* CAN2通信句柄 */

/* ---- CAN硬件引脚配置 ---- */
#define CAN_TX_GPIO_PORT        GPIOB
#define CAN_TX_GPIO_PIN         GPIO_PIN_13
#define CAN_RX_GPIO_PORT        GPIOB
#define CAN_RX_GPIO_PIN         GPIO_PIN_12
#define CAN_TX_GPIO_CLK_ENABLE() __HAL_RCC_GPIOB_CLK_ENABLE()
#define CAN_RX_GPIO_CLK_ENABLE() __HAL_RCC_GPIOB_CLK_ENABLE()

#define CAN_RX0_INT_ENABLE      1              /* 使能CAN2 RX0接收中断 */
#define CAN_RECV_BUF_LEN        256            /* CAN接收/发送缓冲区最大长度 */

typedef struct
{
    uint8_t buf[CAN_RECV_BUF_LEN];             /* 接收帧数据缓冲 */
    uint16_t index;                            /* 缓冲写入位置索引 */
    uint8_t frame_done;                        /* 完整帧接收完成标志: 1=已就绪 */
    uint32_t can_id;                           /* 接收帧的CAN ID */
} can_frame_t;                                 /* CAN帧组装结构: 在中断中逐字节拼装 */
typedef struct {
    uint8_t data[256];                         /* 完整响应帧数据 */
    uint16_t len;                              /* 响应帧实际长度 */
    uint32_t id;                               /* 电机地址(ID) */
    uint32_t can_id;                           /* CAN帧ID */
    int32_t real_pos_pulse;                    /* 当前位置脉冲值(从READ_POS响应中解析) */
} smd_data_t;                                  /* 单电机数据包结构 */
extern SMD_Response receive_data_from[10];     /* 多电机接收响应缓冲区(外部定义于main.c) */

/* ---- 基础控制命令 ---- */
void smd_cal_encoder(uint8_t addr);
uint8_t smd_init(void);
void smd_restart(uint8_t addr);
void smd_reset_factory(uint8_t addr);
void smd_param_save(uint8_t addr);
uint8_t can_receive_msg(uint32_t id, uint8_t *buf);
/* ---- 读取类命令 ---- */
void smd_read_soft_hard_ver(uint8_t addr);
void smd_read_psi(uint8_t addr);
void smd_read_phase_res_ind(uint8_t addr);
void smd_read_phase_ma(uint8_t addr);
void smd_read_vol(uint8_t addr);
void smd_read_ma_pid(uint8_t addr);
void smd_read_speed_pid(uint8_t addr);
void smd_read_pos_pid(uint8_t addr);
void smd_read_tatal_pulse(uint8_t addr);
void smd_read_rotate_speed(uint8_t addr);
void smd_read_pos(uint8_t addr);
void smd_read_pos_error(uint8_t addr);
void smd_read_motor_sta(uint8_t addr);
void smd_read_clog_flag(uint8_t addr);
void smd_read_clog_current(uint8_t addr);
void smd_read_enable_sta(uint8_t addr);
void smd_read_arrived_sta(uint8_t addr);
void smd_read_sys_params(uint8_t addr);
void smd_read_drive_params(uint8_t addr);

/* ---- 设置类命令 ---- */
void smd_set_slave_add(uint8_t addr, uint8_t new_addr);
void smd_set_group_add(uint8_t addr, uint8_t new_addr);
void smd_set_mode(uint8_t addr, uint8_t mode);
void smd_set_pos_pid(uint8_t addr, uint32_t kp, uint32_t ki, uint32_t kd);
void smd_set_pos_torque(uint8_t addr, int16_t torque);
void smd_set_step(uint8_t addr, uint16_t step);
void smd_set_ma(uint8_t addr, int16_t ma);
void smd_set_uart_baud(uint8_t addr, uint32_t baud);
void smd_set_can_baud(uint8_t addr, uint16_t baud);
void smd_set_modbus(uint8_t addr, uint8_t modbus);
void smd_set_clog_pro(uint8_t addr, uint8_t en);
void smd_set_clog_current(uint8_t addr, int16_t ma);
void smd_set_can_id(uint8_t addr, uint32_t id);
void smd_set_dir_level(uint8_t addr,uint8_t dir);
void smd_set_en_level(uint8_t addr,uint8_t en);
void smd_set_cmd_echo(uint8_t addr,uint8_t echo);
void smd_set_key_lock(uint8_t addr, uint8_t lock);
void smd_set_auto_not_display(uint8_t addr, uint8_t en);
void smd_set_io_start_level(uint8_t addr, uint8_t level);
void smd_set_speed_pid(uint8_t addr, uint32_t kp, uint32_t ki, uint32_t kd);

void smd_origin_set_left_pos(uint8_t addr, int32_t pos);
void smd_origin_homing_by_limit(uint8_t addr, uint8_t limit_enable, uint8_t dir, int32_t speed_rpm, int16_t curr_limit);
void smd_origin_trig(uint8_t addr, uint8_t mode);
void smd_origin_break(uint8_t addr);
void smd_origin_read_params(uint8_t addr);
void smd_origin_set_params(uint8_t addr, uint32_t timout);
void smd_origin_read_sta(uint8_t addr);
void smd_origin_aoto_zero(uint8_t addr, uint8_t flag);
void smd_origin_set_right_pos(uint8_t addr, int32_t pos);
void smd_origin_l_r_switch(uint8_t addr, uint8_t ctrl);
    
void smd_torque_mode(uint8_t addr, uint8_t dir, uint16_t current);
void smd_speed_mode(uint8_t addr, uint8_t dir, uint8_t acc, float speed);
void smd_pos_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses);
void smd_pos_rel_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses);
void smd_pulse_mode(uint8_t addr);
void smd_pulse_width_pos_mode(uint8_t addr, uint16_t topw_max, uint16_t topw_min, int32_t top_pos, int32_t down_pos);
void smd_pulse_width_ma_mode(uint8_t addr, uint16_t topw_max, uint16_t topw_min, int32_t top_ma, int32_t down_ma);
void smd_pulse_width_speed_mode(uint8_t addr, uint16_t topw_max, uint16_t topw_min, int32_t top_speed, int32_t down_speed);
void smd_ol_speed_mode(uint8_t addr, uint8_t dir, uint8_t acc, float speed);
void smd_ol_pos_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses);
void smd_ol_pos_rel_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses);
void smd_ol_pulse_mode(uint8_t addr);
void smd_io_run_ctrl(uint8_t addr, uint8_t dir, uint8_t acc, float speed);

void smd_angle_to_zero(uint8_t addr);
void smd_remove_clog_protect(uint8_t addr);
void smd_motor_enable(uint8_t addr, uint8_t en);
void smd_clear_sta(uint8_t addr);
void smd_stop_now(uint8_t addr);

/* ---- 通用通信与同步函数 ---- */
void smd_send_cmd(uint8_t addr, FUN_CODE_TYPE fun_code, uint8_t *data, uint8_t length);
uint8_t smd_bus_wait_idle(uint32_t timeout_ms);
uint8_t smd_exec_cmd_sync(uint8_t addr, FUN_CODE_TYPE fun_code, const uint8_t *payload, uint8_t payload_len, SMD_Response *resp, uint32_t timeout_ms);
uint8_t smd_call_serialized(smd_addr_cmd_fn_t cmd_fn, uint8_t addr, uint8_t expect_func, SMD_Response *resp, uint32_t timeout_ms);
uint8_t smd_call_serialized_auto(smd_addr_cmd_fn_t cmd_fn, uint8_t addr, SMD_Response *resp, uint32_t timeout_ms);
uint8_t wait_smd_response(uint32_t id, SMD_Response *resp, uint32_t timeout_ms);
uint8_t wait_smd_response_by_func(uint32_t id, uint8_t func, SMD_Response *resp, uint32_t timeout_ms);
uint8_t smd_checksum(const uint8_t *data, uint8_t length);
/* ---- 机械臂专用控制 ---- */
void robotic_arm_control(int32_t* angle_data);
void robotic_arm_reset_target_cache(void);
void Read_robotic_arm_real_angle(void);
uint8_t smd_get_last_rx_motor_addr(void);
uint32_t smd_get_last_rx_can_id(void);
void robotic_stop();

/* --- Non-blocking smart-grip state machine --- */
void grip_start(uint8_t addr, uint8_t dir,
                int16_t start_torque, int16_t max_torque,
                int16_t torque_step, int16_t contact_threshold,
                uint32_t poll_interval_ms, uint32_t timeout_ms);
void grip_tick(void);
void grip_reset(void);
uint8_t grip_is_busy(void);
uint8_t grip_is_done(void);
uint8_t grip_timed_out(void);
void robotic_move_to(void);
#endif

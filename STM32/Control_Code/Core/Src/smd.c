/* SMD (Smart Motor Driver) Э��ͨ��ģ��
 * ========================================
 * ����CAN���ߵ�SMD���������ͨ��Э��:
 *   ֡��ʽ: FRAME_HEAD(0xC5) + addr + func_code + payload + checksum + FRAME_TAIL(0x5C)
 *   ͨ�ŷ�ʽ: CAN2��չ֡, 500kbps
 *   ֧��60+������: λ��/�ٶ�/����ģʽ����, PID����, ԭ��ع�, ������д��
 *
 * �ؼ����:
 *   - �ٽ��������������� (smd_try_lock/smd_unlock), ��ֹ������ͬʱ����CAN����
 *   - �������кŵ���Ӧ�ȴ�����, �����ж�����ѯ֮��ľ�̬
 *   - CAN�ж������֡����������У����洢, ��ѭ��ͨ�����кű仯��֪������
 *   - ֧������ͬ��(smd_exec_cmd_sync)�����л����ƶ�(smd_call_serialized)���ֵ���ģʽ
 */
/*
* SMD (Smart Motor Driver) Э��ͨ��ģ��
* һ��ʼ����ֻ�������˶�������ֻ�������˶���Ҫ�Ƕ�����Ϊ0��ô4�᲻������


*/
#include "smd.h"
#include "string.h"
#include "can.h"
#include "OLED.h"
#include "usart.h"
#include <stdio.h>
#include "stm32f4xx_hal.h"
#include <stm32f4xx_hal_can.h>

int16_t robotic_run_speed = 50;  // ��е�������ٶ� (��λ: mm/s)
int16_t robotic_run_acc = 10;  // ��е�����м��ٶ� (��λ: N)
CAN_HandleTypeDef   g_canx_handler;       /* CANͨ�ž��(ָ��hcan2) */
CAN_TxHeaderTypeDef g_canx_txheader;      /* CAN����֡ͷģ�� */
CAN_RxHeaderTypeDef g_canx_rxheader;      /* CAN����֡ͷ���� */
extern  can_frame_t g_can_frame ;         /* ȫ��CAN֡������, ��main.c�ж��� */
extern smd_data_t data[12];               /* 12·�����������, ��main.c�ж��� */

#define SMD_ADDR_COUNT 12U                /* �����ַ����: 1~11��Ч, 0/12+���� */
#define SMD_MIN_FRAME_LEN 5U              /* ��С֡����: head+addr+func+checksum+tail = 5�ֽ� */

/* ---- ��̬����: CAN֡��������װ ---- */
static uint8_t s_rx_assembly_buf[CAN_RECV_BUF_LEN];        /* ֡��װ������: ��CANԭʼ��������ƴ������֡ */
static uint16_t s_rx_assembly_len = 0U;                     /* ֡��װ��������ǰд��λ�� */

/* ---- ��̬����: ��Ӧ׷�� ---- */
static volatile uint32_t s_rx_seq[SMD_ADDR_COUNT];          /* ÿ·�������Ӧ���к�: ÿ���յ�����֡ʱ���� */
static volatile uint8_t s_last_rx_motor_addr = 0U;          /* ���һ���յ���Ӧ�ĵ����ַ */
static volatile uint32_t s_last_rx_can_id = 0U;             /* ���һ���յ���Ӧ��CAN ID */

/* ---- ��̬����: ������ ---- */
static volatile uint8_t s_smd_trans_lock = 0U;              /* CAN����������: 1=ռ����, 0=���� */

/* ---- ��̬����: ��е������� (����ȥ���Ż�) ---- */
static uint32_t s_arm_last_cmd_pulse[6] = {0U, 0U, 0U, 0U, 0U, 0U}; /* ÿ���ؽ��ϴη��͵������� */
static uint8_t s_arm_last_cmd_dir[6] = {0U, 0U, 0U, 0U, 0U, 0U};    /* ÿ���ؽ��ϴη��͵ķ��� */
static uint8_t s_arm_cmd_valid[6] = {0U, 0U, 0U, 0U, 0U, 0U};       /* ÿ���ؽ��ϴ������Ƿ���Ч */

void smd_send_data(uint8_t *data, uint8_t len);

/* smd_try_lock �� ���Ի�ȡCAN����������
 * �ٽ�������: �ȹ��ж�, ���/��λ����־, �ٻָ�ԭ�ж�״̬
 * ����1=��ȡ�ɹ�, 0=���ѱ�ռ��
 */
static uint8_t smd_try_lock(void)
{
    uint8_t locked = 0U;
    uint32_t primask = __get_PRIMASK();

    __disable_irq();
    if (s_smd_trans_lock == 0U)
    {
        s_smd_trans_lock = 1U;
        locked = 1U;
    }

    if (primask == 0U)
    {
        __enable_irq();
    }

    return locked;
}

/* smd_unlock �� �ͷ�CAN����������
 * �ٽ�������: ���жϺ��������־, �ָ�ԭ�ж�״̬
 */
static void smd_unlock(void)
{
    uint32_t primask = __get_PRIMASK();

    __disable_irq();
    s_smd_trans_lock = 0U;

    if (primask == 0U)
    {
        __enable_irq();
    }
}

/* smd_acquire_lock �� ����ʱ������ȡ, ��ѯֱ���ɹ���ʱ
 * ���������ȴ�CAN���߿���
 */
static uint8_t smd_acquire_lock(uint32_t timeout_ms)
{
    uint32_t tick_start = HAL_GetTick();

    while ((HAL_GetTick() - tick_start) < timeout_ms)
    {
        if (smd_try_lock() != 0U)
        {
            return 1U;
        }
    }

    return 0U;
}

/* smd_get_seq_snapshot �� ԭ�Ӷ�ȡĳ��ַ�������Ӧ���к�
 * �ٽ�������: ȷ��32λ���к����ж��в��������޸�
 */
static uint32_t smd_get_seq_snapshot(uint8_t addr)
{
    uint32_t seq;
    uint32_t primask = __get_PRIMASK();

    __disable_irq();
    seq = s_rx_seq[addr];
    if (primask == 0U)
    {
        __enable_irq();
    }

    return seq;
}

/* smd_wait_response_by_func_from_seq �� �������кŵȴ�ָ���������Ӧ
 * �Ƚϵ�ǰ���к����׼���к�, �������仯ʱ��ʾ�յ�����Ӧ֡
 * У��: ֡ͷ(FRAME_HEAD)��֡β(FRAME_TAIL)��������ƥ��
 * ����0=��ʱδ�յ�, >0=�յ���֡����
 */
static uint8_t smd_wait_response_by_func_from_seq(uint32_t id, uint8_t func, uint32_t seq_base, SMD_Response *resp, uint32_t timeout_ms)
{
    uint32_t tick_start;
    uint32_t last_seq = seq_base;

    if ((id == 0U) || (id >= SMD_ADDR_COUNT))
    {
        return 0U;
    }

    tick_start = HAL_GetTick();

    while ((HAL_GetTick() - tick_start) < timeout_ms)
    {
        if (s_rx_seq[id] != last_seq)
        {
            uint16_t rx_len;
            uint16_t copy_len;
            uint8_t frame_buf[CAN_RECV_BUF_LEN];
            uint8_t matched = 0U;
            uint32_t primask = __get_PRIMASK();

            __disable_irq();
            rx_len = data[id].len;
            copy_len = rx_len;
            if (copy_len > CAN_RECV_BUF_LEN)
            {
                copy_len = CAN_RECV_BUF_LEN;
            }
            memcpy(frame_buf, data[id].data, copy_len);
            rx_len = copy_len;
            last_seq = s_rx_seq[id];
            if (primask == 0U)
            {
                __enable_irq();
            }

            if ((rx_len >= SMD_MIN_FRAME_LEN) &&
                (frame_buf[0] == FRAME_HEAD) &&
                (frame_buf[rx_len - 1U] == FRAME_TAIL) &&
                ((func == 0U) || (frame_buf[2] == func)))
            {
                matched = 1U;
            }

            if (matched == 0U)
            {
                continue;
            }

            if (resp != NULL)
            {
                uint16_t out_len = rx_len;
                if (out_len > sizeof(resp->data))
                {
                    out_len = sizeof(resp->data);
                }
                memcpy(resp->data, frame_buf, out_len);
                resp->len = out_len;
            }

            if (rx_len > 255U)
            {
                rx_len = 255U;
            }
            return (uint8_t)rx_len;
        }
    }

    return 0U;
}

/* smd_infer_expect_func �� ���������ָ���ƶ϶�Ӧ�Ĺ�����
 * ���� smd_call_serialized �Զ��ƶ���������Ӧ������
 * ����1=�ƶϳɹ�, 0=δ֪�����
 */
static uint8_t smd_infer_expect_func(smd_addr_cmd_fn_t cmd_fn, uint8_t *func)
{
    if (func == NULL)
    {
        return 0U;
    }

    if (cmd_fn == smd_cal_encoder) { *func = FCT_CAL_ENCODER; return 1U; }
    if (cmd_fn == smd_restart) { *func = FCT_RESTART; return 1U; }
    if (cmd_fn == smd_reset_factory) { *func = FCT_RESET_FACTORY; return 1U; }
    if (cmd_fn == smd_param_save) { *func = FCT_PARAM_SAVE; return 1U; }

    if (cmd_fn == smd_read_soft_hard_ver) { *func = FCT_READ_SOFT_HARD_VER; return 1U; }
    if (cmd_fn == smd_read_psi) { *func = FCT_READ_PSI; return 1U; }
    if (cmd_fn == smd_read_phase_res_ind) { *func = FCT_READ_PHASE_RES_IND; return 1U; }
    if (cmd_fn == smd_read_phase_ma) { *func = FCT_READ_PHASE_MA; return 1U; }
    if (cmd_fn == smd_read_vol) { *func = FCT_READ_VOL; return 1U; }
    if (cmd_fn == smd_read_ma_pid) { *func = FCT_READ_MA_PID; return 1U; }
    if (cmd_fn == smd_read_speed_pid) { *func = FCT_READ_SPEED_PID; return 1U; }
    if (cmd_fn == smd_read_pos_pid) { *func = FCT_READ_POS_PID; return 1U; }
    if (cmd_fn == smd_read_tatal_pulse) { *func = FCT_READ_TOTAL_PULSE; return 1U; }
    if (cmd_fn == smd_read_rotate_speed) { *func = FCT_READ_ROTATE_SPEED; return 1U; }
    if (cmd_fn == smd_read_pos) { *func = FCT_READ_POS; return 1U; }
    if (cmd_fn == smd_read_pos_error) { *func = FCT_READ_POS_ERROR; return 1U; }
    if (cmd_fn == smd_read_motor_sta) { *func = FCT_READ_MOTOR_STA; return 1U; }
    if (cmd_fn == smd_read_clog_flag) { *func = FCT_READ_CLOG_FLAG; return 1U; }
    if (cmd_fn == smd_read_clog_current) { *func = FCT_READ_CLOG_CUR; return 1U; }
    if (cmd_fn == smd_read_enable_sta) { *func = FCT_READ_ENABLE_STA; return 1U; }
    if (cmd_fn == smd_read_arrived_sta) { *func = FCT_READ_ARRIVED_STA; return 1U; }
    if (cmd_fn == smd_read_sys_params) { *func = FCT_READ_SYS_PARAM; return 1U; }
    if (cmd_fn == smd_read_drive_params) { *func = FCT_READ_DRIVE_PARAMS; return 1U; }

    if (cmd_fn == smd_origin_break) { *func = FCT_ORIGIN_BREAK; return 1U; }
    if (cmd_fn == smd_origin_read_params) { *func = FCT_ORIGIN_READ_PARAMS; return 1U; }
    if (cmd_fn == smd_origin_read_sta) { *func = FCT_ORIGIN_READ_STA; return 1U; }

    if (cmd_fn == smd_pulse_mode) { *func = FCT_PULSES_MODE; return 1U; }
    if (cmd_fn == smd_ol_pulse_mode) { *func = FCT_OL_PULSES_MODE; return 1U; }
    if (cmd_fn == smd_angle_to_zero) { *func = FCT_ANGLE_ZERO; return 1U; }
    if (cmd_fn == smd_remove_clog_protect) { *func = FCT_CLEAR_CLOG_PRO; return 1U; }
    if (cmd_fn == smd_clear_sta) { *func = FCT_CLEAR_STATE; return 1U; }
    if (cmd_fn == smd_stop_now) { *func = FCT_STOP_NOW; return 1U; }

    return 0U;
}

/* smd_get_can_id_from_header �� ��CAN֡ͷ��ȡID (��չ֡����) */
static uint32_t smd_get_can_id_from_header(const CAN_RxHeaderTypeDef *rx_header)
{
    if (rx_header->IDE == CAN_ID_EXT)
    {
        return rx_header->ExtId;
    }
    return rx_header->StdId;
}

/* smd_store_reassembled_frame �� ��У��ͨ��������֡����ȫ�ֻ������͵����������
 * ͬʱ�������к�, ֪ͨ�ȴ��е���ѯ�����������ݵ���
 * ���⴦��: ����Ƕ�λ����Ӧ(FCT_READ_POS), ����4�ֽ�����ֵ�� real_pos_pulse
 */
static void smd_store_reassembled_frame(const uint8_t *frame, uint16_t frame_len, uint32_t can_id)
{
    uint16_t copy_len = frame_len;
    uint8_t addr;

    if ((frame == NULL) || (frame_len < SMD_MIN_FRAME_LEN))
    {
        return;
    }

    if (copy_len > CAN_RECV_BUF_LEN)
    {
        copy_len = CAN_RECV_BUF_LEN;
    }

    memcpy(g_can_frame.buf, frame, copy_len);
    g_can_frame.index = copy_len;
    g_can_frame.frame_done = 1U;
    g_can_frame.can_id = can_id;

    addr = frame[1];
    if ((addr == 0U) || (addr >= SMD_ADDR_COUNT))
    {
        return;
    }    if (copy_len > sizeof(data[addr].data))
    {
        copy_len = sizeof(data[addr].data);
    }

    memcpy(data[addr].data, frame, copy_len);
    data[addr].len = copy_len;
    data[addr].id = addr;
    data[addr].can_id = can_id;
    if ((copy_len >= 7U) && (frame[2] == FCT_READ_POS))
    {
        data[addr].real_pos_pulse = ((int32_t)frame[4] << 24) | 
                                    ((int32_t)frame[5] << 16) | 
                                    ((int32_t)frame[6] << 8) | 
                                    ((int32_t)frame[7]);
    }

    s_rx_seq[addr]++;
    s_last_rx_motor_addr = addr;
    s_last_rx_can_id = can_id;
}

/* smd_init �� ��ʼ��SMDͨ��ģ��: ��CAN2��� */
uint8_t smd_init(void)
{
    g_canx_handler = hcan2;
    return 0;
}

/* HAL_CAN_RxFifo0MsgPendingCallback �� CAN2 RX FIFO0 ��Ϣ����ص� (�ж�������)
 * ============================================================================
 * ��ISR��ɴ�CANԭʼ��������SMDЭ��֡�������������:
 *   1. ��ѯFIFOֱ��Ϊ��: �������д�������Ϣ
 *   2. �ֽڼ�״̬��: ����FRAME_HEAD(0xC5)��ʼ, ����ֽ���仺����
 *   3. ����FRAME_TAIL(0x5C)ʱ����У��:
 *      a. �����С֡���� >= 5�ֽ�
 *      b. У��checksum: ֡�����ۼӺ��Ƿ���У���ֽ�һ��
 *      c. У��ͨ�������� smd_store_reassembled_frame �洢��֪ͨ
 *   4. �쳣����: ��������������ò�������һ��֡ͷ
 *      checksum��ƥ�����Ĭ����֡
 */
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
    if (hcan->Instance == CAN2)
    {
        uint8_t rxbuf[8];
        CAN_RxHeaderTypeDef rx_header;

        while (HAL_CAN_GetRxFifoFillLevel(hcan, CAN_RX_FIFO0) > 0U)
        {
            if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rx_header, rxbuf) != HAL_OK)
            {
                break;
            }

            {
                uint8_t len = rx_header.DLC;
                uint32_t can_id = smd_get_can_id_from_header(&rx_header);

                if (len > 8U)
                {
                    len = 8U;
                }

                for (uint8_t i = 0U; i < len; i++)
                {
                    uint8_t byte = rxbuf[i];

                    if (s_rx_assembly_len == 0U)
                    {
                        if (byte == FRAME_HEAD)
                        {
                            s_rx_assembly_buf[0] = byte;
                            s_rx_assembly_len = 1U;
                        }
                        continue;
                    }

                    if (s_rx_assembly_len >= CAN_RECV_BUF_LEN)
                    {
                        s_rx_assembly_len = 0U;
                        if (byte == FRAME_HEAD)
                        {
                            s_rx_assembly_buf[0] = byte;
                            s_rx_assembly_len = 1U;
                        }
                        continue;
                    }

                    s_rx_assembly_buf[s_rx_assembly_len++] = byte;

                    if (byte == FRAME_TAIL)
                    {
                        if (s_rx_assembly_len >= SMD_MIN_FRAME_LEN)
                        {
                            uint16_t checksum_pos = s_rx_assembly_len - 2U;
                            uint8_t checksum = smd_checksum(s_rx_assembly_buf, (uint8_t)checksum_pos);
                            if (checksum == s_rx_assembly_buf[checksum_pos])
                            {
                                smd_store_reassembled_frame(s_rx_assembly_buf, s_rx_assembly_len, can_id);
                                s_rx_assembly_len = 0U;
                            }
                        }
                    }
                }
            }
        }
    }
}
/* can_send_msg �� ͨ��CAN2���͵�֡��Ϣ (���8�ֽ�)
 * ���ͺ������ȴ������䴫�����, ��ʱ50ms
 * ����0=�ɹ�, 1=ʧ��(����ʧ�ܻ�ʱ)
 */
uint8_t can_send_msg(uint32_t id, uint8_t *msg, uint8_t len)
{
    uint32_t TxMailbox = CAN_TX_MAILBOX0;
    CAN_TxHeaderTypeDef tx_header;
    tx_header.StdId = 0x00;
    tx_header.ExtId = id;
    tx_header.IDE = CAN_ID_EXT;
    tx_header.RTR = CAN_RTR_DATA;
    tx_header.DLC = len;
    tx_header.TransmitGlobalTime = DISABLE;

    if (HAL_CAN_AddTxMessage(&hcan2, &tx_header, msg, &TxMailbox) != HAL_OK)
    {
        return 1;
    }

    /* Wait this mailbox transmission complete instead of waiting all mailboxes idle. */
    {
        uint32_t tick_start = HAL_GetTick();
        while (HAL_CAN_IsTxMessagePending(&hcan2, TxMailbox) != 0U)
        {
            if ((HAL_GetTick() - tick_start) > 50U)
            {
                return 1;
            }
        }
    }

    return 0;
}

/* can_send_long_msg �� �ֶη��ͳ�֡: ÿ�����8�ֽ�, ��εȴ�������� */
uint8_t can_send_long_msg(uint32_t id, uint8_t *data, uint16_t len)
{
    uint16_t offset = 0;
    uint8_t ret = 0;
    while (offset < len)
    {
        uint8_t send_len = (len - offset >= 8) ? 8 : (len - offset);
        ret = can_send_msg(id, data + offset, send_len);
        if (ret != 0)
        {
            return 1;
        }
        offset += send_len;
    }
    return 0;
}

/* smd_bus_wait_idle �� �ȴ�CAN���߿��� (3����������ȫ������)
 * COMM_TYPE==1(CANģʽ)ʱ��Ч, ����ģʽֱ�ӷ��سɹ�
 */
uint8_t smd_bus_wait_idle(uint32_t timeout_ms)
{
#if COMM_TYPE == 1
    uint32_t tick_start = HAL_GetTick();

    while ((HAL_GetTick() - tick_start) < timeout_ms)
    {
        if (HAL_CAN_GetTxMailboxesFreeLevel(&hcan2) == 3U)
        {
            return 1U;
        }
    }
    return 0U;
#else
    (void)timeout_ms;
    return 1U;
#endif
}

/* smd_send_cmd �� ��װSMDЭ��֡������ (����, �޵ȴ���Ӧ)
 * ֡��ʽ: [HEAD(1B)] [addr(1B)] [func_code(1B)] [payload(NB)] [checksum(1B)] [TAIL(1B)]
 * У��: �ۼӺ� (���ֽ�0��У��λǰһ���ֽ�)
 */
void smd_send_cmd(uint8_t addr, FUN_CODE_TYPE fun_code, uint8_t *data, uint8_t length)
{
    uint8_t cmd[CAN_RECV_BUF_LEN];
    uint16_t frame_len;

    if (length > (CAN_RECV_BUF_LEN - 5U))
    {
        return;
    }

    cmd[0] = FRAME_HEAD;
    cmd[1] = addr;
    cmd[2] = (uint8_t)fun_code;

    if ((data != NULL) && (length > 0U))
    {
        memcpy(&cmd[3], data, length);
    }

    cmd[3U + length] = smd_checksum(cmd, (uint8_t)(3U + length));
    cmd[4U + length] = FRAME_TAIL;

    frame_len = (uint16_t)length + 5U;
    smd_send_data(cmd, (uint8_t)frame_len);
}

/* smd_exec_cmd_sync �� ͬ��ִ��SMD����: �������ȴ����߿��С����͡��ȴ���Ӧ������
 * ����������������ִ������, �ṩ�˵��˵Ŀɿ�ͨ�ű���
 * ���� SMD_TRANS_OK(0)=�ɹ�, SMD_TRANS_FAIL(1)=ʧ��
 */
uint8_t smd_exec_cmd_sync(uint8_t addr, FUN_CODE_TYPE fun_code, const uint8_t *payload, uint8_t payload_len, SMD_Response *resp, uint32_t timeout_ms)
{
    uint8_t cmd[CAN_RECV_BUF_LEN];
    uint16_t frame_len;
    uint32_t lock_timeout = timeout_ms;
    uint32_t start_seq;
    SMD_Response tmp_resp = {0};

    if ((addr == 0U) || (addr >= SMD_ADDR_COUNT))
    {
        return SMD_TRANS_FAIL;
    }

    if (payload_len > (CAN_RECV_BUF_LEN - 5U))
    {
        return SMD_TRANS_FAIL;
    }

    if (lock_timeout == 0U)
    {
        lock_timeout = 100U;
    }

    if (smd_acquire_lock(lock_timeout) == 0U)
    {
        return SMD_TRANS_FAIL;
    }

    if (smd_bus_wait_idle(50U) == 0U)
    {
        smd_unlock();
        return SMD_TRANS_FAIL;
    }

    start_seq = smd_get_seq_snapshot(addr);

    cmd[0] = FRAME_HEAD;
    cmd[1] = addr;
    cmd[2] = (uint8_t)fun_code;
    if ((payload != NULL) && (payload_len > 0U))
    {
        memcpy(&cmd[3], payload, payload_len);
    }
    cmd[3U + payload_len] = smd_checksum(cmd, (uint8_t)(3U + payload_len));
    cmd[4U + payload_len] = FRAME_TAIL;
    frame_len = (uint16_t)payload_len + 5U;

#if COMM_TYPE == 1
    if (can_send_long_msg(CAN_EXTID, cmd, frame_len) != 0U)
    {
        smd_unlock();
        return SMD_TRANS_FAIL;
    }
#else
    smd_send_data(cmd, (uint8_t)frame_len);
#endif

    if (timeout_ms > 0U)
    {
        uint8_t rx_len;
        SMD_Response *out = (resp != NULL) ? resp : &tmp_resp;

        rx_len = smd_wait_response_by_func_from_seq(addr, (uint8_t)fun_code, start_seq, out, timeout_ms);
        if (rx_len == 0U)
        {
            smd_unlock();
            return SMD_TRANS_FAIL;
        }
    }

    smd_unlock();
    return SMD_TRANS_OK;
}

/* smd_call_serialized �� �����л���ʽ������������ȴ���Ӧ
 * �� smd_exec_cmd_sync ������: ͨ������ָ�����, �Զ��ƶ�������Ӧ������
 * expect_func=0ʱ�Զ�ͨ�� smd_infer_expect_func �ƶ�
 */
uint8_t smd_call_serialized(smd_addr_cmd_fn_t cmd_fn, uint8_t addr, uint8_t expect_func, SMD_Response *resp, uint32_t timeout_ms)
{
    uint8_t rx_len = 0U;
    uint32_t lock_timeout = timeout_ms;
    uint32_t start_seq;
    uint8_t wait_func = expect_func;

    if ((cmd_fn == NULL) || (addr == 0U) || (addr >= SMD_ADDR_COUNT))
    {
        return SMD_TRANS_FAIL;
    }

    if (lock_timeout == 0U)
    {
        lock_timeout = 100U;
    }

    if (smd_acquire_lock(lock_timeout) == 0U)
    {
        return SMD_TRANS_FAIL;
    }

    if (smd_bus_wait_idle(50U) == 0U)
    {
        smd_unlock();
        return SMD_TRANS_FAIL;
    }

    start_seq = smd_get_seq_snapshot(addr);

    cmd_fn(addr);

    if (timeout_ms > 0U)
    {
        if (wait_func == 0U)
        {
            if (smd_infer_expect_func(cmd_fn, &wait_func) == 0U)
            {
                smd_unlock();
                return SMD_TRANS_FAIL;
            }
        }

        rx_len = smd_wait_response_by_func_from_seq(addr, wait_func, start_seq, resp, timeout_ms);
        if (rx_len == 0U)
        {
            smd_unlock();
            return SMD_TRANS_FAIL;
        }
    }

    smd_unlock();
    return SMD_TRANS_OK;
}

/* smd_call_serialized_auto �� smd_call_serialized �ļ򻯰�: �Զ��ƶϹ����� */
uint8_t smd_call_serialized_auto(smd_addr_cmd_fn_t cmd_fn, uint8_t addr, SMD_Response *resp, uint32_t timeout_ms)
{
    return smd_call_serialized(cmd_fn, addr, 0U, resp, timeout_ms);
}

/* can_receive_msg �� ��CAN2 FIFO0��ȡһ����Ϣ (���ڵ���, ���ӡ����OLED) */
uint8_t can_receive_msg(uint32_t id, uint8_t *buf)
{
    CAN_RxHeaderTypeDef rx_header;
    if (HAL_CAN_GetRxFifoFillLevel(&hcan2, CAN_RX_FIFO0) == 0)
    {
        OLED_ShowString(0, 6, "No CAN Msg");
        return 0;
    }
    if (HAL_CAN_GetRxMessage(&hcan2, CAN_RX_FIFO0, &rx_header, buf) != HAL_OK)
    {
        OLED_ShowString(0, 6, "Get CAN Msg Err");
        return 0;
    }
    if (rx_header.ExtId != id)
    {
        OLED_ShowString(0, 6, "ID Err");
        return 0;
    }
    return rx_header.DLC;
}

/* float/uint8_t ������: ���ڽ�float�ٶ�ֵ���Ϊ4�ֽ�CAN���� */
union
{
    float f;
    uint8_t b[4];
} data_u;

/* smd_send_data �� �ײ����ݷ���: COMM_TYPE=0������, COMM_TYPE=1��CAN */
void smd_send_data(uint8_t *data, uint8_t len)
{
#if COMM_TYPE == 0
    usart2_send_cmd(data, len);
#elif COMM_TYPE == 1
    can_send_long_msg(CAN_EXTID, data, len);
#endif
}

/* smd_checksum �� ����SMDЭ��У���: ���ۼӺ� (ȡ��8λ) */
uint8_t smd_checksum(const uint8_t *data, uint8_t length)
{
    uint8_t sum = 0;
    for (uint8_t i = 0; i < length; i++)
    {
        sum += data[i];
    }
    return sum;
}

/* get_smd_response �� �ȴ�ָ��ID�������Ӧ (�޹��������) */
uint8_t get_smd_response(uint32_t id, SMD_Response *resp, uint32_t timeout_ms)
{
    return wait_smd_response(id, resp, timeout_ms);
}

/* ============================================================================
 * ������������ (��payload, 5�ֽ�֡)
 * ֡��ʽ: HEAD + addr + func_code + checksum + TAIL
 * ============================================================================ */

/* smd_cal_encoder �� ������У׼���� */
void smd_cal_encoder(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_CAL_ENCODER;            
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_restart �� ������������� */
void smd_restart(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_RESTART;                
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_reset_factory �� �ָ��������� */
void smd_reset_factory(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_RESET_FACTORY;          
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_param_save �� ���������Flash */
void smd_param_save(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_PARAM_SAVE;             
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* ============================================================================
 * ��ȡ������ �� ��ѯ���״̬��������PID�� (��payload, 5�ֽ�֡)
 * ============================================================================ */

/* smd_read_soft_hard_ver �� ��ȡ��Ӳ���汾�� */
void smd_read_soft_hard_ver(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_SOFT_HARD_VER;     
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_psi �� ��ȡ�����/��в��� */
void smd_read_psi(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_PSI;               
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_phase_res_ind �� ��ȡ�����/��� */
void smd_read_phase_res_ind(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_PHASE_RES_IND;     
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_phase_ma �� ��ȡ��ǰ����� (mA), ���ڼ�צ��������� */
void smd_read_phase_ma(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_PHASE_MA;          
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_vol �� ��ȡ���ߵ�ѹ */
void smd_read_vol(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_VOL;               
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_ma_pid �� ��ȡ������PID���� */
void smd_read_ma_pid(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_MA_PID;            
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_speed_pid �� ��ȡ�ٶȻ�PID���� */
void smd_read_speed_pid(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_SPEED_PID;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_pos_pid �� ��ȡλ�û�PID���� */
void smd_read_pos_pid(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_POS_PID;           
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_tatal_pulse �� ��ȡ�ۼ������� */
void smd_read_tatal_pulse(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_TOTAL_PULSE;       
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_rotate_speed �� ��ȡ��ǰת�� */
void smd_read_rotate_speed(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_ROTATE_SPEED;      
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_pos �� ��ȡ��ǰ����λ�� (������), ������ĵĶ�ȡ���� */
void     smd_read_pos(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_POS;               
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_pos_error �� ��ȡλ����� */
void smd_read_pos_error(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_POS_ERROR;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_motor_sta �� ��ȡ���״̬�� */
void smd_read_motor_sta(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_MOTOR_STA;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_clog_flag �� ��ȡ��ת��־ */
void smd_read_clog_flag(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_CLOG_FLAG;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_clog_current �� ��ȡ��ת���� */
void smd_read_clog_current(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_CLOG_CUR;          
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_enable_sta �� ��ȡʹ��״̬ */
void smd_read_enable_sta(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_ENABLE_STA;        
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_arrived_sta �� ��ȡ��λ״̬ */
void smd_read_arrived_sta(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_ARRIVED_STA;       
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_sys_params �� ��ȡϵͳ���� */
void smd_read_sys_params(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_SYS_PARAM;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* smd_read_drive_params �� ��ȡ���������� */
void smd_read_drive_params(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_READ_DRIVE_PARAMS;      
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                 

    smd_send_data(cmd, 5);
}

/* ============================================================================
 * ���������� �� ���õ������ (��payload, 6~17�ֽ�֡)
 * ============================================================================ */

/* smd_set_slave_add �� ���ôӻ���ַ */
void smd_set_slave_add(uint8_t addr, uint8_t new_addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_SLAVE_ADD;          
    cmd[3] =  new_addr;                   
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_group_add �� �������ַ */
void smd_set_group_add(uint8_t addr, uint8_t new_addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_GROUP_ADD;          
    cmd[3] =  new_addr;                   
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_mode �� ���ÿ���ģʽ (λ��/�ٶ�/���ص�) */
void smd_set_mode(uint8_t addr, uint8_t mode)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_MODE;               
    cmd[3] =  mode;                       
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_pos_pid �� ����λ�û�PID���� (Kp, Ki, Kd��4�ֽ�) */
void smd_set_pos_pid(uint8_t addr, uint32_t kp, uint32_t ki, uint32_t kd)
{
    uint8_t cmd[32] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_POS_PID;               
    cmd[3] = (uint8_t)((kp >> 24) & 0xFF);  
    cmd[4] = (uint8_t)((kp >> 16) & 0xFF);
    cmd[5] = (uint8_t)((kp >> 8) & 0xFF);
    cmd[6] = (uint8_t)((kp >> 0) & 0xFF);
    
    cmd[7] = (uint8_t)((ki >> 24) & 0xFF);  
    cmd[8] = (uint8_t)((ki >> 16) & 0xFF);
    cmd[9] = (uint8_t)((ki >> 8) & 0xFF);
    cmd[10] = (uint8_t)((ki >> 0) & 0xFF);

    cmd[11] = (uint8_t)((kd >> 24) & 0xFF);  
    cmd[12] = (uint8_t)((kd >> 16) & 0xFF);
    cmd[13] = (uint8_t)((kd >> 8) & 0xFF);
    cmd[14] = (uint8_t)((kd >> 0) & 0xFF);

    cmd[15] =  smd_checksum(cmd, 15);       
    cmd[16] =  FRAME_TAIL;                 

    smd_send_data(cmd, 17);
}

/* smd_set_pos_torque �� ����λ��ģʽ�µ��������� */
void smd_set_pos_torque(uint8_t addr, int16_t torque)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_POS_TORQUE;         
    cmd[3] = (uint8_t)((torque >> 8) & 0xFF);  
    cmd[4] = (uint8_t)((torque >> 0) & 0xFF);
    cmd[5] =  smd_checksum(cmd, 5);       
    cmd[6] =  FRAME_TAIL;                 

    smd_send_data(cmd, 7);
}

/* smd_set_step �� ���ò��������� */
void smd_set_step(uint8_t addr, uint16_t step)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                  
    cmd[1] =  addr;                        
    cmd[2] =  FCT_SET_STEP;                
    cmd[3] =  (uint8_t)((step >> 8) & 0xFF);
    cmd[4] =  (uint8_t)((step >> 0) & 0xFF);
    cmd[5] =  smd_checksum(cmd, 5);        
    cmd[6] =  FRAME_TAIL;                  

    smd_send_data(cmd, 7);
}

/* smd_set_ma �� �������������� (mA) */
void smd_set_ma(uint8_t addr, int16_t ma)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_MA;                 
    cmd[3] = (uint8_t)((ma >> 8) & 0xFF);  
    cmd[4] = (uint8_t)((ma >> 0) & 0xFF);
    
    cmd[5] =  smd_checksum(cmd, 5);       
    cmd[6] =  FRAME_TAIL;                 

    smd_send_data(cmd, 7);
}

/* smd_set_uart_baud �� ����UART������ */
void smd_set_uart_baud(uint8_t addr, uint32_t baud)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_UART_BAUD;          
    cmd[3] = (uint8_t)((baud >> 24) & 0xFF);  
    cmd[4] = (uint8_t)((baud >> 16) & 0xFF);
    cmd[5] = (uint8_t)((baud >> 8) & 0xFF);
    cmd[6] = (uint8_t)((baud >> 0) & 0xFF); 
    cmd[7] =  smd_checksum(cmd, 7);       
    cmd[8] =  FRAME_TAIL;                 

    smd_send_data(cmd, 9);
}

/* smd_set_can_baud �� ����CAN������ */
void smd_set_can_baud(uint8_t addr, uint16_t baud)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_CAN_BAUD;           
    cmd[3] = (uint8_t)((baud >> 8) & 0xFF);  
    cmd[4] = (uint8_t)((baud >> 0) & 0xFF);
    cmd[5] =  smd_checksum(cmd, 5);       
    cmd[6] =  FRAME_TAIL;                 

    smd_send_data(cmd, 7);
}

/* smd_set_modbus �� ����Modbus��ַ */
void smd_set_modbus(uint8_t addr, uint8_t modbus)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_MODBUS;             
    cmd[3] =  modbus;                     
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_clog_pro �� ʹ��/���ö�ת���� */
void smd_set_clog_pro(uint8_t addr, uint8_t en)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_CLOG_PRO;           
    cmd[3] =  en;                         
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_clog_current �� ���ö�ת������ֵ */
void smd_set_clog_current(uint8_t addr, int16_t ma)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_CLOG_CUR;           
    cmd[3] =  (uint8_t)((ma >> 8) & 0xFF);
    cmd[4] =  (uint8_t)((ma >> 0) & 0xFF);
    cmd[5] =  smd_checksum(cmd, 5);       
    cmd[6] =  FRAME_TAIL;                 

    smd_send_data(cmd, 7);
}

/* smd_set_can_id �� ����CANͨ��ID */
void smd_set_can_id(uint8_t addr, uint32_t id)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_CAN_ID;             
    cmd[3] = (uint8_t)((id >> 24) & 0xFF);        
    cmd[4] = (uint8_t)((id >> 16) & 0xFF); 
    cmd[5] = (uint8_t)((id >> 8) & 0xFF);  
    cmd[6] = (uint8_t)((id >> 0) & 0xFF); 
    cmd[7] =  smd_checksum(cmd, 7);       
    cmd[8] =  FRAME_TAIL;                 

    smd_send_data(cmd, 9);
}

/* smd_set_dir_level �� ���÷������ŵ�ƽ���� */
void smd_set_dir_level(uint8_t addr,uint8_t dir)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_DIR_LEVEL;          
    cmd[3] =  dir;                        
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_en_level �� ����ʹ�����ŵ�ƽ���� */
void smd_set_en_level(uint8_t addr,uint8_t en)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_EN_LEVEL;           
    cmd[3] =  en;                         
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_cmd_echo �� ����������Կ��� */
void smd_set_cmd_echo(uint8_t addr,uint8_t echo)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_CMD_ECHO;           
    cmd[3] =  echo;                       
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_key_lock �� ���ð������� */
void smd_set_key_lock(uint8_t addr, uint8_t lock)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_KEY_LOCK;           
    cmd[3] =  lock;                       
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_auto_not_display �� �����Զ���Ϣ��ʾ���� */
void smd_set_auto_not_display(uint8_t addr, uint8_t en)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_AUTO_NOT_DISPLAY;   
    cmd[3] =  en;                         
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_io_start_level �� ����IO������ƽ */
void smd_set_io_start_level(uint8_t addr, uint8_t level)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_IO_START_LEVEL;     
    cmd[3] =  level;                      
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                 

    smd_send_data(cmd, 6);
}

/* smd_set_speed_pid �� �����ٶȻ�PID���� (Kp, Ki, Kd��4�ֽ�) */
void smd_set_speed_pid(uint8_t addr, uint32_t kp, uint32_t ki, uint32_t kd)
{
    uint8_t cmd[32] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SET_SPEED_PID;          
    cmd[3] = (uint8_t)((kp >> 24) & 0xFF);  
    cmd[4] = (uint8_t)((kp >> 16) & 0xFF);
    cmd[5] = (uint8_t)((kp >> 8) & 0xFF);
    cmd[6] = (uint8_t)((kp >> 0) & 0xFF);
    
    cmd[7] = (uint8_t)((ki >> 24) & 0xFF);  
    cmd[8] = (uint8_t)((ki >> 16) & 0xFF);
    cmd[9] = (uint8_t)((ki >> 8) & 0xFF);
    cmd[10] = (uint8_t)((ki >> 0) & 0xFF);

    cmd[11] = (uint8_t)((kd >> 24) & 0xFF);  
    cmd[12] = (uint8_t)((kd >> 16) & 0xFF);
    cmd[13] = (uint8_t)((kd >> 8) & 0xFF);
    cmd[14] = (uint8_t)((kd >> 0) & 0xFF);

    cmd[15] =  smd_checksum(cmd, 15);      
    cmd[16] =  FRAME_TAIL;                

    smd_send_data(cmd, 17);
}

/* ============================================================================
 * ԭ��ع����� �� ���ú�ִ�е���������
 * ============================================================================ */

/* smd_origin_set_left_pos �� ��������λλ�� */
void smd_origin_set_left_pos(uint8_t addr, int32_t pos)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                  
    cmd[1] =  addr;                        
    cmd[2] =  FCT_ORIGIN_SET_LEFT_POS;     
    cmd[3] = (uint8_t)((pos >> 24) & 0xFF); 
    cmd[4] = (uint8_t)((pos >> 16) & 0xFF);
    cmd[5] = (uint8_t)((pos >> 8) & 0xFF);
    cmd[6] = (uint8_t)((pos >> 0) & 0xFF); 
    cmd[7] =  smd_checksum(cmd, 7);        
    cmd[8] =  FRAME_TAIL;                  

    smd_send_data(cmd, 9);
}

/* smd_origin_homing_by_limit �� ͨ����λ����ִ��ԭ��ع� */
void smd_origin_homing_by_limit(uint8_t addr, uint8_t limit_enable, uint8_t dir, int32_t speed_rpm, int16_t curr_limit)
{
    uint8_t cmd[32] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_LIMIT_HOME;      
    cmd[3] = limit_enable;                
    cmd[4] = dir;                         
    cmd[5] = (uint8_t)((speed_rpm >> 24) & 0xFF);    
    cmd[6] = (uint8_t)((speed_rpm >> 16) & 0xFF);
    cmd[7] = (uint8_t)((speed_rpm >> 8) & 0xFF);    
    cmd[8] = (uint8_t)((speed_rpm >> 0) & 0xFF);
    
    cmd[9] = (uint8_t)((curr_limit >> 8) & 0xFF);    
    cmd[10] = (uint8_t)((curr_limit >> 0) & 0xFF);
    
    cmd[11] =  smd_checksum(cmd, 11);     
    cmd[12] =  FRAME_TAIL;                

    smd_send_data(cmd, 13);
}

/* smd_origin_trig �� ����ԭ��ع���� */
void smd_origin_trig(uint8_t addr, uint8_t mode)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_TRIG;            
    cmd[3] =  mode;                       
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                

    smd_send_data(cmd, 6);
}

/* smd_origin_break �� ֹͣԭ��ع� */
void smd_origin_break(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_BREAK;           
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_origin_read_params �� ��ȡԭ��ع���� */
void smd_origin_read_params(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_READ_PARAMS;     
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_origin_set_params �� ����ԭ��ع鳬ʱʱ�� */
void smd_origin_set_params(uint8_t addr, uint32_t timout)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_SET_PARAMS;      
    cmd[3] = (uint8_t)((timout >> 24) & 0xFF);  
    cmd[4] = (uint8_t)((timout >> 16) & 0xFF);
    cmd[5] = (uint8_t)((timout >> 8) & 0xFF);
    cmd[6] = (uint8_t)((timout >> 0) & 0xFF); 
    cmd[7] =  smd_checksum(cmd, 7);       
    cmd[8] =  FRAME_TAIL;                

    smd_send_data(cmd, 9);
}

/* smd_origin_read_sta �� ��ȡԭ��ع�״̬ */
void smd_origin_read_sta(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_READ_STA;        
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_origin_aoto_zero �� �����ϵ��Զ����㿪�� */
void smd_origin_aoto_zero(uint8_t addr, uint8_t flag)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_AOTO_ZERO;       
    cmd[3] =  flag;                       
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                

    smd_send_data(cmd, 6);
}

/* smd_origin_set_right_pos �� ��������λλ�� */
void smd_origin_set_right_pos(uint8_t addr, int32_t pos)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                  
    cmd[1] =  addr;                        
    cmd[2] =  FCT_ORIGIN_SET_RIGHT_POS;    
    cmd[3] = (uint8_t)((pos >> 24) & 0xFF); 
    cmd[4] = (uint8_t)((pos >> 16) & 0xFF);
    cmd[5] = (uint8_t)((pos >> 8) & 0xFF);
    cmd[6] = (uint8_t)((pos >> 0) & 0xFF); 
    cmd[7] =  smd_checksum(cmd, 7);        
    cmd[8] =  FRAME_TAIL;                  

    smd_send_data(cmd, 9);
}

/* smd_origin_l_r_switch �� �л�������λģʽ */
void smd_origin_l_r_switch(uint8_t addr, uint8_t ctrl)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ORIGIN_SWITCH;          
    cmd[3] =  ctrl;                       
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                

    smd_send_data(cmd, 6);
}

/* ============================================================================
 * �ջ��˶��������� �� ��λ��/�ٶ�/����PID�ջ�
 * ============================================================================ */

/* smd_torque_mode �� ����ģʽ: ���趨�������ж��� (���ڼ�צ������) */
void smd_torque_mode(uint8_t addr, uint8_t dir, uint16_t current)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_TORQUE_MODE;            
    cmd[3] =  dir;                        
    cmd[4] = (uint8_t)((current >> 8) & 0xFF);  
    cmd[5] = (uint8_t)((current >> 0) & 0xFF); 
    cmd[6] =  smd_checksum(cmd, 6);       
    cmd[7] =  FRAME_TAIL;                

    smd_send_data(cmd, 8);
}

/* smd_speed_mode �� �ٶ�ģʽ: ���趨�ٶȳ�����ת */
void smd_speed_mode(uint8_t addr, uint8_t dir, uint8_t acc, float speed)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_SPEED_MODE;             
    cmd[3] =  dir;                        
    cmd[4] =  acc;                        
    data_u.f = speed;                     
    cmd[5] =  data_u.b[3];                
    cmd[6] =  data_u.b[2];   
    cmd[7] =  data_u.b[1];    
    cmd[8] =  data_u.b[0];     
    cmd[9] =  smd_checksum(cmd, 9);       
    cmd[10] =  FRAME_TAIL;                

    smd_send_data(cmd, 11);
}

/* smd_pos_mode �� λ��ģʽ(����): �ƶ�����������λ�� (����ĵ��˶���������) */
void smd_pos_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses)
{
    uint8_t payload[9];

    payload[0] = dir;
    payload[1] = acc;
    payload[2] = (uint8_t)((speed >> 8) & 0xFFU);
    payload[3] = (uint8_t)(speed & 0xFFU);
    payload[4] = (uint8_t)((pulses >> 24) & 0xFFU);
    payload[5] = (uint8_t)((pulses >> 16) & 0xFFU);
    payload[6] = (uint8_t)((pulses >> 8) & 0xFFU);
    payload[7] = (uint8_t)(pulses & 0xFFU);
    payload[8] = 0U; /* sync flag: 0=disable multi-device sync */

    /* timeout=0: only send, do not wait response */
    (void)smd_exec_cmd_sync(addr, FCT_POS_MODE, payload, sizeof(payload), NULL, 0U);
}

/* smd_pos_rel_mode �� λ��ģʽ(���): �ƶ���������� */
void smd_pos_rel_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                  
    cmd[1] =  addr;                        
    cmd[2] =  FCT_POS_REL_MODE;                
    cmd[3] =  dir;                         
    cmd[4] =  acc;                         
    cmd[5] =  (uint8_t)((speed >> 8) & 0xFF);   
    cmd[6] =  (uint8_t)((speed >> 0) & 0xFF);   
    cmd[7] =  (uint8_t)((pulses >> 24) & 0xFF); 
    cmd[8] =  (uint8_t)((pulses >> 16) & 0xFF); 
    cmd[9] =  (uint8_t)((pulses >> 8) & 0xFF);  
    cmd[10] = (uint8_t)((pulses >> 0) & 0xFF);  
    cmd[11] = 0U; /* sync flag: 0=disable multi-device sync */
    cmd[12] = smd_checksum(cmd, 12);       
    cmd[13] = FRAME_TAIL;                  

    smd_send_data(cmd, 14);
}

/* smd_pulse_mode �� �л�������λ��ģʽ (�ⲿ�������) */
void smd_pulse_mode(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_PULSES_MODE;            
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_pulse_width_pos_mode �� ����λ��ģʽ: ͨ��PWM��������λ�� */
void smd_pulse_width_pos_mode(uint8_t addr, uint16_t topw_max, uint16_t topw_min, int32_t top_pos, int32_t down_pos)
{
    uint8_t cmd[32] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_PULSE_WIDTH_POS_MODE;   
    cmd[3] = (uint8_t)((topw_max >> 8) & 0xFF); 
    cmd[4] = (uint8_t)((topw_max >> 0) & 0xFF); 
    cmd[5] = (uint8_t)((topw_min >> 8) & 0xFF); 
    cmd[6] = (uint8_t)((topw_min >> 0) & 0xFF); 
    cmd[7] =  (uint8_t)((top_pos >> 24) & 0xFF);
    cmd[8] =  (uint8_t)((top_pos >> 16) & 0xFF);
    cmd[9] =  (uint8_t)((top_pos >> 8) & 0xFF); 
    cmd[10] = (uint8_t)((top_pos >> 0) & 0xFF); 
    cmd[11] =  (uint8_t)((down_pos >> 24) & 0xFF);  
    cmd[12] =  (uint8_t)((down_pos >> 16) & 0xFF);  
    cmd[13] =  (uint8_t)((down_pos >> 8) & 0xFF);   
    cmd[14] = (uint8_t)((down_pos >> 0) & 0xFF);    
    cmd[15] =  smd_checksum(cmd, 15);       
    cmd[16] =  FRAME_TAIL;                

    smd_send_data(cmd, 17);
}

/* smd_pulse_width_ma_mode �� ��������ģʽ: ͨ��PWM�������Ƶ��� */
void smd_pulse_width_ma_mode(uint8_t addr, uint16_t topw_max, uint16_t topw_min, int32_t top_ma, int32_t down_ma)
{
    uint8_t cmd[32] = {0};
    
    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_PULSE_WIDTH_MA_MODE;    
    cmd[3] = (uint8_t)((topw_max >> 8) & 0xFF); 
    cmd[4] = (uint8_t)((topw_max >> 0) & 0xFF); 
    cmd[5] = (uint8_t)((topw_min >> 8) & 0xFF); 
    cmd[6] = (uint8_t)((topw_min >> 0) & 0xFF); 
    cmd[7] =  (uint8_t)((top_ma >> 24) & 0xFF);
    cmd[8] =  (uint8_t)((top_ma >> 16) & 0xFF);
    cmd[9] =  (uint8_t)((top_ma >> 8) & 0xFF); 
    cmd[10] = (uint8_t)((top_ma >> 0) & 0xFF); 
    cmd[11] =  (uint8_t)((down_ma >> 24) & 0xFF);  
    cmd[12] =  (uint8_t)((down_ma >> 16) & 0xFF);  
    cmd[13] =  (uint8_t)((down_ma >> 8) & 0xFF);   
    cmd[14] = (uint8_t)((down_ma >> 0) & 0xFF);    
    cmd[15] =  smd_checksum(cmd, 15);       
    cmd[16] =  FRAME_TAIL;                 

    smd_send_data(cmd, 17);
}

/* smd_pulse_width_speed_mode �� �����ٶ�ģʽ: ͨ��PWM���������ٶ� */
void smd_pulse_width_speed_mode(uint8_t addr, uint16_t topw_max, uint16_t topw_min, int32_t top_speed, int32_t down_speed)
{
    uint8_t cmd[32] = {0};
    
    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_PULSE_WIDTH_SPEED_MODE; 
    cmd[3] = (uint8_t)((topw_max >> 8) & 0xFF); 
    cmd[4] = (uint8_t)((topw_max >> 0) & 0xFF); 
    cmd[5] = (uint8_t)((topw_min >> 8) & 0xFF); 
    cmd[6] = (uint8_t)((topw_min >> 0) & 0xFF); 
    cmd[7] =  (uint8_t)((top_speed >> 24) & 0xFF);
    cmd[8] =  (uint8_t)((top_speed >> 16) & 0xFF);
    cmd[9] =  (uint8_t)((top_speed >> 8) & 0xFF); 
    cmd[10] = (uint8_t)((top_speed >> 0) & 0xFF); 
    cmd[11] =  (uint8_t)((down_speed >> 24) & 0xFF);  
    cmd[12] =  (uint8_t)((down_speed >> 16) & 0xFF);  
    cmd[13] =  (uint8_t)((down_speed >> 8) & 0xFF);   
    cmd[14] = (uint8_t)((down_speed >> 0) & 0xFF);    
    cmd[15] =  smd_checksum(cmd, 15);      
    cmd[16] =  FRAME_TAIL;                

    smd_send_data(cmd, 17);
}

/* ============================================================================
 * �����˶��������� �� ��λ�ñջ�, �����ڼ򵥿��Ƴ���
 * ============================================================================ */

/* smd_ol_speed_mode �� �����ٶ�ģʽ */
void smd_ol_speed_mode(uint8_t addr, uint8_t dir, uint8_t acc, float speed)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_OL_SPEED_MODE;          
    cmd[3] =  dir;                        
    cmd[4] =  acc;                        
    data_u.f = speed;                     
    cmd[5] =  data_u.b[3];                
    cmd[6] =  data_u.b[2];   
    cmd[7] =  data_u.b[1];    
    cmd[8] =  data_u.b[0];     
    cmd[9] =  smd_checksum(cmd, 9);       
    cmd[10] =  FRAME_TAIL;                

    smd_send_data(cmd, 11);
}

/* smd_ol_pos_mode �� ��������λ��ģʽ */
void smd_ol_pos_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                  
    cmd[1] =  addr;                        
    cmd[2] =  FCT_OL_POS_MODE;             
    cmd[3] =  dir;                         
    cmd[4] =  acc;                         
    cmd[5] =  (uint8_t)((speed >> 8) & 0xFF);   
    cmd[6] =  (uint8_t)((speed >> 0) & 0xFF);   
    cmd[7] =  (uint8_t)((pulses >> 24) & 0xFF); 
    cmd[8] =  (uint8_t)((pulses >> 16) & 0xFF); 
    cmd[9] =  (uint8_t)((pulses >> 8) & 0xFF);  
    cmd[10] = (uint8_t)((pulses >> 0) & 0xFF);  
    cmd[11] = 0U; /* sync flag: 0=disable multi-device sync */
    cmd[12] = smd_checksum(cmd, 12);       
    cmd[13] = FRAME_TAIL;                  

    smd_send_data(cmd, 14);
}

/* smd_ol_pos_rel_mode �� �������λ��ģʽ */
void smd_ol_pos_rel_mode(uint8_t addr, uint8_t dir, uint8_t acc, uint16_t speed, uint32_t pulses)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                  
    cmd[1] =  addr;                        
    cmd[2] =  FCT_OL_POS_REL_MODE;         
    cmd[3] =  dir;                         
    cmd[4] =  acc;                         
    cmd[5] =  (uint8_t)((speed >> 8) & 0xFF);   
    cmd[6] =  (uint8_t)((speed >> 0) & 0xFF);   
    cmd[7] =  (uint8_t)((pulses >> 24) & 0xFF); 
    cmd[8] =  (uint8_t)((pulses >> 16) & 0xFF); 
    cmd[9] =  (uint8_t)((pulses >> 8) & 0xFF);  
    cmd[10] = (uint8_t)((pulses >> 0) & 0xFF);  
    cmd[11] = 0U; /* sync flag: 0=disable multi-device sync */
    cmd[12] = smd_checksum(cmd, 12);       
    cmd[13] = FRAME_TAIL;                  

    smd_send_data(cmd, 14);
}

/* smd_ol_pulse_mode �� �л�����������ģʽ */
void smd_ol_pulse_mode(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_OL_PULSES_MODE;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* ============================================================================
 * ������������
 * ============================================================================ */

/* smd_io_run_ctrl �� IO���п���: ͨ��IO�źſ��Ƶ�� */
void smd_io_run_ctrl(uint8_t addr, uint8_t dir, uint8_t acc, float speed)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_IO_RUN_MODE;            
    cmd[3] =  dir;                        
    cmd[4] =  acc;                        
    data_u.f = speed;                     
    cmd[5] =  data_u.b[3];                
    cmd[6] =  data_u.b[2];   
    cmd[7] =  data_u.b[1];    
    cmd[8] =  data_u.b[0];     
    cmd[9] =  smd_checksum(cmd, 9);       
    cmd[10] =  FRAME_TAIL;                

    smd_send_data(cmd, 11);
}

/* smd_angle_to_zero �� ���㵱ǰλ�� (����ǰ�Ƕ���Ϊ0) */
void smd_angle_to_zero(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_ANGLE_ZERO;             
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_remove_clog_protect �� �����ת����״̬ */
void smd_remove_clog_protect(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_CLEAR_CLOG_PRO;         
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_motor_enable �� ʹ��/���õ�� */
void smd_motor_enable(uint8_t addr, uint8_t en)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_MOTOR_ENABLE;           
    cmd[3] =  en;                         
    cmd[4] =  smd_checksum(cmd, 4);       
    cmd[5] =  FRAME_TAIL;                

    smd_send_data(cmd, 6);
}

/* smd_clear_sta �� ������״̬ (�������/����) */
void smd_clear_sta(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_CLEAR_STATE;            
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* smd_stop_now �� ����ֹͣ��� (��ͣ) */
void smd_stop_now(uint8_t addr)
{
    uint8_t cmd[16] = {0};

    cmd[0] =  FRAME_HEAD;                 
    cmd[1] =  addr;                       
    cmd[2] =  FCT_STOP_NOW;               
    cmd[3] =  smd_checksum(cmd, 3);       
    cmd[4] =  FRAME_TAIL;                

    smd_send_data(cmd, 5);
}

/* wait_smd_response �� �ȴ�ָ��ID�������Ӧ (ͨ��, �޹��������) */
uint8_t wait_smd_response(uint32_t id, SMD_Response *resp, uint32_t timeout_ms)
{
    return wait_smd_response_by_func(id, 0U, resp, timeout_ms);
}

/* wait_smd_response_by_func �� �ȴ�ָ��ID�͹��������Ӧ
 * func=0ʱ�����˹�����, �����κ���Ӧ
 */
uint8_t wait_smd_response_by_func(uint32_t id, uint8_t func, SMD_Response *resp, uint32_t timeout_ms)
{
    uint32_t start_seq;

    if ((id == 0U) || (id >= SMD_ADDR_COUNT))
    {
        return 0U;
    }

    start_seq = smd_get_seq_snapshot((uint8_t)id);
    return smd_wait_response_by_func_from_seq(id, func, start_seq, resp, timeout_ms);
}

/* smd_get_last_rx_motor_addr �� ��ȡ���һ���յ���Ӧ�ĵ����ַ */
uint8_t smd_get_last_rx_motor_addr(void)
{
    return s_last_rx_motor_addr;
}

/* smd_get_last_rx_can_id �� ��ȡ���һ���յ���Ӧ��CAN ID */
uint32_t smd_get_last_rx_can_id(void)
{
    return s_last_rx_can_id;
}

/* robotic_arm_reset_target_cache �� �����е��Ŀ�������
 * ������Ҫǿ�����·�������ĳ��� (��Ƕ�ģʽ�л�)
 */
void robotic_arm_reset_target_cache(void)
{
    memset(s_arm_last_cmd_pulse, 0, sizeof(s_arm_last_cmd_pulse));
    memset(s_arm_last_cmd_dir, 0, sizeof(s_arm_last_cmd_dir));
    memset(s_arm_cmd_valid, 0, sizeof(s_arm_cmd_valid));
}

/* COG_EXTENSION_THRESHOLD �� ���������ж���ֵ
 * ���2(�粿)�͵��3(�ⲿ)�����˻�е�۵�ˮƽ��չ����
 * �����߾�������λ��֮�ͳ�������ֵʱ, ��Ϊ��������
 * ��ʱ�������(addr 1)����ƶ�, ����ؽ��Ȼ����Խ����㸲����
 */
#define COG_EXTENSION_THRESHOLD  10000U

/* robotic_arm_is_cog_extended �� ͨ�����2+3�Ļ���λ�ù��������Ƿ�����
 * ���� 0=�������ڲ�/��������, 1=��������
 */
static uint8_t robotic_arm_is_cog_extended(void)
{
    int32_t pos2 = data[2].real_pos_pulse;  /* motor 2 �C shoulder */
    int32_t pos3 = data[3].real_pos_pulse;  /* motor 3 �C elbow   */
    int32_t ext;

    if (pos2 < 0L) { pos2 = -pos2; }
    if (pos3 < 0L) { pos3 = -pos3; }

    ext = pos2 + pos3;

    return (ext > (int32_t)COG_EXTENSION_THRESHOLD) ? 1U : 0U;
}

/* robotic_arm_get_order �� ��������λ�ü�����ִ��˳��
 * order[0..5] = 0-based�������
 * �����ڲ� �� ���1�ȶ� �� [0,1,2,3,4,5] (1��6˳��)
 * �������� �� ���1��� �� [1,2,3,4,5,0] (��ؽ��Ȼ���, �������)
 */
static void robotic_arm_get_order(uint8_t order[6])
{
    if (robotic_arm_is_cog_extended() != 0U)
    {
        order[0] = 1;  /* motor 2 */
        order[1] = 2;  /* motor 3 */
        order[2] = 3;  /* motor 4 */
        order[3] = 4;  /* motor 5 */
        order[4] = 5;  /* motor 6 */
        order[5] = 0;  /* motor 1 (base) �C last */
    }
    else
    {
        uint8_t i;
        for (i = 0U; i < 6U; i++)
        {
            order[i] = i;  /* 0,1,2,3,4,5 �� motor 1..6 in natural order */
        }
    }
}

static uint8_t robotic_get_run_acc(void)
{
    if (robotic_run_acc <= 0)
    {
        return 0U;
    }
    if (robotic_run_acc > 255)
    {
        return 255U;
    }

    return (uint8_t)robotic_run_acc;
}

static uint16_t robotic_get_motor_run_speed(uint8_t motor_idx)
{
    int32_t speed = robotic_run_speed;

    if (speed <= 0)
    {
        return 0U;
    }

    switch (motor_idx)
    {
    case 0U:
        speed = speed * 2L;              /* motor 0: 2.0x */
        break;
    case 1U:
    case 2U:
        speed = (speed * 3L + 1L) / 2L;  /* motor 1/2: 1.5x */
        break;
    case 4U:
        speed = (speed + 1L) / 2L;       /* motor 4: 0.5x */
        break;
    default:
        break;
    }

    if (speed > 65535L)
    {
        speed = 65535L;
    }

    return (uint16_t)speed;
}
/* robotic_arm_control �� 6���е���������� (�������� + ����ȥ�� + ���Ի���)
 * 1. ͨ�� CoG ����ȷ���ؽ�ִ��˳�� (��������ʱ�������)
 * 2. �������ϴγɹ�������ͬ��Ŀ�� (ȥ���Ż�)
 * 3. ÿ���ؽ��������3��, �ȴ���Ӧȷ��
 */
void robotic_arm_control(int32_t* pulse_data)
{
    uint8_t i;
    uint8_t order[6];
    uint8_t run_acc;

    if (pulse_data == NULL)
    {
        return;
    }

    /* Determine execution order from current CoG estimation */
    robotic_arm_get_order(order);
    run_acc = robotic_get_run_acc();

    for (i = 0U; i < 6U; i++)
    {
        uint8_t motor_idx = order[i];
        uint8_t addr = motor_idx + 1U;
        uint16_t run_speed = robotic_get_motor_run_speed(motor_idx);
        int32_t pulse_val = pulse_data[motor_idx];
        uint32_t target_pulse;
        uint8_t dir;
        uint8_t sent_ok = 0U;
        uint8_t retry;

        /* Positive => dir=0, Negative => dir=1 */
        if (pulse_val >= 0)
        {
            dir = 0U;
        }
        else
        {
            dir = 1U;
            pulse_val = -pulse_val;
        }

        target_pulse = (uint32_t)pulse_val;

        /* Same target as last successful command: do not resend. */
        if ((s_arm_cmd_valid[motor_idx] != 0U) && (target_pulse == s_arm_last_cmd_pulse[motor_idx]) && (dir == s_arm_last_cmd_dir[motor_idx]))
        {
            continue;
        }

        for (retry = 0U; retry < 3U; retry++)
        {
            uint32_t seq_base = smd_get_seq_snapshot(addr);

            smd_pos_mode(addr, dir, run_acc, run_speed, target_pulse);
            if (smd_wait_response_by_func_from_seq(addr, (uint8_t)FCT_POS_MODE, seq_base, NULL, 100U) != 0U)
            {
                sent_ok = 1U;
                break;
            }
        }

        if (sent_ok == 0U)
        {
            continue;
        }

        s_arm_last_cmd_pulse[motor_idx] = target_pulse;
        s_arm_last_cmd_dir[motor_idx] = dir;
        s_arm_cmd_valid[motor_idx] = 1U;
    }
}

/* Read_robotic_arm_real_angle �� ���ζ�ȡ6������ľ���λ��
 * ������Ͷ�λ������ȴ���Ӧ, ����洢�� data[addr].real_pos_pulse
 */
void Read_robotic_arm_real_angle(void)
{
    uint8_t i;

    for (i = 0U; i < 6U; i++)
    {
        uint8_t addr = (uint8_t)(i + 1U);

        /* Send one read command and wait until this address response is fully received
           before sending the next one. The ISR stores the completed frame into data[addr]. */
        if (smd_call_serialized(smd_read_pos, addr, FCT_READ_POS, NULL, 50U) != SMD_TRANS_OK)
        {
            continue;
        }
    }
}
/**
 * @brief ????????7????????+��???????????delay??????
 *        ?????????????????????????څ???-30?????????��?��??��???��????????
 * @param torque_clamp ?��????????????????څ?
 * @param current_threshold ???????????څ????-30??
 */

/* ---- ���������ܼ�צ״̬�� (�������� + ����ģʽ����) ----
 *
 * ����:
 *   S1: С���رƽ�              �� S2: ��ѯ����, ���Ӵ�
 *   �� S3: ���ص��������ֵ      �� MONITOR: �������ȶ�
 *   �� HOLD: ������������, ά��������ģʽ (��������������)
 *
 * ���ʼ�ղ��뿪����ģʽ �� �г�����ʵʱ������ά�ֶ���λ��PID
 * �����廬�����µ����仯, ���ں�����չ�м��
 */
typedef enum {
    GRIP_IDLE = 0,
    GRIP_S1_SEND,
    GRIP_BASELINE,
    GRIP_S2_POLL,
    GRIP_S3_RAMP,
    GRIP_MONITOR,
    GRIP_HOLD,
    GRIP_DONE,
    GRIP_TIMEOUT
} grip_state_t;

static grip_state_t s_gs = GRIP_IDLE;
static uint8_t  s_g_addr;
static uint8_t  s_g_dir;
static int16_t  s_g_start_tq, s_g_max_tq, s_g_tq_step, s_g_contact_delta;
static uint32_t s_g_poll_ms, s_g_timeout_ms;
static int16_t  s_g_cur_tq;
static uint32_t s_g_tick_start, s_g_last_ms;
static int16_t  s_g_last_mA;       /* previous filtered current reading */
static int16_t  s_g_filtered_mA;   /* filtered absolute phase current */
static int16_t  s_g_baseline_mA;   /* no-load current measured during low-torque closing */
static int32_t  s_g_baseline_sum;
static uint8_t  s_g_baseline_cnt;
static uint8_t  s_g_contact_cnt;   /* consecutive contact detections */
static uint8_t  s_g_stable_cnt;    /* consecutive stable readings */
static uint32_t s_g_monitor_start; /* when MONITOR phase began */

/* ---- tunable constants (change as needed) ---- */
#define GRIP_HOLD_RATIO      70    /* hold at 70% of max torque */
#define GRIP_STABLE_WINDOW   50    /* ��50 mA = current is stable */
#define GRIP_STABLE_NEEDED   5     /* need N consecutive stable readings */
#define GRIP_MONITOR_TIMEOUT 3000  /* max ms in MONITOR before forcing HOLD */

#define GRIP_BASELINE_SAMPLES       4U
#define GRIP_CONTACT_MIN_DELTA_MA   120
#define GRIP_CONTACT_NEEDED         3U
#define GRIP_ADAPT_STABLE_WINDOW    50
#define GRIP_ADAPT_STABLE_NEEDED    4U
#define GRIP_ADAPT_MONITOR_TIMEOUT  1200U
#define GRIP_HOLD_MARGIN_MA         80

static void grip_dbg(const char *msg, int val)
{
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%s %d\r\n", msg, val);
    if (len > 0) HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)len, 100);
}

static int16_t grip_abs_i16(int16_t value)
{
    if (value < 0)
    {
        if (value == (int16_t)0x8000)
        {
            return 32767;
        }
        return (int16_t)(-value);
    }

    return value;
}

static uint8_t grip_send_torque(uint16_t current)
{
    uint8_t payload[3];

    payload[0] = s_g_dir;
    payload[1] = (uint8_t)((current >> 8) & 0xFFU);
    payload[2] = (uint8_t)(current & 0xFFU);

    return smd_exec_cmd_sync(s_g_addr, FCT_TORQUE_MODE, payload, sizeof(payload), NULL, 0U);
}

static uint8_t grip_read_abs_current(int16_t *abs_mA)
{
    SMD_Response resp = {0};
    int16_t raw_mA;

    if (abs_mA == NULL)
    {
        return 0U;
    }

    if (smd_exec_cmd_sync(s_g_addr, FCT_READ_PHASE_MA, NULL, 0U, &resp, 50U) != SMD_TRANS_OK)
    {
        return 0U;
    }

    if ((resp.len < 5U) || (resp.data[2] != FCT_READ_PHASE_MA))
    {
        return 0U;
    }

    raw_mA = (int16_t)(((uint16_t)resp.data[3] << 8) | (uint16_t)resp.data[4]);
    *abs_mA = grip_abs_i16(raw_mA);
    return 1U;
}

static int16_t grip_filter_current(int16_t abs_mA)
{
    if (s_g_filtered_mA == 0)
    {
        s_g_filtered_mA = abs_mA;
    }
    else
    {
        s_g_filtered_mA = (int16_t)(((int32_t)s_g_filtered_mA * 3 + abs_mA) / 4);
    }

    return s_g_filtered_mA;
}

static void grip_enter_timeout(void)
{
    grip_dbg("GRIP timeout", 0);
    (void)grip_send_torque(0U);
    s_gs = GRIP_TIMEOUT;
}

/* grip_start �� ������������צץȡ����
 * @param addr              �����ַ(1~11)
 * @param dir               �˶����� (0��1)
 * @param start_torque      ��ʼ���� (mA)
 * @param max_torque        ������� (mA)
 * @param torque_step       ÿ�ε������� (mA)
 * @param contact_threshold �Ӵ�������ֵ (mA, ���ڴ�ֵ��Ϊ�Ӵ�����)
 * @param poll_interval_ms  ������ѯ���
 * @param timeout_ms        �ܳ�ʱʱ��
 */
void grip_start(uint8_t addr, uint8_t dir,
                int16_t start_torque, int16_t max_torque,
                int16_t torque_step, int16_t contact_threshold,
                uint32_t poll_interval_ms, uint32_t timeout_ms)
{
    if ((addr == 0U) || (addr >= SMD_ADDR_COUNT)) return;

    s_g_addr         = addr;
    s_g_dir          = dir;
    s_g_start_tq     = start_torque;
    s_g_max_tq       = max_torque;
    s_g_tq_step      = torque_step;
    s_g_contact_delta = grip_abs_i16(contact_threshold);
    if (s_g_contact_delta < GRIP_CONTACT_MIN_DELTA_MA)
    {
        s_g_contact_delta = GRIP_CONTACT_MIN_DELTA_MA;
    }
    s_g_poll_ms      = poll_interval_ms;
    s_g_timeout_ms   = timeout_ms;
    s_g_cur_tq       = start_torque;
    s_g_tick_start   = HAL_GetTick();
    s_g_last_ms      = 0U;
    s_g_last_mA      = 0;
    s_g_filtered_mA  = 0;
    s_g_baseline_mA  = 0;
    s_g_baseline_sum = 0;
    s_g_baseline_cnt = 0U;
    s_g_contact_cnt  = 0U;
    s_g_stable_cnt   = 0U;
    s_g_monitor_start = 0U;
    s_gs             = GRIP_S1_SEND;
}

/* grip_reset �� ��λ��צ״̬��������״̬ */
void grip_reset(void)
{
    s_gs = GRIP_IDLE;
}

/* grip_is_busy �� ��ѯ��צ״̬���Ƿ��������� */
uint8_t grip_is_busy(void)
{
    return (uint8_t)(s_gs != GRIP_IDLE && s_gs != GRIP_DONE && s_gs != GRIP_TIMEOUT);
}

/* grip_is_done �� ��ѯ��צ�Ƿ������ץȡ */
uint8_t grip_is_done(void)
{
    return (uint8_t)(s_gs == GRIP_DONE);
}

/* grip_timed_out �� ��ѯ��צ�Ƿ�ʱ */
uint8_t grip_timed_out(void)
{
    return (uint8_t)(s_gs == GRIP_TIMEOUT);
}

/* grip_tick �� ��צ״̬����ѭ��: ÿ1ms��TIM1�ص�����, �ƽ�״̬��ת */
#if 0
void grip_tick(void)
{
    uint32_t now = HAL_GetTick();
    int16_t mA;

    switch (s_gs)
    {
    case GRIP_IDLE:
    case GRIP_DONE:
    case GRIP_TIMEOUT:
        return;

    /* ---- Stage 1: send start torque, move to polling ---- */
    case GRIP_S1_SEND:
        grip_dbg("GRIP S1 tq", (int)s_g_start_tq);
        smd_torque_mode(s_g_addr, s_g_dir, (uint16_t)s_g_start_tq);
        s_g_last_ms = now;
        s_gs = GRIP_S2_POLL;
        return;

    /* ---- Stage 2: poll phase current until contact ---- */
    case GRIP_S2_POLL:
        if ((now - s_g_tick_start) >= s_g_timeout_ms)
        {
            grip_dbg("GRIP timeout", 0);
            s_gs = GRIP_TIMEOUT;
            return;
        }
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        s_g_rx_seq_snap = s_rx_seq[s_g_addr];
        smd_read_phase_ma(s_g_addr);
        s_g_last_ms = now;
        s_gs = GRIP_S2_WAIT_RESP;
        return;

    case GRIP_S2_WAIT_RESP:
        if ((now - s_g_tick_start) >= s_g_timeout_ms)
        {
            grip_dbg("GRIP timeout", 0);
            s_gs = GRIP_TIMEOUT;
            return;
        }
        if (s_rx_seq[s_g_addr] == s_g_rx_seq_snap)
        {
            if ((now - s_g_last_ms) < 50U) return;
            s_g_rx_seq_snap = s_rx_seq[s_g_addr];
            smd_read_phase_ma(s_g_addr);
            s_g_last_ms = now;
            return;
        }
        if (data[s_g_addr].len >= 5U &&
            data[s_g_addr].data[2] == FCT_READ_PHASE_MA)
        {
            mA = (int16_t)(((uint16_t)data[s_g_addr].data[3] << 8) |
                           ((uint16_t)data[s_g_addr].data[4]));
            grip_dbg("GRIP S2 mA", (int)mA);

            if (mA <= s_g_contact_thr)
            {
                grip_dbg("GRIP contact", (int)mA);
                s_g_last_mA = mA;
                s_g_last_ms = now;
                s_gs = GRIP_S3_RAMP;
                return;
            }
        }
        s_g_last_ms = now;
        s_gs = GRIP_S2_POLL;
        return;

    /* ---- Stage 3: ramp torque, then monitor current stability ---- */
    case GRIP_S3_RAMP:
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        s_g_cur_tq += s_g_tq_step;
        if (s_g_cur_tq > s_g_max_tq)
        {
            s_g_cur_tq = s_g_max_tq;
        }
        grip_dbg("GRIP S3 tq", (int)s_g_cur_tq);
        smd_torque_mode(s_g_addr, s_g_dir, (uint16_t)s_g_cur_tq);
        s_g_last_ms = now;

        if (s_g_cur_tq >= s_g_max_tq)
        {
            /* Max torque reached �� start monitoring current stability */
            grip_dbg("GRIP monitor start", (int)s_g_max_tq);
            s_g_stable_cnt   = 0U;
            s_g_monitor_start = now;
            s_g_last_ms       = now;
            s_gs = GRIP_MONITOR;
        }
        return;

    /* ---- Monitor: poll current, check stability at max torque ---- */
    case GRIP_MONITOR:
        /* Force HOLD if monitoring takes too long */
        if ((now - s_g_monitor_start) >= GRIP_MONITOR_TIMEOUT)
        {
            grip_dbg("GRIP monitor force hold", 0);
            s_gs = GRIP_HOLD;
            return;
        }
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        s_g_rx_seq_snap = s_rx_seq[s_g_addr];
        smd_read_phase_ma(s_g_addr);
        s_g_last_ms = now;
        s_gs = GRIP_MONITOR_WAIT;
        return;

    case GRIP_MONITOR_WAIT:
        if ((now - s_g_monitor_start) >= GRIP_MONITOR_TIMEOUT)
        {
            grip_dbg("GRIP monitor force hold", 0);
            s_gs = GRIP_HOLD;
            return;
        }
        if (s_rx_seq[s_g_addr] == s_g_rx_seq_snap)
        {
            if ((now - s_g_last_ms) < 50U) return;
            s_g_rx_seq_snap = s_rx_seq[s_g_addr];
            smd_read_phase_ma(s_g_addr);
            s_g_last_ms = now;
            return;
        }
        if (data[s_g_addr].len >= 5U &&
            data[s_g_addr].data[2] == FCT_READ_PHASE_MA)
        {
            mA = (int16_t)(((uint16_t)data[s_g_addr].data[3] << 8) |
                           ((uint16_t)data[s_g_addr].data[4]));
            {
                int16_t diff = mA - s_g_last_mA;
                if (diff < 0) diff = -diff;
                grip_dbg("GRIP mon mA", (int)mA);

                if (diff <= GRIP_STABLE_WINDOW)
                {
                    s_g_stable_cnt++;
                    grip_dbg("GRIP stable", (int)s_g_stable_cnt);
                    if (s_g_stable_cnt >= GRIP_STABLE_NEEDED)
                    {
                        s_gs = GRIP_HOLD;
                        return;
                    }
                }
                else
                {
                    s_g_stable_cnt = 0U;
                }
                s_g_last_mA = mA;
            }
        }
        s_g_last_ms = now;
        s_gs = GRIP_MONITOR;
        return;

    /* ---- Hold: reduce to hold torque, stay in torque mode ---- */
    case GRIP_HOLD:
    {
        int16_t hold_tq = (int16_t)((int32_t)s_g_max_tq * GRIP_HOLD_RATIO / 100);
        grip_dbg("GRIP HOLD tq", (int)hold_tq);
        smd_torque_mode(s_g_addr, s_g_dir, (uint16_t)hold_tq);
        s_gs = GRIP_DONE;
        return;
    }

    default:
        return;
    }
}
#endif

void grip_tick(void)
{
    uint32_t now = HAL_GetTick();
    int16_t mA;
    int16_t filtered;

    switch (s_gs)
    {
    case GRIP_IDLE:
    case GRIP_DONE:
    case GRIP_TIMEOUT:
        return;

    case GRIP_S1_SEND:
        grip_dbg("GRIP S1 tq", (int)s_g_start_tq);
        if (grip_send_torque((uint16_t)s_g_start_tq) != SMD_TRANS_OK)
        {
            grip_enter_timeout();
            return;
        }
        s_g_last_ms = now;
        s_gs = GRIP_BASELINE;
        return;

    case GRIP_BASELINE:
        if ((now - s_g_tick_start) >= s_g_timeout_ms)
        {
            grip_enter_timeout();
            return;
        }
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        if (grip_read_abs_current(&mA) == 0U)
        {
            s_g_last_ms = now;
            return;
        }

        s_g_baseline_sum += mA;
        s_g_baseline_cnt++;
        s_g_last_ms = now;

        if (s_g_baseline_cnt >= GRIP_BASELINE_SAMPLES)
        {
            s_g_baseline_mA = (int16_t)(s_g_baseline_sum / (int32_t)s_g_baseline_cnt);
            s_g_filtered_mA = s_g_baseline_mA;
            s_g_last_mA = s_g_baseline_mA;
            s_g_contact_cnt = 0U;
            grip_dbg("GRIP base mA", (int)s_g_baseline_mA);
            s_gs = GRIP_S2_POLL;
        }
        return;

    case GRIP_S2_POLL:
        if ((now - s_g_tick_start) >= s_g_timeout_ms)
        {
            grip_enter_timeout();
            return;
        }
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        if (grip_read_abs_current(&mA) == 0U)
        {
            s_g_last_ms = now;
            return;
        }

        filtered = grip_filter_current(mA);
        grip_dbg("GRIP close mA", (int)filtered);

        if (filtered >= (int16_t)(s_g_baseline_mA + s_g_contact_delta))
        {
            s_g_contact_cnt++;
            if (s_g_contact_cnt >= GRIP_CONTACT_NEEDED)
            {
                grip_dbg("GRIP contact", (int)filtered);
                s_g_last_mA = filtered;
                s_g_stable_cnt = 0U;
                s_g_monitor_start = now;
                s_g_last_ms = now;
                s_gs = GRIP_S3_RAMP;
                return;
            }
        }
        else if (s_g_contact_cnt > 0U)
        {
            s_g_contact_cnt--;
        }

        s_g_last_ms = now;
        return;

    case GRIP_S3_RAMP:
        if ((now - s_g_tick_start) >= s_g_timeout_ms)
        {
            s_gs = GRIP_HOLD;
            return;
        }
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        s_g_cur_tq += s_g_tq_step;
        if (s_g_cur_tq > s_g_max_tq)
        {
            s_g_cur_tq = s_g_max_tq;
        }

        grip_dbg("GRIP S3 tq", (int)s_g_cur_tq);
        if (grip_send_torque((uint16_t)s_g_cur_tq) != SMD_TRANS_OK)
        {
            grip_enter_timeout();
            return;
        }

        s_g_stable_cnt = 0U;
        s_g_monitor_start = now;
        s_g_last_ms = now;
        s_gs = GRIP_MONITOR;
        return;

    case GRIP_MONITOR:
        if ((now - s_g_monitor_start) >= GRIP_ADAPT_MONITOR_TIMEOUT)
        {
            if (s_g_cur_tq >= s_g_max_tq)
            {
                grip_dbg("GRIP force hold", (int)s_g_cur_tq);
                s_gs = GRIP_HOLD;
            }
            else
            {
                grip_dbg("GRIP add tq", (int)s_g_cur_tq);
                s_gs = GRIP_S3_RAMP;
            }
            return;
        }
        if ((now - s_g_last_ms) < s_g_poll_ms) return;

        if (grip_read_abs_current(&mA) == 0U)
        {
            s_g_last_ms = now;
            return;
        }

        filtered = grip_filter_current(mA);
        {
            int16_t diff = (int16_t)(filtered - s_g_last_mA);
            if (diff < 0) diff = (int16_t)(-diff);
            grip_dbg("GRIP mon mA", (int)filtered);

            if (diff <= GRIP_ADAPT_STABLE_WINDOW)
            {
                s_g_stable_cnt++;
                grip_dbg("GRIP stable", (int)s_g_stable_cnt);
                if (s_g_stable_cnt >= GRIP_ADAPT_STABLE_NEEDED)
                {
                    s_gs = GRIP_HOLD;
                    return;
                }
            }
            else
            {
                s_g_stable_cnt = 0U;
            }
            s_g_last_mA = filtered;
        }

        s_g_last_ms = now;
        return;

    case GRIP_HOLD:
    {
        int16_t hold_margin = (s_g_tq_step > GRIP_HOLD_MARGIN_MA) ? s_g_tq_step : GRIP_HOLD_MARGIN_MA;
        int16_t hold_tq = (int16_t)(s_g_cur_tq + hold_margin);

        if (hold_tq > s_g_max_tq)
        {
            hold_tq = s_g_max_tq;
        }
        if (hold_tq < s_g_start_tq)
        {
            hold_tq = s_g_start_tq;
        }

        grip_dbg("GRIP HOLD tq", (int)hold_tq);
        (void)grip_send_torque((uint16_t)hold_tq);
        s_gs = GRIP_DONE;
        return;
    }

    default:
        return;
    }
}

/* robotic_stop �� 6�����ֹͣ: ��CoG˳���ÿ���������stop_now���� (�������3��) */
void robotic_stop()
{
    uint8_t i;
    uint8_t order[6];
    robotic_arm_get_order(order);
    for (i = 0U; i < 6U; i++)
    {
        uint8_t addr = order[i] + 1U;
        uint8_t retry;
        for (retry = 0U; retry < 3U; retry++)
        {
            smd_call_serialized(smd_stop_now, addr, 0U, NULL, 0U);
            HAL_Delay(100);
        }
    }
}
extern smd_data_t data[12];
/* robotic_move_to �� �ƶ�6�ᵽ���Ե�ǰ real_pos_pulse λ��
 * �����״̬, �ٷ���λ������, ÿ���������3��
 */
void robotic_move_to(void)
{
    uint8_t i;
    for (i = 0U; i < 6U; i++)
    {
        uint8_t addr = i + 1U;
        uint8_t retry;
        for (retry = 0U; retry < 3U; retry++)
        {
            smd_call_serialized(smd_clear_sta, addr, 0U, NULL, 0U);
            HAL_Delay(70);
        }
        for (retry = 0U; retry < 3U; retry++)
        {
            smd_pos_mode(addr, 0, robotic_get_run_acc(), robotic_get_motor_run_speed(i), data[addr].real_pos_pulse);
            HAL_Delay(70);
        }
    }
}

#ifndef __OLED_H__
#define __OLED_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */



/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */
//void OLED_SetImage(uint8_t *A)
//void OLED_Setpixel(uint8_t x,uint8_t y);
/* ---- OLED显示驱动函数 ----
 * 使用SSD1306控制器, I2C接口, 128x64分辨率
 * 帧缓冲区: GARM[8][128] — 8行 x 128列 = 1024字节
 */
void OLED_NewFrame(void);                     /* 开始新一帧: 清空帧缓冲区 */
void OLED_ShowFrame(void);                    /* 将帧缓冲区刷新到OLED屏幕 */
void OLED_Clear(void);                        /* 直接清除OLED屏幕(不经过缓冲区) */
void OLED_Init(void);                         /* 初始化OLED: 配置SSD1306寄存器序列 */
void OLED_ShowString(uint8_t x, uint8_t y, char *data);       /* 在(x,y)显示ASCII字符串 */
void OLED_ShowNum(uint8_t x, uint8_t y, int num);              /* 显示有符号整数(单数字) */
void OLED_ShowLongNum(uint8_t x, uint8_t y, uint16_t num);     /* 显示无符号16位整数(多位数) */
void OLED_ShowChar(uint8_t x, uint8_t y, char data);           /* 在(x,y)显示单个字符 */
void OLED_ShowImage(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t *image); /* 显示位图图像 */
void OLED_ShowHex(uint8_t x, uint8_t y, uint32_t num, uint8_t digits);  /* 以十六进制显示数值 */
void OLED_ShowHexArray(uint8_t x, uint8_t y, const uint8_t *data, uint8_t len); /* 以十六进制显示字节数组 */
	/* USER CODE BEGIN Prototypes */

/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __OLED_H__ */


# STM32 Low-Level Controller & Hardware Design

**中文 (Default):** [README.md](README.md)

This directory contains the **STM32 embedded low-level master control subsystem** for the Natural-Language Interactive Safe HRC for Industrial Arms project. It includes the custom control board PCB circuit design (EasyEDA Pro project & schematic) and the real-time motion control firmware based on the STM32F407 MCU.

> **Module Maintainer**: [@RM-wD55](https://github.com/RM-wD55) (MY-GIRL) (Responsible for EasyEDA Pro mainboard PCB schematic/layout design, STM32F407 firmware, CAN 6-axis SMD motor driving, serial protocol stack, and hardware watchdog)

---

## Table of Contents

- [System Overview](#system-overview)
- [Hardware & Electrical Design](#hardware--electrical-design)
  - [3D PCB Rendering](#3d-pcb-rendering)
  - [Schematic Diagram](#schematic-diagram)
  - [Key Hardware Specifications & Pinout](#key-hardware-specifications--pinout)
- [Firmware Architecture](#firmware-architecture)
  - [Core Module Breakdown](#core-module-breakdown)
  - [Clock Tree & Interrupt Priorities](#clock-tree--interrupt-priorities)
- [Communication Protocols](#communication-protocols)
  - [1. Host ROS 2 Unified Serial Protocol (USART1)](#1-host-ros-2-unified-serial-protocol-usart1)
  - [2. CAN2 SMD Motor Bus Protocol](#2-can2-smd-motor-bus-protocol)
  - [3. K230 Edge Vision Protocol (USART3)](#3-k230-edge-vision-protocol-usart3)
  - [4. Vacuum Pump Control Protocol (UART4)](#4-vacuum-pump-control-protocol-uart4)
- [Safety & Protection Mechanisms](#safety--protection-mechanisms)
- [Build, Flashing & Development](#build-flashing--development)
- [Directory Layout](#directory-layout)

---

## System Overview

The STM32 low-level controller serves as the **deterministic execution and hardware safety foundation** of the entire collaborative robotic system. The upstream ROS 2 motion planning layer (`ros2/src/main/arm_state.py`) communicates bidirectionally with the STM32 over high-speed serial links. The STM32 executes:
1. **Multi-Axis Motion Control**: Drives 6-axis SMD intelligent closed-loop stepper motor drivers over CAN bus at 500kbps with smooth incremental trajectory commands.
2. **End-Effector Driving**: Controls the vacuum pump suction and release timing via an isolated serial line and power driver circuit.
3. **Edge Vision Aggregation**: Receives and aggregates multi-target object pixel coordinates streamed from the Kendryte K230 vision edge sensor.
4. **Hardware Safety Boundary**: Enforces hardware-level communication watchdog timers (2s timeout auto-estop), power-on zero-point offset verification, Center of Gravity (CoG) aware motion ordering, and emergency stop arbitration.
5. **On-Board Telemetry Display**: Drives an SSD1306 128×64 OLED screen for real-time visualization of joint pulse positions, command counters, and system health status.

---

## Hardware & Electrical Design

The control board is a custom 2-layer PCB designed specifically for 6-DOF industrial robot manipulators, integrating the MCU, isolated communication transceivers, and power management units.

### 3D PCB Rendering

![STM32 Controller 3D PCB Rendering](3D_PCB1_2026-06-28.png)

### Schematic Diagram

![STM32 Controller Circuit Schematic](SCH_2026-06-28.png)

> **EDA Source File**: `jlc.epro2` is an EasyEDA Pro (立创EDA专业版) project file. It can be opened directly in EasyEDA Pro for schematic modifications, PCB layout editing, BOM export, and PCB manufacturing.

### Key Hardware Specifications & Pinout

| Module / Interface | Part / Specification | Pin Assignment | Description |
| --- | --- | --- | --- |
| **Main MCU** | STM32F407VET6 (LQFP-100) | - | ARM Cortex-M4 @ 168MHz, 512KB Flash, 192KB SRAM |
| **CAN Bus** | CAN2 Transceiver (500kbps) | PB12 (CAN2_RX), PB13 (CAN2_TX) | Cascaded control for Joint 1-6 SMD drivers + smart gripper |
| **Host Link** | USART1 (115200bps, DMA RX/TX) | PA9 (TX), PA10 (RX) | ROS 2 / Industrial PC host communication link |
| **Vision Sensor** | USART3 (115200bps, DMA RX) | PB10 (TX), PB11 (RX) | Receives target pixel coordinate stream from K230 module |
| **Vacuum End-Effector** | UART4 (115200bps, DMA TX) | PC10 (TX), PC11 (RX) | Sends pulse commands to vacuum pump & solenoid valve |
| **Status Display** | I2C1 (400kHz Fast Mode) | PB6 (SCL), PB7 (SDA) | 0.96" SSD1306 128×64 OLED display (I2C address `0x78`) |
| **SWD Debug** | Standard SWD Header | PA13 (SWDIO), PA14 (SWCLK) | ST-Link / DAP-Link debugging and firmware flashing |
| **Power Supply** | DC-DC Buck + LDO Regulators | - | Wide DC input voltage, step-down to 5V and isolated 3.3V |

---

## Firmware Architecture

The firmware is located in [`Control_Code/`](Control_Code/), built upon the STM32CubeMX HAL framework with Keil MDK-ARM project support.

```text
Control_Code/
├── Core/
│   ├── Inc/
│   │   ├── main.h               # Global definitions and configurations
│   │   ├── smd.h                # SMD CAN protocol stack (60+ command codes)
│   │   ├── K230_UART.h          # K230 multi-target parser and aggregator
│   │   ├── pump.h               # Vacuum pump UART driver
│   │   ├── oled.h               # SSD1306 OLED display driver
│   │   ├── can.h, usart.h, ...  # Peripheral HAL headers
│   │   └── tim.h, dma.h, ...    # Timer and DMA interfaces
│   └── Src/
│       ├── main.c               # Main loop, packet parser, watchdog & CoG sequencer
│       ├── smd.c                # SMD CAN protocol, frame reassembly, mutex locks
│       ├── K230_UART.c          # Vision coordinate stream parser
│       ├── pump.c               # Vacuum pump command framing & keep-alive refresh
│       ├── oled.c               # OLED dual-framebuffer rendering
│       └── ...
├── Drivers/                     # STM32F4xx HAL Driver & CMSIS
├── MDK-ARM/                     # Keil uVision5 project (test.uvprojx)
└── test.ioc                     # STM32CubeMX hardware pin and clock configuration
```

### Core Module Breakdown

1. **`main.c` (Super-Loop Scheduler & Safety Core)**:
   - 2-second host communication watchdog (`s_last_ok_tick`).
   - Power-on zero offset calibration with tolerance threshold verification (`verify_and_capture_zero_offset()`).
   - Unified frame decoder `U,...*CHK\r\n` handling joint angles, emergency stop, pump control, and speed scaling.
   - **CoG (Center of Gravity) aware joint motion ordering** according to current J2/J3 configuration.
2. **`smd.c` / `smd.h` (SMD Motor CAN Protocol Stack)**:
   - Implements 60+ functional codes for SMD intelligent drivers (position, velocity, torque, PID, current telemetry).
   - Thread-safe `smd_try_lock()` / `smd_unlock()` critical-section locks preventing CAN transmission race conditions.
   - Robust CAN interrupt frame reassembly with multi-byte big-endian decoding and checksum verification.
3. **`K230_UART.c` (Vision Stream Aggregation)**:
   - USART3 DMA Idle reception of individual target coordinates `EA,x,yP` ~ `ED,x,yP`.
   - Aggregates targets into the standard 8-slot array `E<x1,y1,x2,y2,x3,y3,x4,y4>P` for upstream ROS 2 inverse kinematics (`ik_control`).
4. **`pump.c` (Vacuum End-Effector Control)**:
   - Sends `#005P2500Tx000!` frames via UART4 DMA.
   - Supports 1~9s timed suction and continuous mode with automatic 8-second keep-alive refresh.
5. **`oled.c` (Telemetry Display)**:
   - 128×64 dual-buffer rendering for ASCII text, 6-axis joint pulse values, hex/octal debugging data, and command frame statistics.

### Clock Tree & Interrupt Priorities

- **System Clock**: PLL倍频 to **168 MHz** (APB1 = 42MHz, APB2 = 84MHz).
- **Interrupt Preemption Priorities**:

| Interrupt Source | Preemption Priority | Purpose |
| --- | :---: | --- |
| **SysTick** | 0 | 1ms global timebase |
| **CAN2 RX0** | 0 | High-speed SMD motor frame reception and reassembly |
| **USART1** | 0 | ROS 2 host control frame DMA idle interrupt |
| **USART3** | 0 | K230 vision coordinate DMA idle interrupt |
| **UART4 / DMA** | 0 | Vacuum pump DMA transmission complete interrupt |
| **TIM1** | 1 | 1kHz position polling and non-blocking state machine |

---

## Communication Protocols

### 1. Host ROS 2 Unified Serial Protocol (USART1)

- **Baudrate**: `115200 bps`, 8 data bits, 1 stop bit, no parity.
- **DMA Ring Buffer**: Double-buffered DMA with remainder splicing against fragmented packets.

#### Control Frame Format (ROS 2 → STM32)

```text
U,<DOG>,<p1>,<p2>,<p3>,<p4>,<p5>,<p6>,<STOP>,<PUMP>,<SPEED>*CHK\r\n
```

- **Header / Tail**: Starts with `U`, ends with `*CHK\r\n`.
- **Fields** (11 comma-separated fields):
  1. `<DOG>`: Watchdog heartbeat. `OK` to feed watchdog; `MM` for no-op.
  2. `<p1> ~ <p6>`: 6-axis relative target pulse values, or all `MM` (hold current position).
  3. `<STOP>`: Stop arbitration. `CD` for immediate emergency stop; `EF` for motion resume; `MM` for no-op.
  4. `<PUMP>`: Vacuum pump. `CPQE` for continuous suction; `PUT` for release; `MM` for no-op.
  5. `<SPEED>`: Speed scale percentage (`100` for 100%, `50` for 50%), or `MM`.

*Examples*:
```text
U,OK,0,380000,360000,0,-10000,0,MM,CPQE,100*CHK\r\n   # Move + Pick + Feed Dog
U,OK,MM,MM,MM,MM,MM,MM,CD,PUT,MM*CHK\r\n               # Estop + Release Pump
```

#### Firmware ID Handshake

- **Query (ROS 2 → STM32)**: `Q,ID*CHK\r\n`
- **Response (STM32 → ROS 2)**: `R,ID,stm32-f407-unified-v2-p6*CHK\r\n`

#### Echo Confirmation (STM32 → ROS 2)

Upon successfully decoding and accepting a target angle frame:
```text
OK <p1>,<p2>,<p3>,<p4>,<p5>,<p6>\r\n
```

---

### 2. CAN2 SMD Motor Bus Protocol

- **CAN ID**: `0x1000` (29-bit Extended Frame)
- **Baudrate**: `500 kbps`

```text
 0      1       2       3  ...  N-2        N-1
+------+------+------+-----+-----+------+------+
| 0xC5 | addr | func |  payload  | chk  | 0x5C |
+------+------+------+-----+-----+------+------+
  HEAD                              checksum  TAIL
```

- **Motor Addresses**: `1` ~ `6` for manipulator Joints 1-6; `7` for smart gripper.
- **Byte Order**: Big-endian (MSB first).

---

### 3. K230 Edge Vision Protocol (USART3)

- **Input (K230 → STM32)**: `E<Class>,<x>,<y>P` (e.g. `EA,320,240P`).
- **Aggregated Output (STM32 Buffer)**:
  ```text
  E<Ax>,<Ay>,<Bx>,<By>,<Cx>,<Cy>,<Dx>,<Dy>P
  ```

---

### 4. Vacuum Pump Control Protocol (UART4)

- **Frame**: `#005P2500T<time>!`
  - `<time>` = `9000`: Continuous suction (refreshed automatically every 8s).
  - `<time>` = `1000` ~ `9000`: Timed suction for 1~9 seconds.
  - `PUT`: Stop pump power and release vacuum pressure.

---

## Safety & Protection Mechanisms

1. **Hardware Communication Watchdog**:
   - The super-loop tracks elapsed time. If no valid control packet or `OK` heartbeat is received for **2000ms**, the MCU immediately latches emergency stop (`s_emergency_stop = 1`) and cuts motor drive output.
2. **Power-On Zero Offset Calibration**:
   - Reads encoder pulse coordinates from all 6 motors at boot, verifies deviation within tolerance (±500 pulses), and retries up to 3 times before enabling motion.
3. **Center of Gravity (CoG) Aware Motion Sequencing**:
   - Estimates center of gravity extension via the sum of Joint 2 (shoulder) and Joint 3 (elbow) pulses.
   - **Outward Extension**: Base and inner joints move first, outer joints extend last.
   - **Retraction**: Outer joints retract first, base returns last to prevent high overturning moments on the arm mounting base.
4. **CAN Bus Critical Section Locking**:
   - `smd_try_lock()` and `smd_unlock()` protect the CAN mailboxes against interrupt/super-loop collisions.

---

## Build, Flashing & Development

### Prerequisites
- **IDE**: Keil MDK-ARM v5.30+
- **Configuration Tool**: STM32CubeMX v6.10+ (`Control_Code/test.ioc`)
- **Hardware Debugger**: ST-Link V2 / DAP-Link / J-Link

### Build Steps
1. Open the Keil project: `Control_Code/MDK-ARM/test.uvprojx`
2. Click **Rebuild All Target Files** (`F7`).
3. Connect ST-Link to SWD pins (`3V3`, `SWDIO`, `SWCLK`, `GND`).
4. Click **Download** (`F8`) to flash the firmware.
5. On boot, the OLED display will render the firmware status and initial 6-axis coordinates.

---

## Directory Layout

```text
STM32/
├── 3D_PCB1_2026-06-28.png        # Controller 3D PCB rendering
├── SCH_2026-06-28.png            # Controller circuit schematic diagram
├── jlc.epro2                     # EasyEDA Pro project file
├── README.md                     # Chinese documentation
├── README.en.md                  # English documentation
└── Control_Code/                 # Embedded firmware source tree
    ├── Core/                     # Application source and headers
    ├── Drivers/                  # STM32 HAL and CMSIS libraries
    ├── MDK-ARM/                  # Keil uVision project and scripts
    ├── test.ioc                  # STM32CubeMX pin and clock configuration
    └── README.md                 # Firmware subfolder guide
```

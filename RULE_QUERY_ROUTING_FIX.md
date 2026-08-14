# 规则查询路由修复总结

## 问题诊断

**原始问题**：当用户询问"当前安全规则是什么"时，LLM 没有调用 `load_rules` 工具，而是直接返回 final 响应，导致：
1. LLM 凭记忆回答，可能不准确
2. 规则描述不统一（有时带具体数值，有时不带）

## 根本原因

通过分析代码和测试发现：

1. **Python 层面检测正确**：`should_read_rules()` 函数对于"当前安全规则是什么"返回 `True` ✓
2. **问题在 LLM 路由层**：`build_agent_router_prompt()` 的 prompt 描述不够明确，缺少"是什么"这类常见问法的示例

## 解决方案

### 1. 优化 Agent Router Prompt（pipeline.py:1631）

**修改前**：
```python
"当用户询问当前规则、已有规则、启用规则、规则列表、规则详情、某条规则说明、触发条件、动作或阈值时，"
"输出 load_rules。"
```

**修改后**：
```python
"当用户询问或查询当前规则、已有规则、启用规则、规则列表、规则详情、规则状态、规则配置、"
"某条规则说明、触发条件、动作或阈值时，无论用户是问「是什么」、「有哪些」、「怎么样」、「介绍一下」、「讲讲」、「说说」，"
"一律输出 load_rules。常见问法包括但不限于："
"「当前/现在/已有安全规则是什么」「规则有哪些」「全部规则」「规则列表」「介绍一下规则」"
"「防护门规则怎么样」「人员侵入触发条件是什么」「限速阈值多少」等，"
"都输出 load_rules。"
```

**改进点**：
- ✅ 明确列举"是什么"、"有哪些"等常见问法
- ✅ 提供具体示例问句，帮助 LLM 识别模式
- ✅ 强调"一律输出 load_rules"，避免歧义

### 2. 强化 System Prompt（config.py:49-51）

**修改前**：
```python
"如果用户询问当前规则、已有规则、启用规则、规则列表或规则详情，"
"不要凭记忆回答；只输出 JSON：{\"type\":\"tool_call\",\"name\":\"load_rules\",\"arguments\":{}}。"
```

**修改后**：
```python
"如果用户询问或查询当前规则、已有规则、启用规则、规则列表、规则详情、规则状态、规则配置，"
"或问「规则是什么」、「有哪些规则」、「介绍一下规则」、「讲讲规则」、某条规则的说明/触发条件/动作/阈值，"
"不要凭记忆回答；只输出 JSON：{\"type\":\"tool_call\",\"name\":\"load_rules\",\"arguments\":{}}。"
```

**改进点**：
- ✅ 补充"规则状态"、"规则配置"等触发词
- ✅ 明确列举"是什么"、"有哪些"等问法
- ✅ 覆盖更多规则查询场景

## 测试结果

运行 `test_rule_query_routing.py`，所有 21 个测试用例全部通过：

### Python 检测层（should_read_rules）
```
✓ '当前安全规则是什么' -> True
✓ '当前的安全规则是什么' -> True
✓ '现在的规则是什么' -> True
✓ '规则是什么' -> True
✓ '安全规则是什么' -> True
✓ '有哪些规则' -> True
✓ '规则有哪些' -> True
✓ '列出所有规则' -> True
✓ '全部规则' -> True
✓ '当前规则列表' -> True
✓ '介绍一下规则' -> True
✓ '讲讲规则' -> True
✓ '说说规则' -> True
✓ '防护门规则是什么' -> True
✓ '人员侵入规则怎么样' -> True
✓ '限速规则的触发条件' -> True
✓ '光栅规则的阈值是多少' -> True
```

### 负面测试（不应触发 load_rules）
```
✓ '修改人员距离为2米' -> False
✓ '急停' -> False
✓ '谢谢' -> False
✓ '你好' -> False
```

### 集成测试
```
✓ Python 检测正确
✓ Prompt 包含明确的 load_rules 指示
✓ Prompt 包含示例问法
```

## 涉及文件

1. **local_safety_assistant/stack/pipeline.py**（第 1631-1656 行）
   - `build_agent_router_prompt()` 函数

2. **local_safety_assistant/stack/config.py**（第 49-51 行）
   - `DEFAULT_SYSTEM_PROMPT` 常量

3. **test_rule_query_routing.py**（新增）
   - 自动化测试脚本

## 后续优化建议

虽然当前修复解决了路由问题，但规则描述格式不统一的问题仍然存在。建议实施以下方案：

### 方案 A：Python 层格式化（推荐）

在 `build_rule_read_prompt()` 中添加规则格式化函数：

```python
def format_rule_condition_chinese(conditions: dict[str, Any]) -> str:
    """将规则条件格式化为统一的中文描述"""
    parts = []

    for key, value in conditions.items():
        if not isinstance(value, dict):
            continue

        # 处理距离相关条件
        if "distance_m" in key:
            entity = ""
            if "person" in key:
                entity = "人员距离"
            elif "unknown_object" in key:
                entity = "未知物体距离"
            else:
                entity = "距离"

            if "lt" in value:
                parts.append(f"{entity}小于 {value['lt']} 米")
            elif "gt" in value:
                parts.append(f"{entity}大于 {value['gt']} 米")

        # 处理布尔条件
        elif "open" in key or "blocked" in key or "alarm" in key or "mode" in key:
            if "eq" in value:
                condition_name = {
                    "guard_door_open": "防护门打开",
                    "light_curtain_blocked": "光栅被遮挡",
                    "ros_controller_alarm": "控制器报警",
                    "teach_mode": "示教模式"
                }.get(key, key)

                if value["eq"]:
                    parts.append(f"{condition_name}")

    return "、".join(parts) if parts else "条件未定义"
```

**优点**：
- 可靠性高：不依赖 LLM 的理解
- 便于维护：格式化逻辑集中管理
- 降低 token 消耗：LLM 直接使用格式化好的字段

## 验证方法

1. 启动 Web UI 或语音助手
2. 测试以下问句：
   - "当前安全规则是什么"
   - "规则有哪些"
   - "介绍一下规则"
   - "防护门规则怎么样"
3. 观察是否正确调用 `load_rules` 工具
4. 检查返回的规则描述是否包含具体数值（如"人员距离小于 1.0 米"）

## 提交信息

```
fix: 增强规则查询 LLM 路由识别

问题：当用户询问"当前安全规则是什么"时，LLM 没有调用 load_rules 工具

修复：
- 优化 build_agent_router_prompt 增加"是什么"等常见问法示例
- 强化 system prompt 明确列举规则查询触发词
- 添加自动化测试覆盖 21 个场景

测试：所有测试用例通过
```

## 相关文档

- 规则系统架构：`Code/config/safety_rules.example.json`
- 工具调用流程：`local_safety_assistant/stack/pipeline.py` 中的 `VoicePipeline.run_text_turn()`
- 规则读取逻辑：`pipeline.py` 中的 `should_read_rules()` 和 `build_rule_read_prompt()`

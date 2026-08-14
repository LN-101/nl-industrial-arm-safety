#!/usr/bin/env python3
"""测试规则查询路由修复效果"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from local_safety_assistant.stack.pipeline import (
    should_read_rules,
    build_agent_router_prompt,
    normalize_asr_text,
)


def test_rule_query_detection():
    """测试规则查询检测逻辑"""

    test_cases = [
        # 原始问题：这些应该触发 load_rules
        ("当前安全规则是什么", True),
        ("当前的安全规则是什么", True),
        ("现在的规则是什么", True),
        ("规则是什么", True),
        ("安全规则是什么", True),

        # 其他常见问法
        ("有哪些规则", True),
        ("规则有哪些", True),
        ("列出所有规则", True),
        ("全部规则", True),
        ("当前规则列表", True),
        ("介绍一下规则", True),
        ("讲讲规则", True),
        ("说说规则", True),

        # 具体规则查询
        ("防护门规则是什么", True),
        ("人员侵入规则怎么样", True),
        ("限速规则的触发条件", True),
        ("光栅规则的阈值是多少", True),

        # 规则修改（应该走 edit 而不是 read）
        ("修改人员距离为2米", False),  # should_edit_rules 返回 True 会让 should_read_rules 返回 False

        # 非规则查询（不应该触发）
        ("急停", False),
        ("谢谢", False),
        ("你好", False),
    ]

    print("=" * 80)
    print("测试 should_read_rules() 函数")
    print("=" * 80)

    passed = 0
    failed = 0

    for text, expected in test_cases:
        result = should_read_rules(text)
        status = "✓" if result == expected else "✗"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} '{text}' -> {result} (期望: {expected})")

    print(f"\n通过: {passed}/{len(test_cases)}, 失败: {failed}/{len(test_cases)}")
    print()


def test_router_prompt_content():
    """测试路由器 prompt 内容是否包含关键触发词"""

    print("=" * 80)
    print("测试 build_agent_router_prompt() 内容")
    print("=" * 80)

    test_text = "当前安全规则是什么"
    prompt = build_agent_router_prompt(test_text)

    # 检查关键词是否出现在 prompt 中
    keywords = [
        "是什么",
        "有哪些",
        "规则状态",
        "规则配置",
        "介绍一下",
        "讲讲",
        "说说",
        "当前/现在/已有安全规则是什么",
    ]

    print(f"测试输入: '{test_text}'")
    print(f"\nPrompt 长度: {len(prompt)} 字符\n")

    print("关键词检查:")
    for keyword in keywords:
        # 规范化检查（去掉标点）
        normalized_keyword = keyword.replace("/", "").replace("、", "")
        if normalized_keyword in prompt or keyword in prompt:
            print(f"  ✓ 包含: {keyword}")
        else:
            print(f"  ✗ 缺失: {keyword}")

    print("\n完整 Prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    print()


def test_integration():
    """集成测试：验证完整的检测 + prompt 流程"""

    print("=" * 80)
    print("集成测试")
    print("=" * 80)

    test_text = "当前安全规则是什么"

    # Step 1: Python 层面检测
    should_load = should_read_rules(test_text)
    print(f"1. Python 检测 should_read_rules('{test_text}'): {should_load}")

    # Step 2: 生成路由 prompt
    prompt = build_agent_router_prompt(test_text)
    print(f"2. 生成路由 prompt (长度: {len(prompt)} 字符)")

    # Step 3: 检查 prompt 中是否有明确的指示
    has_explicit_instruction = "是什么" in prompt and "load_rules" in prompt
    print(f"3. Prompt 包含明确的 load_rules 指示: {has_explicit_instruction}")

    # Step 4: 检查是否列举了示例
    has_examples = "当前/现在/已有安全规则是什么" in prompt or "规则有哪些" in prompt
    print(f"4. Prompt 包含示例问法: {has_examples}")

    print("\n总结:")
    if should_load and has_explicit_instruction and has_examples:
        print("  ✓ 修复成功！Python 检测正确 + Prompt 包含明确指示和示例")
    else:
        print("  ✗ 仍有问题:")
        if not should_load:
            print("    - Python 检测未通过")
        if not has_explicit_instruction:
            print("    - Prompt 缺少明确指示")
        if not has_examples:
            print("    - Prompt 缺少示例问法")
    print()


if __name__ == "__main__":
    test_rule_query_detection()
    test_router_prompt_content()
    test_integration()

    print("=" * 80)
    print("测试完成")
    print("=" * 80)

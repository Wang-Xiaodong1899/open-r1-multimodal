import re

pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

text = """<think>Thinking process</think><answer>\nFinal answer.\n</answer>
"""

# 使用 re.findall 查找所有匹配的内容
matches = re.match(pattern, text, re.DOTALL)
# matches = re.match(pattern, text)

print(matches)
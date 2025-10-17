"""测试库存阻止是否真的在执行"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("测试库存阻止机制")
print("="*80)

env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

print("\n运行第1步 (应该看到[DEBUG]消息如果阻止生效):")
action = np.zeros(env.action_space.shape)
obs, reward, terminated, truncated, info = env.step(action)

print("\n运行第2步:")
obs, reward, terminated, truncated, info = env.step(action)

print("\n运行第3步:")
obs, reward, terminated, truncated, info = env.step(action)

print("\n="*80)
print("如果上面看到[DEBUG]消息，说明阻止机制正在工作")
print("如果没有看到，说明没有公司的库存超过阈值，或者机制没有生效")
print("="*80)


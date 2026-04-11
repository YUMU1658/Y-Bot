"""NapCat API 字段测试脚本（临时/独立）。

连接到 NapCat 的 WebSocket 端口，调用 get_login_info、get_group_member_info、
get_group_info、get_friend_list 等 API，打印原始响应以确认字段格式。

使用方法:
    1. 确保 NapCat 已启动并配置了正向 WebSocket
    2. 修改下方的 WS_URL、GROUP_ID、USER_ID
    3. 运行: python test_napcat_api.py

注意: 此脚本为一次性测试用途，不属于 Y-BOT 正式代码。
"""

import asyncio
import json
from uuid import uuid4

import websockets

# ===== 配置区域 =====
WS_URL = "ws://localhost:3001"  # NapCat 正向 WebSocket 地址
GROUP_ID = 0  # 要测试的群号（填入实际群号）
USER_ID = 0  # 要测试的用户 QQ 号（填入实际 QQ 号）
# ====================


async def call_api(ws, action: str, params: dict) -> dict:
    """发送 API 请求并等待响应。"""
    echo = uuid4().hex
    payload = json.dumps({"action": action, "params": params, "echo": echo})
    await ws.send(payload)

    # 等待匹配的响应
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(raw)
        if data.get("echo") == echo:
            return data


async def main():
    print(f"正在连接 {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        print("已连接\n")

        # 1. get_login_info
        print("=" * 60)
        print(">>> get_login_info")
        print("=" * 60)
        try:
            resp = await call_api(ws, "get_login_info", {})
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"失败: {e}")
        print()

        # 2. get_group_info
        if GROUP_ID:
            print("=" * 60)
            print(f">>> get_group_info (group_id={GROUP_ID})")
            print("=" * 60)
            try:
                resp = await call_api(ws, "get_group_info", {"group_id": GROUP_ID})
                print(json.dumps(resp, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"失败: {e}")
            print()

        # 3. get_group_member_info
        if GROUP_ID and USER_ID:
            print("=" * 60)
            print(f">>> get_group_member_info (group={GROUP_ID}, user={USER_ID})")
            print("=" * 60)
            try:
                resp = await call_api(
                    ws,
                    "get_group_member_info",
                    {"group_id": GROUP_ID, "user_id": USER_ID},
                )
                print(json.dumps(resp, ensure_ascii=False, indent=2))
                # 重点关注的字段
                data = resp.get("data", {})
                print(f"\n--- 重点字段 ---")
                print(
                    f"  level = {data.get('level')!r}  (type: {type(data.get('level')).__name__})"
                )
                print(f"  title = {data.get('title')!r}")
                print(f"  role  = {data.get('role')!r}")
                print(f"  card  = {data.get('card')!r}")
                print(f"  nickname = {data.get('nickname')!r}")
            except Exception as e:
                print(f"失败: {e}")
            print()

        # 4. get_friend_list (前 5 条)
        print("=" * 60)
        print(">>> get_friend_list (前 5 条)")
        print("=" * 60)
        try:
            resp = await call_api(ws, "get_friend_list", {})
            data = resp.get("data", [])
            if isinstance(data, list):
                print(f"共 {len(data)} 个好友")
                for f in data[:5]:
                    print(json.dumps(f, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(resp, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"失败: {e}")

    print("\n测试完成。")


if __name__ == "__main__":
    asyncio.run(main())

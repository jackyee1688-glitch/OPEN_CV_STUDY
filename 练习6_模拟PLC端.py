# 第6课 模拟PLC端：监听TCP，接收"判断+结果"，执行动作
# 运行方法：先启动本程序，再启动 练习6_视觉端.py
import socket

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 5001))
srv.listen(1)
print("模拟PLC已启动，等待视觉系统连接（监听 127.0.0.1:5001）...")
print("按 Ctrl+C 可停止")

conn, addr = srv.accept()
print(f"视觉系统已连接: {addr}")

ok_count = 0
ng_count = 0
while True:
    data = conn.recv(1024)
    if not data:
        break
    for line in data.decode("utf-8").splitlines():
        parts_line = line.split(",")          # 数据格式：判断,测量值,原因
        verdict = parts_line[0]
        measure = parts_line[1] if len(parts_line) > 1 else "?"
        reason = parts_line[2] if len(parts_line) > 2 else ""
        if verdict == "OK":
            ok_count += 1
            print(f"收到 OK  直径={measure}mm {reason} -> 动作：放行通过（累计 OK={ok_count} NG={ng_count}）")
        elif verdict == "NG":
            ng_count += 1
            print(f"收到 NG  直径={measure}mm {reason} -> 动作：气缸剔除！驱动输出 Y0（累计 OK={ok_count} NG={ng_count}）")
        else:
            print(f"收到未知数据: {line!r}")

conn.close()
srv.close()
print(f"视觉端已断开，本次共 OK={ok_count} 个，NG={ng_count} 个")
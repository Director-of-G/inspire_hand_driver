# from inspire_dds import inspire_hand_touch,inspire_hand_ctrl,inspire_hand_state
# from inspire_dds import inspire_hand_touch,inspire_hand_ctrl,inspire_hand_state
import sys
from inspire_sdkpy import inspire_sdk,inspire_hand_defaut
import time
# import inspire_sdkpy
if __name__ == "__main__":
    
    
    # handler=inspire_sdk.ModbusDataHandler(ip=inspire_hand_defaut.defaut_ip,LR='r',device_id=1)
    handler=inspire_sdk.ModbusDataHandler(ip='192.168.123.211',LR='l',device_id=1)
    time.sleep(0.5)

    call_count = 0  # 记录调用次数
    start_time = time.perf_counter()  # 记录开始时间

    try:
        while True:
            data_dict = handler.read()  # 读取数据

            call_count += 1  # 增加调用计数
            time.sleep(0.001)  # 暂停 5 毫秒

            # 每秒计算并打印一次调用频率
            if call_count % 10 == 0:  # 每 200 次调用计算一次频率
                elapsed_time = time.perf_counter() - start_time  # 计算总耗时
                frequency = call_count / elapsed_time  # 计算频率 (Hz)
                print(f"当前频率: {frequency:.2f} Hz, 调用次数: {call_count}, 耗时: {elapsed_time:.6f} 秒")
    except KeyboardInterrupt:
        elapsed_time = time.perf_counter() - start_time  # 计算总耗时
        frequency = call_count / elapsed_time if elapsed_time > 0 else 0  # 计算最终频率
        print(f"程序结束. 总调用次数: {call_count}, 总耗时: {elapsed_time:.6f} 秒, 最终频率: {frequency:.2f} Hz")

from inspire_sdkpy import inspire_sdk_double, inspire_hand_defaut
import time

if __name__ == "__main__":
    
    ## publish All Data
    # states_structure = [
    #         ('pos_act', 1534, 6, 'short'),
    #         ('angle_act', 1546, 6, 'short'),
    #         ('force_act', 1582, 6, 'short'),
    #         ('current', 1594, 6, 'short'),
    #         ('err', 1606, 3, 'byte'),
    #         ('status', 1612, 3, 'byte'),
    #         ('temperature', 1618, 3, 'byte')
    #     ]
    
    ## Only publish this data to increase publishing frequency
    states_structure = [
            ('angle_act', 1546, 6, 'short'),
            # ('force_act', 1582, 6, 'short'),
            ('status', 1612, 3, 'byte'),
        ]
    
    handler = inspire_sdk_double.ModbusDataHandlerDouble(device_id=[2,1], use_serial=True, serial_port='/dev/ttyUSB0',states_structure=states_structure) # l r
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

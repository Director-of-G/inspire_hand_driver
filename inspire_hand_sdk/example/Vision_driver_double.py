import multiprocessing
import time
from inspire_sdkpy import qt_tabs,inspire_sdk,inspire_hand_defaut
import sys

def worker(ip,LR,name,network=None):
    app = qt_tabs.QApplication(sys.argv)
    handler=inspire_sdk.ModbusDataHandler(network=network,ip=ip, LR=LR, device_id=1)
    window = qt_tabs.MainWindow(data_handler=handler,dt=20,name="Hand Vision Driver")
    window.reflash()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 使用默认IP地址的示例

    process_r = multiprocessing.Process(target=worker, args=('192.168.123.211','r',"右手进程"))
    process_l = multiprocessing.Process(target=worker, args=('192.168.11.210','l',"左手进程"))

    process_r.start()
    time.sleep(0.6)
    process_l.start()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        process_r.terminate()
        process_l.terminate()

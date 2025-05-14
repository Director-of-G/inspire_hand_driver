# from inspire_dds import inspire_hand_touch,inspire_hand_ctrl,inspire_hand_state
# from inspire_dds import inspire_hand_touch,inspire_hand_ctrl,inspire_hand_state
import sys
from inspire_sdkpy import qt_tabs,inspire_sdk,inspire_hand_defaut
# import inspire_sdkpy
if __name__ == "__main__":
    app = qt_tabs.QApplication(sys.argv)
    # handler=inspire_sdk.ModbusDataHandler(ip=inspire_hand_defaut.defaut_ip,LR='r',device_id=1)
    handler=inspire_sdk.ModbusDataHandler(LR='r',device_id=1,use_serial=True)
    window = qt_tabs.MainWindow(data_handler=handler,dt=80,name="Hand Vision Driver",Plot_touch=False,run_time=False)
    window.reflash()
    window.show()
    sys.exit(app.exec_())
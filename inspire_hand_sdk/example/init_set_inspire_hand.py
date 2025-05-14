from pymodbus.client import ModbusTcpClient
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel
from pymodbus.exceptions import ConnectionException

from inspire_sdkpy import defaut_ip

registers = {
    1000: {"name": "HAND_ID", "description": "灵巧手 ID", "length": 1},
    1002: {"name": "REDU_RATIO", "description": "波特率设置", "length": 1},
    1032: {"name": "DEFAULT_SPEED_SET", "description": "各自由度的上电速度设置值", "length": 6},
    1044: {"name": "DEFAULT_FORCE_SET", "description": "各自由度的上电力控阈值设置值", "length": 6},
    1700: {"name": "ip", "description": "ip part", "length": 2},
}

register_set={
    1005: {"name": "SAVE", "description": "保存数据至 Flash", "length": 1},
    1006: {"name": "RESET_PARA", "description": "恢复出厂设置", "length": 1},
    1009: {"name": "GESTURE_FORCE_CLB", "description": "受力传感器校准", "length": 1}

}

baud_rates = {
    0: 115200,
    1: 57600,
    2: 19200,
    3: 921600
}
baud_rates_reverse = {value: key for key, value in baud_rates.items()}



class ModbusHandler:
    def __init__(self, ip, port, id=1):
        self.client = ModbusTcpClient(ip, port)
        try:
            if not self.client.connect():
                raise ConnectionException(f"无法连接到设备: {ip}:{port}")
            print(f"成功连接到设备: {ip}:{port}, ID: {id}")
        except Exception as e:
            print(f"连接错误: {e}")
            self.client = None  # 设置为 None，便于后续检查是否连接成功
        self.id = id

    def read_register(self, address, count):
        response = self.client.read_holding_registers(address, count,self.id)
        if response.isError():
            print("Error reading register:", response)
            return None
        return response.registers

    def write_register(self, address, value):
        response = self.client.write_register(address, value,self.id)
        if response.isError():
            print("Error writing register:", response)
            return False
        return True
    def write_registers(self, address, value):
        response = self.client.write_registers(address, value,self.id)
        if response.isError():
            print("Error writing register:", response)
            return False
        return True

    def close(self):
        if self.client:
            self.client.close()
            print("连接已关闭")

class MainWindow(QMainWindow):
    def __init__(self,ip=defaut_ip,port=6000):
        super().__init__()
        self.id=self.find_online_devices(ip,port)
        self.modbus = ModbusHandler(ip, port,self.id)  # 替换为实际的 IP 和端口
        self.initUI()
        self.read_registers()
        
    def find_online_devices(self,ip=defaut_ip,port=6000):
        for i in range(100):  # 假设设备 ID 范围为 0 到 99
            self.modbus = ModbusHandler(ip, port,i)  # 替换为实际的 IP 和端口
            res = self.modbus.read_register(1000, 1)  # 尝试读取寄存器 1000
            device_id = 0
            if res is not None:
                device_id = res[0]
                print(f'找到在线设备: ID = {device_id}')
            else:
                print(f'未找到在线设备: ID = {i}')
            self.modbus.close()  # 关闭连接
            return device_id
            
    def initUI(self):
        self.setWindowTitle('灵巧手设置')

        layout = QVBoxLayout()

        read_button = QPushButton('读取设置')
        read_button.clicked.connect(self.read_registers)
        layout.addWidget(read_button)

        write_button = QPushButton('写入设置')
        write_button.clicked.connect(self.save_registers)
        layout.addWidget(write_button)
        
        save_button = QPushButton('保存设置')
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button)
        
        reset_button = QPushButton('恢复出场设置')
        reset_button.clicked.connect(self.reset_para)
        layout.addWidget(reset_button)
        
        clb_button = QPushButton('校准力传感器')
        clb_button.clicked.connect(self.cesture_force_clb)
        layout.addWidget(clb_button)
        
        clean_button = QPushButton('清除错误')
        clean_button.clicked.connect(self.clean_error)
        layout.addWidget(clean_button)

        self.register_inputs = {}
        for i ,(address, info) in enumerate(registers.items()):
            layout.addWidget(QLabel(info['description']))
            if not info['name']=='ip':
                inputs = [QLineEdit() for _ in range(info['length'])]
            else :
                inputs = [QLineEdit() for _ in range(info['length']*2)]

            for input_field in inputs:
                layout.addWidget(input_field)
            self.register_inputs[address]=inputs

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.show()
    def save(self):
        self.modbus.write_register(1005, 1)
        print("设置保存寄存器")

        pass
    
    def reset_para(self):
        self.modbus.write_register(1006, 1)
        pass
    
    def cesture_force_clb(self):
        self.modbus.write_registers(1486,[1000]*6)
        self.modbus.write_register(1009,1)
        pass
    def clean_error(self):
        self.modbus.write_register(1004,1)

    def read_registers(self):
        print("读取所有设置")
        for address, info in registers.items():
            if info["length"] == 1:
                values = self.modbus.read_register(address, info["length"] )
                if values is not None:
                    if info['name']=='REDU_RATIO':
                        self.register_inputs[address][0].setText(str(baud_rates[values[0]]))  # 假设每个寄存器只有一个输入框
                    else:
                        self.register_inputs[address][0].setText(str(values[0]))  # 假设每个寄存器只有一个输入框
            elif info["length"] == 6:
                values = self.modbus.read_register(address, info["length"] )
                if values is not None:
                    for j in range(6):
                        self.register_inputs[address][j].setText(str(values[j]))
            elif info['name']=='ip':
                values = self.modbus.read_register(address, 2)
                print(f'ip 寄存器： {values}')
                values = self.read_and_parse_ip(values)
                if values is not None:
                    for j in range(4):
                        self.register_inputs[address][j].setText(str(values[j]))
                        
            print(f'寄存器: {info["name"]} = {values}')
            
    def read_and_parse_ip(self,values):
        if values is not None and len(values) == 2:
            byte1 = values[0] & 0xFF
            byte2 = (values[0] >> 8) & 0xFF
            byte3 = values[1] & 0xFF
            byte4 = (values[1] >> 8) & 0xFF
            
            ip_bytes = [byte1, byte2, byte3, byte4]
            return ip_bytes
        else:
            print('读取失败或返回值不正确')
            return None
    def bytes_to_short(self, values):
        # 将 4 个字节合并为 2 个短整型
        short1 = (values[1] << 8) | values[0]  # 高字节在前，低字节在后
        short2 = (values[3] << 8) | values[2]  # 高字节在前，低字节在后
        return [short1, short2]
    
    def save_registers(self):
        for address, info in registers.items():
            if info["length"] == 1:
                if info['name']=='REDU_RATIO':
                    value = baud_rates_reverse[int(self.register_inputs[address][0].text())]
                else:
                    value = int(self.register_inputs[address][0].text())

                self.modbus.write_register(address, value)
            elif info["length"] == 6:
                values = [int(input_field.text()) for input_field in self.register_inputs[address]]
                self.modbus.write_registers(address,values)
            elif info['name']=='ip':
                values = [int(input_field.text()) for input_field in self.register_inputs[address]]
                values=self.bytes_to_short(values)
                print(f'写入ip :{self.read_and_parse_ip(values)} , 寄存器 : {values}')
                self.modbus.write_registers(address,values)
                      

            pass
        print("写入所有设置")


    def closeEvent(self, event):
        self.modbus.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # window = MainWindow(ip=defaut_ip)
    window = MainWindow(ip='192.168.123.211')
    sys.exit(app.exec_())

import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import Thread

from inspire_sdkpy import inspire_hand_defaut,inspire_dds
import numpy as np

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    # Create a publisher to publish the data defined in UserData class
    pubr = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
    pubr.Init()
    
    publ = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
    publ.Init()
    cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
    short_value=1000


    cmd.angle_set=[0,0,0,0,1000,1000]
    cmd.mode=0b0001
    publ.Write(cmd)
    pubr.Write(cmd)

    time.sleep(1.0)

    cmd.angle_set=[0,0,0,0,0,1000]
    cmd.mode=0b0001
    publ.Write(cmd)
    pubr.Write(cmd)

    time.sleep(3.0)

    for cnd in range(100000): 

            # 寄存器起始地址，0x05CE 对应的是 1486
        start_address = 1486            
        num_registers = 6  # 6 个寄存器
        # 生成要写入的值列表，每个寄存器为一个 short 值

        if (cnd+1) % 10 == 0:
            short_value = 1000-short_value  # 要写入的 short 值



        values_to_write = [short_value] * num_registers
        values_to_write[-1]=1000-values_to_write[-1]
        values_to_write[-2]=1000-values_to_write[-2]

        value_to_write_np=np.array(values_to_write)
        value_to_write_np=np.clip(value_to_write_np,200,800)
        # value_to_write_np[3]=800

        # 将组合模式按二进制方式实现
        # mode 0：0000（无操作）
        # mode 1：0001（角度）
        # mode 2：0010（位置）
        # mode 3：0011（角度 + 位置）
        # mode 4：0100（力控）
        # mode 5：0101（角度 + 力控）
        # mode 6：0110（位置 + 力控）
        # mode 7：0111（角度 + 位置 + 力控）
        # mode 8：1000（速度）
        # mode 9：1001（角度 + 速度）
        # mode 10：1010（位置 + 速度）
        # mode 11：1011（角度 + 位置 + 速度）
        # mode 12：1100（力控 + 速度）
        # mode 13：1101（角度 + 力控 + 速度）
        # mode 14：1110（位置 + 力控 + 速度）
        # mode 15：1111（角度 + 位置 + 力控 + 速度）  
        cmd.angle_set=value_to_write_np.tolist()
        cmd.mode=0b0001
        #Publish message
        if  publ.Write(cmd) and pubr.Write(cmd):
            # print("Publish success. msg:", cmd.crc)
            pass
        else:
            print("Waitting for subscriber.")

        time.sleep(0.1)
        

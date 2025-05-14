// /**
//  * @file inspire_hand.cpp
//  * @brief This is an example of how to control the Unitree H1 (Inspire) Hand using unitree_sdk2.
//  */

// // DDS Channel
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/common/thread/thread.hpp>
#include "unitree/idl/go2/LowState_.hpp"

#include "inspire_hand_ctrl.hpp"
#include "inspire_hand_state.hpp"
#include "inspire_hand_touch.hpp"

#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <algorithm> // 需要引入这个头文件


// /**
//  * Main Function
//  */
inspire::inspire_hand_ctrl hand_cmd;
inspire::inspire_hand_state hand_state;
inspire::inspire_hand_touch hand_touch;
int main(int argc, char **argv)
{

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     H1 Hand Example      \n\n";

    // Initialize the DDS Channel
    std::string networkInterface = argc > 1 ? argv[1] : "";
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

    unitree::robot::ChannelPublisherPtr<inspire::inspire_hand_ctrl> handcmd;

    unitree::robot::ChannelSubscriberPtr<inspire::inspire_hand_touch> handtouch;
    unitree::robot::ChannelSubscriberPtr<inspire::inspire_hand_state> handstate;

    handcmd = std::make_shared<unitree::robot::ChannelPublisher<inspire::inspire_hand_ctrl>>("rt/inspire_hand/ctrl/r");

    // cmd.cmds().resize(12);
    handtouch = std::make_shared<unitree::robot::ChannelSubscriber<inspire::inspire_hand_touch>>("rt/inspire_hand/touch/r");

    handstate = std::make_shared<unitree::robot::ChannelSubscriber<inspire::inspire_hand_state>>("rt/inspire_hand/state/r");

    handcmd->InitChannel();

    handstate->InitChannel([](const void *message)
                            {
    hand_state = *(inspire::inspire_hand_state*)message; 
    std::cout<<hand_state<<std::endl;

    });

    handtouch->InitChannel([](const void *message)
                            {
    hand_touch = *(inspire::inspire_hand_touch*)message; 
    std::cout<<hand_touch<<std::endl;
    });

    int cnt = 0;

    Eigen::Matrix<float, 6, 1> lq, rq;
    lq.fill(1);
    // lq.setZero();
    rq.fill(1);
    // rq.setZero();
    double PERIOD = 4;
    double PI = 3.14159265;

    double short_value = 1;

    while (true)
    {
        usleep(50000);
        if (cnt++ % 30 == 0)
        {
            short_value = 1 - short_value;
        }
        lq = Eigen::Matrix<float, 6, 1>::Ones() *short_value;
        rq = Eigen::Matrix<float, 6, 1>::Ones() *short_value;

        lq[5] = 1 - lq[5];
        lq[4] = 1 - lq[4];

        rq[5] = 1 - rq[5];
        rq[4] = 1 - rq[4];

        rq=rq*1000;
        rq[3]=1000;

        // # 将组合模式按二进制方式实现
        // # mode 0：0000（无操作）
        // # mode 1：0001（角度）
        // # mode 2：0010（位置）
        // # mode 3：0011（角度 + 位置）
        // # mode 4：0100（力控）
        // # mode 5：0101（角度 + 力控）
        // # mode 6：0110（位置 + 力控）
        // # mode 7：0111（角度 + 位置 + 力控）
        // # mode 8：1000（速度）
        // # mode 9：1001（角度 + 速度）
        // # mode 10：1010（位置 + 速度）
        // # mode 11：1011（角度 + 位置 + 速度）
        // # mode 12：1100（力控 + 速度）
        // # mode 13：1101（角度 + 力控 + 速度）
        // # mode 14：1110（位置 + 力控 + 速度）
        // # mode 15：1111（角度 + 位置 + 力控 + 速度）  
        hand_cmd.angle_set().resize(6);
        hand_cmd.mode(0b0001);
        for (size_t i(0); i < 6; i++)
        {
            hand_cmd.angle_set()[i]= rq[i];
        }
        handcmd->Write(hand_cmd);

    }

    return 0;
}


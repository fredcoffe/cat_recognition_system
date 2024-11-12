#include <Servo.h>
//W,F,S
// 引脚定义
int irSensorPin = 7;      // 红外传感器引脚
/*
int Swater = 13;          // 水泵控制引脚
int SW = 12;          // 
*/
Servo myservo;            // 舵机对象
int pos = 0;              // 舵机当前角度
bool isReversing = false; // 标志舵机是否反转

void setup() {
  // 等待串口通信准备好
  Serial.begin(9600);
  while (!Serial) {
    ; // 等待串口连接，适用于部分板子
  }

  delay(500);  // 延迟 500 毫秒，确保串口通信已完全初始化

  // 初始化红外传感器引脚
  pinMode(irSensorPin, INPUT);
  
  // 初始化水泵引脚
  pinMode(Swater, OUTPUT);
  digitalWrite(Swater, LOW);  // 初始状态关闭水泵
  
  // 舵机初始化
  myservo.attach(9);
  myservo.write(pos);  // 舵机初始位置设为 0 度

  // 输出初始化提示信息
  Serial.println("输入W给水10秒，输入F控制舵机旋转");
}

void loop() {
  // 检查是否有串口输入
  if (Serial.available() > 0) {
    char command = Serial.read();  // 读取串口输入字符
/*
    // 处理W命令：启动水泵5秒
    if (command == 'W') {
      Serial.println("水泵启动 5 秒");
      digitalWrite(Swater, HIGH);  // 启动水泵
      delay(5000);                // 延迟 10 秒
      digitalWrite(Swater, LOW);   // 关闭水泵
    }

        // 处理S命令：启动1秒
    if (command == 'S') {
      Serial.println("启动 1 秒");
      digitalWrite(SW, HIGH);  // 启动水泵
      delay(1000);                // 延迟 10 秒
      digitalWrite(SW, LOW);   // 关闭水泵
    }
*/
    // 处理F命令：控制舵机旋转，正转或反转
    else if (command == 'F') {
      if (!isReversing) {
        pos += 60;  // 增加60度
        if (pos >= 180) {
          pos = 180;       // 限制最大角度为180度
          isReversing = true;  // 达到180度后开始反转
        }
      } else {
        pos -= 60;  // 减少60度（反转）
        if (pos <= 0) {
          pos = 0;          // 限制最小角度为0度
          isReversing = false; // 达到0度后开始正转
        }
      }

      // 设置舵机到新的角度并输出当前状态
      myservo.write(pos);
      Serial.print("舵机当前角度: ");
      Serial.println(pos);
    }

    // 每次输入完指令后，输出提示信息
    Serial.println("输入W给水10秒，输入F控制舵机旋转");

    // 清空串口缓冲区，避免后续命令无法执行
    while (Serial.available() > 0) {
      Serial.read();  // 清除所有缓冲区中的字符
    }
  }

  // 读取红外传感器状态
  bool irSensorOutput = digitalRead(irSensorPin);

  // 当红外传感器被触发时输出 "1"
  if (irSensorOutput == HIGH) {
    Serial.println("1");  // 红外传感器触发，输出 1
  }

  delay(10000);  // 循环延迟，避免读取过快
}

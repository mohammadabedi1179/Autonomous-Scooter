#include <TimerOne.h>

/* ================= CONFIG ================= */
const unsigned long Ts = 5000;
const unsigned long pulseTimeout = 200000;
const float pulsesPerRev = 45.0;

const float integralMax = 1000.0;
const int PWM_MAX = 200;

/* ================= PINS ================= */
const int HALL_PIN1 = 2;
const int PWM_PIN1  = 5;
const int DIR_PIN1  = 34;

const int HALL_PIN2 = 3;
const int PWM_PIN2  = 6;
const int DIR_PIN2  = 35;

const int HALL_PIN3 = 18;
const int PWM_PIN3  = 44;
const int DIR_PIN3  = 36;

const int HALL_PIN4 = 19;
const int PWM_PIN4  = 45;
const int DIR_PIN4  = 37;

/* ================= RUNTIME ================= */
volatile unsigned long lastTime[4]={0};
volatile unsigned long period[4]={0};
volatile unsigned long lastPulseTime[4]={0};
volatile bool newPulse[4]={false};

float rpm[4]={0};
float setpoint[4]={0};

float error[4]={0};
float lastError[4]={0};
float integral[4]={0};

int lastSpSign[4]={0};

/* ================= PID GAINS ================= */
float Kp[4]={0.3447,0.3060,0.3020,0.2648};
float Ki[4]={0.8469,1.0924,1.0696,0.9471};
float Kd[4]={0.00336,0.00295,0.00291,0.00255};

/* ================= LOOP ================= */
unsigned long lastLoop=0;
unsigned long lastPrint=0;
const unsigned long printPeriod=100000;

String inputString="";

/* ================= SETUP ================= */
void setup(){

  int hallPins[4]={HALL_PIN1,HALL_PIN2,HALL_PIN3,HALL_PIN4};
  int pwmPins[4]={PWM_PIN1,PWM_PIN2,PWM_PIN3,PWM_PIN4};
  int dirPins[4]={DIR_PIN1,DIR_PIN2,DIR_PIN3,DIR_PIN4};

  for(int i=0;i<4;i++){
    pinMode(hallPins[i],INPUT_PULLUP);
    pinMode(pwmPins[i],OUTPUT);
    pinMode(dirPins[i],OUTPUT);
    analogWrite(pwmPins[i],0);
  }

  attachInterrupt(digitalPinToInterrupt(HALL_PIN1), [](){onPulse(0);}, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN2), [](){onPulse(1);}, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN3), [](){onPulse(2);}, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN4), [](){onPulse(3);}, RISING);

  Serial.begin(115200);
  lastLoop = micros();
}

/* ================= MAIN LOOP ================= */
void loop(){

  /* ===== SERIAL ===== */
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\n' || c=='\r'){
      if(inputString.length()) parseCommand(inputString);
      inputString="";
    }else inputString+=c;
  }

  unsigned long now = micros();
  if(now-lastLoop < Ts) return;
  lastLoop = now;

  float out[4];

  for(int i=0;i<4;i++){

    /* ===== RPM CALC ===== */
    if(now-lastPulseTime[i] > pulseTimeout) rpm[i]=0;

    if(newPulse[i] && period[i]>0){
      noInterrupts();
      unsigned long T = period[i];
      newPulse[i] = false;
      interrupts();
      rpm[i] = (60e6)/(pulsesPerRev*T);
    }

    /* ===== SIGN ===== */
    int spSign;
    if(setpoint[i] > 0) spSign = 1;
    else if(setpoint[i] < 0) spSign = -1;
    else spSign = 0;

    /* ===== PIN SELECT ===== */
    int pwmPin;
    if(i==0) pwmPin = PWM_PIN1;
    else if(i==1) pwmPin = PWM_PIN2;
    else if(i==2) pwmPin = PWM_PIN3;
    else pwmPin = PWM_PIN4;

    int dirPin;
    if(i==0) dirPin = DIR_PIN1;
    else if(i==1) dirPin = DIR_PIN2;
    else if(i==2) dirPin = DIR_PIN3;
    else dirPin = DIR_PIN4;

    /* ===== STOP ===== */
    if(spSign==0){
      integral[i]=0;
      lastError[i]=0;
      error[i]=0;
      out[i]=0;
      analogWrite(pwmPin,0);
      continue;
    }

    /* ===== SIGN CHANGE RESET ===== */
    if(spSign != lastSpSign[i]){
      integral[i]=0;         // reset integral to prevent windup
      lastError[i]=0;        // reset derivative
      out[i]=0;              // zero PWM for 1 cycle (soft start)
      analogWrite(pwmPin,0);
      lastSpSign[i] = spSign;
      continue;              // skip this loop to avoid spike
    }
    lastSpSign[i] = spSign;

    /* ===== PID ===== */
    error[i] = setpoint[i] - rpm[i];

    float dt = Ts/1e6;
    float dErr = (error[i]-lastError[i]) / dt;
    lastError[i] = error[i];

    float out_unsat = Kp[i]*error[i] + Ki[i]*integral[i] + Kd[i]*dErr;

    if(abs(out_unsat) < PWM_MAX){
      integral[i] += error[i]*dt;
      integral[i] = constrain(integral[i], -integralMax, integralMax);
      out[i] = out_unsat;
    }else{
      out[i] = constrain(out_unsat, -PWM_MAX, PWM_MAX);
    }

    /* ===== DIR (CCW = +, CW = -) ===== */
    if(out[i] >= 0){
      digitalWrite(dirPin, LOW);   // CCW
    }else{
      digitalWrite(dirPin, HIGH);  // CW
    }

    int pwmVal = abs(out[i]);
    if(pwmVal > PWM_MAX) pwmVal = PWM_MAX;

    analogWrite(pwmPin, pwmVal);
  }

  /* ===== PRINT ===== */
  if(micros()-lastPrint >= printPeriod){
    lastPrint = micros();

    for(int i=0;i<4;i++){
      Serial.print(rpm[i]);
      if(i<3) Serial.print(",");
    }

    Serial.print(",");

    for(int i=0;i<4;i++){
      Serial.print(error[i]);
      if(i<3) Serial.print(",");
    }

    Serial.print(",");

    for(int i=0;i<4;i++){
      Serial.print(out[i]);
      if(i<3) Serial.print(",");
    }

    Serial.println();
  }
}

/* ================= ISR ================= */
void onPulse(int i){
  unsigned long t = micros();
  period[i] = t - lastTime[i];
  lastTime[i] = t;
  lastPulseTime[i] = t;
  newPulse[i] = true;
}

/* ================= COMMAND ================= */
void parseCommand(String cmd){

  if(!cmd.startsWith("SP#") || cmd.length()!=22){
    Serial.println("ERR");
    return;
  }

  if(cmd.charAt(3)=='+') setpoint[0] = cmd.substring(4,7).toInt();
  else setpoint[0] = -cmd.substring(4,7).toInt();

  if(cmd.charAt(8)=='+') setpoint[1] = cmd.substring(9,12).toInt();
  else setpoint[1] = -cmd.substring(9,12).toInt();

  if(cmd.charAt(13)=='+') setpoint[2] = cmd.substring(14,17).toInt();
  else setpoint[2] = -cmd.substring(14,17).toInt();

  if(cmd.charAt(18)=='+') setpoint[3] = cmd.substring(19,22).toInt();
  else setpoint[3] = -cmd.substring(19,22).toInt();
}

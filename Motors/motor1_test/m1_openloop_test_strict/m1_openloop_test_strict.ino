/*
  M1 Front Right open-loop PWM test - STRICT / SAFE version
  Board: Arduino Mega 2560

  Motor mapping from your main controller:
    M1 PWM = pin 6
    M1 DIR = pin 35

  This sketch does NOT use PID and does NOT use Hall RPM feedback.
  It keeps M0, M2, M3 forced OFF.

  Serial protocol @ 115200:
    HELLO
    ARM
    LIMIT,60
    PWM,+30
    PWM,-30
    STOP
    DISARM

  Safety features:
    - Motor starts DISARMED.
    - PWM defaults to zero.
    - Hard PWM limit = 80.
    - Watchdog stops motor if serial commands stop.
    - Slew-rate limiting.
    - Direction reversal ramps to zero first.
*/

const int PWM_PIN1 = 5;
const int PWM_PIN2 = 6;    // M1 Front Right
const int PWM_PIN3 = 44;
const int PWM_PIN4 = 45;

const int DIR_PIN1 = 34;
const int DIR_PIN2 = 35;   // M1 Front Right
const int DIR_PIN3 = 36;
const int DIR_PIN4 = 37;

const int DIR_POSITIVE = LOW;   // same convention as your main controller
const int DIR_NEGATIVE = HIGH;

const int HARD_PWM_LIMIT = 80;
int softPwmLimit = 60;

const int PWM_SLEW_STEP = 2;
const unsigned long UPDATE_DT_MS = 20;
const unsigned long WATCHDOG_MS = 600;

bool armed = false;
int targetSignedPwm = 0;
int currentPwm = 0;
int currentSign = 0;   // -1, 0, +1

String inputString = "";
unsigned long lastCommandMs = 0;
unsigned long lastUpdateMs = 0;
unsigned long lastPrintMs = 0;
unsigned long lastReadyMs = 0;

int signOf(int v) {
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
}

void allMotorsOff() {
  analogWrite(PWM_PIN1, 0);
  analogWrite(PWM_PIN2, 0);
  analogWrite(PWM_PIN3, 0);
  analogWrite(PWM_PIN4, 0);
}

void setup() {
  pinMode(PWM_PIN1, OUTPUT);
  pinMode(PWM_PIN2, OUTPUT);
  pinMode(PWM_PIN3, OUTPUT);
  pinMode(PWM_PIN4, OUTPUT);

  pinMode(DIR_PIN1, OUTPUT);
  pinMode(DIR_PIN2, OUTPUT);
  pinMode(DIR_PIN3, OUTPUT);
  pinMode(DIR_PIN4, OUTPUT);

  allMotorsOff();
  digitalWrite(DIR_PIN2, DIR_POSITIVE);

  Serial.begin(115200);
  delay(300);

  lastCommandMs = millis();
  lastUpdateMs = millis();
  lastPrintMs = millis();
  lastReadyMs = millis();

  Serial.println("M1_OPEN_LOOP_READY");
  Serial.println("BAUD,115200");
  Serial.println("PINS,M1_PWM=6,M1_DIR=35");
}

void loop() {
  readSerial();

  unsigned long now = millis();

  // Safety watchdog: if the PC stops talking, ramp down to zero.
  if (armed && (now - lastCommandMs > WATCHDOG_MS)) {
    targetSignedPwm = 0;
  }

  if (now - lastUpdateMs >= UPDATE_DT_MS) {
    lastUpdateMs = now;
    updateM1Output();
  }

  // Print READY periodically while disarmed so the Python script can verify the port.
  if (!armed && (now - lastReadyMs >= 1000)) {
    lastReadyMs = now;
    Serial.println("M1_OPEN_LOOP_READY");
  }

  if (now - lastPrintMs >= 200) {
    lastPrintMs = now;
    printStatus();
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputString.length() > 0) {
        parseCommand(inputString);
        inputString = "";
      }
    } else {
      inputString += c;
      if (inputString.length() > 40) inputString = "";
    }
  }
}

void parseCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "HELLO") {
    lastCommandMs = millis();
    Serial.println("OK,HELLO,M1_OPEN_LOOP");
    return;
  }

  if (cmd == "ARM") {
    lastCommandMs = millis();
    armed = true;
    targetSignedPwm = 0;
    currentPwm = 0;
    currentSign = 0;
    allMotorsOff();
    Serial.println("OK,ARMED");
    return;
  }

  if (cmd == "DISARM") {
    armed = false;
    targetSignedPwm = 0;
    currentPwm = 0;
    currentSign = 0;
    allMotorsOff();
    Serial.println("OK,DISARMED");
    return;
  }

  if (cmd == "STOP") {
    lastCommandMs = millis();
    targetSignedPwm = 0;
    Serial.println("OK,STOP");
    return;
  }

  if (cmd.startsWith("LIMIT,")) {
    lastCommandMs = millis();
    int lim = cmd.substring(6).toInt();
    lim = constrain(lim, 0, HARD_PWM_LIMIT);
    softPwmLimit = lim;
    if (targetSignedPwm > softPwmLimit) targetSignedPwm = softPwmLimit;
    if (targetSignedPwm < -softPwmLimit) targetSignedPwm = -softPwmLimit;
    Serial.print("OK,LIMIT,");
    Serial.println(softPwmLimit);
    return;
  }

  if (cmd.startsWith("PWM,")) {
    lastCommandMs = millis();

    if (!armed) {
      targetSignedPwm = 0;
      allMotorsOff();
      Serial.println("ERR,NOT_ARMED");
      return;
    }

    int req = cmd.substring(4).toInt();
    req = constrain(req, -softPwmLimit, softPwmLimit);
    req = constrain(req, -HARD_PWM_LIMIT, HARD_PWM_LIMIT);

    int reqSign = signOf(req);

    // Never reverse instantly. First ramp to zero.
    if (reqSign != 0 && currentSign != 0 && reqSign != currentSign && currentPwm > 0) {
      targetSignedPwm = 0;
      Serial.println("WARN,REVERSAL_BLOCKED_RAMPING_TO_ZERO");
      return;
    }

    targetSignedPwm = req;
    Serial.print("OK,PWM,");
    Serial.println(targetSignedPwm);
    return;
  }

  Serial.print("ERR,UNKNOWN,");
  Serial.println(cmd);
}

void updateM1Output() {
  // Hard-force all other motors off on every update.
  analogWrite(PWM_PIN1, 0);
  analogWrite(PWM_PIN3, 0);
  analogWrite(PWM_PIN4, 0);

  if (!armed) {
    targetSignedPwm = 0;
  }

  int targetSign = signOf(targetSignedPwm);
  int targetMag = abs(targetSignedPwm);

  if (targetSign == 0) {
    if (currentPwm > 0) {
      currentPwm -= PWM_SLEW_STEP;
      if (currentPwm < 0) currentPwm = 0;
    } else {
      currentPwm = 0;
      currentSign = 0;
    }
    analogWrite(PWM_PIN2, currentPwm);
    return;
  }

  if (currentPwm == 0) {
    currentSign = targetSign;
    digitalWrite(DIR_PIN2, currentSign > 0 ? DIR_POSITIVE : DIR_NEGATIVE);
  }

  if (currentSign != targetSign) {
    if (currentPwm > 0) {
      currentPwm -= PWM_SLEW_STEP;
      if (currentPwm < 0) currentPwm = 0;
    }
    analogWrite(PWM_PIN2, currentPwm);
    return;
  }

  if (currentPwm < targetMag) {
    currentPwm += PWM_SLEW_STEP;
    if (currentPwm > targetMag) currentPwm = targetMag;
  } else if (currentPwm > targetMag) {
    currentPwm -= PWM_SLEW_STEP;
    if (currentPwm < targetMag) currentPwm = targetMag;
  }

  currentPwm = constrain(currentPwm, 0, HARD_PWM_LIMIT);
  analogWrite(PWM_PIN2, currentPwm);
}

void printStatus() {
  Serial.print("T,armed=");
  Serial.print(armed ? 1 : 0);
  Serial.print(",target=");
  Serial.print(targetSignedPwm);
  Serial.print(",current=");
  Serial.print(currentPwm);
  Serial.print(",sign=");
  Serial.print(currentSign);
  Serial.print(",limit=");
  Serial.print(softPwmLimit);
  Serial.print(",wd=");
  Serial.println(millis() - lastCommandMs);
}

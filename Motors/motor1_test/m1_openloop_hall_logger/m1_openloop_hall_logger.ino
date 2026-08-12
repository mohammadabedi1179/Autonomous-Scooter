/*
  M1 Front Right open-loop PWM + Hall RPM logger
  Board: Arduino Mega 2560

  Purpose:
    - Drive only M1 in open-loop PWM.
    - Do NOT use PID.
    - Log M1 Hall pulses to diagnose RPM measurement noise.
    - Force M0, M2, M3 OFF at all times.

  Your mapping:
    M1 PWM  = pin 6
    M1 DIR  = pin 35
    M1 Hall = pin 3  (HALL_PIN2 in your main controller)

  Serial protocol @ 115200:
    HELLO
    ARM
    LIMIT,60
    PWM,+40
    PWM,-40
    STOP
    DISARM

  Telemetry output:
    DATA,ms,armed,target,current,sign,limit,wd,raw_cnt,valid_cnt,reject_cnt,last_raw_us,last_valid_us,rpm_raw,rpm_valid,rpm_filt,pulse_age_ms,hall

  Safety:
    - Starts disarmed.
    - Watchdog ramps to zero if PC stops sending commands.
    - Hard PWM limit = 80.
    - Soft PWM limit default = 60.
    - Slew-rate limiting.
    - Direction reversal is blocked unless PWM has ramped to zero.
*/

// ---------------- PIN MAP ----------------
const int PWM_PIN1 = 5;
const int PWM_PIN2 = 6;    // M1 Front Right
const int PWM_PIN3 = 44;
const int PWM_PIN4 = 45;

const int DIR_PIN1 = 34;
const int DIR_PIN2 = 35;   // M1 Front Right
const int DIR_PIN3 = 36;
const int DIR_PIN4 = 37;

const int HALL_PIN2 = 3;   // M1 Hall signal to Arduino

// Direction convention copied from your main controller.
const int DIR_POSITIVE = LOW;
const int DIR_NEGATIVE = HIGH;

// ---------------- SAFETY LIMITS ----------------
const int HARD_PWM_LIMIT = 80;
int softPwmLimit = 60;

const int PWM_SLEW_STEP = 2;                  // PWM units per update
const unsigned long UPDATE_DT_MS = 20;        // output update period
const unsigned long WATCHDOG_MS = 600;        // stop if serial command lost
const unsigned long PRINT_DT_MS = 50;         // 20 Hz telemetry

// ---------------- HALL/RPM SETTINGS ----------------
const float PULSES_PER_REV = 45.0f;
const float MAX_CREDIBLE_RPM = 200.0f;
const unsigned long MIN_VALID_PERIOD_US = (unsigned long)(60000000.0f / (PULSES_PER_REV * MAX_CREDIBLE_RPM));
const unsigned long RPM_TIMEOUT_US = 200000;  // RPM -> 0 if no valid pulse for 200 ms
const float RPM_EMA_ALPHA = 0.25f;            // filtered RPM = alpha*new + (1-alpha)*old

// ---------------- RUNTIME ----------------
String inputString = "";

bool armed = false;
int targetSignedPwm = 0;
int currentPwm = 0;
int currentSign = 0;   // -1, 0, +1

unsigned long lastCommandMs = 0;
unsigned long lastUpdateMs = 0;
unsigned long lastPrintMs = 0;
unsigned long lastReadyMs = 0;

// ---------------- HALL ISR STATE ----------------
volatile unsigned long rawPulseCount = 0;
volatile unsigned long validPulseCount = 0;
volatile unsigned long rejectedPulseCount = 0;

volatile unsigned long lastRawPulseUs = 0;
volatile unsigned long lastValidPulseUs = 0;
volatile unsigned long lastRawPeriodUs = 0;
volatile unsigned long lastValidPeriodUs = 0;
volatile bool newValidPulse = false;

float rpmRaw = 0.0f;
float rpmValid = 0.0f;
float rpmFilt = 0.0f;

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

void resetHallStats() {
  noInterrupts();
  rawPulseCount = 0;
  validPulseCount = 0;
  rejectedPulseCount = 0;
  lastRawPulseUs = 0;
  lastValidPulseUs = 0;
  lastRawPeriodUs = 0;
  lastValidPeriodUs = 0;
  newValidPulse = false;
  interrupts();

  rpmRaw = 0.0f;
  rpmValid = 0.0f;
  rpmFilt = 0.0f;
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

  pinMode(HALL_PIN2, INPUT_PULLUP);

  allMotorsOff();
  digitalWrite(DIR_PIN2, DIR_POSITIVE);

  attachInterrupt(digitalPinToInterrupt(HALL_PIN2), onM1HallPulse, RISING);

  Serial.begin(115200);
  delay(300);

  lastCommandMs = millis();
  lastUpdateMs = millis();
  lastPrintMs = millis();
  lastReadyMs = millis();

  Serial.println("M1_HALL_OPEN_LOOP_READY");
  Serial.println("BAUD,115200");
  Serial.println("PINS,M1_PWM=6,M1_DIR=35,M1_HALL=3");
  Serial.print("HALL,PPREV="); Serial.print(PULSES_PER_REV, 1);
  Serial.print(",MIN_VALID_PERIOD_US="); Serial.println(MIN_VALID_PERIOD_US);
  Serial.println("DATA_HEADER,ms,armed,target,current,sign,limit,wd,raw_cnt,valid_cnt,reject_cnt,last_raw_us,last_valid_us,rpm_raw,rpm_valid,rpm_filt,pulse_age_ms,hall");
}

void loop() {
  readSerial();

  unsigned long nowMs = millis();

  // Safety watchdog: if PC stops talking, ramp down to zero.
  if (armed && (nowMs - lastCommandMs > WATCHDOG_MS)) {
    targetSignedPwm = 0;
  }

  if (nowMs - lastUpdateMs >= UPDATE_DT_MS) {
    lastUpdateMs = nowMs;
    updateM1Output();
  }

  updateRpmEstimate();

  // Print READY periodically while disarmed so Python can verify the correct sketch/port.
  if (!armed && (nowMs - lastReadyMs >= 1000)) {
    lastReadyMs = nowMs;
    Serial.println("M1_HALL_OPEN_LOOP_READY");
  }

  if (nowMs - lastPrintMs >= PRINT_DT_MS) {
    lastPrintMs = nowMs;
    printDataLine();
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
      if (inputString.length() > 50) inputString = "";
    }
  }
}

void parseCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "HELLO") {
    lastCommandMs = millis();
    Serial.println("OK,HELLO,M1_HALL_OPEN_LOOP");
    return;
  }

  if (cmd == "ARM") {
    lastCommandMs = millis();
    armed = true;
    targetSignedPwm = 0;
    currentPwm = 0;
    currentSign = 0;
    allMotorsOff();
    resetHallStats();
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

  if (cmd == "RESET_HALL") {
    lastCommandMs = millis();
    resetHallStats();
    Serial.println("OK,RESET_HALL");
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

  // If stopped, accept the requested direction.
  if (currentPwm == 0) {
    currentSign = targetSign;
    digitalWrite(DIR_PIN2, (currentSign > 0) ? DIR_POSITIVE : DIR_NEGATIVE);
  }

  // If direction mismatch, ramp to zero first.
  if (currentSign != targetSign) {
    if (currentPwm > 0) {
      currentPwm -= PWM_SLEW_STEP;
      if (currentPwm < 0) currentPwm = 0;
    }
    analogWrite(PWM_PIN2, currentPwm);
    return;
  }

  // Slew current PWM toward target magnitude.
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

void updateRpmEstimate() {
  unsigned long nowUs = micros();

  unsigned long rawPeriodCopy;
  unsigned long validPeriodCopy;
  unsigned long lastValidPulseCopy;
  bool haveNewValid;

  noInterrupts();
  rawPeriodCopy = lastRawPeriodUs;
  validPeriodCopy = lastValidPeriodUs;
  lastValidPulseCopy = lastValidPulseUs;
  haveNewValid = newValidPulse;
  newValidPulse = false;
  interrupts();

  if (rawPeriodCopy > 0) {
    rpmRaw = 60000000.0f / (PULSES_PER_REV * (float)rawPeriodCopy);
    if (rpmRaw > 9999.0f) rpmRaw = 9999.0f;  // keep CSV readable; raw glitches will still be obvious
  } else {
    rpmRaw = 0.0f;
  }

  if (lastValidPulseCopy == 0 || (nowUs - lastValidPulseCopy > RPM_TIMEOUT_US)) {
    rpmValid = 0.0f;
  } else if (haveNewValid && validPeriodCopy > 0) {
    rpmValid = 60000000.0f / (PULSES_PER_REV * (float)validPeriodCopy);
    if (rpmValid > MAX_CREDIBLE_RPM) rpmValid = MAX_CREDIBLE_RPM;
  }

  // EMA filtering only on the valid RPM estimate.
  rpmFilt = (RPM_EMA_ALPHA * rpmValid) + ((1.0f - RPM_EMA_ALPHA) * rpmFilt);
}

void printDataLine() {
  unsigned long rawCntCopy;
  unsigned long validCntCopy;
  unsigned long rejectCntCopy;
  unsigned long rawPeriodCopy;
  unsigned long validPeriodCopy;
  unsigned long lastValidPulseCopy;

  noInterrupts();
  rawCntCopy = rawPulseCount;
  validCntCopy = validPulseCount;
  rejectCntCopy = rejectedPulseCount;
  rawPeriodCopy = lastRawPeriodUs;
  validPeriodCopy = lastValidPeriodUs;
  lastValidPulseCopy = lastValidPulseUs;
  interrupts();

  unsigned long nowUs = micros();
  float pulseAgeMs = -1.0f;
  if (lastValidPulseCopy > 0) {
    pulseAgeMs = (nowUs - lastValidPulseCopy) / 1000.0f;
  }

  Serial.print("DATA,");
  Serial.print(millis()); Serial.print(",");
  Serial.print(armed ? 1 : 0); Serial.print(",");
  Serial.print(targetSignedPwm); Serial.print(",");
  Serial.print(currentPwm); Serial.print(",");
  Serial.print(currentSign); Serial.print(",");
  Serial.print(softPwmLimit); Serial.print(",");
  Serial.print(millis() - lastCommandMs); Serial.print(",");
  Serial.print(rawCntCopy); Serial.print(",");
  Serial.print(validCntCopy); Serial.print(",");
  Serial.print(rejectCntCopy); Serial.print(",");
  Serial.print(rawPeriodCopy); Serial.print(",");
  Serial.print(validPeriodCopy); Serial.print(",");
  Serial.print(rpmRaw, 3); Serial.print(",");
  Serial.print(rpmValid, 3); Serial.print(",");
  Serial.print(rpmFilt, 3); Serial.print(",");
  Serial.print(pulseAgeMs, 1); Serial.print(",");
  Serial.println(digitalRead(HALL_PIN2));
}

void onM1HallPulse() {
  unsigned long t = micros();

  rawPulseCount++;

  unsigned long rawPeriod = 0;
  if (lastRawPulseUs > 0) {
    rawPeriod = t - lastRawPulseUs;
    lastRawPeriodUs = rawPeriod;
  }
  lastRawPulseUs = t;

  // First pulse cannot create an RPM period yet.
  if (rawPeriod == 0) return;

  // Reject physically impossible fast pulses. This does NOT affect motor control;
  // it only classifies the Hall measurement as valid/noisy.
  if (rawPeriod < MIN_VALID_PERIOD_US) {
    rejectedPulseCount++;
    return;
  }

  validPulseCount++;
  lastValidPeriodUs = rawPeriod;
  lastValidPulseUs = t;
  newValidPulse = true;
}

/* =============================================================
   combined_controller_window_rpm.ino
   4-motor velocity controller + non-blocking ultrasonic.

   IMPORTANT UPDATE
   ----------------
   This version replaces the old single-pulse-period RPM feedback
   with a pulse-count-window RPM estimator.

   Why:
     Open-loop M1 tests showed that the motor/driver path was OK,
     but the old period-based RPM signal had spikes/dropouts. The
     window estimator produced much cleaner feedback.

   Control design in this version:
     1. Hall ISR only accepts/counts pulses.
     2. Every 200 ms, RPM is computed from pulse count delta.
     3. The window RPM is EMA-filtered.
     4. PID uses filtered window RPM only.
     5. PID period is slowed to 50 ms for safer first closed-loop tests.
     6. Kd is set to zero initially.
     7. Serial commands are ramp-limited internally before entering PID.
     8. Gains are reduced after first 30/50/70 rpm test showed overshoot.

   SERIAL OUTPUT FORMAT kept compatible:
       rpm0,rpm1,rpm2,rpm3,err0,err1,err2,err3,out0,out1,out2,out3
       S:d0,d1,d2,d3,d4

   SERIAL INPUT FORMAT kept compatible:
       SP#+RRR#+RRR#+RRR#+RRR\n
   Test first with wheels lifted and small setpoints.
   ============================================================= */

/* ==================== PID / RPM CONFIG ==================== */
const unsigned long PID_PERIOD_US = 50000;     // 50 ms
const unsigned long RPM_STALE_TIMEOUT_US = 500000; // 500 ms

// Adaptive RPM window:
// At low setpoints the 200 ms window only captures 0 or 1 pulse per window,
// which gives quantised RPM steps of 6.67 RPM and makes the PWM chatter.
// Using a longer window at low speed recovers measurement resolution:
//   200 ms → 6.67 RPM/pulse    (good above ~20 RPM)
//   500 ms → 2.67 RPM/pulse    (good from ~8–20 RPM)
//  1000 ms → 1.33 RPM/pulse    (good below ~8 RPM)
const float  LOW_SPEED_THRESHOLD_1 = 20.0f;   // RPM below which window extends to 500 ms
const float  LOW_SPEED_THRESHOLD_2 = 8.0f;    // RPM below which window extends to 1000 ms
const unsigned long RPM_WINDOW_FAST   = 200000;  // 200 ms  – normal speed
const unsigned long RPM_WINDOW_MED    = 500000;  // 500 ms  – low speed
const unsigned long RPM_WINDOW_SLOW   = 1000000; // 1000 ms – very low speed

// Internal command ramp. This prevents a 0 -> 70 rpm serial command from
// hitting the PID as an instantaneous step. It is the main protection
// against the overshoot seen in the first window-RPM PID test.
const float MAX_RPM_RATE = 40.0f;               // rpm/s

const float pulsesPerRev = 45.0f;

const float MAX_RPM = 200.0f;
const unsigned long MIN_PULSE_PERIOD = (unsigned long)(60e6f / (pulsesPerRev * MAX_RPM));
const unsigned long PULSE_RATIO_GUARD = 3;

// rpm_filt = alpha*old + (1-alpha)*window_rpm
const float RPM_FILTER_ALPHA = 0.65f;
const float RPM_NO_PULSE_DECAY = 0.85f;

const float integralMax = 150.0f; // Tightened from 300 – limits windup headroom
const int   PWM_MAX     = 200;
const int   DZ_KICK_PWM = 12;

// Anti-windup guard: if the error is larger than this threshold, the motor
// has not started moving yet (still in dead-zone during ramp). Suspend
// integral accumulation to prevent the integral from loading up before
// the motor breaks free. Once moving, error stays well below this value.
const float ANTIWINDUP_ERROR_LIMIT = 15.0f;  // RPM

// PID period is 50 ms, so PWM_SLEW=10 means 0→200 PWM in about 1.0 s.
const int   PWM_SLEW    = 10;

/* ==================== PID PINS ==================== */
const int HALL_PIN1 = 2;  const int PWM_PIN1 = 5;  const int DIR_PIN1 = 34;
const int HALL_PIN2 = 3;  const int PWM_PIN2 = 6;  const int DIR_PIN2 = 35;
const int HALL_PIN3 = 18; const int PWM_PIN3 = 44; const int DIR_PIN3 = 36;
const int HALL_PIN4 = 19; const int PWM_PIN4 = 45; const int DIR_PIN4 = 37;

/* ==================== ULTRASONIC PINS ==================== */
// Sensor order: Right Front, Left Front, Left Side, Right Side, Rear
const int TRIG_PINS[5] = {22, 24, 26, 28, 30};
const int ECHO_PINS[5] = {23, 25, 27, 29, 31};

/* ==================== PID / RPM RUNTIME ==================== */
volatile unsigned long pulseCount[4]        = {0, 0, 0, 0};
volatile unsigned long rejectedPulseCount[4]= {0, 0, 0, 0};
volatile unsigned long lastAcceptedTime[4]  = {0, 0, 0, 0};
volatile unsigned long lastPulsePeriod[4]   = {0, 0, 0, 0};
volatile unsigned long lastPulseTime[4]     = {0, 0, 0, 0};

unsigned long prevPulseCount[4] = {0, 0, 0, 0};
unsigned long prevRejectedPulseCount[4] = {0, 0, 0, 0};

// PID feedback RPM = filtered pulse-count-window RPM magnitude.
float rpm[4]       = {0, 0, 0, 0};
float rpmWindow[4] = {0, 0, 0, 0};

// cmdSetpoint is the direct serial command.
// setpoint is the internally ramp-limited setpoint used by the PID.
float cmdSetpoint[4] = {0, 0, 0, 0};
float setpoint[4]    = {0, 0, 0, 0};
float error[4]       = {0, 0, 0, 0};
float lastError[4] = {0, 0, 0, 0};
float integral[4]  = {0, 0, 0, 0};
float out[4]       = {0, 0, 0, 0};

int lastSpSign[4]  = {0, 0, 0, 0};
int prevPwm[4]     = {0, 0, 0, 0};

/* ==================== PID GAINS ==================== */
// Kd remains zero. Add derivative only after clean step tests.
// M2 Ki reduced from 0.25 → 0.10 to prevent windup overshoot.
// M2 Kff reduced from 0.38 → 0.33 (motor characteristic scatter shows
//   it needs less feed-forward than M3).
float Kp[4]  = {0.35,  0.30,  0.35,  0.35};
float Ki[4]  = {0.25,  0.25,  0.18,  0.25};  // M2 halved
float Kd[4]  = {0.0,   0.0,   0.0,   0.0};
float Kff[4] = {0.40,  0.45,  0.36,  0.38};  // M2 Kff pulled down

/* ==================== LOOP TIMING ==================== */
unsigned long lastPidLoop   = 0;
unsigned long lastRpmWindow = 0;
unsigned long lastPrint     = 0;
const unsigned long printPeriod = 100000; // 100 ms

// Active RPM window duration - updated each PID tick based on max |setpoint|.
unsigned long activeRpmWindow = RPM_WINDOW_FAST;

String inputString = "";

/* ==================== ULTRASONIC STATE MACHINE ==================== */
enum UsPhase {
  US_SETTLE,
  US_TRIG_HIGH,
  US_TRIG_LOW,
  US_WAIT_HIGH,
  US_WAIT_LOW,
  US_DONE
};

UsPhase      usPhase       = US_SETTLE;
int          usSensor      = 0;
unsigned long usPhaseStart = 0;
unsigned long usEchoStart  = 0;

float distances[5] = {-1, -1, -1, -1, -1};

unsigned long lastUsPrint = 0;
const unsigned long US_PRINT_PERIOD = 200000; // 200 ms

const unsigned long US_ECHO_WAIT_TIMEOUT  = 5000;
const unsigned long US_ECHO_PULSE_TIMEOUT = 38000;

/* ==================== SETUP ==================== */
void setup() {
  int hallPins[4] = {HALL_PIN1, HALL_PIN2, HALL_PIN3, HALL_PIN4};
  int pwmPins[4]  = {PWM_PIN1,  PWM_PIN2,  PWM_PIN3,  PWM_PIN4};
  int dirPins[4]  = {DIR_PIN1,  DIR_PIN2,  DIR_PIN3,  DIR_PIN4};

  for (int i = 0; i < 4; i++) {
    pinMode(hallPins[i], INPUT_PULLUP);
    pinMode(pwmPins[i],  OUTPUT);
    pinMode(dirPins[i],  OUTPUT);
    analogWrite(pwmPins[i], 0);
  }

  for (int i = 0; i < 5; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
    digitalWrite(TRIG_PINS[i], LOW);
  }

  attachInterrupt(digitalPinToInterrupt(HALL_PIN1), [](){onPulse(0);}, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN2), [](){onPulse(1);}, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN3), [](){onPulse(2);}, RISING);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN4), [](){onPulse(3);}, RISING);

  Serial.begin(115200);
  unsigned long now = micros();
  lastPidLoop   = now;
  lastRpmWindow = now;
  usPhaseStart  = now;
}


/* ==================== SMALL HELPERS ==================== */
float approachFloat(float current, float target, float maxStep) {
  if (current < target) {
    current += maxStep;
    if (current > target) current = target;
  } else if (current > target) {
    current -= maxStep;
    if (current < target) current = target;
  }
  return current;
}

/* ==================== MAIN LOOP ==================== */
void loop() {

  /* --- Serial command parsing --- */
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputString.length()) parseCommand(inputString);
      inputString = "";
    } else {
      inputString += c;
    }
  }

  updateUltrasonic();

  unsigned long now = micros();

  /* --- Window RPM estimator (adaptive window) ---
   * The active window length is chosen by the PID block (see below) based
   * on the current max |setpoint|. A longer window at low speed gives finer
   * pulse-count resolution and eliminates the PWM chatter seen at 5–10 RPM.
   */
  if (now - lastRpmWindow >= activeRpmWindow) {
    unsigned long dtUs = now - lastRpmWindow;
    lastRpmWindow = now;

    unsigned long cnt[4];
    unsigned long rej[4];
    unsigned long lastPulse[4];

    noInterrupts();
    for (int i = 0; i < 4; i++) {
      cnt[i]       = pulseCount[i];
      rej[i]       = rejectedPulseCount[i];
      lastPulse[i] = lastPulseTime[i];
    }
    interrupts();

    for (int i = 0; i < 4; i++) {
      unsigned long dCnt = cnt[i] - prevPulseCount[i];
      prevPulseCount[i] = cnt[i];
      prevRejectedPulseCount[i] = rej[i];

      bool stale = (lastPulse[i] == 0) || (now - lastPulse[i] > RPM_STALE_TIMEOUT_US);

      if (stale) {
        rpmWindow[i] = 0.0f;
        rpm[i] = 0.0f;
      } else {
        rpmWindow[i] = ((float)dCnt) * 60e6f / (pulsesPerRev * (float)dtUs);
        rpmWindow[i] = min(rpmWindow[i], MAX_RPM);

        if (dCnt == 0) {
          // Short pulse gap, not stale yet. Decay gently; do not snap to zero.
          rpm[i] *= RPM_NO_PULSE_DECAY;
        } else if (rpm[i] <= 0.1f) {
          rpm[i] = rpmWindow[i];
        } else {
          rpm[i] = RPM_FILTER_ALPHA * rpm[i] + (1.0f - RPM_FILTER_ALPHA) * rpmWindow[i];
        }
      }
    }
  }

  /* --- PID at fixed slower period --- */
  if (now - lastPidLoop >= PID_PERIOD_US) {
    unsigned long pidDtUs = now - lastPidLoop;
    lastPidLoop = now;
    float dt = (float)pidDtUs / 1e6f;
    if (dt <= 0.0f || dt > 0.2f) dt = PID_PERIOD_US / 1e6f;

    // Ramp external serial command toward internal PID setpoint.
    float maxSetpointStep = MAX_RPM_RATE * dt;
    float maxAbsSp = 0.0f;
    for (int i = 0; i < 4; i++) {
      setpoint[i] = approachFloat(setpoint[i], cmdSetpoint[i], maxSetpointStep);
      if (fabs(setpoint[i]) < 0.05f && fabs(cmdSetpoint[i]) < 0.05f) {
        setpoint[i] = 0.0f;
      }
      if (fabs(setpoint[i]) > maxAbsSp) maxAbsSp = fabs(setpoint[i]);
    }

    // ---- Adaptive RPM window selection ----
    // Choose window based on the highest active setpoint across all motors.
    // This keeps all four motors on the same window boundary, avoiding
    // conflicting window durations. The window only shrinks once all motors
    // are running fast; it grows immediately if any motor drops to low speed.
    if (maxAbsSp < LOW_SPEED_THRESHOLD_2) {
      activeRpmWindow = RPM_WINDOW_SLOW;   // 1000 ms: 1.33 RPM/pulse
    } else if (maxAbsSp < LOW_SPEED_THRESHOLD_1) {
      activeRpmWindow = RPM_WINDOW_MED;    // 500 ms:  2.67 RPM/pulse
    } else {
      activeRpmWindow = RPM_WINDOW_FAST;   // 200 ms:  6.67 RPM/pulse
    }

    for (int i = 0; i < 4; i++) {
      int spSign = (setpoint[i] > 0.05f) ? 1 : (setpoint[i] < -0.05f) ? -1 : 0;

      int pwmPin = (i==0) ? PWM_PIN1 : (i==1) ? PWM_PIN2 : (i==2) ? PWM_PIN3 : PWM_PIN4;
      int dirPin = (i==0) ? DIR_PIN1 : (i==1) ? DIR_PIN2 : (i==2) ? DIR_PIN3 : DIR_PIN4;

      if (spSign == 0) {
        integral[i] = 0.0f;
        lastError[i] = 0.0f;
        error[i] = 0.0f;
        out[i] = 0.0f;
        prevPwm[i] = 0;
        lastSpSign[i] = 0;
        analogWrite(pwmPin, 0);
        continue;
      }

      if (spSign != lastSpSign[i]) {
        integral[i] = 0.0f;
        lastError[i] = 0.0f;
        error[i] = 0.0f;
        out[i] = 0.0f;
        prevPwm[i] = 0;
        analogWrite(pwmPin, 0);
        lastSpSign[i] = spSign;
        continue;
      }
      lastSpSign[i] = spSign;

      float signedRpm = (float)spSign * rpm[i];
      error[i] = setpoint[i] - signedRpm;

      float dErr = (error[i] - lastError[i]) / dt;
      lastError[i] = error[i];

      float ff = Kff[i] * setpoint[i];
      float out_unsat = ff + Kp[i]*error[i] + Ki[i]*integral[i] + Kd[i]*dErr;
      float effort = (float)spSign * out_unsat;

      digitalWrite(dirPin, (spSign > 0) ? LOW : HIGH);

      if (effort >= 0.0f) {
        // Anti-windup guard: if |error| is very large the motor has not
        // started moving yet (dead-zone during ramp). Suspend integral
        // accumulation to prevent loading up before motor breaks free.
        // Normal tracking error stays well within ANTIWINDUP_ERROR_LIMIT.
        bool motorMoving = (fabs(error[i]) < ANTIWINDUP_ERROR_LIMIT);
        if (abs(out_unsat) < PWM_MAX && motorMoving) {
          integral[i] += error[i] * dt;
          integral[i] = constrain(integral[i], -integralMax, integralMax);
          out[i] = out_unsat;
        } else if (!motorMoving) {
          // Motor not moving yet: clamp output but don't accumulate integral.
          out[i] = constrain(out_unsat, -PWM_MAX, PWM_MAX);
        } else {
          // Saturated: clamp without accumulation (standard anti-windup).
          out[i] = constrain(out_unsat, -PWM_MAX, PWM_MAX);
        }

        int pwm_cmd = (int)abs(out[i]);
        pwm_cmd = max(pwm_cmd, DZ_KICK_PWM);
        pwm_cmd = min(pwm_cmd, PWM_MAX);
        pwm_cmd = constrain(pwm_cmd, prevPwm[i] - PWM_SLEW, prevPwm[i] + PWM_SLEW);
        prevPwm[i] = pwm_cmd;
        analogWrite(pwmPin, pwm_cmd);

      } else {
        // Direction-blind Hall feedback: coast on overshoot; do not reverse-brake.
        // Always accumulate integral during coast so it can pull the output back up.
        integral[i] += error[i] * dt;
        integral[i] = constrain(integral[i], -integralMax, integralMax);
        out[i] = 0.0f;
        prevPwm[i] = 0;
        analogWrite(pwmPin, 0);
      }
    }
  }

  /* --- Motor telemetry at 100 ms --- */
  if (micros() - lastPrint >= printPeriod) {
    lastPrint = micros();
    for (int i = 0; i < 4; i++) { Serial.print(rpm[i]);   Serial.print(","); }
    for (int i = 0; i < 4; i++) { Serial.print(error[i]); Serial.print(","); }
    for (int i = 0; i < 4; i++) {
      Serial.print(out[i]);
      if (i < 3) Serial.print(",");
    }
    Serial.println();
  }

  /* --- Ultrasonic print at 200 ms --- */
  if (micros() - lastUsPrint >= US_PRINT_PERIOD) {
    lastUsPrint = micros();
    Serial.print("S:");
    for (int i = 0; i < 5; i++) {
      Serial.print(distances[i], 1);
      if (i < 4) Serial.print(",");
    }
    Serial.println();
  }
}
/* ==================== ULTRASONIC STATE MACHINE ==================== */
void updateUltrasonic() {
  unsigned long now = micros();

  switch (usPhase) {

    case US_SETTLE:
      // Pull TRIG low; wait 2 µs to ensure a clean pulse
      digitalWrite(TRIG_PINS[usSensor], LOW);
      if (now - usPhaseStart >= 2) {
        usPhase      = US_TRIG_HIGH;
        usPhaseStart = now;
        digitalWrite(TRIG_PINS[usSensor], HIGH);
      }
      break;

    case US_TRIG_HIGH:
      // Hold TRIG high for 10 µs
      if (now - usPhaseStart >= 10) {
        digitalWrite(TRIG_PINS[usSensor], LOW);
        usPhase      = US_WAIT_HIGH;
        usPhaseStart = now;
      }
      break;

    case US_WAIT_HIGH:
      // Wait for ECHO to go HIGH (sensor starts echo pulse)
      if (digitalRead(ECHO_PINS[usSensor]) == HIGH) {
        usEchoStart  = now;
        usPhase      = US_WAIT_LOW;
      } else if (now - usPhaseStart > US_ECHO_WAIT_TIMEOUT) {
        // No response → mark as timeout
        distances[usSensor] = -1.0;
        usPhase = US_DONE;
      }
      break;

    case US_WAIT_LOW:
      // Wait for ECHO to go LOW (end of echo pulse)
      if (digitalRead(ECHO_PINS[usSensor]) == LOW) {
        unsigned long echoWidth = now - usEchoStart;
        distances[usSensor] = echoWidth / 58.0; // µs → cm
        usPhase = US_DONE;
      } else if (now - usEchoStart > US_ECHO_PULSE_TIMEOUT) {
        // Object too far / out of range
        distances[usSensor] = -1.0;
        usPhase = US_DONE;
      }
      break;

    case US_DONE:
      // Advance to next sensor
      usSensor     = (usSensor + 1) % 5;
      usPhase      = US_SETTLE;
      usPhaseStart = now;
      break;
  }
}


/* ==================== HALL ISR ==================== */
void onPulse(int i) {
  unsigned long t = micros();

  if (lastAcceptedTime[i] != 0) {
    unsigned long p = t - lastAcceptedTime[i];

    if (p < MIN_PULSE_PERIOD) {
      rejectedPulseCount[i]++;
      return;
    }

    if (lastPulsePeriod[i] > 0 && p < lastPulsePeriod[i] / PULSE_RATIO_GUARD) {
      rejectedPulseCount[i]++;
      return;
    }

    lastPulsePeriod[i] = p;
  }

  lastAcceptedTime[i] = t;
  lastPulseTime[i] = t;
  pulseCount[i]++;
}

/* ==================== COMMAND PARSER ==================== */

void parseCommand(String cmd) {
  if (!cmd.startsWith("SP#") || cmd.length() != 22) {
    Serial.println("ERR");
    return;
  }

  // Write serial command into cmdSetpoint only.
  // The PID uses setpoint[], which is ramp-limited toward cmdSetpoint[].
  cmdSetpoint[0] = (cmd.charAt(3)=='+')  ?  cmd.substring(4,7).toInt()
                                         : -cmd.substring(4,7).toInt();
  cmdSetpoint[1] = (cmd.charAt(8)=='+')  ?  cmd.substring(9,12).toInt()
                                         : -cmd.substring(9,12).toInt();
  cmdSetpoint[2] = (cmd.charAt(13)=='+') ?  cmd.substring(14,17).toInt()
                                         : -cmd.substring(14,17).toInt();
  cmdSetpoint[3] = (cmd.charAt(18)=='+') ?  cmd.substring(19,22).toInt()
                                         : -cmd.substring(19,22).toInt();
}

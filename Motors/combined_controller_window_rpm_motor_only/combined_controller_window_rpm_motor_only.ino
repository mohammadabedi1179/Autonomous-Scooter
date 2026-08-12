/*
     Motor-only keyboard test version:
     - Ultrasonic reading/printing disabled with ENABLE_ULTRASONIC=false.
     - Wheel sign convention is handled in Python keyboard script.
 =============================================================
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
const unsigned long PID_PERIOD_US       = 50000;   // 50 ms
// BUG FIX: with pulsesPerRev=45, the natural inter-pulse period at low RPM
// exceeds the OLD 500ms timeout: period_ms = 60000/(RPM*45), so at 2.67 RPM
// the true gap between pulses is already 500ms, and below that speed the
// gap is LONGER than the timeout -- meaning a genuinely-still-turning wheel
// gets falsely zeroed between every pulse. 2000ms gives margin down to
// ~0.67 RPM before false-staling, comfortably below any real operating
// speed, at the cost of taking longer to correctly report an ACTUAL stop
// (up to ~2s instead of ~0.5s) -- an acceptable tradeoff since a stopped
// wheel reporting nonzero for an extra second is far less harmful than a
// slowly-turning wheel intermittently reporting zero.
const unsigned long RPM_STALE_TIMEOUT_US = 2000000; // 2000 ms

// Motor-only testing: ultrasonic reading/printing disabled.
const bool ENABLE_ULTRASONIC = false;

// Adaptive RPM window — fixes PWM chatter at 5-10 RPM.
// At low setpoints the fixed 200 ms window only captures 0-1 pulse, giving
// quantised RPM steps of 6.67 RPM that make the PWM chatter.
// Longer windows recover resolution:
//   200 ms → 6.67 RPM/pulse  (good above ~20 RPM)
//   500 ms → 2.67 RPM/pulse  (good  8-20 RPM)
//  1000 ms → 1.33 RPM/pulse  (good below 8 RPM)
const float        LOW_SPEED_THRESHOLD_1 = 20.0f;    // RPM: below → 500 ms
const float        LOW_SPEED_THRESHOLD_2 =  8.0f;    // RPM: below → 1000 ms
const unsigned long RPM_WINDOW_FAST      = 200000UL; // 200 ms
const unsigned long RPM_WINDOW_MED       = 500000UL; // 500 ms
const unsigned long RPM_WINDOW_SLOW      = 1000000UL;// 1000 ms

// Internal command ramp — prevents 0→70 RPM hitting PID as a true step.
const float MAX_RPM_RATE = 40.0f;   // RPM/s

const float pulsesPerRev = 45.0f;

const float MAX_RPM = 200.0f;
const unsigned long MIN_PULSE_PERIOD = (unsigned long)(60e6f / (pulsesPerRev * MAX_RPM));
const unsigned long PULSE_RATIO_GUARD = 3;

// rpm_filt = alpha*old + (1-alpha)*window_rpm
const float RPM_FILTER_ALPHA   = 0.65f;
const float RPM_NO_PULSE_DECAY = 0.85f;

// Anti-windup guard: while |error| > this threshold the motor is still in
// its dead-zone during ramp-up. Suspend integral accumulation to prevent
// the integral loading up before the motor breaks free. Normal tracking
// error stays well below this value once moving.
const float ANTIWINDUP_ERROR_LIMIT = 15.0f;  // RPM

const float integralMax = 150.0f; // tightened from 300 — limits windup headroom
const int   PWM_MAX     = 200;

// TEMP STOPGAP (see hand-spin test findings): M1's gearbox/bearing has
// noticeably more internal friction than M0/M2/M3 when spun by hand with
// power off. This is a mechanical issue -- the real fix is servicing M1 --
// but until that's possible, a higher dead-zone "kick" floor for M1 only
// gives it more initial PWM to break static friction, without touching
// Kp/Ki (which were already tuned DOWN for M1 to fix an earlier overshoot
// problem -- raising them back up risks reintroducing that overshoot).
// Start at ~1.5x default and increase in small steps (2-3 at a time) while
// testing; revert to {12,12,12,12} once M1 is serviced.
const int   DZ_KICK_PWM[4] = {12, 18, 12, 12};

// PWM_SLEW=10 at 50 ms PID period → 0→200 PWM in ~1.0 s
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
// First safe closed-loop pass after switching to filtered window RPM.
// Kd is intentionally zero. Add derivative only after clean tests.
// Conservative first gains for filtered window-RPM feedback.
 // The previous gains were too aggressive with the slower 200 ms RPM estimator.
 // M1 was specifically reduced after the 30/50/70 rpm test showed large overshoot.
float Kp[4]  = {0.35,  0.30,  0.35,  0.35};
float Ki[4]  = {0.9,  1.0,  1.0,  1.0};  // M2 halved
float Kd[4]  = {0.0,   0.0,   0.0,   0.0};
float Kff[4] = {0.40,  0.45,  0.36,  0.38};  // M2 Kff pulled down

/* ==================== LOOP TIMING ==================== */
unsigned long lastPidLoop   = 0;
unsigned long lastRpmWindow = 0;
unsigned long lastPrint     = 0;
const unsigned long printPeriod = 100000; // 100 ms

// Starts at fast; PID block updates it every tick based on max |setpoint|.
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
  }  if (ENABLE_ULTRASONIC) {


  for (int i = 0; i < 5; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
    digitalWrite(TRIG_PINS[i], LOW);
  }
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

  if (ENABLE_ULTRASONIC) updateUltrasonic();

  unsigned long now = micros();

  /* --- Window RPM estimator (adaptive window) --- */
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

    // Adaptive RPM window: choose integration time based on max active setpoint.
    // All motors share one window boundary to stay synchronised.
    if      (maxAbsSp < LOW_SPEED_THRESHOLD_2) activeRpmWindow = RPM_WINDOW_SLOW;
    else if (maxAbsSp < LOW_SPEED_THRESHOLD_1) activeRpmWindow = RPM_WINDOW_MED;
    else                                        activeRpmWindow = RPM_WINDOW_FAST;

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
        // BUG FIX: the RPM window/stale-timeout logic above runs on a
        // shared activeRpmWindow that jumps to its SLOWEST setting
        // (1000ms) the instant every motor's setpoint hits 0 -- meaning
        // the reported RPM for a stopped motor could stay frozen at its
        // last (nonzero) value for up to ~1-2s after the command dropped
        // to zero, since nothing here previously told the RPM measurement
        // that we already know this motor is commanded to stop. That
        // stale nonzero RPM fed straight into Python-side velocity and
        // got integrated into the path every time, across every stop
        // event in a session -- a real source of path overestimation.
        // We authoritatively know PWM=0 here, so zero the reported RPM
        // immediately rather than waiting for the window/stale-timeout
        // to catch up; any residual coast-down pulses aren't needed for
        // control purposes.
        rpm[i] = 0.0f;
        rpmWindow[i] = 0.0f;
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
        // Anti-windup guard: large error means motor is still in dead-zone
        // during ramp-up — suspend integral until the motor is actually moving.
        bool motorMoving = (fabs(error[i]) < ANTIWINDUP_ERROR_LIMIT);
        if (motorMoving && abs(out_unsat) < PWM_MAX) {
          // Normal tracking — accumulate integral
          integral[i] += error[i] * dt;
          integral[i] = constrain(integral[i], -integralMax, integralMax);
          out[i] = out_unsat;
        } else {
          // Saturated or dead-zone: clamp output, do not accumulate integral
          out[i] = constrain(out_unsat, -PWM_MAX, PWM_MAX);
        }

        // Speed-proportional kick floor.
        // DZ_KICK_PWM[i] (12 default, 18 for M1) is correct for stiction at
        // moderate speeds but becomes a limit-cycle trigger at low setpoints:
        // at 5 RPM the correct steady-state PWM is ~2-4, so forcing the full
        // kick value causes the motor to
        // overshoot → coast → overshoot indefinitely.
        // Below LOW_KICK_THRESHOLD_RPM, scale the floor down with the setpoint
        // so the floor never exceeds the expected FF output.
        const float LOW_KICK_THRESHOLD_RPM = 15.0f;
        int kick_floor;
        if (fabs(setpoint[i]) < LOW_KICK_THRESHOLD_RPM) {
          // Scale: FF output at this setpoint, clamped 4..DZ_KICK_PWM[i]
          kick_floor = (int)constrain(Kff[i] * fabs(setpoint[i]),
                                      4.0f, (float)DZ_KICK_PWM[i]);
        } else {
          kick_floor = DZ_KICK_PWM[i];
        }

        int pwm_cmd = (int)abs(out[i]);
        pwm_cmd = max(pwm_cmd, kick_floor);
        pwm_cmd = min(pwm_cmd, PWM_MAX);
        pwm_cmd = constrain(pwm_cmd, prevPwm[i] - PWM_SLEW, prevPwm[i] + PWM_SLEW);
        prevPwm[i] = pwm_cmd;
        analogWrite(pwmPin, pwm_cmd);

      } else {
        // Coast on overshoot — always let integral accumulate so it
        // can pull output back up after the coast phase.
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

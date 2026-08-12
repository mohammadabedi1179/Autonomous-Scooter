/* =============================================================
   combined_controller.ino
   PID + Feed-Forward velocity controller (4 motors)
   + non-blocking ultrasonic (5 HC-SR04 sensors).

   CHANGES vs previous versions
   ----------------------------
   v1 → v2 (signed ff + coast-on-overshoot):
     • ff uses setpoint[i] not abs(setpoint[i]) so direction pin
       always follows setpoint sign — fixed 350 RPM runaway.
     • Coast-on-overshoot: when effort = spSign × out_unsat < 0
       apply PWM=0 instead of reversing.  Safe with direction-blind
       hall sensors.
     • Direction pin = (spSign>0)?LOW:HIGH  always.
     • DZ_KICK only when effort ≥ 0.

   v2 → v3 (damping the coast oscillation):
     • Kd reduced: 0.010 → 0.003 (eliminated derivative spike).
     • Kff reduced per-motor from scatter-plot centroids:
         M0: 0.50→0.40  M1: 0.50→0.42  M2: 0.46→0.38  M3: 0.46→0.38
     • Integral NOW accumulates during coast so the oscillation
       converges instead of sustaining indefinitely.

   v3 → v4 (first-pass hall-sensor noise protection):
     • Fixed ISR debounce: reject pulses where p < MIN_PULSE_PERIOD.
     • RPM sanity cap: min(raw_rpm, MAX_RPM = 200).
     • PWM slew-rate limiter: ±20 PWM per 5 ms cycle.

   v4 → v5 (this version — stronger adaptive noise rejection):
     v4's fixed ISR debounce (6,667 µs floor) only catches glitches
     arriving within 6.7 ms of the last real pulse.  At 60 RPM real
     pulses are 22 ms apart, so a glitch at t = 8 ms slips through
     and computes 167 RPM — enough to trigger a coast cycle and tick.

     Layer 1b — Adaptive ISR debounce (new):
       if (period[i] > 0 && p < period[i] / 3) return;
       Rejects any pulse implying a 3× speed jump — physically
       impossible given motor inertia.  Scales with actual speed:
         60 RPM → rejects glitches computing ≥ 180 RPM
         30 RPM → rejects glitches computing ≥  90 RPM

     Layer 2b — Rate-of-change filter in main loop (new):
       if (rpm[i] > 5 && raw_rpm > rpm[i] + 40) → discard pulse
       Final safety net for any spike that passes both ISR stages.

       Even if a spike passes layers 1 & 2, the controller cannot
       make a sudden large current step — no jerk, no audible tick.
       Also smooths the step-response rise (0→200 PWM takes 50 ms).

     • Kd reduced: 0.010 → 0.003.
       Old Kd caused a ~160 PWM derivative spike on the first PID
       cycle of every step (dErr = 80/0.005 = 16000 RPM/s) which
       saturated the output and caused large initial overshoot.
     • Kff reduced per-motor from scatter-plot centroid readings:
         M0: 0.50 → 0.40   M1: 0.50 → 0.42
         M2: 0.46 → 0.38   M3: 0.46 → 0.38
       Lower Kff reduces the coast threshold (how much above
       setpoint before the controller coasts).
     • Integral NOW accumulates during coast.
       Without this: integral is frozen during every overshoot,
       so each drive cycle produces the same overshoot →
       sustained ±10–20 RPM oscillation forever.
       With this: negative error during coast winds integral down;
       after a few cycles the integral offsets any ff calibration
       error and steady-state oscillation disappears.

   SERIAL OUTPUT FORMAT (unchanged)
   --------------------
   Motor telemetry (every 100 ms):
       rpm0,rpm1,rpm2,rpm3,err0,err1,err2,err3,out0,out1,out2,out3
   Ultrasonic distances (every 200 ms):
       S:d0,d1,d2,d3,d4      (cm, -1 = timeout / no object)

   SERIAL INPUT FORMAT (unchanged)
   --------------------------------
   SP#+RRR#+RRR#+RRR#+RRR\n   (22 chars)
   ============================================================= */

/* ==================== PID CONFIG ==================== */
const unsigned long Ts           = 5000;    // PID period (µs)
const unsigned long pulseTimeout = 200000;  // RPM → 0 if no pulse for this long
const float         pulsesPerRev = 45.0;

// ── Hall-sensor noise protection (layer 1 + 2) ──────────────
// Maximum credible motor speed.  RPM values above this are the
// result of a spurious glitch pulse, not real motion.
const float         MAX_RPM          = 200.0f;

// Minimum time between two valid hall pulses (µs) — fixed floor.
// Catches fast glitches (PWM switching noise at ~490 Hz / 2 ms).
//   MIN_PULSE_PERIOD = 60e6 / (pulsesPerRev × MAX_RPM) = 6,667 µs
const unsigned long MIN_PULSE_PERIOD = (unsigned long)(60e6f / (pulsesPerRev * MAX_RPM));

// Adaptive ISR debounce ratio.
// Also rejects a pulse if it arrives faster than 1/PULSE_RATIO_GUARD
// of the previous inter-pulse period.  Motor inertia cannot triple
// the rotational speed between two adjacent pulses — so any apparent
// speed jump of 3× or more is electrical noise.
// At 60 RPM (period = 22,222 µs): rejects glitches at t < 7,407 µs
// (would imply ≥ 180 RPM).  Works at any speed, unlike the fixed floor.
const unsigned long PULSE_RATIO_GUARD = 3;  // reject if p < period/3

// Rate-of-change RPM plausibility cap (main loop, layer 2 supplement).
// Even a glitch that passes both ISR debounce stages cannot cause an
// RPM jump larger than this per measurement cycle.
// Physics: at full PWM the motor accelerates ~300 RPM/s; in one
// 22 ms inter-pulse interval that's ≈ 7 RPM.  40 RPM gives ample
// margin for genuine dynamics while rejecting noise.
const float         RPM_JUMP_MAX     = 40.0f;

// Tighter clamp: feed-forward carries steady-state load;
// integrator only trims small residuals.
const float integralMax  = 300.0;
const int   PWM_MAX      = 200;

// Minimum PWM applied whenever setpoint != 0.
// Overcomes gearbox stiction (~9 RPM threshold from dead-zone probe).
const int   DZ_KICK_PWM  = 12;

// ── PWM slew-rate limiter (layer 3) ─────────────────────────
// Maximum PWM change per 5 ms PID cycle.
// Even if a glitch pulse slips past layers 1 & 2, the controller
// cannot make a step change in motor current large enough to cause
// an audible jerk or couple noise back into the hall sensor.
const int   PWM_SLEW     = 20;

/* ==================== PID PINS ==================== */
const int HALL_PIN1 = 2;  const int PWM_PIN1 = 5;  const int DIR_PIN1 = 34;
const int HALL_PIN2 = 3;  const int PWM_PIN2 = 6;  const int DIR_PIN2 = 35;
const int HALL_PIN3 = 18; const int PWM_PIN3 = 44; const int DIR_PIN3 = 36;
const int HALL_PIN4 = 19; const int PWM_PIN4 = 45; const int DIR_PIN4 = 37;

/* ==================== ULTRASONIC PINS ==================== */
// Sensor order: Right Front, Left Front, Left Side, Right Side, Rear
const int TRIG_PINS[5] = {22, 24, 26, 28, 30};
const int ECHO_PINS[5] = {23, 25, 27, 29, 31};

/* ==================== PID RUNTIME ==================== */
volatile unsigned long lastTime[4]      = {0};
volatile unsigned long period[4]        = {0};
volatile unsigned long lastPulseTime[4] = {0};
volatile bool          newPulse[4]      = {false};

float rpm[4]      = {0};
float setpoint[4] = {0};
float error[4]    = {0};
float lastError[4]= {0};
float integral[4] = {0};
int   lastSpSign[4]= {0};

// Previous PWM output — used by the slew-rate limiter.
// Reset to 0 on zero-setpoint or direction-change events.
int   prevPwm[4]  = {0};

/* ==================== PID GAINS ==================== */
// Kp: unchanged — proportional response is working well.
// Ki: unchanged — integral authority is appropriate.
// Kd: reduced 0.010 → 0.003.
//     The old value added ~160 PWM derivative spike on the very
//     first step cycle (dErr = 80RPM / 0.005s = 16000 RPM/s),
//     saturating the output at 200 and causing large initial
//     overshoot.  With coast-on-overshoot there is no longer a
//     need for heavy derivative damping; 0.003 gives a small
//     noise-filtering benefit without the spike.
float Kp[4]  = {1.20,  1.10,  1.10,  1.00};
float Ki[4]  = {1.20,  1.40,  1.40,  1.20};
float Kd[4]  = {0.003, 0.003, 0.003, 0.003};

// Feed-forward gains (PWM per RPM) — re-read from RPM-vs-PWM scatter.
// Previous values (0.50 / 0.46) were too high; the scatter centroid
// for the high-SP cluster puts the true slope at:
//   M0 Front Left  ≈ 0.40  (dark cluster ~70 RPM / 28 PWM)
//   M1 Front Right ≈ 0.42  (wider scatter; slightly higher)
//   M2 Back Left   ≈ 0.38  (dark cluster ~70 RPM / 26 PWM)
//   M3 Back Right  ≈ 0.38  (tightest cluster; slope clearest)
//
// Why this matters for coast-on-overshoot:
//   Coast kicks in when  Kff*sp + Kp*(sp - rpm) < 0
//   → rpm > sp * (1 + Kff/Kp)
//   Old Kff=0.50, Kp=1.20 → coast threshold = 142 % of setpoint
//   New Kff=0.40, Kp=1.20 → coast threshold = 133 % of setpoint
//   AND with integral unwinding during coast (see below) the
//   effective ff reduces further each cycle until stable.
//
// If a motor over- or under-shoots consistently after a few runs,
// trim its Kff by ±0.02 and re-run the diagnostic.
float Kff[4] = {0.40,  0.42,  0.38,  0.38};

/* ==================== LOOP TIMING ==================== */
unsigned long lastLoop  = 0;
unsigned long lastPrint = 0;
const unsigned long printPeriod = 100000; // 100 ms

String inputString = "";

/* ==================== ULTRASONIC STATE MACHINE ==================== */
// One sensor is measured at a time; after DONE the next sensor starts.
// Typical full round-trip for 5 sensors: ~5 × (2+10+38000) µs ≈ 190 ms.
// That matches well with the 200 ms print period.

enum UsPhase {
  US_SETTLE,      // pull TRIG low for 2 µs
  US_TRIG_HIGH,   // pulse TRIG high for 10 µs
  US_TRIG_LOW,    // TRIG back low; wait for ECHO to rise
  US_WAIT_HIGH,   // waiting for ECHO rising edge
  US_WAIT_LOW,    // measuring ECHO pulse width
  US_DONE         // compute distance, advance to next sensor
};

UsPhase      usPhase       = US_SETTLE;
int          usSensor      = 0;          // 0..4, current sensor under test
unsigned long usPhaseStart = 0;          // micros() when we entered this phase
unsigned long usEchoStart  = 0;          // micros() when ECHO went HIGH

float distances[5] = {-1, -1, -1, -1, -1};

unsigned long lastUsPrint = 0;
const unsigned long US_PRINT_PERIOD = 200000; // 200 ms

// Timeout limits
const unsigned long US_ECHO_WAIT_TIMEOUT = 5000;   // 5 ms: if ECHO doesn't rise → timeout
const unsigned long US_ECHO_PULSE_TIMEOUT = 38000; // 38 ms: max echo pulse

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
  lastLoop     = micros();
  usPhaseStart = micros();
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

  /* --- Non-blocking ultrasonic state machine ---
     Called every loop() iteration so µs-level timing is accurate.
  */
  updateUltrasonic();

  /* --- PID at fixed 5 ms period --- */
  unsigned long now = micros();
  if (now - lastLoop < Ts) return;
  lastLoop = now;

  float out[4];

  for (int i = 0; i < 4; i++) {

    /* RPM from hall pulses */
    if (now - lastPulseTime[i] > pulseTimeout) rpm[i] = 0;

    if (newPulse[i] && period[i] > 0) {
      noInterrupts();
      unsigned long T = period[i];
      newPulse[i] = false;
      interrupts();
      float raw_rpm = (60e6f) / (pulsesPerRev * (float)T);

      // Layer 2a: absolute sanity cap.
      raw_rpm = min(raw_rpm, MAX_RPM);

      // Layer 2b: rate-of-change plausibility filter.
      // If the new reading would require the motor to jump more than
      // RPM_JUMP_MAX in one pulse interval, it is a noise event.
      // Only active once the motor is spinning (rpm[i] > 5) to allow
      // clean startup from rest.
      // On rejection we keep the previous rpm[i] — this is conservative
      // but safe: the integrator will adjust within one or two cycles.
      if (rpm[i] > 5.0f && raw_rpm > rpm[i] + RPM_JUMP_MAX) {
        // Spurious spike: discard — rpm[i] unchanged
      } else {
        rpm[i] = raw_rpm;
      }
    }

    /* Direction sign from setpoint */
    int spSign = (setpoint[i] > 0) ? 1 : (setpoint[i] < 0) ? -1 : 0;

    /* Pin selection */
    int pwmPin = (i==0) ? PWM_PIN1 : (i==1) ? PWM_PIN2 : (i==2) ? PWM_PIN3 : PWM_PIN4;
    int dirPin = (i==0) ? DIR_PIN1 : (i==1) ? DIR_PIN2 : (i==2) ? DIR_PIN3 : DIR_PIN4;

    /* Zero setpoint → stop */
    if (spSign == 0) {
      integral[i] = 0; lastError[i] = 0; error[i] = 0; out[i] = 0;
      prevPwm[i]  = 0;
      analogWrite(pwmPin, 0);
      continue;
    }

    /* Direction change → soft reset */
    if (spSign != lastSpSign[i]) {
      integral[i] = 0; lastError[i] = 0; out[i] = 0;
      prevPwm[i]  = 0;
      analogWrite(pwmPin, 0);
      lastSpSign[i] = spSign;
      continue;
    }
    lastSpSign[i] = spSign;

    /* PID + Feed-Forward
       ─────────────────────────────────────────────────────────
       Hall sensors are direction-blind — rpm[i] is always a
       positive magnitude.  Multiply by spSign so the error is
       computed correctly in both directions.

       Feed-forward (SIGNED):
         ff = Kff[i] * setpoint[i]     ← includes the sign
         For forward setpoint ff > 0, for reverse ff < 0.
         This keeps out_unsat in the same sign domain as the
         setpoint at steady state, so the direction pin always
         follows the setpoint — never the PID correction.

         IMPORTANT: do NOT use abs(setpoint) here.  abs() would
         make ff always positive.  At a reverse setpoint the PID
         correction goes negative (more backward drive), so the
         ff and PID correction would cancel and the direction
         pin would flip to forward at steady state — causing
         oscillation or runaway.

       Coast-on-overshoot:
         "effort" = spSign × out_unsat  tells us whether the
         controller wants to push in the setpoint direction (> 0)
         or slow down (< 0).
         When effort < 0 we coast (PWM = 0) instead of reversing.
         Reversing while the hall sensor is direction-blind would
         look like increasing forward speed and trigger runaway.

       DZ_KICK is applied only when actively driving (effort ≥ 0)
       so it cannot accidentally drive the motor backward during a
       coast/overshoot correction.                                */

    float signedRpm = (float)spSign * rpm[i];
    error[i]        = setpoint[i] - signedRpm;
    float dt        = Ts / 1e6;
    float dErr      = (error[i] - lastError[i]) / dt;
    lastError[i]    = error[i];

    // SIGNED feed-forward: negative for reverse setpoints.
    float ff        = Kff[i] * setpoint[i];

    float out_unsat = ff + Kp[i]*error[i] + Ki[i]*integral[i] + Kd[i]*dErr;

    // effort > 0: drive in setpoint direction.
    // effort < 0: motor is above setpoint speed → coast.
    float effort = (float)spSign * out_unsat;

    // Direction pin is ALWAYS governed by setpoint sign, never by
    // the PID output sign.  This is the only safe choice with a
    // direction-blind hall sensor.
    digitalWrite(dirPin, (spSign > 0) ? LOW : HIGH);

    if (effort >= 0.0f) {
      // ── Driving in setpoint direction ──
      // Anti-windup: only integrate when output is not saturated.
      if (abs(out_unsat) < PWM_MAX) {
        integral[i] += error[i] * dt;
        integral[i]  = constrain(integral[i], -integralMax, integralMax);
        out[i]       = out_unsat;
      } else {
        out[i] = constrain(out_unsat, -PWM_MAX, PWM_MAX);
      }
      // Dead-zone kick: floor only when actively driving.
      // Layer 3: slew-rate limit — constrain PWM change per cycle.
      // Even a glitch that passes layers 1 & 2 cannot produce a
      // sudden large current step through the motor.
      int pwm_cmd = (int)abs(out[i]);
      pwm_cmd = max(pwm_cmd, DZ_KICK_PWM);
      pwm_cmd = min(pwm_cmd, PWM_MAX);
      pwm_cmd = constrain(pwm_cmd,
                          prevPwm[i] - PWM_SLEW,
                          prevPwm[i] + PWM_SLEW);
      prevPwm[i] = pwm_cmd;
      analogWrite(pwmPin, pwm_cmd);

    } else {
      // ── Coast: motor is above setpoint speed ──
      // Apply zero PWM and let friction/inertia slow the motor.
      //
      // CRITICAL — integrate during coast:
      //   During overshoot, error < 0 (motor above setpoint).
      //   If we freeze the integral here, every drive cycle produces
      //   the same overshoot and the system never converges —
      //   the sustained ±10–20 RPM oscillation observed in testing.
      //
      //   Allowing the integral to accumulate negative error during
      //   coast means Ki × integral decreases each overshoot cycle.
      //   After a few oscillations:
      //     ff + Ki × integral  →  exactly the PWM the motor needs
      //   The overshoot amplitude shrinks to zero and the motor
      //   settles at steady state without oscillation.
      //
      //   At steady state with ff slightly over-calibrated (e.g. 0.40
      //   vs true 0.38 PWM/RPM): the integral converges to a small
      //   negative value that compensates the ff excess automatically.
      integral[i] += error[i] * dt;
      integral[i]  = constrain(integral[i], -integralMax, integralMax);
      out[i]  = 0.0f;
      prevPwm[i] = 0;   // slew restarts from 0 after coast
      analogWrite(pwmPin, 0);
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
  unsigned long p = t - lastTime[i];

  // Layer 1a: fixed debounce floor.
  // Rejects pulses that arrive within MIN_PULSE_PERIOD (6,667 µs) of
  // the previous one — catches fast PWM-switching glitches.
  if (p < MIN_PULSE_PERIOD) return;

  // Layer 1b: adaptive debounce.
  // The fixed floor only catches glitches that arrive within 6.7 ms of
  // the last pulse.  At 60 RPM real pulses are 22 ms apart, so a glitch
  // at t = 8 ms passes the fixed gate and computes 167 RPM — enough to
  // trigger a coast cycle and cause an audible tick.
  //
  // This second gate rejects any pulse that makes the motor appear to
  // have tripled its speed since the last measurement.  Motor inertia
  // cannot produce a 3× speed jump between adjacent pulses.
  //
  //   At 60 RPM (period = 22,222 µs):
  //     threshold = 22,222 / 3 = 7,407 µs  → rejects up to 180 RPM glitches
  //   At 30 RPM (period = 44,444 µs):
  //     threshold = 14,815 µs               → rejects up to  90 RPM glitches
  //
  // During genuine rapid acceleration (e.g. 10 → 30 RPM) the period
  // halves (133,333 → 44,444 µs).  Half < one-third?  No — so fast
  // acceleration is always accepted; only 3× jumps are blocked.
  if (period[i] > 0 && p < period[i] / PULSE_RATIO_GUARD) return;

  period[i]        = p;
  lastTime[i]      = t;
  lastPulseTime[i] = t;
  newPulse[i]      = true;
}

/* ==================== COMMAND PARSER ==================== */
void parseCommand(String cmd) {
  if (!cmd.startsWith("SP#") || cmd.length() != 22) {
    Serial.println("ERR");
    return;
  }
  setpoint[0] = (cmd.charAt(3)=='+') ?  cmd.substring(4,7).toInt()
                                      : -cmd.substring(4,7).toInt();
  setpoint[1] = (cmd.charAt(8)=='+') ?  cmd.substring(9,12).toInt()
                                      : -cmd.substring(9,12).toInt();
  setpoint[2] = (cmd.charAt(13)=='+') ?  cmd.substring(14,17).toInt()
                                       : -cmd.substring(14,17).toInt();
  setpoint[3] = (cmd.charAt(18)=='+') ?  cmd.substring(19,22).toInt()
                                       : -cmd.substring(19,22).toInt();
}

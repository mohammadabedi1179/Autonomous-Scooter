// ============================================================
// 5x SRF05 — Arduino Mega, parallel group firing
// Group A: S1 right-front (22/23), S3 left-side (26/27), S5 rear (30/31)
// Group B: S2 left-front (24/25), S4 right-side (28/29)
// ============================================================
// Technique: simultaneous trigger + multi-pin polling loop
// replaces pulseIn() (single-blocking) with a tight while()
// that watches all echo pins in the group at the same time.
// No interrupts or pin moves required — works on pins 22-31 as-is.
//
// Loop time:  ~100 ms  (groupA 50ms + groupB 50ms)
// Update rate: ~10 Hz  (vs ~4 Hz sequential)
// ============================================================

#define NUM_SENSORS   5
#define MEDIAN_WINDOW 5
#define ECHO_TIMEOUT  30000UL   // µs — covers 4 m round trip
#define GROUP_GAP_MS  50        // ms between group A and group B

// ── pin tables ──────────────────────────────────────────────
const int TRIG[NUM_SENSORS] = {22, 24, 26, 28, 30};
const int ECHO[NUM_SENSORS] = {23, 25, 27, 29, 31};

// Group A: indices 0,2,4  (S1 right-front, S3 left-side, S5 rear)
// Group B: indices 1,3    (S2 left-front,  S4 right-side)
const int GROUP_A[]   = {0, 2, 4};
const int GROUP_B[]   = {1, 3};
const int GROUP_A_LEN = 3;
const int GROUP_B_LEN = 2;

// ── median filter state ─────────────────────────────────────
float buf[NUM_SENSORS][MEDIAN_WINDOW];
int   bufIdx[NUM_SENSORS];
bool  bufFull[NUM_SENSORS];


// ── fire a group simultaneously, poll all echo pins ─────────
// indices[] : sensor indices in this group
// len       : number of sensors in group
// out[]     : raw distance results written here (cm, or -1)
void measureGroup(const int* indices, int len, float* out)
{
    // 1. Pull all triggers LOW first (clean state)
    for (int i = 0; i < len; i++)
        digitalWrite(TRIG[indices[i]], LOW);
    delayMicroseconds(2);

    // 2. Fire all triggers HIGH together
    for (int i = 0; i < len; i++)
        digitalWrite(TRIG[indices[i]], HIGH);
    delayMicroseconds(10);

    // 3. Pull all triggers LOW — sensors now emitting
    for (int i = 0; i < len; i++)
        digitalWrite(TRIG[indices[i]], LOW);

    // 4. Poll all echo pins simultaneously
    unsigned long startTime[len];
    unsigned long duration[len];
    bool started[len];
    bool done[len];

    for (int i = 0; i < len; i++) {
        startTime[i] = 0;
        duration[i]  = 0;
        started[i]   = false;
        done[i]      = false;
    }

    unsigned long deadline = micros() + ECHO_TIMEOUT + 1000UL;

    while (micros() < deadline) {
        bool allDone = true;

        for (int i = 0; i < len; i++) {
            if (done[i]) continue;
            allDone = false;

            int pin = ECHO[indices[i]];
            bool high = digitalRead(pin);

            if (!started[i] && high) {
                startTime[i] = micros();
                started[i]   = true;
            } else if (started[i] && !high) {
                duration[i] = micros() - startTime[i];
                done[i]     = true;
            }
        }

        if (allDone) break;
    }

    // 5. Convert durations to cm
    for (int i = 0; i < len; i++) {
        if (!started[i] || duration[i] == 0)
            out[indices[i]] = -1.0;
        else
            out[indices[i]] = duration[i] * 0.0343 / 2.0;
    }
}


// ── median filter (same as before) ──────────────────────────
float pushMedian(int sensorIdx, float raw)
{
    buf[sensorIdx][bufIdx[sensorIdx]] = raw;
    bufIdx[sensorIdx] = (bufIdx[sensorIdx] + 1) % MEDIAN_WINDOW;
    if (bufIdx[sensorIdx] == 0) bufFull[sensorIdx] = true;

    int filled = bufFull[sensorIdx] ? MEDIAN_WINDOW : bufIdx[sensorIdx];
    if (filled < 3) return raw;

    float valid[MEDIAN_WINDOW];
    int vCount = 0;
    for (int i = 0; i < filled; i++)
        if (buf[sensorIdx][i] >= 0) valid[vCount++] = buf[sensorIdx][i];

    if (vCount == 0) return -1.0;

    // Insertion sort
    for (int i = 1; i < vCount; i++) {
        float key = valid[i]; int j = i - 1;
        while (j >= 0 && valid[j] > key) { valid[j+1] = valid[j]; j--; }
        valid[j+1] = key;
    }
    return valid[vCount / 2];
}


void setup()
{
    Serial.begin(115200);

    for (int i = 0; i < NUM_SENSORS; i++) {
        pinMode(TRIG[i], OUTPUT);
        pinMode(ECHO[i], INPUT);
        digitalWrite(TRIG[i], LOW);
        bufIdx[i]  = 0;
        bufFull[i] = false;
    }

    delay(1000);
}


void loop()
{
    float raw[NUM_SENSORS];
    float filtered[NUM_SENSORS];

    // Fire group A (S1, S3, S5) simultaneously
    measureGroup(GROUP_A, GROUP_A_LEN, raw);
    delay(GROUP_GAP_MS);

    // Fire group B (S2, S4) simultaneously
    measureGroup(GROUP_B, GROUP_B_LEN, raw);

    // Apply median filter to all 5
    for (int i = 0; i < NUM_SENSORS; i++)
        filtered[i] = pushMedian(i, raw[i]);

    // CSV: right_front, left_front, left_side, right_side, rear
    for (int i = 0; i < NUM_SENSORS; i++) {
        Serial.print(filtered[i], 1);
        if (i < NUM_SENSORS - 1) Serial.print(",");
    }
    Serial.println();
}

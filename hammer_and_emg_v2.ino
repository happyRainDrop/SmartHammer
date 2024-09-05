#include <Arduino.h>

bool checklights = true;

// consts
const int HAMMER_SENSOR_PIN = A0;
const int EMG_SENSOR_PIN = A7;
const int SEND_TRIGGER_PIN = 2;
const int DATA_LENGTH = 9000;
const int MAX_DEV = 50;
const int CHECKLIGHT_PIN = 51;
const int NUM_READINGS_PAST_THRESH = 5;

// variables
unsigned int hammer_data[DATA_LENGTH];
unsigned int emg_data[DATA_LENGTH];
bool triggered = false;

// use average to find resting value of sensors.
unsigned long sum1 = 0;
unsigned int numSamples = 0;

unsigned long mytime = 0;

//* digitalWriteFast function: from Github, takes abt 23 ns (instead of 4 us for digitalWrite())
boolean NonConstantUsed(void) __attribute__ (( error("") ));
void LTO_Not_Enabled(void) __attribute__ (( error("")));

#define digitalWriteFast(pin, val)          \
    if (__builtin_constant_p(pin) /*&& __builtin_constant_p(val)*/) {_dwfast(pin, val);} else {NonConstantUsed();}

static inline  __attribute__((always_inline)) void _dwfast(int pin, int val) {
    if (val) {
        digitalPinToPort(pin)->PIO_SODR = digitalPinToBitMask(pin);
    } else {
        digitalPinToPort(pin)->PIO_CODR = digitalPinToBitMask(pin);
    }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(SEND_TRIGGER_PIN, OUTPUT);
  digitalWriteFast(SEND_TRIGGER_PIN, LOW);
  if (checklights) {
    pinMode(CHECKLIGHT_PIN, OUTPUT);
    digitalWriteFast(CHECKLIGHT_PIN, LOW);
  }
}

void print_avgs() {
  Serial.println("== AVERAGE VALS: ===");
  Serial.print(sum1/float(numSamples));
  Serial.println(", ");
}

void transmit() {
  for (int i = 0; i < DATA_LENGTH; i++) {
    Serial.print(i * 1.0 * mytime/DATA_LENGTH * 1/1000.0);
    Serial.print(", ");
    Serial.print(hammer_data[i] * 3.3/1024);
    Serial.print(", ");
    Serial.println(emg_data[i] * 5.0/1024);
  }
  Serial.println("================");

  // digitalWriteFast(SEND_TRIGGER_PIN, LOW);
  // if (checklights) digitalWriteFast(CHECKLIGHT_PIN, LOW);
}

void loop() {
  //*
  // put your main code here, to run repeatedly:
  triggered = true;
  mytime = micros();
  for (int i = 0; i < NUM_READINGS_PAST_THRESH; i++) {
    hammer_data[i] = analogRead(HAMMER_SENSOR_PIN);  // read the input pin
    emg_data[i] = analogRead(EMG_SENSOR_PIN); 
    if (!(numSamples > 5 && abs(hammer_data[i] - sum1/float(numSamples)) > MAX_DEV)) triggered = false;
  }

  if (triggered) {

    /* Used for determining what triggered it
    for (int i = 0; i < NUM_READINGS_PAST_THRESH; i++) {
      Serial.print(hammer_data[i]*3.3/1024);
      Serial.print(" - ");
      Serial.print((3.3/1024)*(sum1/float(numSamples)));
      Serial.print(" = ");
      Serial.print((3.3/1024)*abs(hammer_data[0] - sum1/float(numSamples)));
      Serial.print(" > ");
      Serial.print(MAX_DEV*(3.3/1024));
      Serial.print("; ");
    }
    Serial.println("");
    //*/

 
    digitalWriteFast(SEND_TRIGGER_PIN, HIGH);
    if (checklights) digitalWriteFast(CHECKLIGHT_PIN, HIGH);

    // It's time for us to collect data
    for (int i = NUM_READINGS_PAST_THRESH; i < DATA_LENGTH; i++) {
      hammer_data[i] = analogRead(HAMMER_SENSOR_PIN);  // read the input pin
      emg_data[i] = analogRead(EMG_SENSOR_PIN);  // read the input pin
    }
    mytime = micros() - mytime;

    // We were collecting data, but now we're done
    digitalWriteFast(SEND_TRIGGER_PIN, LOW);
    if (checklights) digitalWriteFast(CHECKLIGHT_PIN, LOW);
    transmit();

    // reset
    memset(hammer_data, 0, DATA_LENGTH);
    memset(emg_data, 0, DATA_LENGTH);
    sum1 = 0;
    mytime = 0;
    numSamples = 0;

  } else {

    // We are calibrating!
    for (int i = 0; i < NUM_READINGS_PAST_THRESH; i++) {
      sum1 += hammer_data[i];
      numSamples += 1;
    }
  }

}

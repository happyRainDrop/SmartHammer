#include <Arduino.h>

// consts
const int HAMMER_SENSOR_PIN = A0;
const int EMG_SENSOR_PIN = A7;
const int SEND_TRIGGER_PIN = 2;
const int DATA_LENGTH = 9000;
const int MAX_DEV = 50;

// variables
unsigned int hammer_data[DATA_LENGTH];
unsigned int emg_data[DATA_LENGTH];

// use average to find resting value of sensors.
unsigned long sum1 = 0;
unsigned int numSamples = 0;

unsigned long mytime = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(SEND_TRIGGER_PIN, OUTPUT);
  digitalWrite(SEND_TRIGGER_PIN, LOW);
  
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
    Serial.println(emg_data[i] * 3.3/1024);
  }
  Serial.println("================");
  digitalWrite(SEND_TRIGGER_PIN, LOW);
}

void loop() {
  //*
  // put your main code here, to run repeatedly:
  hammer_data[0] = analogRead(HAMMER_SENSOR_PIN);  // read the input pin
  emg_data[0] = analogRead(EMG_SENSOR_PIN); 
  mytime = micros();

  if (numSamples > 5 && abs(hammer_data[0] - sum1/float(numSamples)) > MAX_DEV) {
    /*
    Serial.print(hammer_data[0]);
    Serial.print(" - ");
    Serial.print(sum1/float(numSamples));
    Serial.print(" = ");
    Serial.print(abs(hammer_data[0] - sum1/float(numSamples)));
    Serial.print(" > ");
    Serial.println(MAX_DEV);
    //*/

 
    digitalWrite(SEND_TRIGGER_PIN, HIGH);

    // It's time for us to collect data
    for (int i = 1; i < DATA_LENGTH; i++) {
      hammer_data[i] = analogRead(HAMMER_SENSOR_PIN);  // read the input pin
      emg_data[i] = analogRead(EMG_SENSOR_PIN);  // read the input pin
    }
    mytime = micros() - mytime;

    // We were collecting data, but now we're done
    digitalWrite(SEND_TRIGGER_PIN, LOW);
    transmit();

    // reset
    memset(hammer_data, 0, DATA_LENGTH);
    memset(emg_data, 0, DATA_LENGTH);
    sum1 = 0;
    mytime = 0;
    numSamples = 0;

  } else {

    // We are calibrating!
    sum1 += hammer_data[0];
    numSamples += 1;
  }

}
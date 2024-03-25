#include <Arduino.h>

// consts
const int SENSOR_1_PIN = A1;
const int SENSOR_2_PIN = A2;
const int SENSOR_3_PIN = A3;
const int SENSOR_4_PIN = A4;
const int THRESH_1 = 40;
const int THRESH_2 = 500;
const int THRESH_3 = 500;
const int THRESH_4 = 1024;
const int DATA_LENGTH = 400;

// variables
int data_1[DATA_LENGTH];
int data_2[DATA_LENGTH];
int data_3[DATA_LENGTH];
int data_4[DATA_LENGTH];
long times[DATA_LENGTH];

int idx = 0;
unsigned long time = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void transmit() {
  for (int i = 0; i < DATA_LENGTH; i++) {
    Serial.print(times[i]);
    Serial.print(", ");
    Serial.print(data_1[i]);
    Serial.print(", ");
    Serial.print(data_2[i]);
    Serial.print(", ");
    Serial.print(data_3[i]);
    Serial.print(", ");
    Serial.println(data_4[i]);
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  data_1[idx] = analogRead(SENSOR_1_PIN);  // read the input pin
  data_2[idx] = analogRead(SENSOR_2_PIN);  // read the input pin
  data_3[idx] = analogRead(SENSOR_3_PIN);  // read the input pin
  data_4[idx] = analogRead(SENSOR_4_PIN);  // read the input pin
  times[idx] = micros();

  if ((data_1[idx] > THRESH_1 || data_2[idx] > THRESH_2 || data_3[idx] > THRESH_3 || data_4[idx] > THRESH_4) && idx < DATA_LENGTH) {
    // It's time for us to collect data
    idx += 1;
  } else if (idx > 0) {
    // We were collecting data, but now we're done
    transmit();
    // reset
    memset(data_1, 0, DATA_LENGTH);
    memset(data_2, 0, DATA_LENGTH);
    memset(data_3, 0, DATA_LENGTH);
    memset(data_4, 0, DATA_LENGTH);
    idx = 0;
  }
}
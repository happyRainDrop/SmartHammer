#include <Arduino.h>

// consts
const int SENSOR_1_PIN = A0;
const int SENSOR_2_PIN = A1;
const int SENSOR_3_PIN = A2;
const int SENSOR_4_PIN = A3;
const int SEND_TRIGGER_PIN = 2;
const int DATA_LENGTH = 200;
const int MAX_DEV = 100;

// variables
unsigned int data_1[DATA_LENGTH];
unsigned int data_2[DATA_LENGTH];
unsigned int data_3[DATA_LENGTH];
unsigned int data_4[DATA_LENGTH];

// use average to find resting value of sensors.
unsigned long sum1 = 0;
unsigned long sum2 = 0;
unsigned long sum3 = 0;
unsigned long sum4 = 0;
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
  Serial.print(", ");
  Serial.print(sum2/float(numSamples));
  Serial.print(", ");
  Serial.print(sum3/float(numSamples));
  Serial.print(", ");
  Serial.println(sum4/float(numSamples));
}

void transmit() {
  for (int i = 0; i < DATA_LENGTH; i++) {
    Serial.print(i * 1.0 * mytime/DATA_LENGTH * 1/1000.0);
    Serial.print(", ");
    Serial.println(data_1[i] * 3.3/1024);
    /*
    Serial.print(", ");
    Serial.print(data_2[i]  * 3.3/1024);
    Serial.print(", ");
    Serial.print(data_3[i]  * 3.3/1024);
    Serial.print(", ");
    Serial.println(data_4[i]  * 3.3/1024);
    //*/
  }
}

void loop() {
  //*
  // put your main code here, to run repeatedly:
  data_1[0] = analogRead(SENSOR_1_PIN);  // read the input pin
  data_2[0] = 0; // analogRead(SENSOR_2_PIN);  // read the input pin
  data_3[0] = 0; // analogRead(SENSOR_3_PIN);  // read the input pin
  data_4[0] = 0; // analogRead(SENSOR_4_PIN);  // read the input pin
  mytime = micros();

  if (numSamples > 5 && 
      abs(data_1[0] - sum1/float(numSamples)) > MAX_DEV || 
      abs(data_2[0] - sum2/float(numSamples)) > MAX_DEV || 
      abs(data_3[0] - sum3/float(numSamples)) > MAX_DEV || 
      abs(data_4[0] - sum4/float(numSamples)) > MAX_DEV) {
 
    digitalWrite(SEND_TRIGGER_PIN, HIGH);

    // It's time for us to collect data
    for (int i = 1; i < DATA_LENGTH; i++) {
      data_1[i] = analogRead(SENSOR_1_PIN);  // read the input pin
      data_2[i] = analogRead(SENSOR_2_PIN);  // read the input pin
      data_3[i] = analogRead(SENSOR_3_PIN);  // read the input pin
      data_4[i] = analogRead(SENSOR_4_PIN);  // read the input pin
    }
    mytime = micros() - mytime;
    // We were collecting data, but now we're done
    digitalWrite(SEND_TRIGGER_PIN, LOW);
    print_avgs();
    transmit();
    // reset
    memset(data_1, 0, DATA_LENGTH);
    memset(data_2, 0, DATA_LENGTH);
    memset(data_3, 0, DATA_LENGTH);
    memset(data_4, 0, DATA_LENGTH);
    sum1 = 0; sum2 = 0; sum3 = 0; sum4 = 0; numSamples = 0;
    mytime = 0;
  } else {
    // We are calibrating!
    sum1 += data_1[0];
    sum2 += data_2[0];
    sum3 += data_3[0];
    sum4 += data_4[0];
    numSamples += 1;
  }
  //*/

}
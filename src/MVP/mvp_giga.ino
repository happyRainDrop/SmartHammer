// 4-element transmit, recieve, and Wi-Fi code
// Need to modify internal RAM files to work: https://github.com/arduino/ArduinoCore-mbed/pull/995/files
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUDP.h>
#include <GIGA_digitalWriteFast.h>
#include "STMSpeeduino.h"
#include "SDRAM.h"

/////////////////////////////////////////// Transmission side things
bool wired_connection = true;
bool testing_timing = true;
volatile bool turn_on_osc_pin_state = false;
const int TURN_ON_OSC_PIN = 53;
const int MAX_TIMER_CYCLES = 108; // 52*2 kHz timer --> 1 ms passes after 108 cycles
const int MAX_NUM_PULSES = 10; // send a burst of 10 square waves
int numTimerCyclesPassed = -1;
volatile short pendingSaveRequests = 0;

/////////////////////////////////////////// Recieving side things

// Constants - Simulatenous ADC
const int ADC0Channel = A0; //ADC 1 channel -- hammer
const int ADC1Channel = A1; //ADC 2 channel -- cuff1
const int ADC2Channel = A2; //ADC 3 channel -- cuff2
const int ADC3Channel = A3; //ADC 4 channel -- cuff3
const int ADC4Channel = A4; //ADC 5 channel -- EMG
const int Resolution = 16; //8, 10, 12, 14, 16
const double ClockSpeed = 40; //Clock speed in mhz, stable up to 40mhz, may decrease range further
const int SampleTime = 0; //0 to 7
const int Samplenum = 0; //Number of samples, is 1 more, 0 to 1023
bool Differential = 0; //A10 has to be used as input positive on giga r1, A11 as negative

// Constants - WiFi
const char* ssid = "Forest fire";
const char* password = "firewall";
const char* host = "10.83.186.237";  // Laptop IP
const int hostPort = 4210;
const int localPort = 2390;  // Port Arduino uses to send from

// Constants - hammer
const int NUM_READINGS_PAST_THRESH = 100;
const float MAX_DEV = 0.18; // max deviation of analogRead in volts to trigger

// Constants - cuff
// pins
const int CALIBRATION_PIN = 23;
const int CHECKLIGHT_PIN = 51;
// data
const int DATA_LENGTH = 260;
const int DATA_LENGTH_BEFORE_HAMMER = 180;
const int NUM_PULSES_TO_SAVE = 2000; // total # of pulses
const int NUM_PULSES_TO_SAVE_BEFORE_HAMMER = 5;

// Wifi variables
WiFiUDP udp;
WiFiClient client;
String message = ""; // to transmit datas
bool calibrating = false;
bool checklights = true;

// UDP/TCP variables 
// (TCP: linesPerPacket = 100, lenBuffer = 8000)
// (UDP: linesPerPacket = 10, lenBuffer = 4500)
const int lenBuffer = 8000;
const int linesPerPacket = 100;
char packetBuffer[lenBuffer];
int numPacketsSent = 0;
int bufferIndex = 0;
int lineCount = 0;

// Variables - hammer
long hammerTriggeredTime = 0;
long lastHammerFinishedTime = 0;
double running_sum = 0;
unsigned int num_hammer_samples = 0;
int true_data_length = DATA_LENGTH_BEFORE_HAMMER;

// Variables - cuff
int curr_data_it = 0;
int start_data_it = 0;
long mytime = 0;        // myTime used to determine time difference between analogReads
long analogReadStartTime = 0;
bool hammerTriggered = false;
bool stopReading = false;

// Data arrays, all in V (except time, in ms)
// Use uint16_t and uint32_t for clarity
uint16_t* cuff_reciever_1_data; 
uint16_t* cuff_reciever_2_data;
uint16_t* cuff_reciever_3_data;
uint16_t* hammer_data;

// Temp SRAM data arrays, all in V (except time, in ms)
//*
unsigned short cuff_reciever_1_data_SRAM[DATA_LENGTH];  // units: V
unsigned short cuff_reciever_2_data_SRAM[DATA_LENGTH];  // units: V
unsigned short cuff_reciever_3_data_SRAM[DATA_LENGTH];  // units: V
unsigned short hammer_data_SRAM[DATA_LENGTH]; // units: V
unsigned short emg_data[NUM_PULSES_TO_SAVE];  // units: V
unsigned int start_and_end_times[2 * NUM_PULSES_TO_SAVE];   // units: ms
//*/


void setup() {

    ////////////////////////////////////// Setting pins
    if (wired_connection) Serial.begin(115200);
    pinMode(TURN_ON_OSC_PIN, OUTPUT);
    pinMode(CALIBRATION_PIN, INPUT);
    if (checklights) {
      pinMode(CHECKLIGHT_PIN, OUTPUT);
      digitalWriteFast(CHECKLIGHT_PIN, LOW);
    }

    /////////////////////////////////////////// Transmission side things
    //*
    // === Enable clocks ===
    #define RCC_APB1LENR_TIM6EN (1 << 4)
    RCC->APB1LENR |= RCC_APB1LENR_TIM6EN;

    // TIM6: 52 kHz interrupt
    // 240 MHz clock → divide by 10 → 24 MHz timer clock
    TIM6->PSC = 9;           // 240 MHz / 10 = 24 MHz
    TIM6->ARR = 230;         // 24 MHz / 231 ≈ 103.9 kHz → toggles output every ~9.6 μs
    TIM6->DIER |= TIM_DIER_UIE;       // Enable update interrupt
    TIM6->CR1 |= TIM_CR1_CEN;         // Enable counter
    NVIC_EnableIRQ(TIM6_DAC_IRQn);     // Enable IRQ
    //*/

        ////////////////////////////////////// SDRAM allocation
    //*
    SDRAM.begin();
    // Only allocate what will fit (~7 MB total)
    cuff_reciever_1_data = (uint16_t*)SDRAM.malloc(DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    cuff_reciever_2_data = (uint16_t*)SDRAM.malloc(DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    cuff_reciever_3_data = (uint16_t*)SDRAM.malloc(DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    hammer_data = (uint16_t*)SDRAM.malloc(DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));

    SCB_InvalidateDCache_by_Addr(cuff_reciever_1_data, DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    SCB_InvalidateDCache_by_Addr(cuff_reciever_2_data, DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    SCB_InvalidateDCache_by_Addr(cuff_reciever_3_data, DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    SCB_InvalidateDCache_by_Addr(hammer_data, DATA_LENGTH * NUM_PULSES_TO_SAVE * sizeof(uint16_t));
    //*/

    /////////////////////////////////////////// ADCs + Wifi
    // put your setup code here, to run once:
    ADCBegin(ADC1, ADC1Channel, Resolution, Differential, ClockSpeed, SampleTime, Samplenum);
    ADCBegin(ADC2, ADC2Channel, Resolution, Differential, ClockSpeed, SampleTime, Samplenum);
    ResolutionSet(ADC1, Resolution);
    ResolutionSet(ADC2, Resolution);
    // connectWifi();

    delay(2000); // wait
}

/////////////////////////////////////////// Transmit data things

bool sendTCPPacketAndWaitForAck(uint8_t* data, int length, int numPacketsSent) {
  const int MAX_RETRIES = 5;
  int packetNum = numPacketsSent;
  for (int retry = 0; retry < MAX_RETRIES; retry++) {
    if (wired_connection) {
      // Serial.print("📤 Sending packet #");
      // Serial.print(packetNum);
      // Serial.println();
      if (retry > 0) {
        Serial.print(" (retry ");
        Serial.print(retry);
        Serial.print(")");
        Serial.println();
      }
    }

    // Write data
    unsigned long t_start_0 = millis();
    char header[16];
    snprintf(header, sizeof(header), "PACKET%d\n", numPacketsSent);
    client.write((uint8_t *)header, strlen(header));
    client.write((uint8_t *)data, length);
    
    // Timing -- comment out later
    if (testing_timing) {
      Serial.print("   Took ");
      Serial.print(millis() - t_start_0);
      Serial.print(" ms to write data for packet");
      Serial.println(packetNum);
    }

    // Wait for ACK
    unsigned long t_start = millis();
    while (millis() - t_start < 5000) {
      if (!client.connected()) {
        if (wired_connection) Serial.println("⚠️ Client disconnected while waiting for ACK");
        return false;
      }

      if (client.available()) {
        String ack = client.readStringUntil('\n');
        ack.trim();
        String expectedAck = "ACK" + String(packetNum);
        if (ack == expectedAck) {
          // if (wired_connection) Serial.println("✅ " + ack + " received");
          
          // Timing -- comment out later
          if (testing_timing) {
            Serial.print("      Took ");
            Serial.print(millis() - t_start);
            Serial.print(" ms to recieve ACK for packet ");
            Serial.println(packetNum);
          }
          return true;
        } else {
          if (wired_connection) {
            Serial.print("❌ Unexpected ACK: ");
            Serial.println(ack);
          }
        }
      }

      delay(10);
    }

    if (wired_connection) Serial.println("⏱️ ACK wait timed out");
  }
  
  if (wired_connection) {
    Serial.print("❌ Failed to receive ACK for packet ");
    Serial.println(packetNum);
  }
  return false;
}

void transmitOverTCP() {
  /////////////////////////////////////////////////////////////////// WIFI STUFF
  if (WiFi.status() != WL_CONNECTED) connectWifi();

  if (!client.connected()) {
    if (!client.connect(host, hostPort)) {
      if (wired_connection) Serial.println("TCP connection failed.");
      return;
    }
    if (wired_connection) Serial.println("TCP connected to host.");
  }

  /////////////////////////////////////////////////////////////////// CALCULATING DATA
  int num_pulses_to_save = calibrating ? 1 : NUM_PULSES_TO_SAVE;
  float time_to_print = 0;
  float cuff1_voltage_to_print = 0;
  float cuff2_voltage_to_print = 0;
  float cuff3_voltage_to_print = 0;
  float hammer_voltage_to_print = 0;
  float emg_voltage_to_print = 0;

  int i_initial = start_data_it * DATA_LENGTH;
  int arr_len = DATA_LENGTH * num_pulses_to_save;
  int which_it = start_data_it;
  int last_it = -1;
  long it_start = 0;
  long it_end = 0;

  for (int i = i_initial; i - i_initial < arr_len; i++) {
    which_it = (i / DATA_LENGTH) % num_pulses_to_save;

    if (which_it != last_it) {
      it_start = start_and_end_times[which_it * 2];
      it_end = start_and_end_times[which_it * 2 + 1];
    }

    int i_within_it = i % DATA_LENGTH;
    int num_samples_this_period = DATA_LENGTH;
    if (it_start < hammerTriggeredTime) num_samples_this_period = DATA_LENGTH_BEFORE_HAMMER;
    time_to_print = ((i_within_it * 1.0 * (it_end - it_start)) / (num_samples_this_period*1.0)) / 1000.0 +
                    (it_start - hammerTriggeredTime) / 1000.0;

    if (it_start < hammerTriggeredTime && i_within_it >= DATA_LENGTH_BEFORE_HAMMER) {
      // just print last time and voltage at the end of the file
      time_to_print = ((DATA_LENGTH_BEFORE_HAMMER * 1.0 * (it_end - it_start)) / (num_samples_this_period*1.0)) / 1000.0 +
                    (it_start - hammerTriggeredTime) / 1000.0;
    } else {
      cuff1_voltage_to_print = cuff_reciever_1_data[i % arr_len] * 3.3 / pow(2, Resolution);
      cuff2_voltage_to_print = cuff_reciever_2_data[i % arr_len] * 3.3 / pow(2, Resolution);
      cuff3_voltage_to_print = cuff_reciever_3_data[i % arr_len] * 3.3 / pow(2, Resolution);
      emg_voltage_to_print = emg_data[which_it] * 3.3 / pow(2, Resolution);
    }

    if (hammer_data[i % arr_len] != 0)
      hammer_voltage_to_print = hammer_data[i % arr_len] * 3.3 / pow(2, Resolution);
      
    ///////////////////////////////////////////////////////////////////// SENDING DATA

    char line[100];
    snprintf(line, sizeof(line),
             "%.6f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
             time_to_print, cuff1_voltage_to_print, cuff2_voltage_to_print,
             cuff3_voltage_to_print, hammer_voltage_to_print, emg_voltage_to_print);

    int lineLen = strlen(line);
    if (bufferIndex + lineLen < lenBuffer) {
      memcpy(packetBuffer + bufferIndex, line, lineLen);
      bufferIndex += lineLen;
      lineCount++;
    } else {
      if (wired_connection) {
        Serial.print("Error: bufferIndex = ");
        Serial.print(bufferIndex);
        Serial.print(" + lineLen = ");
        Serial.print(lineLen);
        Serial.print(">=");
        Serial.println(lenBuffer);
      }
    }

    if (lineCount == linesPerPacket) {
      if (!sendTCPPacketAndWaitForAck((uint8_t*)packetBuffer, bufferIndex, numPacketsSent)) {
        // Could retry here more globally, or just give up
        if (wired_connection) Serial.println("  Couldn't send packet. Aborting.");
        bufferIndex = 0;
        lineCount = 0;
        client.stop();
        return;
      }
      bufferIndex = 0;
      lineCount = 0;
      numPacketsSent++;
    }

    last_it = which_it;
  }

  if (lineCount > 0) {  // Send the last packet
    client.write((uint8_t *)packetBuffer, bufferIndex);
  }

  // Send end marker
  client.write("============\n");
  client.stop();
  numPacketsSent = 0;
}

bool sendUDPPacketAndWaitForAck(uint8_t* data, int length, int numPacketsSent) {
  const int MAX_RETRIES = 3;
  int packetNum = numPacketsSent;
  for (int retry = 0; retry < MAX_RETRIES; retry++) {
    if (wired_connection) {
      // Serial.print("📤 Sending packet #");
      // Serial.print(packetNum);
      // Serial.println();
      if (retry > 0) {
        Serial.print(" (retry ");
        Serial.print(retry);
        Serial.print(")");
        Serial.println();
      }
    }

    // Write data
    char header[16];          // header
    snprintf(header, sizeof(header), "PACKET%d\n", numPacketsSent);
    udp.beginPacket(host, hostPort);
    udp.write((uint8_t*)header, strlen(header)); // send header
    udp.write((uint8_t*)data, length); // send rest of packet
    udp.endPacket();

    // Wait for ACK
    unsigned long t_start = millis();
    while (millis() - t_start < 5000) {
      int len = udp.parsePacket();
      if (len > 0) {
        char ack[32];
        udp.read(ack, sizeof(ack));
        ack[len] = '\0';
        char expected[16];
        snprintf(expected, sizeof(expected), "ACK%d", packetNum);
        if (strncmp(ack, expected, strlen(expected)) == 0) return true;
      }
      delay(5);
    }

    Serial.println("⏱️ ACK wait timed out");
  }
  
  if (wired_connection) {
    Serial.print("❌ Failed to receive ACK for packet ");
    Serial.println(packetNum);
  }
  return false;
}

void transmitOverUDP() {

  if (WiFi.status() != WL_CONNECTED) connectWifi();
  udp.begin(localPort);

  /////////////////////////////////////////////////////////////////// CALCULATING DATA
  int num_pulses_to_save = calibrating ? 1 : NUM_PULSES_TO_SAVE;
  float time_to_print = 0;
  float cuff1_voltage_to_print = 0;
  float cuff2_voltage_to_print = 0;
  float cuff3_voltage_to_print = 0;
  float hammer_voltage_to_print = 0;
  float emg_voltage_to_print = 0;

  int i_initial = start_data_it * DATA_LENGTH;
  int arr_len = DATA_LENGTH * num_pulses_to_save;
  int which_it = start_data_it;
  int last_it = -1;
  long it_start = 0;
  long it_end = 0;

  for (int i = i_initial; i - i_initial < arr_len; i++) {
    which_it = (i / DATA_LENGTH) % num_pulses_to_save;

    if (which_it != last_it) {
      it_start = start_and_end_times[which_it * 2];
      it_end = start_and_end_times[which_it * 2 + 1];
    }

    int i_within_it = i % DATA_LENGTH;
    int num_samples_this_period = DATA_LENGTH;
    if (it_start < hammerTriggeredTime) num_samples_this_period = DATA_LENGTH_BEFORE_HAMMER;
    time_to_print = ((i_within_it * 1.0 * (it_end - it_start)) / (num_samples_this_period*1.0)) / 1000.0 +
                    (it_start - hammerTriggeredTime) / 1000.0;

    if (it_start < hammerTriggeredTime && i_within_it >= DATA_LENGTH_BEFORE_HAMMER) {
      // just print last time and voltage at the end of the file
      time_to_print = ((DATA_LENGTH_BEFORE_HAMMER * 1.0 * (it_end - it_start)) / (num_samples_this_period*1.0)) / 1000.0 +
                    (it_start - hammerTriggeredTime) / 1000.0;
    } else {
      cuff1_voltage_to_print = cuff_reciever_1_data[i % arr_len] * 3.3 / pow(2, Resolution);
      cuff2_voltage_to_print = cuff_reciever_2_data[i % arr_len] * 3.3 / pow(2, Resolution);
      cuff3_voltage_to_print = cuff_reciever_3_data[i % arr_len] * 3.3 / pow(2, Resolution);
      emg_voltage_to_print = emg_data[which_it] * 3.3 / pow(2, Resolution);
    }

    if (hammer_data[i % arr_len] != 0)
      hammer_voltage_to_print = hammer_data[i % arr_len] * 3.3 / pow(2, Resolution);
      
    ///////////////////////////////////////////////////////////////////// SENDING DATA

    char line[100];
    snprintf(line, sizeof(line),
             "%.6f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
             time_to_print, cuff1_voltage_to_print, cuff2_voltage_to_print,
             cuff3_voltage_to_print, hammer_voltage_to_print, emg_voltage_to_print);

    int lineLen = strlen(line);
    if (bufferIndex + lineLen < lenBuffer) {
      memcpy(packetBuffer + bufferIndex, line, lineLen);
      bufferIndex += lineLen;
      lineCount++;
    } else {
      if (wired_connection) {
        Serial.print("Error: bufferIndex = ");
        Serial.print(bufferIndex);
        Serial.print(" + lineLen = ");
        Serial.print(lineLen);
        Serial.print(">=");
        Serial.println(lenBuffer);
      }
    }

    // Send if 10 lines collected
    if (lineCount == linesPerPacket) {
      if (!sendUDPPacketAndWaitForAck((uint8_t*)packetBuffer, bufferIndex, numPacketsSent)) {
          // Could retry here more globally, or just give up
          if (wired_connection) Serial.println("  Couldn't send packet. Aborting.");
          bufferIndex = 0;
          lineCount = 0;
          return;
      }

      bufferIndex = 0;
      lineCount = 0;
      numPacketsSent++;   
    }
    
    last_it = which_it;
  }

  if (lineCount > 0) {
    udp.beginPacket(host, hostPort);
    udp.write((uint8_t*)packetBuffer, bufferIndex);
    udp.endPacket();
  }

  // Send end marker
  udp.beginPacket(host, hostPort);
  udp.write("============");
  udp.endPacket();
  numPacketsSent = 0;
}

void transmitOverSerial() {
  // Set up output buffers
  const int linesPerPacket = 10;
  char packetBuffer[1024]; // big enough for 10 lines
  int bufferIndex = 0;
  int lineCount = 0;

  // Prepare variables
  int num_pulses_to_save = calibrating ? 1 : NUM_PULSES_TO_SAVE;
  float time_to_print = 0;
  float cuff1_voltage_to_print = 0;
  float cuff2_voltage_to_print = 0;
  float cuff3_voltage_to_print = 0;
  float hammer_voltage_to_print = 0;
  float emg_voltage_to_print = 0;

  // It's time for us to print data
  int i_initial = start_data_it * DATA_LENGTH;
  int arr_len = DATA_LENGTH * num_pulses_to_save;
  int which_it = start_data_it;
  int last_it = -1;
  long it_start = 0;
  long it_end = 1000;
  it_start = 0;
  it_end = 0;
  for (int i = i_initial; i - i_initial < arr_len; i++) {

      // Correct the values in start_and_end_times. for each iteration, only the stop and start times are saved.
      which_it = (i/DATA_LENGTH) % num_pulses_to_save;
       
      // update start and end time when we have a new iteration
      if (which_it != last_it) {
        //if (wired_connection) Serial.print(which_it); if (wired_connection) Serial.println(" ===========================");
        it_start = start_and_end_times[which_it * 2];
        it_end = start_and_end_times[which_it * 2 + 1]; 
      }

      int i_within_it = i % DATA_LENGTH; 
      time_to_print = ((i_within_it*1.0*(it_end-it_start))/(DATA_LENGTH-1.0))/1000.0 
                                + (it_start-hammerTriggeredTime)/1000.0; 
                                // in ms
      cuff1_voltage_to_print = cuff_reciever_1_data[i % arr_len] * 3.3/pow(2, Resolution);
      cuff2_voltage_to_print = cuff_reciever_2_data[i % arr_len] * 3.3/pow(2, Resolution);
      cuff3_voltage_to_print = cuff_reciever_3_data[i % arr_len] * 3.3/pow(2, Resolution);
      emg_voltage_to_print = emg_data[which_it] * 3.3/pow(2, Resolution);

      hammer_data[DATA_LENGTH * NUM_PULSES_TO_SAVE_BEFORE_HAMMER - 1] * 3.3/pow(2, Resolution);
      if (i % arr_len < DATA_LENGTH * NUM_PULSES_TO_SAVE_BEFORE_HAMMER)
        hammer_voltage_to_print = hammer_data[i % arr_len] * 3.3/pow(2, Resolution);

    //////////////////////////////////////// SEND DATA OVER SERIAL
    Serial.print(time_to_print, 6);
    Serial.print(", ");
    Serial.print(cuff1_voltage_to_print, 3);
    Serial.print(", ");
    Serial.print(cuff2_voltage_to_print, 3);
    Serial.print(", ");
    Serial.print(cuff3_voltage_to_print, 3);
    Serial.print(", ");
    Serial.print(hammer_voltage_to_print, 3);
    Serial.print(", ");
    Serial.println(emg_voltage_to_print, 3);
  }

  Serial.println("=====");
}

void connectWifi() {
  if (wired_connection) {
    Serial.print("Attempting to connect to Wifi: ");
    Serial.println(ssid);
  }
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    if (wired_connection) Serial.print(".");
  }
  if (wired_connection) {
    Serial.print("\nWiFi connected! ");
    Serial.print("Signal strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.print(", Arduino IP address: ");
    Serial.println(WiFi.localIP());
  }
}

//////////////////////////////////////////// Collecting reciever data

void saveData() {

    // Read the data into an array, ONLY SRAM ARRAYS HERE
    mytime = micros();         // save the reading timepoint. 0 = time of start of 10 pulses

        // read emg once per cycle
    ADCChangeChannel(ADC2, ADC4Channel);
    emg_data[curr_data_it] = CatchADCValue(ADC2); // read the EMG pin
    ADCChangeChannel(ADC2, ADC2Channel);

    // stream cuff and hammer data
    if (hammerTriggered) true_data_length = DATA_LENGTH;
    else true_data_length = DATA_LENGTH_BEFORE_HAMMER;

    for (int i = 0; i < true_data_length; i++) {
      cuff_reciever_1_data_SRAM[i] = CatchADCValue(ADC1);
      cuff_reciever_2_data_SRAM[i] = CatchADCValue(ADC2);
      ADCChangeChannel(ADC1, ADC3Channel);
      cuff_reciever_3_data_SRAM[i] = CatchADCValue(ADC1);
      ADCChangeChannel(ADC1, ADC1Channel);
  
      if (hammerTriggered == false) {
        ADCChangeChannel(ADC2, ADC0Channel);
        hammer_data_SRAM[i] = CatchADCValue(ADC2); // read the hammer pin
        ADCChangeChannel(ADC2, ADC2Channel);
      } else hammer_data_SRAM[i] = 0;
    } 

    // UPDATE TIMES
    analogReadStartTime = mytime;
    mytime = micros() - mytime;
    start_and_end_times[curr_data_it * 2] = analogReadStartTime;
    start_and_end_times[curr_data_it * 2 + 1] = analogReadStartTime + mytime;  

    // Copy into SRAM
    memcpy(&cuff_reciever_1_data[curr_data_it * DATA_LENGTH], cuff_reciever_1_data_SRAM, DATA_LENGTH * sizeof(uint16_t));
    memcpy(&cuff_reciever_2_data[curr_data_it * DATA_LENGTH], cuff_reciever_2_data_SRAM, DATA_LENGTH * sizeof(uint16_t));
    memcpy(&cuff_reciever_3_data[curr_data_it * DATA_LENGTH], cuff_reciever_3_data_SRAM, DATA_LENGTH * sizeof(uint16_t));
    memcpy(&hammer_data[curr_data_it * DATA_LENGTH], hammer_data_SRAM, DATA_LENGTH * sizeof(uint16_t));

    // Check if the hammer was triggered during this time?
    // We will check this even if the hammer is already triggered, for a constant delay time.
    if (hammerTriggered == false) {
      int aboveThreshInARow = 0;
      
      for (int i = 0; i < DATA_LENGTH_BEFORE_HAMMER; i++) {
        if ((3.3*hammer_data_SRAM[i]/pow(2, Resolution)) > MAX_DEV) aboveThreshInARow += 1;
        else aboveThreshInARow = 0;
        
        // save the time if triggered. The only thing that will actually change the array.
        if (aboveThreshInARow >= NUM_READINGS_PAST_THRESH) {
          unsigned int cuffCycleStartTime = start_and_end_times[curr_data_it * 2];
          unsigned int cuffCycleEndTime = start_and_end_times[curr_data_it * 2 + 1];  
          hammerTriggeredTime = cuffCycleStartTime + i*float(cuffCycleEndTime-cuffCycleStartTime)/DATA_LENGTH_BEFORE_HAMMER;
          hammerTriggered = true;
        }
      }
    }

    // Lastly, update which iteration (block of the array) we'll update next.
    if (calibrating) curr_data_it = 0;
    else curr_data_it = (curr_data_it + 1) % NUM_PULSES_TO_SAVE;

    pendingSaveRequests = 0; // ASSUME IT TOOK LESS TIME TO SAVE DATA THAN FOR NEXT INTERRUPT TO ARRIVE
}

void loop() {
  //*
  // calibrating = (digitalRead(CALIBRATION_PIN) == LOW);
  if (calibrating) {
    // save the time and stop reading
    hammerTriggeredTime = micros();
    start_data_it = 0;
    while (true) {
      if (pendingSaveRequests > 0) {
        saveData();
        break;
      }
    }
    transmitOverTCP();

  } else {

    if (digitalRead(2) == HIGH) { 
      hammerTriggered = true; 
      hammerTriggeredTime = micros();
    }
    if (hammerTriggered) { 

        if (wired_connection) Serial.println("triggered");

        // show on the checklight that we've started reading
        digitalWriteFast(CHECKLIGHT_PIN, HIGH);

        // save some of the readings before the hammer hit
        start_data_it = (curr_data_it - NUM_PULSES_TO_SAVE_BEFORE_HAMMER + NUM_PULSES_TO_SAVE) % NUM_PULSES_TO_SAVE;

        // and reading some more readings after the hammer hit
        for (int i = 0; i < NUM_PULSES_TO_SAVE - NUM_PULSES_TO_SAVE_BEFORE_HAMMER; i++) {
          while (true) {
            if (pendingSaveRequests > 0) {
              saveData();
              break;
            }
          }
        }

        // lastly, print the data.
        if (wired_connection) Serial.println("starting transmit");
        transmitOverTCP();
        if (wired_connection) Serial.println("ending transmit");

        delay(2000); // wait 3 second before checking for another pulse
        pendingSaveRequests = 0;
        curr_data_it = 0;
        hammerTriggered = false;

        // show on the checklight we're ready for another pulse
        digitalWriteFast(CHECKLIGHT_PIN, LOW);

    } else {
      while (true) {
        if (pendingSaveRequests > 0) {
          saveData();
          break;
        }
      }
    }
  }
  //*/
}

/////////////////////////////////////////// Transmitter (burst) signal
// === TIM6 ISR: Toggle pin TURN_ON_OSC at 52*2 kHz ===
extern "C" void TIM6_DAC_IRQHandler(void) {
  if (TIM6->SR & TIM_SR_UIF) {
    TIM6->SR &= ~TIM_SR_UIF;  // Clear interrupt flag

    // Count clock cycles
    numTimerCyclesPassed += 1;
    if (numTimerCyclesPassed > MAX_TIMER_CYCLES) {
      numTimerCyclesPassed = 0;
      pendingSaveRequests += 1;
    }

    // Set digital pin state
    if (numTimerCyclesPassed < 2 * MAX_NUM_PULSES) { // have yet to write 10 pulses
      turn_on_osc_pin_state = !turn_on_osc_pin_state;
      if (turn_on_osc_pin_state) digitalWriteFast(TURN_ON_OSC_PIN, HIGH); 
      else digitalWriteFast(TURN_ON_OSC_PIN, LOW); 
    }

  }
}

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
volatile bool turn_on_osc_pin_state = false;
const int TURN_ON_OSC_PIN = 3;
const int MAX_TIMER_CYCLES = 108; // 52*2 kHz timer --> 1 ms passes after 108 cycles
const int MAX_NUM_PULSES = 10; // send a burst of 10 square waves
int numTimerCyclesPassed = -1;
volatile short pendingSaveRequests = 0;

/////////////////////////////////////////// Recieving side things

// Constants - WiFi
const char* ssid = "Forest fire";
const char* password = "firewall";
const char* host = "192.168.235.237";  // Laptop IP
const int hostPort = 4210;
const int localPort = 2390;  // Port Arduino uses to send from

// Misc variables
WiFiUDP udp;
String message = ""; // to transmit datas
char buffer[50]; // Buffer for float to string conversion
bool calibrating = false;
bool checklights = true;

// Constants - hammer
const int NUM_READINGS_PAST_THRESH = 10;
const float MAX_DEV = 0.75; // max deviation of analogRead in volts to trigger

// Variables - hammer
long hammerTriggeredTime = 0;
double running_sum = 0;
unsigned int num_hammer_samples = 0;

// Variables - cuff
int curr_data_it = 0;
int start_data_it = 0;
long mytime = 0;        // myTime used to determine time difference between analogReads
long analogReadStartTime = 0;
bool hammerTriggered = false;
bool stopReading = false;

// Simulatenous ADC
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

// Constants - cuff
// pins
const int CALIBRATION_PIN = 23;
const int CHECKLIGHT_PIN = 51;
// data
const int DATA_LENGTH = 250;
const int NUM_PULSES_TO_SAVE = 2000; // total # of pulses
const int NUM_PULSES_TO_SAVE_BEFORE_HAMMER = 5;

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
    Serial.println("Begin.");
}

/////////////////////////////////////////// Recieving side things

void transmitOverUDP() {

  if (WiFi.status() != WL_CONNECTED) connectWifi();

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

    //////////////////////////////////////// SEND DATA OVER WIFI
    char line[100]; // individual line buffer
    snprintf(line, sizeof(line),
      "%.6f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
      time_to_print, cuff1_voltage_to_print, cuff2_voltage_to_print, cuff3_voltage_to_print, hammer_voltage_to_print, emg_voltage_to_print
    );

    // Add line to packet buffer
    int lineLen = strlen(line);
    if (bufferIndex + lineLen < sizeof(packetBuffer)) {
      memcpy(packetBuffer + bufferIndex, line, lineLen);
      bufferIndex += lineLen;
      lineCount++;
    }

    // Send if 10 lines collected
    if (lineCount == linesPerPacket) {
      udp.beginPacket(host, hostPort);
      udp.write((uint8_t*)packetBuffer, bufferIndex);
      udp.endPacket();

      bufferIndex = 0;
      lineCount = 0;
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
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    if (wired_connection) Serial.print(".");
  }
  if (wired_connection) Serial.println("\nWiFi connected!");
  udp.begin(localPort);
}

void saveData() {

    // Read the data into an array, ONLY SRAM ARRAYS HERE
    mytime = micros();         // save the reading timepoint. 0 = time of start of 10 pulses

        // read emg once per cycle
    ADCChangeChannel(ADC2, ADC4Channel);
    emg_data[curr_data_it] = CatchADCValue(ADC2); // read the EMG pin
    ADCChangeChannel(ADC2, ADC2Channel);

    // stream cuff and hammer data
    for (int i = 0; i < DATA_LENGTH; i++) {
      cuff_reciever_1_data_SRAM[i] = CatchADCValue(ADC1);
      cuff_reciever_2_data_SRAM[i] = CatchADCValue(ADC2);
      ADCChangeChannel(ADC1, ADC3Channel);
      cuff_reciever_3_data_SRAM[i] = CatchADCValue(ADC1);
      ADCChangeChannel(ADC1, ADC1Channel);
  
      if (hammerTriggered == false) {
        ADCChangeChannel(ADC2, ADC0Channel);
        hammer_data_SRAM[i] = CatchADCValue(ADC2); // read the hammer pin
        ADCChangeChannel(ADC2, ADC2Channel);
      }
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
      int checking_ham_start_it = curr_data_it * DATA_LENGTH;
      int checking_ham_it = 0;
      int checking_exceeds_ham_it = 0;
      for (int i = 0; i < DATA_LENGTH - NUM_READINGS_PAST_THRESH; i++) {
        checking_ham_it = checking_ham_start_it + i;
        boolean this_is_the_trigger_it = true;
        running_sum += hammer_data[checking_ham_it];
        num_hammer_samples += 1;

        // look at the next NUM_READINGS_PAST_THRESH and see if they are all larger than the 
        // running average
        for (int j = 1; j <= NUM_READINGS_PAST_THRESH; j++) {
          checking_exceeds_ham_it = checking_ham_it + j;
          if ((3.3*hammer_data[checking_exceeds_ham_it]/pow(2, Resolution)) - 
          ((3.3*running_sum)/(float(num_hammer_samples)*pow(2, Resolution))) < MAX_DEV) 
            this_is_the_trigger_it = false;
        }

        // save the time if triggered. The only thing that will actually change the array.
        if (this_is_the_trigger_it && !hammerTriggered) {
          hammerTriggered = true;

          unsigned int cuffCycleStartTime = start_and_end_times[curr_data_it * 2];
          unsigned int cuffCycleEndTime = start_and_end_times[curr_data_it * 2 + 1];  
          hammerTriggeredTime = cuffCycleStartTime + float(cuffCycleEndTime - cuffCycleStartTime)/DATA_LENGTH * i;
        }
      }
    }

    // Lastly, update which iteration (block of the array) we'll update next.
    if (calibrating) curr_data_it = 0;
    else curr_data_it = (curr_data_it + 1) % NUM_PULSES_TO_SAVE;

    pendingSaveRequests -= 1;
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
    transmitOverUDP();

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
        transmitOverUDP();
        if (wired_connection) Serial.println("ending transmit");

        // show on the checklight we're done reading
        digitalWriteFast(CHECKLIGHT_PIN, LOW);

        delay(5000); // wait 5 second before checking for another pulse
        pendingSaveRequests = 0;
        hammerTriggered = false;

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

/////////////////////////////////////////// Transmission side things
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

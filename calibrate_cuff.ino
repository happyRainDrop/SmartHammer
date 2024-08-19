bool calibrating = false;

// variables and constants
const int TRIGGER_PIN = 3;
const int TURN_ON_OSC = 2;
const int CUFF_READ_PIN = A0;
const int DATA_LENGTH = 200;
const int DATA_LENGTH_DURING_PULSE = 35;
const int NUM_PULSES_TO_SAVE = 55;
const int NUM_PULSES_TO_SAVE_BEFORE_HAMMER = 5;

// DELAY_LENGTH, NOP_LEN
// 160, 16000 works pretty well (7/12/2024, 52 kHz). write HIGH --> LOW. DATA_LENGTH = 1
// 100, 3860 works pretty well (7/15/2024, 52 kHz). write LOW --> HIGH. DATA_LENGTH = 250
const int DELAY_LEN = 100; // number of microseconds between saveCuffData() calls
const int NOP_LEN = 3860;

float cuff_data[DATA_LENGTH * NUM_PULSES_TO_SAVE];  // units: V
float cuff_times[DATA_LENGTH * NUM_PULSES_TO_SAVE];        // units: ms


int curr_data_it = 0;
int start_data_it = 0;
bool pulseTriggered = false;
bool stopReading = false;

long pulseTriggeredTime = 0;
long startDigitalPulseTime = 0;
long mytime = 0;        // myTime used to determine time difference between analogReads
long mytime_after = 0;
long analogReadStartTime = 0;


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
    pinMode(TURN_ON_OSC, OUTPUT);
    pinMode(TRIGGER_PIN, INPUT);
}

void saveCuffData() {

    // Next, read the data into an array
    mytime = micros();         // save the reading timepoint. 0 = time of start of 10 pulses

    // DIGITAL WRITE PULSE
    digitalWriteFast(TURN_ON_OSC, LOW);
    /*
    for (int i = 0; i < NOP_LEN; i++) {
      __asm__ __volatile__ ("nop\n\t");
    }*/
    for (int i = 0; i < DATA_LENGTH_DURING_PULSE; i++) {
      cuff_data[curr_data_it * DATA_LENGTH + i] = analogRead(CUFF_READ_PIN);  // read the input pin
    }
    digitalWriteFast(TURN_ON_OSC, HIGH);

    for (int i = DATA_LENGTH_DURING_PULSE; i < DATA_LENGTH; i++) {
      cuff_data[curr_data_it * DATA_LENGTH + i] = analogRead(CUFF_READ_PIN);  // read the input pin
    }

    analogReadStartTime = mytime;
    mytime = micros() - mytime;
    cuff_times[curr_data_it * DATA_LENGTH] = analogReadStartTime;
    cuff_times[curr_data_it * DATA_LENGTH + DATA_LENGTH - 1] = analogReadStartTime + mytime;  

    // Lastly, update which iteration (block of the array) we'll update next.
    if (calibrating) curr_data_it = 0;
    else curr_data_it = (curr_data_it + 1) % NUM_PULSES_TO_SAVE;
}

void printCuffData() {
  // adjust cuff_times correctly
      /*
    for (int i = 0; i < DATA_LENGTH; i++) {
      cuff_times[curr_data_it * DATA_LENGTH + i] = i * 1.0 * mytime / (1000.0*DATA_LENGTH) + analogReadStartTime; // in ms
    } */

  int num_pulses_to_save = calibrating ? 1 : NUM_PULSES_TO_SAVE;

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

      // Correct the values in cuff_times. for each iteration, only the stop and start times are saved.
      which_it = (i/DATA_LENGTH) % num_pulses_to_save;
       
      // update start and end time when we have a new iteration
      if (which_it != last_it) {
        // Serial.print(which_it); Serial.println(" ==========");
        it_start = cuff_times[which_it * DATA_LENGTH];
        it_end = cuff_times[which_it * DATA_LENGTH + DATA_LENGTH - 1]; 
      }

      int i_within_it = i % DATA_LENGTH; 
      cuff_times[i % arr_len] = ((i_within_it*1.0*(it_end-it_start))/(DATA_LENGTH-1.0))/1000.0 
                                + (it_start-pulseTriggeredTime)/1000.0; 
                                // in ms
      
      // convert the data to voltages
      cuff_data[i % arr_len] *= 3.3/1024.0;

      // Print values
      Serial.print(cuff_times[i % arr_len], 6); // prints time rel to trigger (time 0)
      Serial.print(", ");
      Serial.println(cuff_data[i % arr_len], 3);

      last_it = which_it;
  }
}

void loop() {
  if (calibrating) {
    // save the time and stop reading
    pulseTriggeredTime = micros();
    start_data_it = 0;
    saveCuffData();
    printCuffData();

  } else {

    pulseTriggered = (digitalRead(TRIGGER_PIN) == HIGH); 
    if (pulseTriggered) { 
        // save the time and stop reading
        pulseTriggeredTime = micros();
        
        // save some of the readings before the hammer hit
        start_data_it = (curr_data_it - NUM_PULSES_TO_SAVE_BEFORE_HAMMER + NUM_PULSES_TO_SAVE) % NUM_PULSES_TO_SAVE;

        // and reading some more readings after the hammer hit
        for (int i = 0; i < NUM_PULSES_TO_SAVE - NUM_PULSES_TO_SAVE_BEFORE_HAMMER; i++) {
          saveCuffData();
          delayMicroseconds(DELAY_LEN);
        }

        // lastly, print the data.
        printCuffData();
        Serial.println("==============");
        delay(1000); // wait 1 second before checking for another pulse

    } else {
      saveCuffData();
      delayMicroseconds(DELAY_LEN);
    }
  }
}

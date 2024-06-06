const byte A_1 = 46;
const byte B_1 = 48;
const byte C_1 = 50;

const byte A_2 = 47;
const byte B_2 = 49;
const byte C_2 = 51;

const byte A_3 = 41;
const byte B_3 = 43;
const byte C_3 = 45;

int state;
long start;
int P = 10;
enum scenarios { INIT,
                 RUN,
                 POWER_CHANGE,
                 ON_OFF,
                 DEMO };
scenarios scenario;


//--------------------------------------------- PARAMETERS TO CHANGE ---------------------------------------------
// First PWM (RUN,POWER_CHANGE & ON_OFF)
float DC = 0.50; // Duty Cycle

// POWER_CHANGE Settings
float DC2 = 0.30; // Duty Cycle
float T_switch = 20;  //s

// ON_OFF: Off delay
float T_off = 300;  //s

// Set scenario
scenarios s = RUN;
//--------------------------------------------- END ---------------------------------------------



void setup() {
  Serial.begin(9600);
  // initialize digital pin LED_BUILTIN as an output
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(A_1, OUTPUT);
  pinMode(A_2, OUTPUT);
  pinMode(A_3, OUTPUT);
  pinMode(B_1, OUTPUT);
  pinMode(B_2, OUTPUT);
  pinMode(B_3, OUTPUT);
  pinMode(C_1, OUTPUT);
  pinMode(C_2, OUTPUT);
  pinMode(C_3, OUTPUT);
  state = 0;
  scenario = INIT;
};


void loop() {
  switch (scenario) {
    case INIT:
      // Serial.println("INIT");
      digitalWrite(B_2, LOW);
      delay(3 * 1000);  // wait 3s before starting scenario
      start = millis();
      scenario = s;
      // Serial.println(s);
      break;

    case RUN:  //Run PWM
      digitalWrite(B_2, HIGH);
      delay(P * DC);
      digitalWrite(B_2, LOW);
      delay(P * (1.0 - DC));    
      break;

    case POWER_CHANGE:  // Run PWM1 for T1s then run PWM2 for
      if (millis() - start < (T_switch * 1000)) {
        
        delay(P * DC);
        digitalWrite(B_2, LOW);
        delay(P * (1.0 - DC));
      } else {
        digitalWrite(B_2, HIGH);
        delay(P * DC2);
        digitalWrite(B_2, LOW);
        delay(P * (1.0 - DC2));
      }
      break;

    case ON_OFF:  // Turn off PWM with a certain duty cycle
      if (millis() - start < (T_switch * 1000)) {
        digitalWrite(B_2, HIGH);
        // digitalWrite(C_3, HIGH);
        // digitalWrite(A_1, HIGH);
        delay(P * DC);
        digitalWrite(B_2, LOW);
        // digitalWrite(C_3, LOW);
        // digitalWrite(A_1, LOW);
        delay(P * (1 - DC));
      } else {
        delay(T_off * 1000);
        start = millis();
      }
      break;
    
    case DEMO: //
      digitalWrite(B_1, HIGH);
      digitalWrite(B_2, HIGH);
      digitalWrite(B_3, HIGH);
      digitalWrite(A_2, HIGH);
      digitalWrite(C_2, HIGH);
      delay(P * DC);
      digitalWrite(B_1, LOW);
      digitalWrite(B_2, LOW);
      digitalWrite(B_3, LOW);
      digitalWrite(A_2, LOW);
      digitalWrite(C_2, LOW);
      delay(P * (1.0 - DC));    
      break;
  }
}

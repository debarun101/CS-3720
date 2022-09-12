// include the library code:
#include <LiquidCrystal.h>
#include <dht.h>
#include <Servo.h>

#define DHT11_PIN A0

// initialize the library with the numbers of the interface pins
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
dht DHT;

// Create a servo object to control a servo
Servo myservo;
int pos = 0; // store the servo position
//temperature = DHT.temperature; //temp initial
//tempprev = temperaturel;

void display(int temperature, int humidity) {
  lcd.clear();
    
  lcd.print("Temp: ");
  lcd.print(temperature);
  lcd.print((char)223);
  lcd.print("C");
 
  lcd.setCursor(0, 1);
  lcd.print("Humidity: ");
  lcd.print(humidity);
  lcd.print("%");
}

void message(int count) {
  lcd.clear();
  lcd.print("Number of People");
  
  lcd.setCursor(0, 1);
  lcd.print(count);
}

void setup() {
  // attaches the servo on pin 9 to the servo object
  myservo.attach(9);
  
  // set up the LCD's number of columns and rows: 
  lcd.begin(16, 2);  
  
  Serial.begin(9600);
  while (!Serial) {
  }
}

int count = 0;
int temperature = 0;
int humidity = 0;

void loop() {
  if (Serial.available() > 0) {
    String inString = Serial.readString();
    Serial.println(inString.c_str());
    count = atoi(inString.c_str());
  }
  
  int chk = DHT.read11(DHT11_PIN);
  
  
  temperature = DHT.temperature;
  //Control Rules: HVAC rate = 10 degC/hr, person rate = 1.78 degC/hr, 
  
  humidity = DHT.humidity;
  
  message(count);
  delay(1000);
  display(temperature, humidity);
  delay(5000);
  
  double stat = count*1.78 + temperature;
  if(stat > 30 ){
    for(pos = 0; pos < 180; pos += 5){
      myservo.write(pos);
      delay(10);
    }
  }
}

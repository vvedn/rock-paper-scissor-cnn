from machine import Pin
import time

led = Pin(2, Pin.OUT)   # most ESP32 boards use GPIO2 for onboard LED

while True:
    led.value(1)
    time.sleep(0.5)
    led.value(0)
    time.sleep(0.5)
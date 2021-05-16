import serial

ser = serial.Serial("COM3",9600)
while True:
    received_data = ser.read()              #read serial port
    data_left = ser.inWaiting()             #check for remaining byte
    received_data += ser.read(data_left)
    print(received_data)

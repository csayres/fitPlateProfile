#!/usr/bin/env python
from telnetlib import Telnet


class MitutoyoConnection(object):
    # telnet connection to terminal server
    host = "10.1.1.35"
    port = 10001
    def __init__(self):
        try:
            self.telnet = Telnet(self.host, self.port)
        except:
            print("Cannot connect to mitutoyo host!! Check connections and try again or use manual entry")
            self.telnet = None

    def queryGauges(self):
        radialFloats = []
        for gaugeNum in range(1,6):
            self.telnet.write("D0%i\r\n"%gaugeNum)
            gaugeOutput = self.telnet.read_until("\r", 1)
            try:
                gauge, val = gaugeOutput.strip().split(":")
                gauge = int(gauge)
                val = float(val)
                assert gauge == gaugeNum
                print("gauge %i: %.4f"%(gauge, val))
            except:
                raise RuntimeError("Failed to parse gauge %i output: %s"%(gaugeNum, gaugeOutput))
            radialFloats.append(val)
        return radialFloats


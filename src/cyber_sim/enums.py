from enum import IntEnum

class Action(IntEnum):
  PASSIVE = 0 # NIST CSF 2.0 Functions Identify + Protect
  ACTIVE = 1 # NIST CSF 2.0 Functions Detect + Respond
  RECOVER = 2 #Recover function

class AttackTarget(IntEnum):
  NONE = 0
  IT = 1
  OT = 2

class Intensity(IntEnum):
  NONE = 0
  LOW = 1
  HIGH = 2
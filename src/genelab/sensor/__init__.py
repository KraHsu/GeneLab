"""Sensor abstractions for cameras, contacts, tactile data, and future streams."""

from genelab.sensor.body_velocity import BodyVelocitySensor, BodyVelocitySensorCfg
from genelab.sensor.sensor import Sensor, SensorCfg

__all__ = [
    "BodyVelocitySensor",
    "BodyVelocitySensorCfg",
    "Sensor",
    "SensorCfg",
]

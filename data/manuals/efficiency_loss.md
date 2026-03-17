# Engine Efficiency Loss Maintenance Manual

## Overview
Engine efficiency loss is indicated by declining efficiency metrics alongside
rising operational demand sensors. Often accompanies HPC degradation.

## Symptoms
- Falling sensor_7 (fan efficiency)
- Falling sensor_12 (bypass ratio)
- Falling sensor_20 (fuel flow ratio)
- Falling sensor_21 (efficiency metric)
- Rising sensor_11 (corrected core speed)

## Recommended Actions by Severity

### CRITICAL (RUL < 30 cycles)
1. Immediate engine removal from service
2. Full efficiency audit of all compressor stages
3. Inspect fuel management system
4. Check for compressor stall indicators
5. Review control system calibration

### WARNING (RUL 30-80 cycles)
1. Increase monitoring frequency to every 5 cycles
2. Perform engine trim check
3. Inspect and clean fuel nozzles
4. Review bleed air system for leaks
5. Check variable stator vane schedule

### HEALTHY (RUL > 80 cycles)
1. Monitor efficiency trends
2. Schedule fuel system inspection at next opportunity
3. Document for fleet trend analysis

## Estimated Repair Time
- Control system recalibration: 2-4 hours
- Fuel system maintenance: 6-12 hours
# HPC Degradation Maintenance Manual

## Overview
High Pressure Compressor (HPC) degradation is the most common fault mode in turbofan engines.
It is characterized by rising outlet temperatures and increasing fan speeds.

## Symptoms
- Rising sensor_3 (LPC outlet temperature)
- Rising sensor_4 (HPC outlet temperature)
- Rising sensor_9 (physical fan speed)
- Rising sensor_14 (corrected fan speed)
- Falling sensor_7 (fan efficiency)

## Recommended Actions by Severity

### CRITICAL (RUL < 30 cycles)
1. Ground the engine immediately
2. Schedule emergency HPC blade inspection
3. Check for blade tip clearance violations
4. Inspect compressor seals for leakage
5. Perform borescope inspection of HPC stages

### WARNING (RUL 30-80 cycles)
1. Schedule HPC inspection within next 20 cycles
2. Monitor sensor_3 and sensor_4 trends closely
3. Perform water wash to remove deposits
4. Check compressor inlet for FOD damage
5. Review fuel nozzle condition

### HEALTHY (RUL > 80 cycles)
1. Continue normal monitoring
2. Schedule routine borescope at next maintenance window
3. Log degradation trend for fleet analysis

## Estimated Repair Time
- Emergency inspection: 4-8 hours
- Full HPC replacement: 48-72 hours
**Author:** Kelan Solomon  
**Date:** 06/06/2024  
**Contact:** kelan.solomon@epfl.ch

# Creating Patterned Heat with a Micro Heater Array for Thermal Vat Polymerization - SOFTWARE

This code is modified from the Seek Camera SDK: [Seek Camera Python SDK](https://github.com/seekthermal/seekcamera-python)

**N.B:** The measurements were taken with the wrong emissivity. The values are therefore not exact. This has since been corrected in the software (seekcamera.py, line 85). This emissivity is for the Al₂O₃ substrate, so it is specific to the micro heaters.

## File architecture
- **Analysis.ipynb**: Python notebook for graphing (KS)
- **functions.py**: Functions used in the graphs above (KS)
- **SeekCamera.py**: Code used to run the camera (Seek - modified by KS)
- **Record_Data.ipynb**: Separate notebook for recording new data (KS)
- **thermography-E5D608D31715.csv**: CSV with last recorded data
- **Microheaters_020524.ino**: Arduino code

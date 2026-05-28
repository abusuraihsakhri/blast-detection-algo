@echo off
title Pathology AI - Warm Start Training (Resume)
color 0B
echo.
echo  =====================================================
echo       PATHOLOGY AI: RESUME WARM-START TRAINING
echo  =====================================================
echo.
echo  This will resume your most recent YOLO training run directly from
echo  your last saved checkpoint, recovering your previous progress!
echo.
echo  Press any key to confirm and RESUME training immediately...
pause > nul

echo.
echo Starting training pipeline in RESUME mode...
cd /d "%~dp0"
python training_pipeline.py --mode warm --resume

echo.
echo Training complete! Press any key to exit.
pause > nul

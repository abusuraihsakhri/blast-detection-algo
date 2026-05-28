@echo off
title Pathology AI - Warm Start Training
color 0B
echo.
echo  =====================================================
echo       PATHOLOGY AI: FULL WARM-START TRAINING
echo  =====================================================
echo.
echo  This will merge all historical images into a unified dataset
echo  and train the YOLO model across all 10 classes (Lymphoid + Myeloid).
echo.
echo  WARNING: This may take 1-2 hours depending on your RTX 3060 load.
echo.
echo  Press any key to confirm and START training immediately...
pause > nul

echo.
echo Starting training pipeline...
cd /d "%~dp0"
python training_pipeline.py --mode warm

echo.
echo Training complete! Press any key to exit.
pause > nul

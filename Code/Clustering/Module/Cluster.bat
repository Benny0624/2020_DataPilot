@echo off  
C:  
set root= "C:\Users\User\anaconda3"
call %root%\Scripts\activate.bat %root%
cd C:\Users\User\Desktop\2020_DataPilot\Code
start python Clustering_Module.py  
exit 

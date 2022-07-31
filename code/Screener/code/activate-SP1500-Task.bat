@echo off

rem This file is UTF-8 encoded, so we need to update the current code page while executing it
for /f "tokens=2 delims=:." %%a in ('"%SystemRoot%\System32\chcp.com"') do (
    set _OLD_CODEPAGE=%%a
)
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" 65001 > nul
)

set VIRTUAL_ENV=c:\Users\User\Documents\wiki\wiki\dev\3.9-JupyterLab

if not defined PROMPT set PROMPT=$P$G

if defined _OLD_VIRTUAL_PROMPT set PROMPT=%_OLD_VIRTUAL_PROMPT%
if defined _OLD_VIRTUAL_PYTHONHOME set PYTHONHOME=%_OLD_VIRTUAL_PYTHONHOME%

set _OLD_VIRTUAL_PROMPT=%PROMPT%
set PROMPT=(3.9-JupyterLab) %PROMPT%

if defined PYTHONHOME set _OLD_VIRTUAL_PYTHONHOME=%PYTHONHOME%
set PYTHONHOME=

if defined _OLD_VIRTUAL_PATH set PATH=%_OLD_VIRTUAL_PATH%
if not defined _OLD_VIRTUAL_PATH set _OLD_VIRTUAL_PATH=%PATH%

set PATH=%VIRTUAL_ENV%\Scripts;%PATH%

:END
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" %_OLD_CODEPAGE% > nul
    set _OLD_CODEPAGE=
)

cd C:\Users\User\Documents\wiki\wiki\dev\python\Python-Stock\code\Screener\code
C:\Users\User\AppData\Local\Programs\3.9-JupyterLab\Scripts\ipython.exe --TerminalIPythonApp.file_to_run=SP1500.ipynb
xcopy ..\data\processed\*.csv "P:\.shortcut-targets-by-id\1UvGLUJlzZfE8ZzfMw0dedjA1v9se4Tlq\STOCK DATA (JOSH AND RICK)\REPORTS\" /Y
xcopy ..\reports\figures\*.png "P:\.shortcut-targets-by-id\1UvGLUJlzZfE8ZzfMw0dedjA1v9se4Tlq\STOCK DATA (JOSH AND RICK)\FIGURES\" /Y

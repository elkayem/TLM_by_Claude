"""
Put a "TLM Mixtape" shortcut on the Windows desktop.

Run it with the Python of the venv the app should use, from the tlm/
project root:

    %USERPROFILE%\\tlm-venv\\Scripts\\python.exe tools\\make_shortcut.py

The shortcut starts that venv's pythonw.exe (pythonw = no console window)
with "-m tlm.gui" in this repo folder, and shows the cassette icon. It
asks Windows where the desktop is, because it isn't always ~/Desktop
(OneDrive, for one, moves it to ~/OneDrive/Desktop). Re-run it if the
repo or the venv moves.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON = os.path.join(REPO, "assets", "mixtape.ico")


def ps_quote(s):
    return "'" + s.replace("'", "''") + "'"   # a PowerShell string literal


def main():
    if os.name != "nt":
        raise SystemExit("make_shortcut.py makes a Windows .lnk shortcut; "
                         "elsewhere, run: python -m tlm.gui")
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(pythonw):
        raise SystemExit(f"no pythonw.exe next to {sys.executable}")
    if not os.path.exists(ICON):
        subprocess.run([sys.executable, os.path.join(REPO, "tools",
                                                     "make_icon.py")],
                       check=True)
    # Windows shortcuts are made through a COM object; PowerShell is the
    # easiest way to reach it without extra Python packages.
    script = f"""
$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'TLM Mixtape.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
$s.TargetPath = {ps_quote(pythonw)}
$s.Arguments = '-m tlm.gui'
$s.WorkingDirectory = {ps_quote(REPO)}
$s.IconLocation = {ps_quote(ICON + ",0")}
$s.Description = 'TLM Mixtape: play with the Stage 4 language model'
$s.Save()
Write-Output $link
"""
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                             "-Command", script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"PowerShell failed:\n{result.stderr}")
    print(f"created {result.stdout.strip()}")
    print(f"  runs: {pythonw} -m tlm.gui\n  in:   {REPO}")


if __name__ == "__main__":
    main()

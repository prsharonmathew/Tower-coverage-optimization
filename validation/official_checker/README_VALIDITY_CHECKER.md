# Validity Checker

Usage:
Linux:    ./validity_checker cities.txt solution.txt
Windows:  validity_checker.exe cities_de_50k.txt best_selected_towers.txt

Input format:
Cities:    name, latitude, longitude
Solution:  latitude, longitude, radius

Output:sssss
- OK: all cities are covered
- NOT_FEASIBLE: prints one uncovered city
- INPUT_ERROR: if format is invalid
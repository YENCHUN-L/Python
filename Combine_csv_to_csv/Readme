# Combine CSV to CSV

## Overview
This script combines multiple CSV files from one folder into a single output file.
It processes files one by one, which helps reduce memory usage compared with loading all files at once.

## File
- Script: combine_csv_to_csv.py
- Output: combined_csv.csv

## How It Works
1. Sets the working directory using os.chdir("/mydir").
2. Finds all CSV files in that directory using glob with pattern *.csv.
3. Loops through each file:
	 - Reads the file with pandas.read_csv().
	 - Appends rows to combined_csv.csv using mode='a'.
	 - Deletes the temporary DataFrame to keep memory usage lower.

## Requirements
- Python 3.x
- pandas

Install dependency:

```bash
pip install pandas
```

## Usage
1. Open combine_csv_to_csv.py.
2. Update this line to your real folder path:

```python
os.chdir("/mydir")
```

3. Place all source CSV files in that folder.
4. Run the script:

```bash
python combine_csv_to_csv.py
```

5. Check the generated file combined_csv.csv in the same folder.

## Notes and Limitations
- The script appends every file using mode='a'.
- The current code writes header rows each time it appends a file. If all input files have headers, the output may contain repeated header lines.
- The script assumes input CSV files are compatible (same columns/order) for a clean combined result.
- Encoding is set to utf-8-sig for output, which is useful when opening in spreadsheet tools.

## Why Use mode='a' Instead of "append" in Pandas?
- In pandas, DataFrame.to_csv() does not have an append=True parameter.
- Appending to a CSV file is controlled by mode='a' (append mode), which is the standard file-write behavior in Python.
- This script writes one input file at a time to the output file, so it avoids building one huge DataFrame in memory.

Important distinction:
- DataFrame.append() (or _append) is about appending rows to another DataFrame in memory, not appending directly to a CSV file on disk.
- DataFrame.append() has been deprecated/removed in modern pandas versions; pandas recommends pandas.concat() instead.
- Even with concat(), combining many large files in memory first can use much more RAM.

So for this use case (many CSV files, memory-friendly processing), writing each chunk with to_csv(..., mode='a') is the practical approach.

## Recommended Improvement
To avoid repeated headers, write the header only once (first file) and append the rest without header.

Example approach:

```python
for i, f in enumerate(all_filenames):
		df = pd.read_csv(f)
		df.to_csv(
				"combined_csv.csv",
				index=False,
				encoding="utf-8-sig",
				mode="a",
				header=(i == 0)
		)
```

## Troubleshooting
- No output file created:
	- Verify os.chdir("/mydir") points to a valid folder.
	- Verify that folder contains .csv files.
- Permission error:
	- Close combined_csv.csv if it is open in Excel or another editor.
- Wrong character display:
	- Keep encoding='utf-8-sig' and open the file with UTF-8 support.

## Summary
This script is a simple, memory-friendly way to combine many CSV files from one directory into one CSV output.


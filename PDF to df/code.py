from tabula import read_pdf

# Getting area range base on page 1
tables = read_pdf(filename, output_format="json", pages=1, silent=True)
top = tables[0]["top"]
left = tables[0]["left"]
bottom = 10000
right = tables[0]["width"] + left

# Adjust area selection:
test_area = [top - 0, left - 5, bottom, right - 740]

# Read_pdf:
df1 = read_pdf(
 filename,
 pages=1,
 area = test_area
 )[0]

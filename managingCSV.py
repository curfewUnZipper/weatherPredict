import csv

# Opeing the file for reading using the file handle f
f = open('data.csv', 'r')

# Instantiating the csv reader object using the file handle
csvreader = csv.reader(f)

# Empty lists
headings = [] # For heading
rows = []  # Foe data

#Reading the heading using next() function
headings=next(csvreader)
print(headings)

# Iterating through the csv reader object to store data in the list
for r in csvreader:
    rows.append(r)

# Displaying the contents of the list 
print(rows[6])

import csv
students = []
import os

csv_path = 'student_marks.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File '{csv_path}' not found. Please make sure it exists in the script directory.")

with open(csv_path) as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        marks = [int(row[subject]) for subject in row if subject not in ['name', 'gender', 'rollno']]
        total = sum(marks)
        average = total / len(marks)
        row['total_marks'] = total
        row['Average'] = round(average, 2)
        students.append(row)

fieldnames = list(students[0].keys())
with open('student_marks_with_totals.csv', 'w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(students)

print("New file 'student_marks_with_totals.csv' created with total and average marks.")
import csv
print("\nThe most general hypothesis : ['?','?','?','?','?','?']\n")
print("\nThe most specific hypothesis : ['0','0','0','0','0','0']\n")
with open('datasets/enjoysport.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    a = []
    for row in reader:
        a.append(row)
        print(row)
num_attributes = len(a[0]) - 1
hypothesis = ['0'] * num_attributes
print('Initial Value of the hypothesis:')
print(hypothesis)
for j in range(num_attributes):
    hypothesis[j] = a[1][j]
print('\nFind S: Finding Maximally specific hypothesis\n')
for i in range(1, len(a)):
    if a[i][num_attributes] == 'yes':
        for j in range(num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]
        print(f"For training example no : {i} the hypothesis is", hypothesis)

print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypothesis)

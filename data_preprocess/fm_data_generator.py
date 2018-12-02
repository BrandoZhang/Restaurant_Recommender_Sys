# Credit: https://pythonprogramming.net/reading-csv-files-python-3/
import csv
import hashlib

result = list()

with open('yelp_academic_dataset_review.csv', 'r') as data:
    readCSV = csv.reader(data, delimiter=',')

    skip = True  # skip the first line
    for row in readCSV:
        if skip:
            skip = False
            continue
        try:
            result.append('{}::{}::{}\n'.format(int(hashlib.sha256(row[8].encode("utf-8")).hexdigest(),32) % (10 ** 9), int(hashlib.sha256(row[0].encode("utf-8")).hexdigest(),32) % (10 ** 9), row[5]))
        except:
            print(row)

with open('fm_data_int.dat', 'w') as fm_data:
    fm_data.writelines(result)

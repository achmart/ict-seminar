def clean_data(filename):
    """ Opens the data file, deletes all entries,
    that do not belong to class cp or im, and returns
    a list of lists with the respective entries"""
    import re
    with open(filename) as input_file:
        raw_data = input_file.readlines()
        data = []
        for line in raw_data:
            if line[-3:-1] == "cp" or line[-3:-1] == "im":
                data.append(re.sub(r" +"," ",line).split())
    return data
import csv

map2 = {}
def readNodes (G , path):
    with open(path,'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            G.add_node(row[0])
def readEdges (G , path):
    w = 1
    with open(path , 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            G.add_edge(row[0], row[1])
            if len(row) > 2:
                if row[2] in map2:
                    G[row[0]][row[1]]['weight'] = map2[row[2]]
                else:
                    map2[row[2]] = w
                    w += 1
                    G[row[0]][row[1]]['weight'] = map2[row[2]]
            else:
                G[row[0]][row[1]]['weight'] = 1
    print(map2)

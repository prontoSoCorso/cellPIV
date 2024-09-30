import sqlite3
import os


def writeTofile(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)


pdb_file = "/Users/imacvega/Desktop/D2018.11.22_S02141_I0141_D18553.pdb"

pdb_name = pdb_file.split("/")[-1].split(".pdb")[0]

dir_out = "/Users/imacvega/Desktop/PDB/PDB_equatore/images/"
print(pdb_name)

con = sqlite3.connect(pdb_file)
cur = con.cursor()
res = cur.execute("SELECT * FROM IMAGES")
images = res.fetchall()


well_list = []
for row in images:
	well_list.append(row[0])
well_list = list(set(well_list))

for well in well_list:
	os.mkdir(dir_out+str(well))


sep = "_"
for row in images:
	if row[2] == 0:
		id_im = pdb_name+sep+str(row[0])+sep+str(row[1])+sep+str(row[2])+sep+str(row[3])
	            
		resumeFile = row[4]

		photoPath = dir_out + str(row[0]) + "/" + id_im + ".jpg"
		writeTofile(resumeFile, photoPath)

cur.close()
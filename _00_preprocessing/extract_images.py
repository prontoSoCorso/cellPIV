import sqlite3
import os
import argparse

"""
Extract frames from PDB files
"""
def writeTofile(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embryo frames from PDB files')
    parser.add_argument('--input_dir', dest='input_dir',
                        type=str,
                        help='input directory with PDB files')
    parser.add_argument('--output_dir', dest='output_dir',
                        type=str,
                        help='output directory')
    args = parser.parse_args()

    dir_out = args.output_dir

    pdb2number_p = {}
    sep = "_"
    try:
        # the first layer is the "year" folder
        for year in os.listdir(args.input_dir):
            if os.path.isdir(args.input_dir+year):
                # enter and look inside each subfolder
                for subfolder in os.listdir(args.input_dir+year):
                    for file in os.listdir(args.input_dir+year+'/'+subfolder):
                        print(file)
                        if os.path.isdir(file):
                            print(args.input_dir+year+'/'+subfolder+'/'+file+' is a directory')
                        else: # it's a file
                            # check if it is PDB
                            if file.endswith('.pdb'):
                                pdb_file = args.input_dir+year+'/'+subfolder+'/'+file
                                pdb_name = pdb_file.split("/")[-1].split(".pdb")[0]
                                print(pdb_name)

                                con = sqlite3.connect(pdb_file)
                                cur = con.cursor()
                                res = cur.execute("SELECT * FROM IMAGES")
                                images = res.fetchall()

                                well_list = []
                                for row in images:
                                    well_list.append(row[0])
                                well_list = list(set(well_list))

                                if pdb_name in pdb2number_p.keys():
                                    print(pdb_name)
                                else:
                                    pdb2number_p[pdb_name] = len(well_list)
                                # each folder -> 1 embryo (max 12 for each PDB)
                                for well in well_list:
                                    os.mkdir(dir_out + str(pdb_name)+ sep + str(well))

                                
                                for row in images:
                                    id_im = pdb_name + sep + str(row[0]) + sep + str(row[1]) + sep + str(
                                        row[2]) + sep + str(row[3])

                                    resumeFile = row[4]

                                    photoPath = dir_out + str(pdb_name) + sep + str(row[0]) + "/" + id_im + ".jpg"
                                    writeTofile(resumeFile, photoPath)

                                cur.close()
            else:
                print(year+' not a folder')

    except Exception as e:
        print(e)

